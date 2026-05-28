from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import fallback_backend
from .mesh_utils import MeshStats
from .plots import plot_topdown_layout, plot_worst_case_heatmap


@dataclass(frozen=True)
class OptimizationResult:
    backend: str
    candidates_m: np.ndarray
    receivers: list[dict]
    path_loss_db: np.ndarray
    rx_power_dbm: np.ndarray
    worst_case_path_loss_db: np.ndarray
    mean_path_loss_db: np.ndarray
    optimal_index: int
    backend_details: dict

    @property
    def optimal_position_m(self) -> np.ndarray:
        return self.candidates_m[self.optimal_index]


def generate_candidate_grid(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_size: tuple[int, int] = (10, 10),
    z_m: float | None = None,
    margin_m: float = 0.15,
) -> np.ndarray:
    nx, ny = grid_size
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_size values must be positive")
    bounds_min = np.asarray(bounds_min, dtype=float)
    bounds_max = np.asarray(bounds_max, dtype=float)
    if z_m is None:
        z_m = float(bounds_min[2] + 0.80 * (bounds_max[2] - bounds_min[2]))
    x0 = float(bounds_min[0] + margin_m)
    x1 = float(bounds_max[0] - margin_m)
    y0 = float(bounds_min[1] + margin_m)
    y1 = float(bounds_max[1] - margin_m)
    if x1 <= x0 or y1 <= y0:
        x0, x1 = float(bounds_min[0]), float(bounds_max[0])
        y0, y1 = float(bounds_min[1]), float(bounds_max[1])
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, float(z_m), dtype=float)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def optimize_placement(
    mesh_path: Path,
    candidates_m: np.ndarray,
    receivers: list[dict],
    frequency_hz: float,
    tx_power_dbm: float,
    backend: str = "auto",
    fallback_obstruction_loss_db: float = 9.0,
    sionna_options: dict | None = None,
) -> OptimizationResult:
    if not receivers:
        raise ValueError("At least one receiver position is required")
    candidates = np.asarray(candidates_m, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidates_m must be an Nx3 array")

    selected_backend = backend
    details: dict
    if backend in {"auto", "sionna"}:
        try:
            from . import sionna_backend

            if sionna_backend.available():
                selected_backend = "sionna"
                path_loss_db, rx_power_dbm, details = sionna_backend.evaluate_candidates(
                    mesh_path=mesh_path,
                    candidates_m=candidates,
                    receivers=receivers,
                    frequency_hz=frequency_hz,
                    tx_power_dbm=tx_power_dbm,
                    **(sionna_options or {}),
                )
            elif backend == "sionna":
                raise RuntimeError("Sionna RT is not installed or cannot be imported")
            else:
                selected_backend = "fallback"
                path_loss_db, rx_power_dbm, details = fallback_backend.evaluate_candidates(
                    mesh_path, candidates, receivers, frequency_hz, tx_power_dbm, fallback_obstruction_loss_db
                )
        except Exception:
            if backend == "sionna":
                raise
            selected_backend = "fallback"
            path_loss_db, rx_power_dbm, details = fallback_backend.evaluate_candidates(
                mesh_path, candidates, receivers, frequency_hz, tx_power_dbm, fallback_obstruction_loss_db
            )
    elif backend == "fallback":
        path_loss_db, rx_power_dbm, details = fallback_backend.evaluate_candidates(
            mesh_path, candidates, receivers, frequency_hz, tx_power_dbm, fallback_obstruction_loss_db
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    worst_case = np.max(path_loss_db, axis=1)
    mean_loss = np.mean(path_loss_db, axis=1)
    optimal_index = int(np.argmin(worst_case))
    return OptimizationResult(
        backend=selected_backend,
        candidates_m=candidates,
        receivers=receivers,
        path_loss_db=path_loss_db,
        rx_power_dbm=rx_power_dbm,
        worst_case_path_loss_db=worst_case,
        mean_path_loss_db=mean_loss,
        optimal_index=optimal_index,
        backend_details=details,
    )


def write_outputs(
    out_dir: Path,
    result: OptimizationResult,
    mesh_stats: MeshStats,
    frequency_hz: float,
    tx_power_dbm: float,
    grid_size: tuple[int, int] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rx_names = [receiver["name"] for receiver in result.receivers]

    csv_path = out_dir / "candidate_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["candidate_index", "x_m", "y_m", "z_m"]
            + [f"path_loss_{name}_db" for name in rx_names]
            + [f"rx_power_{name}_dbm" for name in rx_names]
            + ["worst_case_path_loss_db", "mean_path_loss_db"]
        )
        for idx, tx_pos in enumerate(result.candidates_m):
            writer.writerow(
                [idx, *[f"{value:.6f}" for value in tx_pos]]
                + [f"{value:.6f}" for value in result.path_loss_db[idx]]
                + [f"{value:.6f}" for value in result.rx_power_dbm[idx]]
                + [
                    f"{result.worst_case_path_loss_db[idx]:.6f}",
                    f"{result.mean_path_loss_db[idx]:.6f}",
                ]
            )

    optimal_loss_by_receiver = {
        rx_names[i]: float(result.path_loss_db[result.optimal_index, i]) for i in range(len(rx_names))
    }
    optimal_power_by_receiver = {
        rx_names[i]: float(result.rx_power_dbm[result.optimal_index, i]) for i in range(len(rx_names))
    }
    summary = {
        "backend": result.backend,
        "backend_details": result.backend_details,
        "frequency_hz": frequency_hz,
        "tx_power_dbm": tx_power_dbm,
        "mesh": mesh_stats.to_dict(),
        "receivers": result.receivers,
        "candidate_count": int(len(result.candidates_m)),
        "grid_size": None if grid_size is None else list(grid_size),
        "optimal": {
            "candidate_index": result.optimal_index,
            "position_m": result.optimal_position_m.astype(float).tolist(),
            "worst_case_path_loss_db": float(result.worst_case_path_loss_db[result.optimal_index]),
            "mean_path_loss_db": float(result.mean_path_loss_db[result.optimal_index]),
            "path_loss_by_receiver_db": optimal_loss_by_receiver,
            "rx_power_by_receiver_dbm": optimal_power_by_receiver,
        },
        "candidates": [
            {
                "candidate_index": int(idx),
                "position_m": tx_pos.astype(float).tolist(),
                "worst_case_path_loss_db": float(result.worst_case_path_loss_db[idx]),
                "mean_path_loss_db": float(result.mean_path_loss_db[idx]),
            }
            for idx, tx_pos in enumerate(result.candidates_m)
        ],
    }
    with (out_dir / "optimization_result.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_worst_case_heatmap(
        candidates_m=result.candidates_m,
        values=result.worst_case_path_loss_db,
        optimal_index=result.optimal_index,
        receivers=result.receivers,
        out_path=out_dir / "worst_case_path_loss_heatmap.png",
        grid_size=grid_size,
    )
    plot_topdown_layout(
        mesh_path=Path(mesh_stats.optimizer_mesh),
        candidates_m=result.candidates_m,
        receivers=result.receivers,
        optimal_index=result.optimal_index,
        out_path=out_dir / "topdown_layout.png",
    )
