from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .mesh_utils import load_trimesh


def plot_worst_case_heatmap(
    candidates_m: np.ndarray,
    values: np.ndarray,
    optimal_index: int,
    receivers: list[dict],
    out_path: Path,
    grid_size: tuple[int, int] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=180)
    candidates = np.asarray(candidates_m, dtype=float)
    values = np.asarray(values, dtype=float)
    if grid_size is not None and grid_size[0] * grid_size[1] == len(values):
        nx, ny = grid_size
        x_values = np.unique(candidates[:, 0])
        y_values = np.unique(candidates[:, 1])
        image = values.reshape(ny, nx)
        im = ax.imshow(
            image,
            origin="lower",
            extent=[float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max())],
            interpolation="bilinear",
            cmap="viridis_r",
            aspect="auto",
        )
    else:
        im = ax.scatter(candidates[:, 0], candidates[:, 1], c=values, cmap="viridis_r", s=60)
    cb = fig.colorbar(im, ax=ax, shrink=0.86)
    cb.set_label("Worst-case path loss (dB)")

    optimal = candidates[optimal_index]
    ax.scatter(optimal[0], optimal[1], marker="*", s=210, c="gold", edgecolors="black", label="Optimal AP")
    for receiver in receivers:
        pos = np.asarray(receiver["position_m"], dtype=float)
        ax.scatter(pos[0], pos[1], marker="x", s=70, c="white", linewidths=1.8)
        ax.text(pos[0], pos[1], f" {receiver['name']}", color="white", fontsize=8, weight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Worst-case path loss over AP candidate positions")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _mesh_edge_collection(mesh_path: Path, max_faces: int = 10_000) -> LineCollection:
    mesh = load_trimesh(mesh_path)
    faces = mesh.faces
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces, dtype=int)]
    vertices = np.asarray(mesh.vertices)
    segments = []
    for face in faces:
        tri = vertices[face][:, :2]
        segments.extend([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
    return LineCollection(segments, colors="0.55", linewidths=0.15, alpha=0.2)


def plot_topdown_layout(
    mesh_path: Path,
    candidates_m: np.ndarray,
    receivers: list[dict],
    optimal_index: int,
    out_path: Path,
) -> None:
    candidates = np.asarray(candidates_m, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
    ax.add_collection(_mesh_edge_collection(mesh_path))
    ax.scatter(candidates[:, 0], candidates[:, 1], s=14, c="tab:blue", alpha=0.45, label="AP candidates")
    optimal = candidates[optimal_index]
    ax.scatter(optimal[0], optimal[1], marker="*", s=220, c="gold", edgecolors="black", label="Optimal AP")
    for receiver in receivers:
        pos = np.asarray(receiver["position_m"], dtype=float)
        ax.scatter(pos[0], pos[1], marker="^", s=80, c="tab:green", edgecolors="black")
        ax.text(pos[0], pos[1], f" {receiver['name']}", fontsize=8)
    mesh = load_trimesh(mesh_path)
    bounds = mesh.bounds
    ax.set_xlim(float(bounds[0, 0]), float(bounds[1, 0]))
    ax.set_ylim(float(bounds[0, 1]), float(bounds[1, 1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Room mesh, candidate APs, receivers, and optimum")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
