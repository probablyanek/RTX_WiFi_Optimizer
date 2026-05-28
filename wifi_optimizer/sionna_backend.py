from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from .rf_math import db10, fspl_db


def available() -> bool:
    try:
        import mitsuba as mi  # noqa: F401
        import sionna.rt  # noqa: F401
    except Exception:
        return False
    return True


def _imports():
    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        try:
            mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
        except ImportError:
            mi.set_variant("llvm_ad_mono_polarized")

    from sionna.rt import PathSolver, PlanarArray, RadioMaterial, Receiver, SceneObject, Transmitter, load_scene

    return dr, mi, PathSolver, PlanarArray, RadioMaterial, Receiver, SceneObject, Transmitter, load_scene


def _tensor_to_numpy(tensor) -> np.ndarray:
    return np.asarray(tensor.numpy())


def evaluate_candidates(
    mesh_path: Path,
    candidates_m: np.ndarray,
    receivers: list[dict],
    frequency_hz: float,
    tx_power_dbm: float,
    batch_size: int = 10,
    max_depth: int = 2,
    samples_per_src: int = 300_000,
    max_num_paths_per_src: int = 250_000,
    seed: int = 41,
    material_relative_permittivity: float = 4.5,
    material_conductivity: float = 0.02,
    material_thickness_m: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate candidates using NVIDIA Sionna RT."""
    (
        dr,
        mi,
        PathSolver,
        PlanarArray,
        RadioMaterial,
        Receiver,
        SceneObject,
        Transmitter,
        load_scene,
    ) = _imports()

    scene = load_scene()
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    material = RadioMaterial(
        name="room_lossy_dielectric",
        relative_permittivity=material_relative_permittivity,
        conductivity=material_conductivity,
        thickness=material_thickness_m,
        scattering_coefficient=0.0,
        xpd_coefficient=0.0,
        color=(0.72, 0.72, 0.72),
    )
    scene.edit(
        add=SceneObject(
            fname=str(mesh_path),
            name="reconstructed_room_mesh",
            radio_material=material,
            remove_duplicate_vertices=False,
        )
    )

    grid_center = np.mean(candidates_m, axis=0)
    for receiver in receivers:
        scene.add(
            Receiver(
                name=receiver["name"],
                position=list(receiver["position_m"]),
                look_at=grid_center.tolist(),
                display_radius=0.035,
            )
        )

    solver = PathSolver()
    path_loss_db = np.full((len(candidates_m), len(receivers)), 320.0, dtype=float)
    rx_power_dbm = np.full_like(path_loss_db, -300.0)
    receiver_center = np.mean(np.array([r["position_m"] for r in receivers], dtype=float), axis=0)

    for start in range(0, len(candidates_m), batch_size):
        end = min(start + batch_size, len(candidates_m))
        batch = candidates_m[start:end]
        tx_names: list[str] = []
        for local_index, tx_pos in enumerate(batch):
            tx_name = f"tx_{start + local_index:04d}"
            tx_names.append(tx_name)
            scene.add(
                Transmitter(
                    name=tx_name,
                    position=tx_pos.tolist(),
                    look_at=receiver_center.tolist(),
                    power_dbm=tx_power_dbm,
                    display_radius=0.04,
                )
            )

        paths = solver(
            scene,
            max_depth=max_depth,
            max_num_paths_per_src=max_num_paths_per_src,
            samples_per_src=samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=False,
            edge_diffraction=False,
            seed=seed + start,
        )

        valid = _tensor_to_numpy(paths.valid).astype(bool)
        if valid.shape[-1] > 0:
            coeff = _tensor_to_numpy(paths.a[0]) + 1j * _tensor_to_numpy(paths.a[1])
            for rx_index, receiver in enumerate(receivers):
                rx_pos = np.asarray(receiver["position_m"], dtype=float)
                for tx_local_index, tx_pos in enumerate(batch):
                    mask = valid[rx_index, tx_local_index, :]
                    if not np.any(mask):
                        continue
                    link_coeff = coeff[rx_index, 0, tx_local_index, 0, mask]
                    path_gain = float(np.sum(np.abs(link_coeff) ** 2))
                    if path_gain <= 0.0 or not np.isfinite(path_gain):
                        continue
                    distance_m = float(np.linalg.norm(np.asarray(tx_pos, dtype=float) - rx_pos))
                    loss = max(-db10(path_gain), fspl_db(distance_m, frequency_hz))
                    global_index = start + tx_local_index
                    path_loss_db[global_index, rx_index] = loss
                    rx_power_dbm[global_index, rx_index] = tx_power_dbm - loss

        for tx_name in tx_names:
            if tx_name in scene.transmitters:
                scene.remove(tx_name)
        del paths
        gc.collect()
        try:
            dr.flush_malloc_cache()
        except Exception:
            pass

    details = {
        "model": "sionna_rt_path_solver",
        "mitsuba_variant": mi.variant(),
        "batch_size": batch_size,
        "max_depth": max_depth,
        "samples_per_src": samples_per_src,
        "max_num_paths_per_src": max_num_paths_per_src,
        "seed": seed,
    }
    return path_loss_db, rx_power_dbm, details
