from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh_utils import load_trimesh
from .rf_math import fspl_db


def count_segment_intersections(
    triangles: np.ndarray,
    start_m: np.ndarray,
    end_m: np.ndarray,
    chunk_size: int = 100_000,
    dedup_tolerance_m: float = 0.02,
) -> int:
    """Count unique mesh intersections along a finite TX-RX segment.

    This is a CPU-only geometric approximation used when Sionna RT is not
    available. It is not a replacement for electromagnetic ray tracing.
    """
    start = np.asarray(start_m, dtype=float)
    end = np.asarray(end_m, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return 0
    unit = direction / length
    hit_distances: list[float] = []
    eps = 1e-9
    endpoint_eps = max(1e-4, 0.001 * length)

    for offset in range(0, len(triangles), chunk_size):
        tri = triangles[offset : offset + chunk_size]
        v0 = tri[:, 0, :]
        edge1 = tri[:, 1, :] - v0
        edge2 = tri[:, 2, :] - v0
        h = np.cross(unit, edge2)
        a = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(a) > eps
        if not np.any(valid):
            continue

        v0 = v0[valid]
        edge1 = edge1[valid]
        edge2 = edge2[valid]
        h = h[valid]
        inv_a = 1.0 / a[valid]
        s = start - v0
        u = inv_a * np.einsum("ij,ij->i", s, h)
        valid_u = (u >= -eps) & (u <= 1.0 + eps)
        if not np.any(valid_u):
            continue

        s = s[valid_u]
        edge1 = edge1[valid_u]
        edge2 = edge2[valid_u]
        inv_a = inv_a[valid_u]
        u = u[valid_u]
        q = np.cross(s, edge1)
        v = inv_a * (q @ unit)
        valid_v = (v >= -eps) & ((u + v) <= 1.0 + eps)
        if not np.any(valid_v):
            continue

        q = q[valid_v]
        edge2 = edge2[valid_v]
        inv_a = inv_a[valid_v]
        t = inv_a * np.einsum("ij,ij->i", edge2, q)
        valid_t = (t > endpoint_eps) & (t < length - endpoint_eps)
        hit_distances.extend(float(x) for x in t[valid_t])

    if not hit_distances:
        return 0

    hit_distances.sort()
    unique_hits = [hit_distances[0]]
    for distance in hit_distances[1:]:
        if abs(distance - unique_hits[-1]) > dedup_tolerance_m:
            unique_hits.append(distance)
    return len(unique_hits)


def evaluate_candidates(
    mesh_path: Path,
    candidates_m: np.ndarray,
    receivers: list[dict],
    frequency_hz: float,
    tx_power_dbm: float,
    obstruction_loss_db: float = 9.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evaluate candidates with a deterministic CPU fallback model.

    The model is FSPL plus a fixed loss for each mesh intersection along the
    line segment between transmitter and receiver. It is intended for sanity
    checks and machines without a GPU/Sionna installation.
    """
    mesh = load_trimesh(mesh_path)
    triangles = np.asarray(mesh.triangles, dtype=float)
    path_loss_db = np.empty((len(candidates_m), len(receivers)), dtype=float)
    rx_power_dbm = np.empty_like(path_loss_db)
    obstruction_counts = np.zeros_like(path_loss_db, dtype=int)

    for tx_index, tx_pos in enumerate(candidates_m):
        tx = np.asarray(tx_pos, dtype=float)
        for rx_index, receiver in enumerate(receivers):
            rx = np.asarray(receiver["position_m"], dtype=float)
            distance_m = float(np.linalg.norm(rx - tx))
            obstruction_count = count_segment_intersections(triangles, tx, rx)
            loss = fspl_db(distance_m, frequency_hz) + obstruction_loss_db * obstruction_count
            path_loss_db[tx_index, rx_index] = loss
            rx_power_dbm[tx_index, rx_index] = tx_power_dbm - loss
            obstruction_counts[tx_index, rx_index] = obstruction_count

    details = {
        "model": "fspl_plus_mesh_intersection_penalty",
        "obstruction_loss_db": obstruction_loss_db,
        "obstruction_counts": obstruction_counts.tolist(),
    }
    return path_loss_db, rx_power_dbm, details
