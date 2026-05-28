from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshStats:
    source: str
    optimizer_mesh: str
    original_vertices: int
    original_faces: int
    simulated_vertices: int
    simulated_faces: int
    bounds_min: list[float]
    bounds_max: list[float]
    extents: list[float]
    simplified: bool
    simplification_warning: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {path}, got {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    return mesh


def clean_mesh_for_tracing(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.ColorVisuals(mesh)
    try:
        mesh.remove_degenerate_faces()
    except AttributeError:
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def prepare_mesh(
    mesh_path: Path,
    out_dir: Path,
    max_faces: int = 50_000,
    aggression: int = 7,
    force: bool = False,
) -> tuple[Path, MeshStats]:
    """Prepare a clean triangular PLY cache for Sionna or the CPU fallback."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "mesh_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    face_tag = "full" if max_faces <= 0 else f"{max_faces}f"
    cache_path = cache_dir / f"{mesh_path.stem}_wifi_tri_{face_tag}.ply"

    source_mesh = load_trimesh(mesh_path)
    original_vertices = int(len(source_mesh.vertices))
    original_faces = int(len(source_mesh.faces))
    rebuild = force or not cache_path.exists() or cache_path.stat().st_mtime < mesh_path.stat().st_mtime
    simplified = False
    simplification_warning = None

    if rebuild:
        mesh = clean_mesh_for_tracing(source_mesh)
        if max_faces > 0 and len(mesh.faces) > max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=max_faces, aggression=aggression)
                mesh = clean_mesh_for_tracing(mesh)
                simplified = True
            except Exception as exc:  # pragma: no cover - depends on optional backend
                simplification_warning = (
                    "Mesh simplification failed; using the full cleaned mesh. "
                    "Install fast-simplification for faster tracing. "
                    f"Error: {exc}"
                )
        mesh.export(cache_path, file_type="ply", encoding="binary_little_endian")

    sim_mesh = load_trimesh(cache_path)
    actually_simplified = int(len(sim_mesh.faces)) < original_faces
    bounds = sim_mesh.bounds.astype(float)
    stats = MeshStats(
        source=str(mesh_path),
        optimizer_mesh=str(cache_path),
        original_vertices=original_vertices,
        original_faces=original_faces,
        simulated_vertices=int(len(sim_mesh.vertices)),
        simulated_faces=int(len(sim_mesh.faces)),
        bounds_min=bounds[0].tolist(),
        bounds_max=bounds[1].tolist(),
        extents=np.asarray(sim_mesh.extents, dtype=float).tolist(),
        simplified=bool(simplified or actually_simplified),
        simplification_warning=simplification_warning,
    )
    return cache_path, stats
