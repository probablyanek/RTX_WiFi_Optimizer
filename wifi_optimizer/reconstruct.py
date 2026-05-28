from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_latest_surface_mesh(reconstruction_root: Path, exps_folder: str | Path = "exps") -> Path:
    exps_path = Path(exps_folder)
    if not exps_path.is_absolute():
        exps_path = reconstruction_root / exps_path
    meshes = sorted(exps_path.rglob("vis/surface_*.ply"), key=lambda p: p.stat().st_mtime)
    if not meshes:
        raise FileNotFoundError(f"No reconstructed surface_*.ply files found under {exps_path}")
    return meshes[-1]


def run_visual_reconstruction(
    reconstruction_root: Path,
    conf: Path,
    output_mesh: Path,
    exps_folder: str = "exps",
    scan_id: int | None = None,
    python_executable: str | None = None,
) -> Path:
    """Run the visual reconstruction backend and copy its newest mesh."""
    root = reconstruction_root.resolve()
    code_dir = root / "code"
    if not code_dir.exists():
        raise FileNotFoundError(f"Reconstruction code directory not found: {code_dir}")

    conf_path = conf
    if not conf_path.is_absolute():
        candidate = code_dir / conf_path
        conf_path = candidate if candidate.exists() else (root / conf_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"Reconstruction config not found: {conf}")

    python = python_executable or sys.executable
    cmd = [
        python,
        "training/exp_runner.py",
        "--conf",
        str(conf_path),
        "--exps_folder",
        exps_folder,
    ]
    if scan_id is not None:
        cmd.extend(["--scan_id", str(scan_id)])

    subprocess.run(cmd, cwd=code_dir, check=True)
    latest = find_latest_surface_mesh(root, exps_folder)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, output_mesh)
    return output_mesh
