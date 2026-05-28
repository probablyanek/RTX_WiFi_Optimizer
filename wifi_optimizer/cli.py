from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .images import prepare_image_sequence
from .mesh_utils import prepare_mesh
from .optimizer import generate_candidate_grid, optimize_placement, write_outputs
from .reconstruct import run_visual_reconstruction


DEFAULT_RECEIVERS = [
    {"name": "Rx1", "position_m": [0.283, -0.139, 0.200]},
    {"name": "Rx2", "position_m": [-0.481, 0.096, 0.125]},
    {"name": "Rx3", "position_m": [-0.582, -0.312, 0.100]},
]


def parse_receiver(value: str, index: int) -> dict:
    if ":" in value:
        name, coords = value.split(":", 1)
    else:
        name, coords = f"Rx{index + 1}", value
    parts = [float(part.strip()) for part in coords.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("receiver must be name:x,y,z or x,y,z")
    return {"name": name.strip() or f"Rx{index + 1}", "position_m": parts}


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open() as f:
        return json.load(f)


def receivers_from_args(args: argparse.Namespace, config: dict) -> list[dict]:
    if args.receiver:
        return [parse_receiver(value, idx) for idx, value in enumerate(args.receiver)]
    return config.get("receivers", DEFAULT_RECEIVERS)


def candidate_grid_from_args(args: argparse.Namespace, config: dict, bounds_min: list[float], bounds_max: list[float]) -> tuple[np.ndarray, tuple[int, int]]:
    grid_cfg = config.get("candidate_grid", {})
    grid_size_arg = args.grid_size if args.grid_size is not None else grid_cfg.get("grid_size", [10, 10])
    grid_size = (int(grid_size_arg[0]), int(grid_size_arg[1]))
    z_m = args.tx_height_m if args.tx_height_m is not None else grid_cfg.get("z_m")
    margin_m = args.candidate_margin_m if args.candidate_margin_m is not None else float(grid_cfg.get("margin_m", 0.15))

    if "x_min" in grid_cfg and "x_max" in grid_cfg and "y_min" in grid_cfg and "y_max" in grid_cfg:
        custom_bounds_min = np.array([grid_cfg["x_min"], grid_cfg["y_min"], bounds_min[2]], dtype=float)
        custom_bounds_max = np.array([grid_cfg["x_max"], grid_cfg["y_max"], bounds_max[2]], dtype=float)
        margin_m = 0.0
    else:
        custom_bounds_min = np.asarray(bounds_min, dtype=float)
        custom_bounds_max = np.asarray(bounds_max, dtype=float)

    candidates = generate_candidate_grid(custom_bounds_min, custom_bounds_max, grid_size=grid_size, z_m=z_m, margin_m=margin_m)
    return candidates, grid_size


def add_optimizer_args(parser: argparse.ArgumentParser, mesh_required: bool) -> None:
    parser.add_argument("--mesh", type=Path, required=mesh_required, help="Input room mesh (.ply/.obj/etc.).")
    parser.add_argument("--config", type=Path, help="JSON configuration file.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--backend", choices=["auto", "sionna", "fallback"], default="auto")
    parser.add_argument("--receiver", action="append", help="Receiver as name:x,y,z. Can be repeated.")
    parser.add_argument("--grid-size", nargs=2, type=int, metavar=("NX", "NY"))
    parser.add_argument("--tx-height-m", type=float, help="AP candidate z height. Defaults near mesh ceiling.")
    parser.add_argument("--candidate-margin-m", type=float, help="XY margin from mesh bounds for generated grid.")
    parser.add_argument("--frequency-hz", type=float)
    parser.add_argument("--tx-power-dbm", type=float)
    parser.add_argument("--mesh-max-faces", type=int, help="Decimate mesh cache to this face count. Use 0 for full mesh.")
    parser.add_argument("--mesh-aggression", type=int, default=7)
    parser.add_argument("--force-remesh", action="store_true")
    parser.add_argument("--fallback-obstruction-loss-db", type=float)
    parser.add_argument("--sionna-batch-size", type=int)
    parser.add_argument("--sionna-max-depth", type=int)
    parser.add_argument("--sionna-samples-per-src", type=int)
    parser.add_argument("--sionna-max-paths-per-src", type=int)
    parser.add_argument("--sionna-seed", type=int)


def add_reconstruct_args(parser: argparse.ArgumentParser, conf_required: bool = True) -> None:
    parser.add_argument("--reconstruction-root", type=Path, default=Path("room_reconstruction"))
    parser.add_argument("--reconstruction-conf", type=Path, required=conf_required, help="Visual reconstruction config file.")
    parser.add_argument("--reconstruction-exps-folder", default="exps")
    parser.add_argument("--scan-id", type=int)
    parser.add_argument("--python", dest="python_executable", help="Python executable for the reconstruction environment.")


def run_optimize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    mesh_path = args.mesh
    frequency_hz = float(args.frequency_hz if args.frequency_hz is not None else config.get("frequency_hz", 5.8e9))
    tx_power_dbm = float(args.tx_power_dbm if args.tx_power_dbm is not None else config.get("tx_power_dbm", 20.0))
    mesh_max_faces = int(args.mesh_max_faces if args.mesh_max_faces is not None else config.get("mesh_max_faces", 50_000))
    fallback_loss = float(
        args.fallback_obstruction_loss_db
        if args.fallback_obstruction_loss_db is not None
        else config.get("fallback_obstruction_loss_db", 9.0)
    )
    receivers = receivers_from_args(args, config)

    prepared_mesh, mesh_stats = prepare_mesh(
        mesh_path=mesh_path,
        out_dir=args.out_dir,
        max_faces=mesh_max_faces,
        aggression=args.mesh_aggression,
        force=args.force_remesh,
    )
    if mesh_stats.simplification_warning:
        print(mesh_stats.simplification_warning)

    candidates, grid_size = candidate_grid_from_args(args, config, mesh_stats.bounds_min, mesh_stats.bounds_max)
    sionna_cfg = dict(config.get("sionna", {}))
    if args.sionna_batch_size is not None:
        sionna_cfg["batch_size"] = args.sionna_batch_size
    if args.sionna_max_depth is not None:
        sionna_cfg["max_depth"] = args.sionna_max_depth
    if args.sionna_samples_per_src is not None:
        sionna_cfg["samples_per_src"] = args.sionna_samples_per_src
    if args.sionna_max_paths_per_src is not None:
        sionna_cfg["max_num_paths_per_src"] = args.sionna_max_paths_per_src
    if args.sionna_seed is not None:
        sionna_cfg["seed"] = args.sionna_seed

    result = optimize_placement(
        mesh_path=prepared_mesh,
        candidates_m=candidates,
        receivers=receivers,
        frequency_hz=frequency_hz,
        tx_power_dbm=tx_power_dbm,
        backend=args.backend,
        fallback_obstruction_loss_db=fallback_loss,
        sionna_options=sionna_cfg,
    )
    write_outputs(args.out_dir, result, mesh_stats, frequency_hz, tx_power_dbm, grid_size)

    optimal = result.optimal_position_m
    worst = result.worst_case_path_loss_db[result.optimal_index]
    mean = result.mean_path_loss_db[result.optimal_index]
    print(f"Backend: {result.backend}")
    print(f"Optimal AP position: x={optimal[0]:.3f} m, y={optimal[1]:.3f} m, z={optimal[2]:.3f} m")
    print(f"Worst-case path loss: {worst:.2f} dB")
    print(f"Mean path loss at optimum: {mean:.2f} dB")
    print(f"Outputs written to: {args.out_dir.resolve()}")
    return 0


def run_reconstruct(args: argparse.Namespace) -> int:
    output_mesh = args.output_mesh
    mesh_path = run_visual_reconstruction(
        reconstruction_root=args.reconstruction_root,
        conf=args.reconstruction_conf,
        output_mesh=output_mesh,
        exps_folder=args.reconstruction_exps_folder,
        scan_id=args.scan_id,
        python_executable=args.python_executable,
    )
    print(f"Reconstructed mesh written to: {mesh_path.resolve()}")
    return 0


def run_prepare_images(args: argparse.Namespace) -> int:
    manifest = prepare_image_sequence(args.images_dir, args.out_dir, args.limit)
    print(f"Prepared {manifest['image_count']} images in: {args.out_dir.resolve()}")
    print(f"Manifest written to: {(args.out_dir / 'image_manifest.json').resolve()}")
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    if args.mesh is None:
        if args.reconstruction_conf is None:
            raise SystemExit("--reconstruction-conf is required when pipeline is run without --mesh")
        args.mesh = args.out_dir / "reconstructed_mesh.ply"
        run_visual_reconstruction(
            reconstruction_root=args.reconstruction_root,
            conf=args.reconstruction_conf,
            output_mesh=args.mesh,
            exps_folder=args.reconstruction_exps_folder,
            scan_id=args.scan_id,
            python_executable=args.python_executable,
        )
    return run_optimize(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize indoor WiFi AP placement from a reconstructed room mesh.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser("optimize", help="Optimize AP placement from an existing mesh.")
    add_optimizer_args(optimize_parser, mesh_required=True)
    optimize_parser.set_defaults(func=run_optimize)

    images_parser = subparsers.add_parser("prepare-images", help="Normalize room photos for visual reconstruction preprocessing.")
    images_parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing room photos.")
    images_parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for numbered *_rgb.png files.")
    images_parser.add_argument("--limit", type=int, help="Optional maximum number of images to copy.")
    images_parser.set_defaults(func=run_prepare_images)

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Run visual reconstruction and export the latest mesh.")
    add_reconstruct_args(reconstruct_parser, conf_required=True)
    reconstruct_parser.add_argument("--output-mesh", type=Path, required=True)
    reconstruct_parser.set_defaults(func=run_reconstruct)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run reconstruction if needed, then optimize AP placement.")
    add_optimizer_args(pipeline_parser, mesh_required=False)
    add_reconstruct_args(pipeline_parser, conf_required=False)
    pipeline_parser.set_defaults(func=run_pipeline)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
