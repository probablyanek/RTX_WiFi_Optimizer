# FINAL WiFi Optimizer

This repository is the cleaned end-to-end program for the paper workflow:

1. Use the integrated visual reconstruction backend to reconstruct a room mesh from a captured image sequence.
2. Convert the mesh into a tracing-friendly triangular PLY cache.
3. Sweep candidate WiFi access-point positions.
4. Pick the AP coordinate that minimizes worst-case receiver path loss.

The optimizer uses NVIDIA Sionna RT when available. On machines without a GPU or Sionna, it automatically falls back to a CPU geometric model so the CLI, mesh handling, output files, and plots can still be tested.

## Repository Layout

| Path | Purpose |
|---|---|
| `wifi_optimizer/` | Main Python package and CLI. |
| `configs/room_example.json` | Example receiver/grid/Sionna configuration matching the paper-scale office runs. |
| `room_reconstruction/` | Integrated visual reconstruction backend used for image-sequence-to-mesh reconstruction. |
| `examples/meshes/` | Example reconstructed office meshes copied from the original ray-tracing project. |
| `THIRD_PARTY_NOTICES.md` | Attribution and license notes for included third-party reconstruction components. |
| `requirements.txt` | Minimal CPU-testable dependencies. |
| `requirements-sionna.txt` | Optional Sionna RT dependencies for final ray-tracing runs. |
| `tests/` | CPU tests for mesh preparation and fallback optimization. |

## Install

CPU-testable setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For final Sionna RT runs:

```bash
python -m pip install -r requirements-sionna.txt
```

For reconstruction, use the conda environment from `room_reconstruction/env_yamls/reconstruction.yaml`. The reconstruction backend is GPU-heavy and cannot be fully tested on this CPU-only machine.

## Fast CPU Smoke Test

Run the optimizer on an existing mesh with the fallback backend:

```bash
python -m wifi_optimizer.cli optimize \
  --mesh examples/meshes/reconstructed_office_2.ply \
  --config configs/room_example.json \
  --backend fallback \
  --grid-size 3 3 \
  --mesh-max-faces 2000 \
  --out-dir outputs/smoke_office_2
```

Outputs:

| File | Meaning |
|---|---|
| `optimization_result.json` | Machine-readable summary including the optimal AP coordinate. |
| `candidate_metrics.csv` | Per-candidate path loss and received power. |
| `worst_case_path_loss_heatmap.png` | Heatmap of the min-max objective. |
| `topdown_layout.png` | Mesh footprint, candidate APs, receivers, and optimum. |

## Existing Mesh Workflow

If you already have a `.ply` mesh, run:

```bash
python -m wifi_optimizer.cli optimize \
  --mesh path/to/reconstructed_room.ply \
  --config configs/room_example.json \
  --backend auto \
  --out-dir outputs/room_run
```

`--backend auto` uses Sionna RT if importable, otherwise the CPU fallback.

## Image Sequence Preparation

Start with a directory of room photos or video frames. Normalize them to the image naming convention used by the reconstruction preprocessing scripts:

```bash
python -m wifi_optimizer.cli prepare-images \
  --images-dir path/to/room_photos \
  --out-dir data/my_room/scan1
```

This writes numbered files such as `000000_rgb.png`, `000001_rgb.png`, and `image_manifest.json`. You still need camera poses/intrinsics (`cameras.npz`) and monocular cues/flow before reconstruction. Those GPU-heavy preprocessing tools are included under `room_reconstruction/preprocess/`.

## Full Reconstruction + Optimization Workflow

The reconstruction backend expects a processed dataset layout including RGB frames, `cameras.npz`, monocular depth/normal cues, and optical-flow pairs. The `room_reconstruction` backend contains preprocessing scripts for Replica, 7-Scenes, Azure/COLMAP captures, Omnidata cues, and GMFlow.

After preparing the reconstruction dataset and config, run:

```bash
python -m wifi_optimizer.cli pipeline \
  --reconstruction-root room_reconstruction \
  --reconstruction-conf confs/runconf_demo_1.conf \
  --scan-id 1 \
  --config configs/room_example.json \
  --backend auto \
  --out-dir outputs/full_room_run
```

The pipeline command runs `training/exp_runner.py`, locates the newest `vis/surface_*.ply`, copies it to `outputs/full_room_run/reconstructed_mesh.ply`, and then optimizes AP placement.

## Manual Two-Step Workflow

Reconstruct only:

```bash
python -m wifi_optimizer.cli reconstruct \
  --reconstruction-root room_reconstruction \
  --reconstruction-conf confs/runconf_demo_1.conf \
  --scan-id 1 \
  --output-mesh outputs/reconstructed_mesh.ply
```

Optimize only:

```bash
python -m wifi_optimizer.cli optimize \
  --mesh outputs/reconstructed_mesh.ply \
  --config configs/room_example.json \
  --backend auto \
  --out-dir outputs/wifi_optimization
```

## Configuration

The JSON config controls receiver positions, the candidate AP grid, carrier frequency, transmit power, mesh simplification, fallback loss, and Sionna parameters.

Receivers can also be passed on the command line:

```bash
python -m wifi_optimizer.cli optimize \
  --mesh room.ply \
  --receiver Desk1:0.2,0.1,0.8 \
  --receiver Door:-1.0,0.5,0.8 \
  --tx-height-m 2.4 \
  --grid-size 12 12 \
  --out-dir outputs/custom_room
```

## Backend Notes

`sionna` backend:

Uses Sionna `PathSolver`, specular reflections, refraction, and a lossy dielectric material. This is the backend used for final research-grade results.

`fallback` backend:

Uses free-space path loss plus a fixed penalty for each mesh intersection between the AP and receiver. This is only a deterministic sanity check for CPU-only machines.

## Included Reconstruction Code

The reconstruction backend is integrated under `room_reconstruction/`. Third-party attribution and license details are kept in `THIRD_PARTY_NOTICES.md` and `room_reconstruction/LICENSE`.

## Test

```bash
python -m unittest discover -s tests
```
