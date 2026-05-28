from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_input_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    images = [path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No supported image files found in {images_dir}")
    return images


def prepare_image_sequence(images_dir: Path, out_dir: Path, limit: int | None = None) -> dict:
    """Convert room photos to the numbered RGB image convention."""
    input_images = list_input_images(images_dir)
    if limit is not None:
        input_images = input_images[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, source in enumerate(input_images):
        target = out_dir / f"{index:06d}_rgb.png"
        with Image.open(source) as image:
            image.convert("RGB").save(target)
        records.append({"index": index, "source": str(source), "target": str(target)})

    manifest = {
        "source_dir": str(images_dir),
        "output_dir": str(out_dir),
        "image_count": len(records),
        "images": records,
        "next_steps": [
            "Run COLMAP or another pose-estimation pipeline to create cameras.npz.",
            "Run monocular cue and optical-flow preprocessing.",
            "Run wifi-optimizer pipeline or reconstruct once the processed dataset is ready.",
        ],
    }
    with (out_dir / "image_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
