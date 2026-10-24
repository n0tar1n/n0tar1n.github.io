"""Create smaller working copies of phone photos without changing originals.

The assignment images are preserved under images/.  The warp implementation
works with the copies in working_images/ so a multi-image mosaic fits reliably
in memory on an ordinary laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resize_image(source: Path, destination: Path, max_side: int) -> None:
    with Image.open(source) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        new_size = (round(width * scale), round(height * scale))
        # This is only input preparation. The assignment's projective warp and
        # interpolation remain implemented from scratch in mosaic.py.
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=95)
        print(f"{source} -> {destination} ({new_size[0]}x{new_size[1]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare manageable working copies of input photos.")
    parser.add_argument("--input", default="images", help="Directory containing original photos")
    parser.add_argument("--output", default="working_images", help="Directory for resized copies")
    parser.add_argument("--max-side", type=int, default=1600, help="Maximum width or height in pixels")
    args = parser.parse_args()
    if args.max_side < 100:
        parser.error("--max-side must be at least 100")

    input_root = Path(args.input)
    output_root = Path(args.output)
    if not input_root.is_dir():
        parser.error(f"Input directory does not exist: {input_root}")
    for source in sorted(input_root.rglob("*")):
        if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            destination = output_root / source.relative_to(input_root)
            resize_image(source, destination, args.max_side)


if __name__ == "__main__":
    main()
