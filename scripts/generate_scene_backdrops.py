#!/usr/bin/env python3
"""Build the six animated "scene" page backdrops used by Account settings -> Personalize.

Writes, for each scene, three static WebP files to app/web/static/backgrounds/:
    <scene>-dark.webp    the picture as supplied (night look, shown with the app's dark theme)
    <scene>-light.webp   a high-key "daylight" rendition of the same picture (shown with the light theme)
    <scene>-thumb.webp   a small preview tile for the Personalize picker

Scenes (source picture -> name):
    cosmic-drift, obsidian, midnight-ocean, rainfall-district, silent-sanctuary, misty-forest

Unlike the four seasonal backdrops, these are full-frame, fully opaque pictures held in a fixed layer behind the whole
app; the scrim that keeps text readable is applied in CSS (theme.css), not baked in, so it can be tuned per page.

    python scripts/generate_scene_backdrops.py --src path/to/pictures-folder

The folder must contain one picture per scene named <scene>.(png|jpg|jpeg|webp).  Needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "app" / "web" / "static" / "backgrounds"
SCENES = ("cosmic-drift", "obsidian", "midnight-ocean", "rainfall-district", "silent-sanctuary", "misty-forest")
MAX_W = 1920            # never ship more pixels than a full-HD screen can show; the source is only ~1672 wide anyway
THUMB = (360, 202)
PALE = np.array([232, 239, 246], dtype=np.float32) / 255.0   # the light theme's page colour (see theme.css)


def find_source(folder: Path, scene: str) -> Path:
    for ext in ("webp", "png", "jpg", "jpeg"):
        p = folder / f"{scene}.{ext}"
        if p.exists():
            return p
    raise SystemExit(f"missing source picture for '{scene}' in {folder}")


def to_light(im: Image.Image) -> Image.Image:
    """High-key daylight version: desaturate a little, lift the shadows, squeeze the tones into the top of the range and
    tint toward the app's cool paper colour. Structure (clouds, buildings, trees) stays visible; dark text reads on it."""
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lum = (a * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(axis=2, keepdims=True)
    a = np.clip(lum + (a - lum) * 0.60, 0.0, 1.0) ** 0.52
    out = np.clip((0.50 + 0.50 * a) * (PALE / PALE.max()), 0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder with <scene>.(png|jpg|webp) pictures")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--quality", type=int, default=94)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        im = Image.open(find_source(args.src, scene)).convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        for suffix, pic in (("dark", im), ("light", to_light(im))):
            p = args.out / f"{scene}-{suffix}.webp"
            pic.save(p, "WEBP", quality=args.quality, method=6)
            print("wrote", p.name, f"({p.stat().st_size // 1024} KB)", flush=True)
        th = im.copy()
        th.thumbnail(THUMB, Image.LANCZOS)
        th = th.filter(ImageFilter.UnsharpMask(radius=0.8, percent=40, threshold=2))
        p = args.out / f"{scene}-thumb.webp"
        th.save(p, "WEBP", quality=80, method=6)
        print("wrote", p.name, f"({p.stat().st_size // 1024} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
