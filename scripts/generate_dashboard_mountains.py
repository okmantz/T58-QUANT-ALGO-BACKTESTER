#!/usr/bin/env python3
"""Build the dashboard's mountain backdrop (app/web/static/dashboard-mountains.webp).

The backdrop is the "teal night range" artwork (assets/images/dashboard-mountains-source.webp): a starfield over
dark silhouetted peaks with a teal rim-glow along the ridgeline and mist in the valleys. This script prepares it for
the page:

  * feathers the top, left and right edges a touch and fades the bottom ~38% to transparent so the range dissolves
    into the page background (#03080a) instead of ending in a hard line;
  * writes one static WebP -- zero runtime cost in the app (no filters, no JS, no animation).

    python scripts/generate_dashboard_mountains.py                     # uses the bundled source artwork
    python scripts/generate_dashboard_mountains.py --source my.webp    # or any wide night-mountain image

Needs numpy and Pillow (Pillow comes with qrcode[pil] in config/requirements.txt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "images" / "dashboard-mountains-source.webp"
OUT = ROOT / "app" / "web" / "static" / "dashboard-mountains.webp"
TARGET_W = 2048


def build(source: Path = SOURCE, out: Path = OUT) -> Path:
    im = Image.open(source).convert("RGB")
    if im.width != TARGET_W:
        im = im.resize((TARGET_W, round(im.height * TARGET_W / im.width)), Image.LANCZOS)
    w, h = im.size
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    def smooth(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    bottom = 1.0 - smooth((ys - 0.62) / 0.38)          # solid until 62% of the height, transparent at the bottom
    top = smooth(ys / 0.04)                            # hair-thin feather at the very top
    left = smooth(xs / 0.05)
    right = smooth((1.0 - xs) / 0.02)
    alpha = np.clip(bottom * top * left * right, 0.0, 1.0)

    rgba = np.dstack([np.asarray(im, dtype=np.uint8), (alpha * 255).astype(np.uint8)])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out, "WEBP", quality=86, method=6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    p = build(a.source, a.out)
    print(f"wrote {p} ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
