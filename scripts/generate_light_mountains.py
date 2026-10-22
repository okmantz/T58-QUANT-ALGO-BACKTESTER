#!/usr/bin/env python3
"""Build the LIGHT theme's mountain backdrop (app/web/static/dashboard-mountains-light.webp).

The dark theme has a photographic "teal night range" (see generate_dashboard_mountains.py). The light theme is its
daytime counterpart: a clean, pale, snow-dusted alpine range in cool glacier blues, with mist pooled in the
valleys -- the artwork in assets/images/dashboard-mountains-light-source.webp (white background, pale blue peaks).

This script turns that artwork into a backdrop the page can sit on:

  * the white background becomes TRANSPARENT ("un-multiplied from white"), so the mist stays soft and
    semi-transparent and the page's own pale sky gradient shows through -- no visible rectangle, ever;
  * it is cropped/padded to the same 2048x768 frame as the dark artwork, anchored at the bottom, so the
    CSS (`center bottom / cover`) treats both themes identically;
  * a gentle tonal lift keeps the ridges crisp on the page background, and the bottom ~30% fades to
    transparent so the range dissolves into the page; the left/right edges are feathered a hair.

    python scripts/generate_light_mountains.py                       # writes the WebP the theme uses
    python scripts/generate_light_mountains.py --source my.webp      # or any wide mountains-on-white image
    python scripts/generate_light_mountains.py --out my.webp

Needs numpy and Pillow (Pillow comes with qrcode[pil] in config/requirements.txt). Deterministic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "images" / "dashboard-mountains-light-source.webp"
OUT = ROOT / "app" / "web" / "static" / "dashboard-mountains-light.webp"
W, H = 2048, 768


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fit_frame(img: Image.Image) -> Image.Image:
    """Scale to the frame width, then keep the BOTTOM H rows (the sky is blank, so cropping/padding the top is free)."""
    scale = W / img.width
    img = img.resize((W, max(1, round(img.height * scale))), Image.LANCZOS)
    if img.height >= H:
        return img.crop((0, img.height - H, W, img.height))
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img, (0, H - img.height))
    return canvas


def _white_to_alpha(rgb: np.ndarray, floor: float = 0.012) -> tuple[np.ndarray, np.ndarray]:
    """Un-multiply from a white background: composite(out_rgb, alpha) over white == the original pixel.
    alpha = the strongest channel's distance from white; colour = what that alpha was tinted with."""
    f = rgb.astype(np.float32) / 255.0
    alpha = np.clip(1.0 - f.min(axis=2), 0.0, 1.0)
    alpha = np.where(alpha < floor, 0.0, alpha)
    safe = np.maximum(alpha, 1e-4)[..., None]
    col = np.clip(1.0 - (1.0 - f) / safe, 0.0, 1.0)
    return col, alpha


def build(out: Path = OUT, source: Path = SOURCE) -> Path:
    img = _fit_frame(Image.open(source).convert("RGB"))
    col, alpha = _white_to_alpha(np.asarray(img))

    # tonal lift: a touch more body in the thin mist/shadow so the range reads on the pale page (opacity is set in CSS)
    alpha = np.clip(alpha * 1.18, 0.0, 1.0) ** 0.92
    # keep the cool glacier cast: nudge colour a little toward the theme's slate-blue
    tint = np.array((0.34, 0.50, 0.62), np.float32)[None, None, :]
    col = np.clip(col * 0.86 + tint * 0.14 * (1.0 - col.mean(axis=2, keepdims=True) * 0.4), 0.0, 1.0)

    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    fade = (1.0 - _smooth((yy - 0.70) / 0.30)) * _smooth(xx / 0.07) * _smooth((1.0 - xx) / 0.09)
    alpha = np.clip(alpha * fade, 0.0, 1.0)

    rgba = np.dstack([(col * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out, "WEBP", quality=80, method=6, exact=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--source", type=Path, default=SOURCE)
    a = ap.parse_args()
    p = build(a.out, a.source)
    print(f"wrote {p} ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
