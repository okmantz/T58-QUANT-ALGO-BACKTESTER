#!/usr/bin/env python3
"""Generate the dashboard's mountain backdrop (app/web/static/dashboard-mountains.webp).

Why a generated bitmap instead of CSS/SVG lines: a handful of jagged polygons never reads as
terrain. This paints five receding ranges the way a photograph looks -- ridged-fractal
silhouettes, vertical ravine texture lit from the upper left, snow-dusted summits, a teal rim
light on every ridgeline, and mist pooling between the layers -- then fades the whole thing to
transparent on the left and bottom so it melts into the page background.

It is a static image: zero runtime cost in the app (no filters, no JS, no animation).

    python scripts/generate_dashboard_mountains.py            # writes the .webp next to theme.css
    python scripts/generate_dashboard_mountains.py --seed 7   # a different (but equally calm) range

Needs numpy, scipy and Pillow (all already in config/requirements.txt except Pillow, which
qrcode[pil] installs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

W, H = 2200, 640
OUT = Path(__file__).resolve().parents[1] / "app" / "web" / "static" / "dashboard-mountains.webp"

# Palette -- deep teal-black shadows, one cool rim light, a pale mint for snow, a faint cyan haze.
SHADOW = np.array([2, 9, 11], dtype=np.float32)
MID = np.array([10, 40, 42], dtype=np.float32)
LIT = np.array([40, 150, 140], dtype=np.float32)
RIM = np.array([90, 235, 200], dtype=np.float32)
SNOW = np.array([205, 255, 244], dtype=np.float32)
HAZE = np.array([22, 96, 98], dtype=np.float32)
SKY = np.array([30, 140, 130], dtype=np.float32)


def _norm(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / (a.max() + 1e-9)


def noise2(rng, sy: float, sx: float) -> np.ndarray:
    """Smooth 2-D noise in [0,1] with different blur radii on each axis (tall sy => vertical streaks)."""
    return _norm(gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), (sy, sx), mode="wrap"))


def ridge_profile(rng, base: float, amp: float, rough: float, envelope: np.ndarray) -> np.ndarray:
    """Top edge of one range as a y-coordinate per column (smaller y = higher on screen)."""
    prof = np.zeros(W, dtype=np.float32)
    for octave, (scale, weight) in enumerate([(430, 1.0), (210, .55), (100, .24), (46, .11), (20, .05), (8, .02)]):
        n = gaussian_filter1d(rng.standard_normal(W).astype(np.float32), scale / 2.2, mode="wrap")
        n = n / (np.abs(n).max() + 1e-9)
        prof += weight * (1.0 - np.abs(n)) * (1.0 if octave < 3 else rough)  # ridged => sharp crests
    prof = _norm(prof)
    return base - amp * prof * envelope


def render(seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    xf = np.linspace(0, 1, W, dtype=np.float32)

    # Sky glow sits behind everything (this is the only non-mountain colour; alpha keeps it faint).
    sky_a = np.exp(-(((xs / W - 0.70) / 0.34) ** 2 + ((ys / H - 0.30) / 0.42) ** 2)) * 0.55
    rgb = np.broadcast_to(SKY, (H, W, 3)).copy()
    alpha = sky_a[..., 0] if sky_a.ndim == 3 else sky_a
    alpha = np.broadcast_to(alpha, (H, W)).astype(np.float32).copy()

    # (base y as a fraction of H, relative height, roughness, haze amount, brightness) -- far to near.
    layers = [
        (0.42, 0.34, 1.0, 0.74, 0.55),
        (0.52, 0.36, 1.0, 0.58, 0.68),
        (0.63, 0.34, 1.1, 0.40, 0.82),
        (0.76, 0.30, 1.2, 0.22, 0.95),
        (0.92, 0.24, 1.3, 0.08, 1.05),
    ]
    # Tallest summits cluster right of centre; the range thins out toward the left edge.
    env_far = 0.50 + 0.50 * np.exp(-((xf - 0.66) / 0.30) ** 2)

    for i, (base, rel, rough, haze, bright) in enumerate(layers):
        env = np.clip(env_far * (0.85 + 0.3 * rng.random()), 0, 1.15) if i else env_far
        ridge = ridge_profile(rng, base * H, rel * H, rough, env)
        d = ys - ridge[None, :]                       # depth below the crest, in pixels
        inside = d >= 0
        dn = np.clip(d / (0.55 * H), 0, 1)            # 0 at the crest -> 1 deep in the massif

        # Ravines/crags: noise stretched vertically, plus a finer cross-grain.
        # Faces fan out diagonally from each summit instead of hanging straight down: sample the texture through a
        # warp that leans with the local slope of the crest, so gullies follow the shoulders of the mountain.
        lean = gaussian_filter1d(np.gradient(ridge), 9) * 1.3 + gaussian_filter1d(rng.standard_normal(W), 30) * 0.9
        lean = np.clip(lean, -1.1, 1.1).astype(np.float32)
        warp_x = xs + np.clip(d, 0, H) * lean[None, :] * -1.0
        warp_x = np.broadcast_to(warp_x, (H, W))
        warp_y = np.broadcast_to(ys, (H, W))
        base_tex = 0.55 * noise2(rng, 48, 6) + 0.30 * noise2(rng, 20, 3) + 0.15 * noise2(rng, 26, 26)
        crag = map_coordinates(base_tex, [warp_y, warp_x], order=1, mode="wrap").astype(np.float32)
        crag = _norm(crag)
        crag = _norm(crag * (0.8 + 0.4 * noise2(rng, 4, 60)))
        # Pseudo-lighting from the upper left: faces whose texture rises toward the right catch the light.
        slope_tex = gaussian_filter(np.gradient(crag, axis=1), 1.2)
        slope_tex = np.clip(0.5 + slope_tex * 38, 0, 1)
        slope_ridge = np.gradient(gaussian_filter1d(ridge, 6))      # <0 where the crest climbs rightward
        slope_ridge = np.clip(0.5 - slope_ridge * 0.9, 0, 1)[None, :]
        lit = np.clip(0.55 * slope_tex + 0.45 * slope_ridge, 0, 1)
        face = lit * (1.0 - 0.75 * dn) * (0.55 + 0.7 * crag)
        face = np.clip(face * bright * 1.15, 0, 1)

        col = SHADOW[None, None, :] + (MID - SHADOW)[None, None, :] * np.clip(face * 1.25, 0, 1)[..., None]
        col = col + (LIT - MID)[None, None, :] * np.clip((face - 0.50) * 1.6, 0, 1)[..., None]

        # Snow dusting on the highest summits (only where this layer is genuinely tall), broken up by noise.
        peak_h = np.clip((base * H - ridge) / (rel * H), 0, 1)[None, :]
        snow_depth = (0.05 + 0.22 * peak_h) * H * (0.55 + 0.9 * noise2(rng, 18, 5))
        snow = np.clip(1.0 - d / (snow_depth + 1e-3), 0, 1) ** 1.6 * (peak_h > 0.55) * lit * (1.1 - haze)
        col = col + (SNOW - col) * (snow * 0.85)[..., None]

        # Teal rim light hugging the crest, tighter and brighter on the near layers.
        rim = np.exp(-np.clip(d, 0, None) / (2.2 + 2.0 * (1 - haze))) * inside
        col = col + (RIM - col) * (np.clip(rim, 0, 1) * (0.55 + 0.4 * bright))[..., None]

        # Atmospheric perspective: far ranges wash toward haze; every range pales toward its base (mist).
        base_mist = np.clip((d - 0.30 * H) / (0.5 * H), 0, 1)
        fog = np.clip(haze + 0.55 * base_mist, 0, 0.92)
        col = col + (HAZE - col) * (fog * (0.5 + 0.5 * noise2(rng, 8, 40)))[..., None]
        col = col * (0.48 + 0.40 * bright)

        a = inside.astype(np.float32)
        # 1px of softness on the crest so it is not aliased.
        a = np.clip(d + 0.8, 0, 1.6) / 1.6 * inside + (d < 0) * np.clip(d + 0.8, 0, 1)
        a = np.clip(a, 0, 1)
        rgb = rgb * (1 - a[..., None]) + col * a[..., None]
        alpha = np.maximum(alpha, a)

        # Mist pooled between this layer and the next (a soft horizontal band hugging the foot of the range).
        if i < len(layers) - 1:
            band_y = (layers[i + 1][0] * H) - 0.10 * H
            band = np.exp(-((ys - band_y) / (0.07 * H)) ** 2) * (0.25 + 0.75 * noise2(rng, 14, 90))
            band = band * 0.34
            rgb = rgb + (np.array([150, 235, 225], dtype=np.float32) - rgb) * band[..., None]
            alpha = np.maximum(alpha, band * 0.65)

    # Fade to transparent on the left and bottom (and a touch at the very top) so it dissolves into the page.
    fade_x = np.clip((xs / W - 0.02) / 0.62, 0, 1) ** 1.25
    fade_y = np.clip((1.0 - ys / H) / 0.46, 0, 1) ** 1.1
    fade_top = np.clip(ys / (0.04 * H), 0, 1)
    alpha = alpha * fade_x * fade_y * fade_top

    # Dither so the long gradients don't band once compressed.
    rgb = np.clip(rgb + rng.normal(0, 1.1, rgb.shape).astype(np.float32), 0, 255)
    out = np.dstack([rgb, np.clip(alpha, 0, 1)[..., None] * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()
    img = render(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, "WEBP", quality=args.quality, method=6)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {img.width}x{img.height})")


if __name__ == "__main__":
    main()
