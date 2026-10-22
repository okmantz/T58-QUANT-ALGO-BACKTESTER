#!/usr/bin/env python3
"""Build the LIGHT theme's mountain backdrop (app/web/static/dashboard-mountains-light.webp).

The dark theme has a photographic "teal night range" (see generate_dashboard_mountains.py). The light theme gets
its daytime counterpart: the same composition -- a broad range with the tallest peak right of centre, a rim of teal
light along the ridges, mist pooled in the valleys -- painted as a misty alpine morning in the same cool teal/slate
family, so switching themes reads as the same product at a different time of day.

It is generated, not photographed, so it is reproducible and tiny:

  * five ridge layers, far -> near, each lighter and bluer the further away it is (atmospheric perspective);
  * faceted rock shading (sun from the upper left) and snow caps on the tall peaks of the two far layers;
  * valley mist at the foot of every layer, and a soft teal/white dawn glow behind the peaks;
  * the sky is left transparent, so the page's own pale gradient shows through, and the whole image fades to
    transparent along the bottom so the range dissolves into the page (same 2048x768 frame as the dark artwork).

    python scripts/generate_light_mountains.py                # writes the WebP the theme uses
    python scripts/generate_light_mountains.py --out my.webp --seed 7

Needs numpy and Pillow (Pillow comes with qrcode[pil] in config/requirements.txt). Deterministic for a given seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "web" / "static" / "dashboard-mountains-light.webp"
W, H = 2048, 768

# (peaks as (x 0..1, height above the layer's base 0..1, spread), base line as a fraction of H, shadow RGB, light RGB,
#  haze RGB, ridge roughness, snow?) -- far layer first.
LAYERS = [
    dict(peaks=[(0.10, .30, 1.0), (0.36, .42, 0.9), (0.62, .52, 0.9), (0.90, .46, 1.0)], base=.60, rough=.6,
         shadow=(158, 184, 201), light=(210, 226, 236), haze=(228, 238, 245), snow=False, mist=.20),
    dict(peaks=[(0.05, .38, 1.0), (0.27, .52, 0.85), (0.47, .64, 0.8), (0.72, .78, 0.75), (0.96, .58, 0.9)], base=.70, rough=.8,
         shadow=(112, 150, 172), light=(176, 205, 222), haze=(218, 232, 241), snow=True, mist=.26),
    dict(peaks=[(0.18, .48, 0.9), (0.40, .66, 0.75), (0.60, .82, 0.7), (0.80, 1.00, 0.65), (0.99, .66, 0.8)], base=.82, rough=1.0,
         shadow=(70, 116, 140), light=(132, 174, 195), haze=(204, 225, 236), snow=True, mist=.34),
    dict(peaks=[(0.04, .32, 0.9), (0.33, .40, 0.85), (0.58, .52, 0.75), (0.86, .46, 0.8)], base=.93, rough=.9,
         shadow=(48, 94, 116), light=(94, 142, 164), haze=(186, 212, 226), snow=False, mist=.42),
    dict(peaks=[(0.14, .26, 1.0), (0.52, .22, 1.1), (0.92, .30, 0.9)], base=1.04, rough=.7,
         shadow=(34, 74, 92), light=(66, 112, 132), haze=(168, 198, 213), snow=False, mist=.5),
]
SNOW_LIGHT = np.array((247, 251, 253), np.float32)
SNOW_SHADE = np.array((176, 199, 216), np.float32)
RIM_TEAL = np.array((120, 232, 214), np.float32)


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _noise2d(rng: np.random.Generator, gh: int, gw: int) -> np.ndarray:
    """Smooth value noise in [0, 1], H x W, from a gh x gw random grid (bicubic upscale)."""
    grid = rng.random((gh, gw)).astype(np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _fbm_1d(rng: np.random.Generator, octaves: int = 6, ridged: bool = False) -> np.ndarray:
    """Fractal noise along x in [-1, 1]; the high octaves give the jagged alpine ridgeline. `ridged` folds each
    octave (1 - |n|) so it comes out as sharp crags instead of rounded hills."""
    xs = np.linspace(0, 1, W)
    out = np.zeros(W, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        pts = 4 * 2 ** o
        n = np.interp(xs, np.linspace(0, 1, pts + 1), rng.random(pts + 1) * 2 - 1).astype(np.float32)
        out += amp * ((1.0 - np.abs(n)) * 2 - 1 if ridged else n)
        total += amp
        amp *= 0.55
    return out / total


def _ridge(layer: dict, rng: np.random.Generator) -> np.ndarray:
    """Ridge height in pixels above the layer's base line, one value per column."""
    xs = np.linspace(0, 1, W)
    h = np.zeros(W, np.float32)
    for cx, ph, spread in layer["peaks"]:
        tent = ph * np.clip(1.0 - np.abs(xs - cx) / (0.26 * spread), 0.0, 1.0) ** 1.18   # straight alpine flanks
        h = np.maximum(h, tent)
    rough = layer["rough"]
    crags = _fbm_1d(rng, 8, ridged=True)
    h = h * (0.94 + 0.12 * _fbm_1d(rng, 3)) + 0.085 * rough * crags * np.clip(h * 2.2 + 0.1, 0.1, 1.0) \
        + 0.02 * rough * _fbm_1d(rng, 9)
    return np.clip(h, 0.015, None) * H * 0.62


def _paint_layer(rgb: np.ndarray, alpha: np.ndarray, layer: dict, rng: np.random.Generator) -> None:
    base_y = layer["base"] * H
    ridge_h = _ridge(layer, rng)
    top_y = base_y - ridge_h                                   # ridgeline y per column
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    depth = ys - top_y[None, :]                                # >0 below the ridgeline
    inside = depth >= 0

    # lighting, sun from the upper left. Each summit has an arete (a ridge line) running down its face: left of it the
    # rock is lit, right of it in shade -- with the arete wandering as it descends, so faces read as planes of rock.
    wob = (_noise2d(rng, 3, 40) - 0.5) * 120.0 + (_noise2d(rng, 8, 110) - 0.5) * 46.0
    num = np.zeros((H, W), np.float32)
    den = np.zeros((1, W), np.float32)
    for cx, _ph, spread in layer["peaks"]:
        arete = cx * W + wob + depth * 0.10
        wgt = 1.0 / (1.0 + (np.abs(xs - cx * W) / (0.13 * W * spread)) ** 3)   # each summit rules its own flanks...
        num += wgt * _smooth((arete - xs) / 70.0 + 0.5)                       # 1 = left of the arete (lit face)
        den += wgt
    side = num / den                                           # ...and the faces blend smoothly across the valleys
    slope = np.convolve(np.gradient(top_y), np.ones(15, np.float32) / 15, mode="same")
    near_edge = np.clip(0.5 + slope * 0.9, 0.0, 1.0)[None, :]  # the flank facing the light, right at the ridgeline
    gully = _noise2d(rng, 14, 120) * 0.6 + _noise2d(rng, 40, 380) * 0.4
    grain = _noise2d(rng, 120, 900)
    lit = 0.12 + 0.62 * side + 0.16 * near_edge + 0.34 * (gully - 0.5) + 0.10 * (grain - 0.5)
    lit = np.clip(lit, 0.0, 1.0) * (1.0 - 0.50 * _smooth(depth / (0.5 * H)))   # light flattens toward the foot

    shadow = np.array(layer["shadow"], np.float32)
    light = np.array(layer["light"], np.float32)
    haze = np.array(layer["haze"], np.float32)
    col = shadow[None, None, :] + (light - shadow)[None, None, :] * lit[..., None]

    if layer["snow"]:
        # snow caps: a broken snowline a fixed fraction down from each layer's summits, whiter on the lit faces
        summit_top = top_y.min()
        line = summit_top + (base_y - summit_top) * 0.40
        lim = line + (_noise2d(rng, 9, 120) - 0.5) * 110.0 + (gully - 0.5) * 60.0 - (1.0 - side) * 14.0
        snow = _smooth((lim - ys) / 26.0) * inside
        snow *= _smooth((0.25 + 0.95 * side + 0.5 * (gully - 0.5)) * 1.6)
        tone = np.clip(0.30 + 0.78 * side, 0.0, 1.0)[..., None]
        snow_col = SNOW_SHADE[None, None, :] + (SNOW_LIGHT - SNOW_SHADE)[None, None, :] * tone
        col = col * (1 - snow[..., None]) + snow_col * snow[..., None]

    # a thin teal rim of light along the ridgeline -- the daytime echo of the dark theme's rim-glow
    rim = np.exp(-np.clip(depth, 0, None) / 2.6) * inside * 0.55
    col = col * (1 - rim[..., None] * 0.55) + RIM_TEAL[None, None, :] * (rim[..., None] * 0.55)

    # valley mist: the foot of the layer dissolves into haze, which is what separates it from the next layer
    mist = _smooth(depth / (layer["mist"] * H + 1.0) - 0.15)
    mist = np.clip(mist + 0.10 * (_noise2d(rng, 4, 24) - 0.5), 0, 1) * 0.92
    col = col * (1 - mist[..., None]) + haze[None, None, :] * mist[..., None]

    # soft anti-aliased ridge edge
    cover = np.clip(depth + 0.5, 0.0, 1.0)
    rgb[:] = rgb * (1 - cover[..., None]) + col * cover[..., None]
    alpha[:] = np.maximum(alpha, cover)


def build(out: Path = OUT, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)

    # dawn glow behind the peaks (painted first, so it only shows in the sky): soft teal into white
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((xs - 0.74 * W) / (0.42 * W)) ** 2 + ((ys - 0.50 * H) / (0.30 * H)) ** 2))
    g2 = np.exp(-(((xs - 0.20 * W) / (0.30 * W)) ** 2 + ((ys - 0.62 * H) / (0.22 * H)) ** 2))
    glow = np.clip(g * 0.55 + g2 * 0.25, 0, 1)
    rgb[:] = np.array((176, 238, 226), np.float32)[None, None, :]
    alpha[:] = glow * 0.55

    for layer in LAYERS:
        _paint_layer(rgb, alpha, layer, rng)

    # dissolve into the page: feather the top and sides a hair, fade the bottom ~35% to transparent
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    fade = (1.0 - _smooth((yy - 0.66) / 0.34)) * _smooth(yy / 0.03) * _smooth(xx / 0.04) * _smooth((1.0 - xx) / 0.02)
    alpha = np.clip(alpha * fade, 0.0, 1.0)

    rgba = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out, "WEBP", quality=88, method=6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    p = build(a.out, a.seed)
    print(f"wrote {p} ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
