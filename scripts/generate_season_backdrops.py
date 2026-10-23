#!/usr/bin/env python3
"""Build the four seasonal page backdrops (light + dark each) used by Account settings -> Personalize.

Writes eight static WebP files to app/web/static/backgrounds/:
    spring-dark.webp  spring-light.webp  summer-dark.webp  summer-light.webp
    autumn-dark.webp  autumn-light.webp  winter-dark.webp  winter-light.webp

"dark" = the night scene shown with the app's dark theme, "light" = the day scene shown with the light theme.
Each file is a 2048x768 RGBA image (the same box the default mountain backdrop uses): solid in the middle, easing to
transparent at the top / left / right edges and over the lower ~34%, so it melts into the page background with no hard
edge anywhere.

The artwork itself is painted, not generated: this script slices it out of two source pictures and finishes it.

    --winter FILE   a picture with the winter NIGHT scene stacked on top of the winter DAY scene (two equal halves)
    --grid FILE     a picture with a 4-row x 2-column grid of scenes: rows are spring, summer, winter, autumn (winter is
                    taken from --winter instead when that is given) and columns are night (left) / day (right).
                    The grid is located automatically (the largest non-black block), so a phone screenshot with black
                    bars above and below it works as-is.

    python scripts/generate_season_backdrops.py --winter winter.webp --grid seasons-grid.jpg
    python scripts/generate_season_backdrops.py --winter winter.webp --grid seasons-grid.jpg --preview

Needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "app" / "web" / "static" / "backgrounds"
W, H = 2048, 768
SEASONS = ("spring", "summer", "winter", "autumn")  # row order inside the grid picture
INSET = 2  # px trimmed from every tile edge to drop seam / compression artefacts


def smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def find_grid_box(img: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the biggest contiguous block of non-black rows (ignores status / navigation bars)."""
    lum = np.asarray(img.convert("L"), dtype=np.float32)
    rows = lum.mean(axis=1) > 8.0
    best, start = (0, 0), None
    for i, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    y0, y1 = best
    cols = np.where(lum[y0:y1].mean(axis=0) > 8.0)[0]
    return int(cols.min()), y0, int(cols.max()) + 1, y1


def split_grid(img: Image.Image) -> dict[tuple[str, str], Image.Image]:
    x0, y0, x1, y1 = find_grid_box(img)
    cw, ch = (x1 - x0) / 2.0, (y1 - y0) / float(len(SEASONS))
    out: dict[tuple[str, str], Image.Image] = {}
    for r, season in enumerate(SEASONS):
        for c, mode in enumerate(("dark", "light")):
            box = (round(x0 + c * cw) + INSET, round(y0 + r * ch) + INSET,
                   round(x0 + (c + 1) * cw) - INSET, round(y0 + (r + 1) * ch) - INSET)
            out[(season, mode)] = img.crop(box)
    return out


def split_stack(img: Image.Image) -> dict[tuple[str, str], Image.Image]:
    """Top half = night (dark theme), bottom half = day (light theme)."""
    w, h = img.size
    half = h // 2
    return {("winter", "dark"): img.crop((INSET, INSET, w - INSET, half - INSET)),
            ("winter", "light"): img.crop((INSET, half + INSET, w - INSET, h - INSET))}


def fit_cover(tile: Image.Image) -> Image.Image:
    """Scale to cover W x H (centre crop), with a mild unsharp mask and a whisper of grain when upscaling a lot."""
    tile = tile.convert("RGB")
    s = max(W / tile.width, H / tile.height)
    nw, nh = max(round(tile.width * s), W), max(round(tile.height * s), H)
    if s > 1.6:  # big upscale: go in two Lanczos steps, it keeps edges cleaner than one jump
        mid = tile.resize((round(tile.width * s ** 0.5), round(tile.height * s ** 0.5)), Image.LANCZOS)
        tile = mid.filter(ImageFilter.UnsharpMask(radius=1.0, percent=45, threshold=2))
    big = tile.resize((nw, nh), Image.LANCZOS)
    big = big.filter(ImageFilter.UnsharpMask(radius=1.6 if s > 1.6 else 1.0, percent=55 if s > 1.6 else 25, threshold=2))
    left, top = (nw - W) // 2, (nh - H) // 2
    arr = np.asarray(big.crop((left, top, left + W, top + H)), dtype=np.float32) / 255.0
    if s > 1.6:  # hides the softness of a small source without looking noisy
        rng = np.random.default_rng(58)
        lum = arr.mean(axis=2, keepdims=True)
        g = rng.normal(0.0, 0.010, (H, W, 1)).astype(np.float32)
        arr = np.clip(arr + g * (0.55 + 0.9 * (1.0 - lum)), 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "RGB")


def with_fade(rgb: Image.Image) -> Image.Image:
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    alpha = (1.0 - smooth((ys - 0.66) / 0.34)) * smooth(ys / 0.05) * smooth(xs / 0.06) * smooth((1.0 - xs) / 0.06)
    a = (np.clip(alpha, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    rgba = np.dstack([np.asarray(rgb, dtype=np.uint8), a])
    return Image.fromarray(rgba, "RGBA")


def preview(rgba: Image.Image, path: Path, light: bool) -> None:
    bg = Image.new("RGB", rgba.size, (230, 237, 243) if light else (3, 8, 10))
    bg.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    bg.save(path.with_suffix(".preview.png"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winter", type=Path, help="winter night-over-day picture (two stacked halves)")
    ap.add_argument("--grid", type=Path, help="4x2 grid picture: rows spring/summer/winter/autumn, columns night/day")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--preview", action="store_true", help="also write .preview.png composites next to the WebP files")
    args = ap.parse_args(argv)
    if not (args.winter or args.grid):
        ap.error("give --winter and/or --grid")

    tiles: dict[tuple[str, str], Image.Image] = {}
    if args.grid:
        tiles.update(split_grid(Image.open(args.grid)))
    if args.winter:
        tiles.update(split_stack(Image.open(args.winter)))  # the dedicated winter picture wins over the grid's row

    args.out.mkdir(parents=True, exist_ok=True)
    for (season, mode), tile in sorted(tiles.items()):
        rgba = with_fade(fit_cover(tile))
        p = args.out / f"{season}-{mode}.webp"
        rgba.save(p, "WEBP", quality=86, method=6)
        print("wrote", p, f"({p.stat().st_size // 1024} KB)", flush=True)
        if args.preview:
            preview(rgba, p, light=(mode == "light"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
