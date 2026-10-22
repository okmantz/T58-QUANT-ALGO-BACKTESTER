#!/usr/bin/env python3
"""Build the four seasonal page backdrops (light + dark each) used by Account settings -> Personalize.

Writes eight static WebP files to app/web/static/backgrounds/:
    spring-dark.webp  spring-light.webp  summer-dark.webp  summer-light.webp
    autumn-dark.webp  autumn-light.webp  winter-dark.webp  winter-light.webp

"dark" = the night scene shown with the app's dark theme, "light" = the day scene shown with the light theme.
They are the same 2048x768 RGBA format as the default mountain backdrop (dashboard-mountains.webp): the scene is
solid at the top and dissolves to transparent over the lower ~38% so it melts into the page background.

The scenes are generated procedurally (layered fractal ridgelines, atmospheric haze, season-specific trees, lake
reflections, aurora, stars) -- deterministic from a fixed seed, no source photos, zero runtime cost in the app (one
static image, no filters, no animation). Re-run to regenerate:

    python scripts/generate_season_backdrops.py            # all eight
    python scripts/generate_season_backdrops.py winter     # one season

Needs numpy and Pillow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "app" / "web" / "static" / "backgrounds"
W, H = 2048, 768


def hexc(s: str) -> np.ndarray:
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fbm(n: int, rng: np.random.Generator, octaves: int = 6, base: float = 3.0, gain: float = 0.5) -> np.ndarray:
    """1-D fractal noise in [0, 1], length n."""
    x = np.linspace(0, 1, n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    amp, freq, tot = 1.0, base, 0.0
    for _ in range(octaves):
        pts = rng.random(int(freq) + 3).astype(np.float32)
        idx = x * freq
        i0 = np.floor(idx).astype(int)
        f = smooth(idx - i0)
        out += amp * (pts[i0] * (1 - f) + pts[i0 + 1] * f)
        tot += amp
        amp *= gain
        freq *= 2.0
    out /= tot
    return (out - out.min()) / max(float(out.max() - out.min()), 1e-6)


def blur(arr: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


class Canvas:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.img = np.zeros((H, W, 3), dtype=np.float32)
        self.yy = np.arange(H, dtype=np.float32)[:, None]
        self.xx = np.arange(W, dtype=np.float32)[None, :]

    def sky(self, top: str, mid: str, horizon: str, horizon_y: float):
        t = np.clip(self.yy / horizon_y, 0, 1)
        top, mid, hor = hexc(top), hexc(mid), hexc(horizon)
        a = smooth(t * 2.0)[..., None]
        b = smooth(t * 2.0 - 1.0)[..., None]
        col = top * (1 - a) + mid * a
        col = col * (1 - b) + hor * b
        self.img[:] = np.broadcast_to(col, (H, W, 3))

    def glow(self, cx, cy, rx, ry, color: str, strength: float):
        d = ((self.xx - cx) / rx) ** 2 + ((self.yy - cy) / ry) ** 2
        g = np.exp(-d * 2.2)[..., None] * strength
        self.img += g * hexc(color)

    def disc(self, cx, cy, r, color: str, halo: str, halo_strength: float = 0.5):
        d = np.sqrt((self.xx - cx) ** 2 + (self.yy - cy) ** 2)
        m = (1.0 - smooth((d - r) / 2.5))[..., None]
        self.glow(cx, cy, r * 7, r * 7, halo, halo_strength)
        self.img = self.img * (1 - m) + hexc(color) * m

    def stars(self, count: int, max_y: float, strength: float = 1.0):
        ys = (self.rng.random(count) ** 1.6 * max_y).astype(int)
        xs = self.rng.integers(0, W, count)
        br = (self.rng.random(count) ** 3 * 0.9 + 0.1) * strength
        fade = 1.0 - smooth(ys / max_y) * 0.85
        for y, x, b, f in zip(ys, xs, br, fade):
            self.img[y, x] += b * f
            if b > 0.55:
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    yy, xx = min(max(y + dy, 0), H - 1), min(max(x + dx, 0), W - 1)
                    self.img[yy, xx] += b * f * 0.35

    def ridge(self, base_y: float, amp: float, color_top: str, color_bot: str, haze: str, haze_amt: float,
              rim: str | None = None, rim_strength: float = 0.0, base: float = 3.0, octaves: int = 6,
              sharp: float = 1.0) -> np.ndarray:
        n = fbm(W, self.rng, octaves, base) ** sharp
        ry = base_y - n * amp
        depth = np.clip((self.yy - ry[None, :]) / 260.0, 0, 1)
        mask = (self.yy >= ry[None, :] - 0.5)
        edge = np.clip(1.0 - np.abs(self.yy - ry[None, :]) / 1.6, 0, 1)
        col = hexc(color_top) * (1 - depth[..., None]) + hexc(color_bot) * depth[..., None]
        col = col * (1 - haze_amt) + hexc(haze) * haze_amt
        m = mask[..., None].astype(np.float32)
        # feather the silhouette edge 1px so ridges aren't aliased
        soft = np.clip((self.yy - ry[None, :] + 1.0) / 1.6, 0, 1)[..., None]
        self.img = self.img * (1 - soft) + col * soft
        if rim and rim_strength:
            rim_band = np.clip(1.0 - np.abs(self.yy - ry[None, :] - 1.5) / 5.0, 0, 1)[..., None]
            self.img += rim_band * hexc(rim) * rim_strength
        _ = (m, edge)
        return ry

    def mist(self, y: float, height: float, color: str, strength: float):
        band = np.exp(-(((self.yy - y) / height) ** 2))[..., None]
        wob = (0.6 + 0.4 * fbm(W, self.rng, 4, 6.0))[None, :, None]
        self.img = self.img * (1 - band * wob * strength) + hexc(color) * band * wob * strength

    # ---- trees -------------------------------------------------------------------------------------------------
    def pine(self, x: int, ground_y: int, h: int, color: str, snow: str | None = None, snow_amt: float = 0.0):
        tiers = 5
        w0 = h * 0.34
        col = hexc(color)
        sn = hexc(snow) if snow else None
        for t in range(tiers):
            top = ground_y - h + int(t * h / (tiers + 0.5))
            bot = ground_y - h + int((t + 1.6) * h / (tiers + 0.5))
            bot = min(bot, ground_y)
            tw = w0 * (0.35 + 0.65 * (t + 1) / tiers)
            ys = np.arange(max(top, 0), min(bot, H))
            if len(ys) == 0:
                continue
            for y in ys:
                frac = (y - top) / max(bot - top, 1)
                half = tw * frac
                x0, x1 = int(x - half), int(x + half) + 1
                x0, x1 = max(x0, 0), min(x1, W)
                if x1 <= x0:
                    continue
                c = col
                if sn is not None and frac < 0.45:
                    c = col * (1 - snow_amt) + sn * snow_amt
                self.img[y, x0:x1] = c

    def round_tree(self, x: int, ground_y: int, h: int, trunk: str, canopy: list[str], glow: str | None = None,
                   glow_strength: float = 0.0):
        """Foliage tree: thin trunk + a canopy of many small leaf-dots clustered in an ellipse (reads as
        textured foliage rather than a few big balls), lit from the top."""
        tw = max(1, h // 40)
        ty0 = ground_y - int(h * 0.5)
        self.img[max(ty0, 0):ground_y, max(x - tw, 0):min(x + tw + 1, W)] = hexc(trunk)
        cx, cy = x, ground_y - h * 0.68
        rx, ry = h * 0.34, h * 0.30
        n = int(h * 5)
        ang = self.rng.random(n) * 6.2832
        rad = np.sqrt(self.rng.random(n))
        px = (cx + np.cos(ang) * rad * rx).astype(int)
        py = (cy + np.sin(ang) * rad * ry * 1.05).astype(int)
        size = np.maximum(1, (h * (0.018 + 0.03 * self.rng.random(n))).astype(int))
        top_lit = 0.78 + 0.34 * np.clip(1 - (py - (cy - ry)) / (2 * ry), 0, 1)
        picks = self.rng.integers(0, len(canopy), n)
        cols = np.stack([hexc(c) for c in canopy])
        for i in range(n):
            x0, y0 = px[i], py[i]
            if 0 <= x0 < W - 4 and 0 <= y0 < H - 4:
                sz = size[i]
                self.img[y0:y0 + sz, x0:x0 + sz] = cols[picks[i]] * top_lit[i]
        if glow and glow_strength:
            self.glow(cx, cy, h * 0.9, h * 0.7, glow, glow_strength)

    def water(self, y0: int, tint: str, tint_amt: float, ripple: float, shimmer: str | None = None):
        """Mirror everything above y0 into the lake below it, darkened, rippled and tinted."""
        below = H - y0
        src = self.img[max(2 * y0 - H, 0):y0][::-1]
        if src.shape[0] < below:
            pad = np.repeat(src[-1:], below - src.shape[0], axis=0)
            src = np.concatenate([src, pad], axis=0)
        src = src[:below].copy()
        rows = np.arange(below, dtype=np.float32)
        disp = (np.sin(rows * 0.9 + self.rng.random() * 6) * ripple * (0.3 + rows / below))[:, None]
        xi = np.clip((self.xx + disp).astype(int), 0, W - 1)
        src = np.take_along_axis(src, np.repeat(xi[..., None], 3, axis=2), axis=1)
        src = blur(src, 1.4)
        depth = smooth(rows / below)[:, None, None]
        water = src * (0.72 - 0.30 * depth) * (1 - tint_amt) + hexc(tint) * tint_amt
        self.img[y0:] = water
        if shimmer:
            streak = (self.rng.random((below, 1)) > 0.93).astype(np.float32) * (0.10 + 0.14 * self.rng.random((below, 1)))
            streak = streak * (0.4 + fbm(W, self.rng, 3, 10.0)[None, :])
            self.img[y0:] += streak[..., None] * hexc(shimmer)
        shore = np.clip(1 - np.abs(self.yy[y0:y0 + 6] - y0) / 3.0, 0, 1)[..., None]
        self.img[y0:y0 + 6] += shore * 0.05


def aurora(cv: Canvas, colors: list[str], y_range=(70, 300), strength=0.55):
    for k, c in enumerate(colors):
        base = y_range[0] + (y_range[1] - y_range[0]) * (k + 0.5) / len(colors)
        curve = base + 60 * (fbm(W, cv.rng, 3, 2.0) - 0.5) * 2 + 20 * np.sin(np.linspace(0, 9, W) + k)
        thick = 55 + 40 * fbm(W, cv.rng, 3, 3.0)
        d = (cv.yy - curve[None, :]) / thick[None, :]
        band = np.where(d < 0, np.exp(-(d * 1.6) ** 2 * 2.5), np.exp(-(d * 0.7) ** 2 * 3.0))
        rays = 0.55 + 0.45 * fbm(W, cv.rng, 5, 40.0)[None, :]
        cv.img += (band * rays)[..., None] * hexc(c) * strength


def waterfall(cv: Canvas, x: int, y_top: int, y_bot: int, width: int, color: str, strength: float):
    xs = np.arange(max(x - width * 3, 0), min(x + width * 3, W))
    ys = np.arange(max(y_top, 0), min(y_bot, H))
    if len(xs) == 0 or len(ys) == 0:
        return
    gx = np.exp(-(((xs - x) / width) ** 2))[None, :]
    streaks = 0.65 + 0.35 * cv.rng.random((1, len(xs)))
    fade = (0.55 + 0.45 * smooth((ys - y_top) / 30.0))[:, None]
    m = (gx * streaks * fade * strength)[..., None]
    cv.img[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] += m * hexc(color)
    cv.glow(x, y_bot, width * 7, 38, color, strength * 0.5)  # mist at the base


def finish(cv: Canvas, path: Path) -> None:
    rgb = blur(np.clip(cv.img, 0, 1), 0.7)  # soften vector edges slightly
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    alpha = (1.0 - smooth((ys - 0.66) / 0.34)) * smooth(ys / 0.04) * smooth(xs / 0.05) * smooth((1 - xs) / 0.03)
    rgba = np.dstack([(rgb * 255).astype(np.uint8), (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=82, method=6)


# ---------------------------------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------------------------------
def scatter(cv: Canvas, ry: np.ndarray, n: int, fn, lo=0.04, hi=0.96, min_gap=14):
    xs = np.sort(cv.rng.integers(int(W * lo), int(W * hi), n))
    last = -999
    for x in xs:
        if x - last < min_gap:
            continue
        last = x
        fn(int(x), int(ry[int(x)]))


def scene(season: str, night: bool, seed: int) -> Canvas:
    rng = np.random.default_rng(seed)
    cv = Canvas(rng)
    hy = 430.0  # horizon / lake line region

    if season == "spring":
        if night:
            cv.sky("#070b24", "#2a2150", "#8a5a86", hy)
            cv.stars(700, 260, 0.9)
            cv.disc(1500, 150, 34, "#fff3f6", "#ffb7d5", 0.45)
            far = cv.ridge(430, 210, "#3a3466", "#241f4a", "#8a5a86", 0.55, "#ffc3dc", 0.35, 2.4)
            mid = cv.ridge(500, 190, "#262250", "#17143a", "#6d4a7c", 0.40, "#ff9ec8", 0.22, 3.2)
            waterfall(cv, 560, int(mid[560]) - 30, 560, 9, "#d6c9ff", 0.75)
            near = cv.ridge(585, 120, "#14122f", "#0a0920", "#3a2a5a", 0.20, "#ff8fbf", 0.15, 4.0)
            cv.mist(560, 60, "#c78fb8", 0.35)
            trunk, canopy = "#1d1030", ["#ff9ac6", "#ffb3d6", "#e86fae", "#ffc9e1"]
            glow, gs = "#ff7fb8", 0.10
        else:
            cv.sky("#7fbdf0", "#bfe0f7", "#fbe4ee", hy)
            cv.glow(1450, 160, 340, 190, "#fff7e0", 0.22)
            cv.disc(1450, 150, 38, "#fffdf2", "#fff2c4", 0.5)
            far = cv.ridge(430, 210, "#9fc2dc", "#86abc9", "#f6e2ea", 0.50, "#ffffff", 0.18, 2.4)
            mid = cv.ridge(500, 190, "#79a5a5", "#5f8f93", "#e7dfe8", 0.35, "#f6fff0", 0.15, 3.2)
            waterfall(cv, 560, int(mid[560]) - 30, 560, 9, "#ffffff", 0.9)
            near = cv.ridge(585, 120, "#4f8260", "#3b6a50", "#cfdcd0", 0.18, None, 0, 4.0)
            cv.mist(560, 60, "#ffffff", 0.45)
            trunk, canopy = "#5a3f3f", ["#ffc0d9", "#ffd5e5", "#f4a0c4", "#ffe6ef"]
            glow, gs = None, 0.0
        # blossom trees on the mid / near ridges
        scatter(cv, mid, 60, lambda x, y: cv.round_tree(x, y + 6, int(46 + rng.random() * 36), trunk, canopy), min_gap=22)
        scatter(cv, near, 36, lambda x, y: cv.round_tree(x, y + 8, int(80 + rng.random() * 80), trunk, canopy, glow, gs), min_gap=42)
        # petals drifting
        for _ in range(240):
            px, py = int(rng.integers(0, W)), int(rng.integers(40, 560))
            cv.img[py:py + 2, px:px + 3] += hexc("#ffd0e4") * (0.55 if night else 0.9) * (0.3 + rng.random() * 0.7)

    elif season == "summer":
        if night:
            cv.sky("#050c22", "#0f2f55", "#2b7d93", hy)
            cv.stars(600, 250, 0.85)
            cv.disc(450, 140, 30, "#f4f8ff", "#9bd0ff", 0.4)
            cv.glow(1100, 440, 900, 120, "#ffb06b", 0.35)
            far = cv.ridge(440, 200, "#1e4a6a", "#12304e", "#2b7d93", 0.50, "#7fe3ff", 0.25, 2.6)
            mid = cv.ridge(490, 150, "#102c46", "#0a1d33", "#1f5e78", 0.36, "#5fd0f0", 0.18, 3.4)
            pine_c, snow = "#07151f", None
        else:
            cv.sky("#2f93e0", "#7cc4f2", "#d9f0fa", hy)
            cv.glow(1500, 150, 380, 200, "#fff5cc", 0.22)
            cv.disc(1500, 150, 40, "#fffbe6", "#fff0b0", 0.25)
            far = cv.ridge(440, 200, "#7fb0cc", "#6a9bba", "#d9ecf5", 0.50, "#ffffff", 0.15, 2.6)
            mid = cv.ridge(490, 150, "#4f8f86", "#3c7a73", "#bcdad6", 0.32, None, 0, 3.4)
            pine_c, snow = "#1f5a46", None
        scatter(cv, mid, 70, lambda x, y: cv.pine(x, y + 6, int(46 + rng.random() * 44), pine_c), min_gap=22)
        near = cv.ridge(530, 80, "#0a1a14" if night else "#2c6b4e", "#06110c" if night else "#215640",
                        "#103045" if night else "#9fcab8", 0.16, None, 0, 4.2)
        scatter(cv, near, 46, lambda x, y: cv.pine(x, y + 4, int(70 + rng.random() * 90), "#050f0a" if night else "#164a38"), min_gap=34)
        cv.water(548, "#0c2a44" if night else "#4aa6d6", 0.38, 2.0, "#9fe0ff" if night else "#ffffff")
        cv.mist(548, 20, "#7fb8d0" if night else "#ffffff", 0.18)

    elif season == "autumn":
        if night:
            cv.sky("#0c0816", "#2a1428", "#a2492a", hy)
            cv.stars(420, 240, 0.7)
            cv.disc(1380, 170, 36, "#ffe8c4", "#ff9a52", 0.5)
            cv.glow(900, 450, 1000, 130, "#ff7a3a", 0.45)
            far = cv.ridge(440, 190, "#3d2038", "#251328", "#8a3c2c", 0.45, "#ff9a5a", 0.28, 2.6)
            mid = cv.ridge(500, 150, "#241222", "#150b17", "#6a2c28", 0.32, "#ff7e3d", 0.18, 3.4)
            canopy = ["#ff7a2e", "#ff5a1f", "#ffb02e", "#d9381e"]
            trunk, gl = "#120a10", ("#ff7a2e", 0.10)
        else:
            cv.sky("#e9a863", "#f6cf9c", "#fdeed6", hy)
            cv.glow(1450, 190, 380, 200, "#fff0cc", 0.25)
            cv.disc(1450, 170, 42, "#fffaf0", "#ffe2a8", 0.5)
            far = cv.ridge(440, 190, "#c99373", "#b57f66", "#f7e2c8", 0.50, "#fff3df", 0.15, 2.6)
            mid = cv.ridge(500, 150, "#b5703f", "#9c5b33", "#ecc9a0", 0.32, None, 0, 3.4)
            canopy = ["#e8651d", "#f08a2a", "#f3b83a", "#c93a1e"]
            trunk, gl = "#5a3a28", (None, 0.0)
        scatter(cv, mid, 80, lambda x, y: cv.round_tree(x, y + 6, int(44 + rng.random() * 36), trunk, canopy), min_gap=20)
        near = cv.ridge(530, 80, "#1c0e14" if night else "#8c4a28", "#10080c" if night else "#6e381e",
                        "#3a1a22" if night else "#d8a878", 0.16, None, 0, 4.2)
        scatter(cv, near, 40, lambda x, y: cv.round_tree(x, y + 6, int(84 + rng.random() * 84), trunk, canopy, gl[0], gl[1]), min_gap=40)
        cv.water(548, "#2a1018" if night else "#d9893f", 0.34, 2.4, "#ffb072" if night else "#fff2d6")
        cv.mist(548, 20, "#e08a5a" if night else "#fff2dc", 0.18)

    else:  # winter
        if night:
            cv.sky("#030816", "#0a2236", "#1f5a66", hy)
            cv.stars(900, 280, 1.0)
            aurora(cv, ["#3dffb0", "#35d6ff", "#8a6bff"], (60, 300), 0.50)
            cv.disc(260, 120, 24, "#eef6ff", "#9fc8ff", 0.35)
            far = cv.ridge(440, 210, "#2c4a68", "#1c3350", "#2f6a78", 0.45, "#bfeaff", 0.35, 2.4)
            mid = cv.ridge(500, 170, "#1b3047", "#11202f", "#244f60", 0.32, "#9fdcf5", 0.25, 3.2)
            pine_c, snow_c = "#06121b", "#8fb4c8"
        else:
            cv.sky("#a8cde8", "#d3e6f3", "#f2f8fc", hy)
            cv.glow(1500, 200, 380, 190, "#ffffff", 0.25)
            cv.disc(1500, 180, 36, "#ffffff", "#eaf4ff", 0.5)
            far = cv.ridge(440, 210, "#dbe8f2", "#c5d8e8", "#f2f8fc", 0.42, "#ffffff", 0.3, 2.4)
            mid = cv.ridge(500, 170, "#b9cfe0", "#9db8cf", "#e3eef6", 0.28, "#ffffff", 0.2, 3.2)
            pine_c, snow_c = "#4a6f6a", "#f4fbff"
        scatter(cv, mid, 80, lambda x, y: cv.pine(x, y + 6, int(48 + rng.random() * 46), pine_c, snow_c, 0.45), min_gap=20)
        near = cv.ridge(532, 80, "#0d1a26" if night else "#e8f1f8", "#08111a" if night else "#cbdcea",
                        "#16303f" if night else "#f2f8fc", 0.14, None, 0, 4.2)
        scatter(cv, near, 40, lambda x, y: cv.pine(x, y + 4, int(80 + rng.random() * 100), "#040d14" if night else "#3d6660", snow_c, 0.5), min_gap=34)
        cv.water(552, "#0a2236" if night else "#bcd7ea", 0.45, 1.2, "#a8e8ff" if night else "#ffffff")
        # snowfall
        for _ in range(520):
            px, py = int(rng.integers(0, W)), int(rng.integers(30, 600))
            cv.img[py:py + 2, px:px + 2] += 0.55 * (0.4 + rng.random() * 0.6)

    # gentle overall vignette so the scene never fights page content
    return cv


SEASONS = ("spring", "summer", "autumn", "winter")


def build(seasons=SEASONS, out_dir: Path = OUT_DIR) -> list[Path]:
    written = []
    for si, season in enumerate(SEASONS):
        if season not in seasons:
            continue
        for night in (True, False):
            cv = scene(season, night, seed=7000 + si * 10 + (0 if night else 1))
            p = out_dir / f"{season}-{'dark' if night else 'light'}.webp"
            finish(cv, p)
            written.append(p)
            print("wrote", p.relative_to(ROOT))
    return written


if __name__ == "__main__":
    wanted = tuple(a for a in sys.argv[1:] if a in SEASONS) or SEASONS
    build(wanted)
