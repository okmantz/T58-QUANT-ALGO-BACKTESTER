#!/usr/bin/env python3
"""Build the four seasonal page backdrops (light + dark each) used by Account settings -> Personalize.

Writes eight static WebP files to app/web/static/backgrounds/:
    spring-dark.webp  spring-light.webp  summer-dark.webp  summer-light.webp
    autumn-dark.webp  autumn-light.webp  winter-dark.webp  winter-light.webp

"dark" = the night scene shown with the app's dark theme, "light" = the day scene shown with the light theme.
Same 2048x768 RGBA format as the default mountain backdrop: solid at the top, dissolving to transparent over the
lower ~34% so it melts into the page background.

v2 (photographic look). Everything is still generated in code -- no source photos, no AI model, deterministic from a
fixed seed, zero runtime cost in the app (one static image) -- but rendered like a matte painting rather than a
cartoon: physically-shaded cloud decks, scattering-style sky and sun/moon glow, Milky Way and cratered moon at night,
ridged-noise mountains lit from the sun side with snow, rock and forest zones, haze and depth-of-field that grow with
distance, individually lit trees (needle tiers / leaf clusters, trunks, snow load), a rippled Fresnel lake with a sun /
moon glitter path, then bloom, filmic tone mapping, vignette and fine film grain.

    python scripts/generate_season_backdrops.py            # all eight
    python scripts/generate_season_backdrops.py winter     # one season
    python scripts/generate_season_backdrops.py --preview  # also write PNG previews next to the webp files

Needs numpy, scipy and Pillow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "app" / "web" / "static" / "backgrounds"
W, H = 2048, 768
F32 = np.float32


# ---------------------------------------------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------------------------------------------
def hexc(s: str) -> np.ndarray:
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=F32)


def tuple_mix(a: str, b: str, t: float) -> str:
    ca, cb = hexc(a), hexc(b)
    c = (ca * (1 - t) + cb * t * 1.0)
    return '#%02x%02x%02x' % tuple(int(np.clip(v, 0, 1) * 255) for v in c)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mix(a, b, t):
    return a * (1 - t) + b * t


def gblur(a: np.ndarray, s: float) -> np.ndarray:
    if s <= 0:
        return a
    if a.ndim == 3:
        return np.stack([gaussian_filter(a[..., i], s, mode="nearest") for i in range(a.shape[2])], -1)
    return gaussian_filter(a, s, mode="nearest")


def fbm1(n: int, rng: np.random.Generator, octaves: int = 7, base: float = 3.0, gain: float = 0.5) -> np.ndarray:
    """1-D fractal noise in [0, 1], length n."""
    x = np.linspace(0, 1, n, dtype=F32)
    out = np.zeros(n, dtype=F32)
    amp, freq, tot = 1.0, base, 0.0
    for _ in range(octaves):
        pts = rng.random(int(freq) + 3).astype(F32)
        idx = x * freq
        i0 = np.floor(idx).astype(int)
        f = smooth(idx - i0)
        out += amp * (pts[i0] * (1 - f) + pts[i0 + 1] * f)
        tot += amp
        amp *= gain
        freq *= 2.0
    out /= tot
    return (out - out.min()) / max(float(out.max() - out.min()), 1e-6)


def noise2(rng, h: int, w: int, cy: float, cx: float, octaves: int = 5, gain: float = 0.5) -> np.ndarray:
    """2-D fractal value noise in [0, 1] with anisotropic first-octave cell size (cy rows x cx columns)."""
    out = np.zeros((h, w), dtype=F32)
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        if cy < 1.2 and cx < 1.2:
            break
        cye, cxe = max(cy, 1.0), max(cx, 1.0)
        gh, gw = int(h / cye) + 3, int(w / cxe) + 3
        g = rng.random((gh, gw)).astype(F32)
        im = Image.fromarray(g, mode="F").resize((max(int(round(gw * cxe)), 2), max(int(round(gh * cye)), 2)), Image.BICUBIC)
        out += amp * np.asarray(im, dtype=F32)[:h, :w]
        tot += amp
        amp *= gain
        cy, cx = cy / 2.0, cx / 2.0
    out /= max(tot, 1e-6)
    lo, hi = np.percentile(out, 1.0), np.percentile(out, 99.0)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


# ---------------------------------------------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------------------------------------------
class Canvas:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.img = np.zeros((H, W, 3), dtype=F32)
        self.yy = np.arange(H, dtype=F32)[:, None]
        self.xx = np.arange(W, dtype=F32)[None, :]
        self.sun = (W * 0.72, 150.0)
        self.light = 1.0  # +1: light comes from the right, -1: from the left
        # small reusable noise tiles for leaf / needle texture (cheap per-sprite variation)
        self.bank = [noise2(self.rng, 64, 64, 9, 9, 3, 0.6) for _ in range(10)]
        self.fine = [noise2(self.rng, 64, 64, 2, 2, 2, 0.6) for _ in range(10)]

    def tile(self, bank, h, w):
        t = bank[int(self.rng.integers(0, len(bank)))]
        if self.rng.random() < 0.5:
            t = t[:, ::-1]
        if self.rng.random() < 0.5:
            t = t[::-1]
        reps = (h // 64 + 1, w // 64 + 1)
        return np.tile(t, reps)[:h, :w]

    # -- compositing ----------------------------------------------------------------------------------------
    def over(self, rgb, a, blur: float = 0.0):
        a = a[..., None] if a.ndim == 2 else a
        rgb = np.broadcast_to(rgb, (H, W, 3)).astype(F32)
        a = a.astype(F32)
        if blur > 0:
            pre, aa = gblur(rgb * a, blur), gblur(a, blur)
            self.img = self.img * (1 - aa) + pre
        else:
            self.img = self.img * (1 - a) + rgb * a

    def add_glow(self, cx, cy, rx, ry, color, strength, power=1.0):
        d = np.sqrt(((self.xx - cx) / rx) ** 2 + ((self.yy - cy) / ry) ** 2)
        self.img += (np.exp(-(d ** power) * 2.2) * strength)[..., None] * hexc(color) if isinstance(color, str) \
            else (np.exp(-(d ** power) * 2.2) * strength)[..., None] * color

    # -- sky ------------------------------------------------------------------------------------------------
    def sky(self, zen, mid, hor, hz, sun=None, sun_col="#fff1c9", sun_k=1.0, disc_r=0.0):
        t = np.clip(self.yy / hz, 0, 1) ** 1.35
        a, b = smooth(t / 0.55), smooth((t - 0.55) / 0.45)
        col = hexc(zen) * (1 - a[..., None]) + hexc(mid) * a[..., None]
        col = col * (1 - b[..., None]) + hexc(hor) * b[..., None]
        self.img[:] = np.broadcast_to(col, (H, W, 3))
        # subtle large-scale colour drift so the gradient isn't a perfect ramp
        drift = (noise2(self.rng, H, W, 380, 520, 3) - 0.5) * 0.045
        self.img += drift[..., None]
        if sun:
            self.sun = sun
            sx, sy = sun
            d = np.sqrt((self.xx - sx) ** 2 + ((self.yy - sy) * 1.12) ** 2)
            g = 0.62 * np.exp(-d / (0.045 * W)) + 0.30 * np.exp(-d / (0.17 * W)) + 0.16 * np.exp(-d / (0.55 * W))
            self.img += (g * sun_k)[..., None] * hexc(sun_col)
            self.disc_r = disc_r

    def sun_disc(self):
        r = getattr(self, "disc_r", 0.0)
        if not r:
            return
        sx, sy = self.sun
        d = np.sqrt((self.xx - sx) ** 2 + (self.yy - sy) ** 2)
        m = (1 - smooth((d - r) / 3.0))[..., None]
        self.img = self.img * (1 - m) + np.array([2.2, 2.1, 1.8], dtype=F32) * m

    def clouds(self, hz, lit, shade, cover=0.5, op=0.9, scale=1.0, fade_low=0.16, edge_light=0.9):
        rng = self.rng
        D = noise2(rng, int(hz * 2.3), W, 95 * scale, 330 * scale, 6, 0.56)
        s = (np.clip(self.yy[:, 0] / hz, 0, 1) ** 2.0 * hz * 2.2).astype(int)
        Dp = D[s, :]
        c = smooth((Dp - (1 - cover)) / 0.2)
        sx, sy = self.sun
        vx, vy = sx - W / 2, sy - hz * 0.5
        n = max(np.hypot(vx, vy), 1.0)
        dx, dy = int(round(7 * vx / n)), int(round(7 * vy / n)) or -3
        Dsh = np.roll(np.roll(Dp, -dy, 0), -dx, 1)
        lf = np.clip(0.5 + (Dp - Dsh) * 5.5, 0, 1)
        thick = smooth((Dp - (1 - cover)) / 0.55)
        col = hexc(shade) * (1 - lf[..., None]) + hexc(lit) * lf[..., None]
        col = col * (1 - 0.30 * thick[..., None])
        # silver lining near the light source
        d = np.sqrt((self.xx - sx) ** 2 + (self.yy - sy) ** 2)
        col = col + (np.exp(-d / (0.16 * W)) * (1 - thick) * edge_light * 0.6)[..., None] * hexc(lit)
        fade = smooth((hz - self.yy[:, 0]) / (hz * fade_low))[:, None]
        self.over(col, c * op * fade, blur=1.2)

    def stars(self, n: int, max_y: float, k: float = 1.0, band: bool = True):
        rng = self.rng
        layer = np.zeros((H, W), dtype=F32)
        if band:
            # Milky Way: a soft diagonal band with dust lanes, plus extra star density inside it
            ang = np.deg2rad(-24)
            dist = (-(self.xx - W * 0.55) * np.sin(ang) + (self.yy - 120) * np.cos(ang)) / 150.0
            m = np.exp(-dist ** 2)
            tex = noise2(rng, H, W, 40, 90, 6, 0.6)
            dust = smooth((noise2(rng, H, W, 30, 70, 5) - 0.55) * 4)
            mw = m * (0.15 + 0.85 * tex ** 1.6) * (1 - 0.7 * dust) * (1 - smooth(self.yy / max_y)) * 0.34
            self.img += mw[..., None] * hexc("#c9d4ff") * k
        xs = rng.random(n) * W
        ys = rng.random(n) ** 1.4 * max_y
        if band:  # extra stars concentrated along the Milky Way line
            xb = rng.random(n // 2) * W
            yb = 120 + (xb - W * 0.55) * np.tan(np.deg2rad(-24)) + rng.normal(0, 55, n // 2)
            xs, ys = np.concatenate([xs, xb]), np.concatenate([ys, yb])
        ok = (ys >= 0) & (ys < max_y)
        ys, xs = ys[ok].astype(int), xs[ok].astype(int)
        mag = rng.random(len(ys)) ** 5
        np.add.at(layer, (ys, xs), 0.25 + mag * 1.6)
        sharp = gblur(layer, 0.65) * 3.0
        bloom = gblur(np.where(layer > 1.0, layer, 0), 2.2) * 5.0
        fade = (1 - smooth(self.yy / max_y) * 0.9)
        tint = hexc("#ffffff") * 0.8 + hexc("#cfe0ff") * 0.2
        self.img += ((sharp + bloom) * fade * k)[..., None] * tint

    def moon(self, cx, cy, r, glow="#a9c4ff", glow_k=0.5, tint="#fff6e0"):
        rng = self.rng
        d = np.sqrt((self.xx - cx) ** 2 + (self.yy - cy) ** 2)
        for rad, st in ((r * 9, 0.18), (r * 4, 0.28), (r * 1.9, 0.5)):
            self.img += (np.exp(-(d / rad) ** 1.3 * 2.0) * st * glow_k)[..., None] * hexc(glow)
        box = int(r * 2 + 6)
        yy, xx = np.mgrid[-box:box + 1, -box:box + 1].astype(F32)
        rr = np.sqrt(xx * xx + yy * yy)
        maria = noise2(rng, 2 * box + 1, 2 * box + 1, 14, 14, 4)
        crater = noise2(rng, 2 * box + 1, 2 * box + 1, 3, 3, 3)
        tone = 0.93 - 0.20 * smooth((maria - 0.5) * 3) - 0.05 * crater
        limb = 0.80 + 0.20 * np.sqrt(np.clip(1 - (rr / r) ** 2, 0, 1))
        col = (tone * limb)[..., None] * hexc(tint)
        a = (1 - smooth((rr - r) / 1.6))[..., None]
        y0, x0 = int(cy) - box, int(cx) - box
        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y0 + 2 * box + 1, H), min(x0 + 2 * box + 1, W)
        sub = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
        self.img[ys:ye, xs:xe] = self.img[ys:ye, xs:xe] * (1 - a[sub]) + col[sub] * a[sub] * 1.05

    def aurora(self, colors, y_lo=40, y_hi=330, strength=0.7):
        rng = self.rng
        for k, c in enumerate(colors):
            base = y_lo + (y_hi - y_lo) * (k + 0.5) / len(colors)
            curve = base + 90 * (fbm1(W, rng, 4, 1.6) - 0.5) + 28 * np.sin(np.linspace(0, 8, W) + k * 1.7)
            d = self.yy - curve[None, :]
            prof = np.where(d < 0, np.exp(d / 120.0), np.exp(-d / 26.0))
            rays = noise2(rng, H, W, 380, 16, 3, 0.55) ** 1.3
            drift = 0.55 + 0.45 * fbm1(W, rng, 4, 5.0)[None, :]
            band = prof * (0.45 + 0.9 * rays) * drift
            self.img += gaussian_filter(band * strength, (2.0, 5.0), mode='nearest')[..., None] * hexc(c)

    # -- terrain --------------------------------------------------------------------------------------------
    def profile(self, base_y, amp, base=2.6, octs=8, ridged=0.55, power=0.72):
        f = fbm1(W, self.rng, octs, base)
        r = (1 - np.abs(2 * f - 1)) ** 1.4
        n = (1 - ridged) * f + ridged * r
        n = (n - n.min()) / max(float(n.max() - n.min()), 1e-6)
        return base_y - (n ** power) * amp

    def mountains(self, ry, amp, lit, shadow, haze_col, haze, forest=None, forest_alt=0.0, snow=None, snow_alt=0.0,
                  rim=None, rim_k=0.0, blur=0.8, detail=1.0, base_y=None):
        rng = self.rng
        e = self.yy - ry[None, :]
        inside = e >= -2
        ec = np.clip(e, 0, None)
        n1 = noise2(rng, H, W, 120, 80, 6, 0.55)
        rid = (1 - np.abs(2 * n1 - 1)) ** 1.4
        S = rid * (1 - np.exp(-ec / 85.0))
        gx = np.gradient(S, axis=1)
        facets = np.gradient(gaussian_filter1d(ry, 7))
        fac = (self.light * facets)[None, :] * 0.26 * np.exp(-ec / 190.0)
        shade = 0.50 + self.light * gx * 14.0 * detail + fac * 1.6 + 0.10 * (n1 - 0.5)
        sh = np.clip(shade, 0, 1)[..., None]
        col = hexc(shadow) * (1 - sh) + hexc(lit) * sh
        A = (base_y if base_y is not None else ry.max()) - self.yy
        if forest is not None:
            tex = noise2(rng, H, W, 3.2, 3.2, 3, 0.6)
            streak = noise2(rng, H, W, 40, 9, 3, 0.6)
            fa = forest_alt * (0.85 + 0.3 * noise2(rng, H, W, 90, 110, 3))
            fm = smooth((fa - A) / 38.0 + 0.5)[..., None] * smooth(ec / 14.0)[..., None]
            fcol = hexc(forest) * (0.68 + 0.55 * tex[..., None] ) * (0.72 + 0.5 * sh) * (0.94 + 0.10 * streak[..., None])
            col = col * (1 - fm) + fcol * fm
        if snow is not None:
            sm = 0.85 * smooth((A - snow_alt) / 55.0) * (0.30 + 0.70 * smooth((rid - 0.30) / 0.35))
            sm = (sm * smooth(ec / 3.0))[..., None]
            scol = hexc(snow) * (0.62 + 0.48 * sh)
            col = col * (1 - sm) + scol * sm
        haze = haze * 0.62
        col = col * 0.90
        hv = np.clip(haze * (1 + 0.55 * (1 - smooth(np.clip(ec, 0, None) / max(amp * 1.1, 1.0)))), 0, 0.93)[..., None]
        col = col * (1 - hv) + hexc(haze_col) * hv
        if rim is not None and rim_k:
            fs = np.clip(0.5 + self.light * gaussian_filter1d(np.gradient(ry), 3) * 1.4, 0, 1)[None, :]
            col = col + (np.exp(-ec / 3.2) * fs * rim_k)[..., None] * hexc(rim)
        a = smooth((self.yy - ry[None, :] + 1.0) / 1.8)
        self.over(col, a * inside, blur=blur)

    def treeline(self, ry, kind, hmin, hmax, gap, dark, lit, haze_col, haze, blur=0.9, snow=None):
        """A dense strip of small individually-drawn trees standing along a ridge (distant forest)."""
        rng = self.rng
        pal = [dark, tuple_mix(dark, lit, 0.5), lit]
        xs, x = [], int(rng.integers(0, 10))
        while x < W:
            xs.append(x)
            x += int(max(gap * rng.uniform(0.45, 1.4), 2))
        trees = []
        rows = 3  # the forest climbs the slope: several staggered rows, nearer rows slightly larger
        for rw in range(rows):
            off = rw * hmax * 0.55
            for x in xs:
                xj = int(np.clip(x + rng.integers(-gap, gap + 1) + rw * 3, 0, W - 1))
                trees.append((xj, int(ry[xj] + off) + int(rng.integers(0, 8)), rng.uniform(hmin, hmax) * (1 + 0.12 * rw)))
        trees.sort(key=lambda t: t[1])
        for x, gy, h in trees:
            if kind == "pine":
                self.pine(x, gy, h * 1.35, dark, lit, snow=snow, snow_amt=0.7 if snow else 0.0, haze_col=haze_col, haze=haze)
            else:
                self.broadleaf(x, gy, h * 1.9, pal, trunk=dark, dark_k=0.42, haze_col=haze_col, haze=haze)
        return ry

    # -- individual trees ----------------------------------------------------------------------------------
    def _paste(self, rgb, a, cx, by):
        h, w = a.shape
        x0, y0 = int(cx - w / 2), int(by - h)
        xs, ys = max(x0, 0), max(y0, 0)
        xe, ye = min(x0 + w, W), min(y0 + h, H)
        if xe <= xs or ye <= ys:
            return
        sub = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
        al = a[sub][..., None]
        self.img[ys:ye, xs:xe] = self.img[ys:ye, xs:xe] * (1 - al) + rgb[sub] * al

    def pine(self, cx, ground_y, h, dark, lit, snow=None, snow_amt=0.0, haze_col=None, haze=0.0, trunk="#2a1c14"):
        rng = self.rng
        hh = int(h)
        C = int(h * 0.66) | 1
        mid = C // 2
        u = (np.arange(hh, dtype=F32) / hh)[:, None]
        cols = (np.arange(C, dtype=F32) - mid)[None, :]
        wmax = C / 2 * rng.uniform(0.92, 1.0)
        nt = int(rng.integers(7, 12))
        acol = np.abs(cols) / wmax
        # branches droop: each tier's lower boundary runs lower at the tips, plus per-tree irregularity
        jit = gaussian_filter1d(rng.random(C).astype(F32) - 0.5, 1.2)[None, :] * (0.9 / nt)
        ue = u - (0.55 / nt) * acol ** 1.15 + jit
        ph = (ue * nt) % 1.0
        env = wmax * (0.03 + 0.97 * np.clip(u, 0, 1) ** 0.88)
        env = env * (0.90 + 0.20 * gaussian_filter1d(rng.random(hh).astype(F32), 0.8)[:, None])
        hw = env * (0.55 + 0.45 * ph ** 0.5)
        ragged = (self.tile(self.fine, hh, C) - 0.5) * h * 0.035
        a = smooth((hw + ragged - np.abs(cols)) / 1.2)
        side = np.clip(cols / (hw + 1.0), -1, 1)
        lf = 0.5 + 0.5 * self.light * side
        under = smooth((0.34 - ph) / 0.34)          # shadow under the tier above
        shade = np.clip(0.10 + 0.55 * lf + 0.30 * ph - 0.40 * under, 0, 1)
        needle = self.tile(self.bank, hh, C)
        fine = self.tile(self.fine, hh, C)
        shade = np.clip(shade * (0.70 + 0.55 * needle) * (0.85 + 0.30 * fine), 0, 1)[..., None]
        col = hexc(dark) * (1 - shade) + hexc(lit) * shade
        if snow is not None and snow_amt > 0:
            sm = (smooth((ph - 0.2) / 0.4) * smooth((1.0 - ph) / 0.4) * snow_amt * (0.5 + 0.5 * lf)
                  * smooth((needle - 0.30) * 3.0))[..., None]
            col = col * (1 - sm) + hexc(snow) * (0.78 + 0.22 * lf[..., None]) * sm
        tcol = hexc(trunk)
        if haze > 0 and haze_col:
            col = col * (1 - haze) + hexc(haze_col) * haze
            tcol = tcol * (1 - haze) + hexc(haze_col) * haze
        tr = max(int(h * 0.05), 2)
        rgb = np.concatenate([col, np.broadcast_to(tcol, (tr, C, 3))], 0)
        al = np.concatenate([a, np.zeros((tr, C), dtype=F32)], 0)
        tw = max(int(h / 80), 1)
        al[hh:, mid - tw: mid + tw + 1] = 1.0
        self._paste(rgb, al, cx, ground_y + tr)

    def broadleaf(self, cx, ground_y, h, palette, trunk="#3b281c", dark_k=0.50, haze_col=None, haze=0.0,
                  blossom=False):
        rng = self.rng
        h = float(h)
        small0 = h < 60
        Wd, Hd = int(h * 1.5) | 1, int(h * (1.02 if small0 else 1.30))
        rgb = np.zeros((Hd, Wd, 3), dtype=F32)
        al = np.zeros((Hd, Wd), dtype=F32)
        mid = Wd // 2
        ccy = h * 0.52  # canopy centre (from the top of the patch)
        pal = np.stack([hexc(c) for c in palette])
        # trunk: tapered, slightly curved, lit on the sun side, with bark texture
        small = h < 60
        ty0 = int(ccy)
        rows = np.arange(ty0, Hd)
        sway = np.sin(rows * 0.045 + rng.random() * 6) * h * 0.012
        tw0, tw1 = max(h * 0.030, 1.4), max(h * 0.016, 0.9)
        wid = tw1 + (tw0 - tw1) * (rows - ty0) / max(Hd - ty0, 1)
        cols = np.arange(Wd)[None, :]
        tm = smooth((wid[:, None] - np.abs(cols - mid - sway[:, None])) / 1.0)
        bark = self.tile(self.fine, Hd - ty0, Wd)
        side = np.clip(0.5 + 0.5 * self.light * (cols - mid - sway[:, None]) / (wid[:, None] + 0.5), 0, 1)
        tcol = hexc(trunk)[None, None, :] * (0.40 + 0.95 * side[..., None]) * (0.75 + 0.5 * bark[..., None])
        rgb[ty0:] = tcol
        al[ty0:] = tm
        if not small:  # a couple of visible limbs reaching into the canopy
            for k in range(int(rng.integers(3, 6))):
                ang = rng.uniform(-1.1, 1.1)
                ln = h * rng.uniform(0.18, 0.34)
                for tt in np.linspace(0, 1, int(ln)):
                    px = int(mid + np.sin(ang) * ln * tt)
                    py = int(ty0 + 4 - np.cos(ang) * ln * tt * 0.8)
                    wd = max(int(tw1 * (1.2 - tt)), 1)
                    if 0 <= py < Hd and wd < px < Wd - wd:
                        rgb[py, px - wd:px + wd + 1] = hexc(trunk) * 0.55
                        al[py, px - wd:px + wd + 1] = 1.0
        # dense dark core so the canopy reads solid, then back-to-front leaf clumps lit like small spheres
        yy0, xx0 = np.mgrid[0:Hd, 0:Wd].astype(F32)
        core = np.sqrt(((xx0 - mid) / (h * 0.44)) ** 2 + ((yy0 - ccy) / (h * 0.38)) ** 2)
        cm = smooth((1.0 - core) / 0.35)
        rgb = rgb * (1 - cm[..., None]) + (pal.mean(0) * 0.30)[None, None, :] * cm[..., None]
        al = np.maximum(al, cm)
        n = int((14 + h * 0.55) if not small else (7 + h * 0.2))
        ang = rng.uniform(0, 6.2832, n)
        rad = np.sqrt(rng.random(n))
        cxs = np.cos(ang) * rad * h * 0.40
        cys = np.sin(ang) * rad * h * 0.34
        order = np.argsort(cys + rng.random(n) * h * 0.05)
        for i in order:
            r = h * rng.uniform(0.050, 0.105)
            px, py = mid + cxs[i], ccy + cys[i]
            R = int(r * 1.4) + 2
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(F32)
            sz = 2 * R + 1
            leaf = self.tile(self.bank, sz, sz)
            dd = np.sqrt(xx * xx + yy * yy) / r + (leaf - 0.5) * 0.85
            m = smooth((1.0 - dd) / 0.22)
            nz = np.sqrt(np.clip(1 - (xx / r) ** 2 - (yy / r) ** 2, 0, 1))
            lam = np.clip(0.28 + 0.34 * self.light * xx / r + 0.40 * (-yy / r) + 0.30 * nz, 0, 1)
            macro = np.clip(0.55 + 0.30 * self.light * (px - mid) / (h * 0.40) + 0.40 * (ccy - (py + yy)) / (h * 0.40), 0.18, 1.0)
            base = pal[int(rng.integers(0, len(pal)))]
            speck = 0.70 + 0.55 * self.tile(self.fine, sz, sz)
            c = base[None, None, :] * (dark_k + (1 - dark_k) * (lam * macro)[..., None]) * speck[..., None]
            if blossom:
                c = c + (speck[..., None] > 1.05) * 0.16 * lam[..., None]
            x0, y0 = int(px - R), int(py - R)
            xs, ys = max(x0, 0), max(y0, 0)
            xe, ye = min(x0 + sz, Wd), min(y0 + sz, Hd)
            if xe <= xs or ye <= ys:
                continue
            sub = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
            mm = m[sub][..., None]
            rgb[ys:ye, xs:xe] = rgb[ys:ye, xs:xe] * (1 - mm) + c[sub] * mm
            al[ys:ye, xs:xe] = np.maximum(al[ys:ye, xs:xe], m[sub])
        if haze > 0 and haze_col:
            rgb = rgb * (1 - haze) + hexc(haze_col) * haze
        self._paste(rgb, al, cx, ground_y)

    # -- water ---------------------------------------------------------------------------------------------
    def lake(self, y0, tint, sky_col, glint="#ffffff", glint_x=None, glint_k=0.5, ripple=1.0, base_refl=0.82):
        rng = self.rng
        below = H - y0
        src_rows = np.clip(y0 - 1 - (np.arange(below) * 0.92).astype(int), 0, H - 1)
        refl = self.img[src_rows].copy()  # (below, W, 3) mirrored scene
        t = (np.arange(below, dtype=F32) / below)[:, None]
        wave = noise2(rng, below, W, 4, 70, 4, 0.55) - 0.5
        disp = wave * (4 + 26 * t) * ripple
        fine = (noise2(rng, below, W, 1.5, 24, 3) - 0.5) * (2 + 10 * t) * ripple
        xi = np.clip((self.xx + disp + fine).astype(int), 0, W - 1)
        refl = np.take_along_axis(refl, np.repeat(xi[..., None], 3, 2), axis=1)
        # depth-increasing horizontal blur
        near, far = gblur(refl, 2.6), gblur(refl, 0.8)
        tt = smooth(t * 1.4)[..., None]
        refl = far * (1 - tt) + near * tt
        fres = (base_refl - 0.42 * t)[..., None]
        wcol = hexc(tint)[None, None, :]
        sky_tone = hexc(sky_col)[None, None, :]
        water = refl * fres * 0.92 + wcol * (1 - fres) * 0.85 + sky_tone * 0.10 * (1 - t[..., None])
        # ripple luminance modulation and a darker band under the shoreline
        lum = 0.88 + 0.26 * noise2(rng, below, W, 1.6, 34, 3)
        water = water * lum[..., None]
        shore = np.exp(-np.arange(below, dtype=F32) / 5.0)[:, None, None]
        water = water * (1 - 0.18 * shore)
        if glint_x is not None:
            spread = 14 + 90 * t
            col_g = np.exp(-(((self.xx - glint_x) / spread) ** 2))
            streak = (noise2(rng, below, W, 1.4, 22, 3) > 0.58).astype(F32) * (0.5 + 0.5 * rng.random((below, W)).astype(F32))
            g = col_g * (0.15 + streak * 1.0) * (1 - 0.5 * t) * glint_k
            water = water + g[..., None] * hexc(glint)
        self.img[y0:] = water
        edge = np.exp(-(np.arange(8, dtype=F32)) / 2.2)[:, None, None]
        self.img[y0:y0 + 8] += edge * 0.04

    def grass(self, ry, base, tip, blur=0.0, flowers=None, density=0.0):
        """Meadow ground: from the ridge line to the bottom, with blade texture, depth shading and optional flowers."""
        rng = self.rng
        e = self.yy - ry[None, :]
        tex = noise2(rng, H, W, 1.6, 1.2, 3, 0.65)
        blades = noise2(rng, H, W, 3.0, 2.2, 3, 0.6)
        big = noise2(rng, H, W, 70, 160, 4)
        patch = noise2(rng, H, W, 28, 90, 4)
        t = np.clip(0.30 + 0.50 * big + 0.30 * (patch - 0.5) + 0.40 * (tex - 0.5) + 0.25 * (blades - 0.5) - 0.18 * smooth(e / 220.0), 0, 1)[..., None]
        col = hexc(base) * (1 - t) + hexc(tip) * t
        a = smooth((self.yy - ry[None, :] + 1.0) / 1.8)
        self.over(col, a, blur=blur)
        if flowers and density > 0:
            n = int(density)
            ys = rng.integers(int(ry.min()) + 20, H - 20, n)
            xs = rng.integers(0, W, n)
            for y, x in zip(ys, xs):
                if y <= ry[x] + 6:
                    continue
                r = 0.8 + 2.6 * ((y - ry[x]) / 260.0)
                c = hexc(flowers[rng.integers(0, len(flowers))])
                R = int(r) + 2
                yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
                m = smooth(1 - np.sqrt(xx * xx + yy * yy) / max(r, 0.8))
                x0, y0 = x - R, y - R
                xs0, ys0 = max(x0, 0), max(y0, 0)
                xe, ye = min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
                if xe <= xs0 or ye <= ys0:
                    continue
                sub = (slice(ys0 - y0, ye - y0), slice(xs0 - x0, xe - x0))
                mm = m[sub][..., None]
                self.img[ys0:ye, xs0:xe] = self.img[ys0:ye, xs0:xe] * (1 - mm) + c * mm

    def mist(self, y, height, color, strength, blur=0.0):
        band = np.exp(-(((self.yy - y) / height) ** 2))
        wob = 0.55 + 0.45 * noise2(self.rng, H, W, 60, 280, 3)
        a = band * wob * strength
        self.over(hexc(color), a, blur=blur)

    def particles(self, n, y_lo, y_hi, color, k=0.8, size=(1.0, 2.6), blur_far=True):
        """Snow / petals / pollen: soft discs, larger and brighter when 'closer' (lower on the frame)."""
        rng = self.rng
        ys = rng.integers(y_lo, y_hi, n)
        xs = rng.integers(0, W, n)
        depth = (ys - y_lo) / max(y_hi - y_lo, 1)
        layer = np.zeros((H, W), dtype=F32)
        for y, x, d in zip(ys, xs, depth):
            r = size[0] + (size[1] - size[0]) * d * rng.random() * 1.3
            R = int(r * 2) + 2
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
            m = np.exp(-(xx * xx + yy * yy) / (2 * (r * 0.6) ** 2)) * (0.35 + 0.65 * d) * k
            x0, y0 = x - R, y - R
            xs0, ys0 = max(x0, 0), max(y0, 0)
            xe, ye = min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
            if xe <= xs0 or ye <= ys0:
                continue
            layer[ys0:ye, xs0:xe] += m[ys0 - y0:ye - y0, xs0 - x0:xe - x0]
        self.img += np.clip(layer, 0, 1.2)[..., None] * hexc(color)


# ---------------------------------------------------------------------------------------------------------------
# finishing
# ---------------------------------------------------------------------------------------------------------------
def finish(cv: Canvas, path: Path, exposure=1.0, sat=1.06, bloom=0.28, grain=0.010, preview=False, contrast=1.16) -> None:
    img = np.clip(cv.img, 0, None)
    bright = np.clip(img - 0.72, 0, None)
    img = img + bloom * (gblur(bright, 22) * 0.6 + gblur(bright, 7) * 0.4)
    # filmic tone curve (ACES-like), keeps highlights from clipping to flat white
    x = img * exposure
    img = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    img = np.clip(0.5 + (img - 0.5) * contrast, 0, 1)
    lum = img.mean(axis=2, keepdims=True)
    img = np.clip(lum + (img - lum) * sat, 0, 1)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(F32)
    v = ((xx / W - 0.5) * 1.6) ** 2 + ((yy / H - 0.42) * 1.5) ** 2
    img = img * (1 - 0.18 * np.clip(v, 0, 1))[..., None]
    img = gblur(img, 0.45)
    rng = np.random.default_rng(99)
    g = rng.normal(0, grain, (H, W, 1)).astype(F32) + rng.normal(0, grain * 0.35, (H, W, 3)).astype(F32)
    img = np.clip(img + g * (0.55 + 0.9 * (1 - lum)), 0, 1)
    ys = np.linspace(0, 1, H, dtype=F32)[:, None]
    xs = np.linspace(0, 1, W, dtype=F32)[None, :]
    alpha = (1.0 - smooth((ys - 0.66) / 0.34)) * smooth(ys / 0.04) * smooth(xs / 0.05) * smooth((1 - xs) / 0.03)
    rgba = np.dstack([(img * 255 + 0.5).astype(np.uint8), (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=80, method=6)
    if preview:
        bgc = np.array([244, 246, 248] if lum.mean() > 0.45 else [11, 14, 20], dtype=F32)
        comp = rgba[..., :3].astype(F32) * (rgba[..., 3:] / 255.0) + bgc * (1 - rgba[..., 3:] / 255.0)
        Image.fromarray(comp.astype(np.uint8)).save(path.with_suffix(".preview.png"))


# ---------------------------------------------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------------------------------------------
def scatter_trees(cv: Canvas, ry, n, fn, lo=0.03, hi=0.97, min_gap=20, jitter=8):
    xs = np.sort(cv.rng.integers(int(W * lo), int(W * hi), n))
    last = -999
    items = []
    for x in xs:
        if x - last < min_gap:
            continue
        last = x
        items.append(int(x))
    for x in items:
        fn(x, int(ry[x]) + int(cv.rng.integers(2, jitter)))


def scene(season: str, night: bool, seed: int):
    cv = Canvas(seed)
    rng = cv.rng
    hz = 470.0
    LAKE_Y = 556

    # ======================================================== SPRING =============================================
    if season == "spring":
        cv.light = 1.0 if not night else -1.0
        if night:
            sun = (330, 150)
            cv.sky("#050822", "#251f54", "#7c527f", hz)
            cv.stars(1100, 330, 1.0)
            cv.moon(sun[0], sun[1], 30, glow="#c9b8ff", glow_k=0.55, tint="#fff1f5")
            cv.sun = sun
            cv.clouds(hz, "#8f7fc4", "#1d1a45", cover=0.42, op=0.55, scale=1.1)
            far = cv.profile(450, 230, 2.2)
            cv.mountains(far, 230, "#6d6aa6", "#262554", "#7e5a92", 0.58, forest="#2c2a55", forest_alt=60, snow="#c8c4ee",
                         snow_alt=130, rim="#e9c8ff", rim_k=0.30, blur=1.5, base_y=450)
            mid = cv.profile(520, 150, 3.4)
            cv.mountains(mid, 150, "#4a4684", "#1a1840", "#6b4a86", 0.40, forest="#1c1a3c", forest_alt=95, rim="#ffb7de",
                         rim_k=0.20, blur=1.0, base_y=520)
            cv.treeline(mid, "round", 16, 30, 6, "#1c1838", "#6d4a88", "#6b4a86", 0.38, blur=1.0)
            near = cv.profile(575, 90, 4.0, octs=7, ridged=0.3)
            cv.grass(near, "#0e1a24", "#2a3b56", blur=0.5)
            pal = ["#ff9cc7", "#ffb9da", "#f07cb2", "#ffd0e4", "#d96aa8"]
            scatter_trees(cv, near, 22, lambda x, y: cv.broadleaf(x, y, int(120 + rng.random() * 90), pal, trunk="#1a1226",
                          dark_k=0.34, blossom=True, haze_col="#4b3a70", haze=0.10), min_gap=64)
            cv.mist(560, 46, "#b79ad0", 0.28, blur=2)
            cv.particles(260, 80, 640, "#ffc8e0", 0.8)
            return cv, dict(exposure=1.05, sat=1.08, bloom=0.34)
        sun = (1480, 165)
        cv.sky("#5da7ea", "#a9d4f4", "#f7e3ee", hz, sun=sun, sun_col="#fff0d2", sun_k=1.5, disc_r=24)
        cv.clouds(hz, "#ffffff", "#cdd9ea", cover=0.50, op=0.90, scale=1.0)
        cv.sun_disc()
        far = cv.profile(450, 230, 2.2)
        cv.mountains(far, 230, "#a5b3cc", "#56688f", "#e8eaf2", 0.52, forest="#6f9a8d", forest_alt=60, snow="#ffffff",
                     snow_alt=125, rim="#ffffff", rim_k=0.30, blur=1.5, base_y=450)
        mid = cv.profile(520, 150, 3.4)
        cv.mountains(mid, 150, "#98b9a8", "#4d7a72", "#d7e2e4", 0.36, forest="#4f8059", forest_alt=100, blur=1.0, base_y=520)
        cv.treeline(mid, "round", 16, 30, 6, "#3e6e4c", "#a8cf9a", "#d7e2e4", 0.30, blur=1.0)
        near = cv.profile(575, 90, 4.0, octs=7, ridged=0.3)
        cv.grass(near, "#2c6a3a", "#86b552", blur=0.5, flowers=["#ffffff", "#ffe27a", "#ff9ec2"], density=1400)
        pal = ["#ffc2d9", "#ffd7e6", "#f6a6c6", "#ffe9f1", "#e98fb7"]
        scatter_trees(cv, near, 22, lambda x, y: cv.broadleaf(x, y, int(120 + rng.random() * 90), pal, trunk="#5a4036",
                      dark_k=0.52, blossom=True, haze_col="#dfe8ee", haze=0.06), min_gap=64)
        cv.mist(560, 46, "#ffffff", 0.22, blur=2)
        cv.particles(220, 80, 640, "#ffd6e6", 0.9)
        return cv, dict(exposure=1.0, sat=1.08, bloom=0.30)

    # ======================================================== SUMMER =============================================
    if season == "summer":
        if night:
            cv.light = -1.0
            sun = (430, 140)
            cv.sky("#030a1f", "#0d2b52", "#276d85", hz)
            cv.stars(1300, 340, 1.0)
            cv.moon(sun[0], sun[1], 31, glow="#9cc8ff", glow_k=0.55)
            cv.sun = sun
            cv.clouds(hz, "#6f98c8", "#0b1c36", cover=0.38, op=0.50, scale=1.2)
            far = cv.profile(455, 240, 2.4)
            cv.mountains(far, 240, "#4f86b0", "#173957", "#2c7189", 0.55, forest="#10304a", forest_alt=80, snow="#bfe4ff",
                         snow_alt=140, rim="#8fe4ff", rim_k=0.34, blur=1.5, base_y=455)
            mid = cv.profile(520, 160, 3.4)
            cv.mountains(mid, 160, "#2e6a8c", "#0d2438", "#1f5f78", 0.40, forest="#0a1f2f", forest_alt=110,
                         rim="#6fd0f0", rim_k=0.20, blur=1.0, base_y=520)
            cv.treeline(mid, "pine", 22, 46, 10, "#06141f", "#2c6685", "#1f5f78", 0.32, blur=1.0)
            near = cv.profile(548, 70, 4.2, octs=7, ridged=0.3)
            cv.mountains(near, 70, "#10222d", "#050e14", "#0e2a3a", 0.16, forest="#06121a", forest_alt=200, blur=0.5,
                         base_y=548)
            scatter_trees(cv, near, 40, lambda x, y: cv.pine(x, y + 4, int(90 + rng.random() * 120), "#040c12", "#2e5e70",
                          haze_col="#0e2a3a", haze=0.10), min_gap=36)
            cv.lake(LAKE_Y, "#0a2540", "#1a4a68", glint="#bfe3ff", glint_x=sun[0], glint_k=0.50)
            cv.mist(LAKE_Y - 4, 24, "#7fb8d0", 0.16, blur=2)
            return cv, dict(exposure=1.02, sat=1.08, bloom=0.34)
        cv.light = 1.0
        sun = (1500, 150)
        cv.sky("#2a8de0", "#79c1f2", "#d9eef8", hz, sun=sun, sun_col="#fff4cf", sun_k=1.5, disc_r=26)
        cv.clouds(hz, "#ffffff", "#c2d4ea", cover=0.50, op=0.92, scale=1.0)
        cv.sun_disc()
        far = cv.profile(455, 240, 2.4)
        cv.mountains(far, 240, "#a6bbd3", "#5a7699", "#e3eef6", 0.50, forest="#5e8f90", forest_alt=75, snow="#ffffff",
                     snow_alt=135, rim="#ffffff", rim_k=0.28, blur=1.5, base_y=455)
        mid = cv.profile(520, 160, 3.4)
        cv.mountains(mid, 160, "#89b7aa", "#3d756e", "#cfe3e6", 0.34, forest="#255a44", forest_alt=115, blur=1.0, base_y=520)
        cv.treeline(mid, "pine", 22, 46, 10, "#174a36", "#5c9a74", "#cfe3e6", 0.28, blur=1.0)
        near = cv.profile(548, 70, 4.2, octs=7, ridged=0.3)
        cv.mountains(near, 70, "#4c8f68", "#1f5640", "#a9d0c4", 0.12, forest="#1b4d39", forest_alt=200, blur=0.5, base_y=548)
        scatter_trees(cv, near, 40, lambda x, y: cv.pine(x, y + 4, int(90 + rng.random() * 120), "#0d2e1e", "#3a7d4c",
                      haze_col="#bcd9d4", haze=0.05), min_gap=36)
        cv.lake(LAKE_Y, "#2b7fae", "#8fcbee", glint="#ffffff", glint_x=sun[0], glint_k=0.55)
        cv.mist(LAKE_Y - 4, 24, "#ffffff", 0.14, blur=2)
        return cv, dict(exposure=1.0, sat=1.06, bloom=0.30)

    # ======================================================== AUTUMN =============================================
    if season == "autumn":
        pal_day = ["#e8651d", "#f08a2a", "#f3b83a", "#c93a1e", "#d9541c"]
        pal_night = ["#ff7a2e", "#ff5a1f", "#ffb02e", "#d9381e", "#ff8f3a"]
        if night:
            cv.light = 1.0
            sun = (1390, 175)
            cv.sky("#0a0716", "#2a1530", "#b0502e", hz)
            cv.stars(620, 300, 0.8)
            cv.moon(sun[0], sun[1], 33, glow="#ff9a58", glow_k=0.75, tint="#ffe9c9")
            cv.sun = sun
            cv.add_glow(900, 470, 1100, 120, "#ff7a3a", 0.40)
            cv.clouds(hz, "#d98a63", "#1d1020", cover=0.40, op=0.60, scale=1.1)
            far = cv.profile(455, 230, 2.4)
            cv.mountains(far, 230, "#7a4248", "#2a1530", "#a2492e", 0.55, forest="#3a1d2a", forest_alt=70, rim="#ffa05a",
                         rim_k=0.34, blur=1.5, base_y=455)
            mid = cv.profile(520, 150, 3.4)
            cv.mountains(mid, 150, "#552636", "#170c18", "#7a3028", 0.40, forest="#2a1220", forest_alt=100, rim="#ff7e3d",
                         rim_k=0.22, blur=1.0, base_y=520)
            cv.treeline(mid, "round", 18, 34, 6, "#3a1418", "#d4601f", "#7a3028", 0.30, blur=1.0)
            near = cv.profile(548, 70, 4.2, octs=7, ridged=0.3)
            cv.grass(near, "#150a10", "#4a2420", blur=0.5)
            scatter_trees(cv, near, 26, lambda x, y: cv.broadleaf(x, y + 6, int(130 + rng.random() * 100), pal_night,
                          trunk="#150a0e", dark_k=0.30, haze_col="#4a2024", haze=0.10), min_gap=60)
            cv.lake(LAKE_Y, "#2a1018", "#8a3a28", glint="#ffb87a", glint_x=sun[0], glint_k=0.65)
            cv.mist(LAKE_Y - 4, 24, "#e08a5a", 0.14, blur=2)
            return cv, dict(exposure=1.05, sat=1.10, bloom=0.36)
        cv.light = 1.0
        sun = (1450, 180)
        cv.sky("#df9a58", "#f2c28c", "#fbe9cf", hz, sun=sun, sun_col="#fff0cd", sun_k=1.6, disc_r=28)
        cv.clouds(hz, "#fff0d6", "#e3b48a", cover=0.52, op=0.80, scale=1.0)
        cv.sun_disc()
        far = cv.profile(455, 230, 2.4)
        cv.mountains(far, 230, "#c99d82", "#8a5a4a", "#f6e1c8", 0.52, forest="#a8683f", forest_alt=70, rim="#fff0d8",
                     rim_k=0.28, blur=1.5, base_y=455)
        mid = cv.profile(520, 150, 3.4)
        cv.mountains(mid, 150, "#c98a58", "#8c5232", "#ebc8a0", 0.34, forest="#9a4a22", forest_alt=100, blur=1.0, base_y=520)
        cv.treeline(mid, "round", 18, 34, 6, "#8a3a1c", "#f2a23a", "#ebc8a0", 0.26, blur=1.0)
        near = cv.profile(548, 70, 4.2, octs=7, ridged=0.3)
        cv.grass(near, "#7a4220", "#c58a42", blur=0.5)
        scatter_trees(cv, near, 26, lambda x, y: cv.broadleaf(x, y + 6, int(130 + rng.random() * 100), pal_day,
                      trunk="#4a3022", dark_k=0.46, haze_col="#e8c9a4", haze=0.05), min_gap=60)
        cv.lake(LAKE_Y, "#c97a3a", "#f1c28c", glint="#fff2d6", glint_x=sun[0], glint_k=0.60)
        cv.mist(LAKE_Y - 4, 24, "#fff0d8", 0.14, blur=2)
        cv.particles(120, 100, 620, "#f2a23a", 0.7, size=(1.2, 3.4))
        return cv, dict(exposure=1.0, sat=1.10, bloom=0.34)

    # ======================================================== WINTER =============================================
    if night:
        cv.light = -1.0
        sun = (260, 120)
        cv.sky("#02060f", "#08203a", "#1d5560", hz)
        cv.stars(1500, 340, 1.0)
        cv.aurora(["#38ffb0", "#2fd0ff", "#8a6bff"], 60, 300, 0.46)
        cv.moon(sun[0], sun[1], 26, glow="#a9d0ff", glow_k=0.5, tint="#f3f7ff")
        cv.sun = sun
        cv.clouds(hz, "#5d86a8", "#08141f", cover=0.30, op=0.35, scale=1.3)
        far = cv.profile(455, 240, 2.4)
        cv.mountains(far, 240, "#4a7596", "#183450", "#27606c", 0.50, forest="#102a40", forest_alt=60, snow="#d6ecff",
                     snow_alt=105, rim="#bfeaff", rim_k=0.40, blur=1.5, base_y=455)
        mid = cv.profile(520, 160, 3.4)
        cv.mountains(mid, 160, "#2f5a78", "#0e2234", "#22505e", 0.36, forest="#0a1d2c", forest_alt=105, snow="#bfe0f4",
                     snow_alt=150, rim="#9fdcf5", rim_k=0.26, blur=1.0, base_y=520)
        cv.treeline(mid, "pine", 22, 46, 10, "#06121b", "#3a6a86", "#22505e", 0.28, blur=1.0, snow="#9fc0d4")
        near = cv.profile(548, 70, 4.2, octs=7, ridged=0.3)
        cv.grass(near, "#2f5471", "#a9c9de", blur=0.5)
        scatter_trees(cv, near, 40, lambda x, y: cv.pine(x, y + 4, int(90 + rng.random() * 120), "#050e15", "#3a6a86",
                      snow="#cfe6f4", snow_amt=0.85, haze_col="#183a50", haze=0.08), min_gap=36)
        cv.lake(LAKE_Y, "#0a2236", "#2a6a82", glint="#bfe6ff", glint_x=sun[0], glint_k=0.40, ripple=0.4)
        cv.particles(380, 40, 640, "#dff0ff", 0.85, size=(0.9, 2.6))
        return cv, dict(exposure=1.04, sat=1.06, bloom=0.34)
    cv.light = 1.0
    sun = (1500, 190)
    cv.sky("#9fc7e6", "#d1e5f3", "#f3f8fc", hz, sun=sun, sun_col="#ffffff", sun_k=1.5, disc_r=26)
    cv.clouds(hz, "#ffffff", "#d3e0ee", cover=0.52, op=0.85, scale=1.0)
    cv.sun_disc()
    far = cv.profile(455, 240, 2.4)
    cv.mountains(far, 240, "#cfdce8", "#7f98b3", "#f1f7fb", 0.48, forest="#7d9a9f", forest_alt=55, snow="#ffffff",
                 snow_alt=95, rim="#ffffff", rim_k=0.34, blur=1.5, base_y=455)
    mid = cv.profile(520, 160, 3.4)
    cv.mountains(mid, 160, "#cfe0ee", "#7c9ab4", "#e6f0f7", 0.32, forest="#4f7a76", forest_alt=105, snow="#ffffff",
                 snow_alt=140, rim="#ffffff", rim_k=0.24, blur=1.0, base_y=520)
    cv.treeline(mid, "pine", 22, 46, 10, "#3f6660", "#9bbfc0", "#e6f0f7", 0.24, blur=1.0, snow="#ffffff")
    near = cv.profile(548, 70, 4.2, octs=7, ridged=0.3)
    cv.grass(near, "#c3d6e6", "#f4f9fd", blur=0.5)
    scatter_trees(cv, near, 40, lambda x, y: cv.pine(x, y + 4, int(90 + rng.random() * 120), "#264f4a", "#6a9d92",
                  snow="#ffffff", snow_amt=0.95, haze_col="#e6f0f7", haze=0.05), min_gap=36)
    cv.lake(LAKE_Y, "#a9c8de", "#d6e8f4", glint="#ffffff", glint_x=sun[0], glint_k=0.55, ripple=0.35, base_refl=0.86)
    cv.particles(320, 40, 640, "#ffffff", 0.9, size=(0.9, 2.6))
    return cv, dict(exposure=1.0, sat=1.04, bloom=0.32)


SEASONS = ("spring", "summer", "autumn", "winter")


def build(seasons=SEASONS, out_dir: Path = OUT_DIR, preview: bool = False) -> list[Path]:
    written = []
    for si, season in enumerate(SEASONS):
        if season not in seasons:
            continue
        for night in (True, False):
            cv, fin = scene(season, night, seed=7100 + si * 10 + (0 if night else 1))
            p = out_dir / f"{season}-{'dark' if night else 'light'}.webp"
            finish(cv, p, preview=preview, **fin)
            written.append(p)
            print("wrote", p, f"({p.stat().st_size // 1024} KB)", flush=True)
    return written


if __name__ == "__main__":
    args = sys.argv[1:]
    wanted = tuple(a for a in args if a in SEASONS) or SEASONS
    build(wanted, preview="--preview" in args)
