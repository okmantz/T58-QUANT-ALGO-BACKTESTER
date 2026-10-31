/* T58 scene canvas: real motion for two of the Personalize backdrops, drawn on a canvas in the fixed scene layer.
 *
 *   midnight-ocean  The water in the painting itself moves. The picture is re-drawn every frame in thin horizontal strips below
 *                   the horizon, each strip shifted by a travelling wave whose size and speed grow with distance from the
 *                   horizon (perspective), so the swell, the chop and the moon's reflection ripple the way real water does.
 *                   The old "line drawn over the waves" is gone. Soft procedural clouds drift across the sky and catch the
 *                   moonlight as they pass it.
 *   misty-forest    Layered mist: five depth layers (far haze behind the ridges, two bands in the trees, low mist on the lake
 *                   and a thin foreground veil), each its own tileable cloud texture drifting at its own speed and direction,
 *                   breathing in density and cross-fading with a second copy of itself so it billows instead of streaking.
 *
 * Both draw the SAME picture the CSS uses (background-size: cover at --scene-pos), so the canvas sits seamlessly over it; the
 * CSS scrim (readable text) is re-applied on top by a sibling <i class="scrim"> that reads the same --sc1..3 variables.
 * Cost: ~30 fps, paused while the tab is hidden, skipped for reduced-motion / data-saver, thinner on phones. If anything fails
 * (no canvas, image blocked) it removes itself and the static picture underneath is exactly what you had before. */
(function () {
  'use strict';

  var HORIZON = 0.485;                       // horizon line of the ocean picture, as a fraction of the picture's height
  var MOON = { x: 0.665, y: 0.185 };         // moon centre in picture fractions
  var LAKE_TOP = 0.60;                       // where the lake begins in the forest picture

  function rnd(a, b) { return a + Math.random() * (b - a); }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function smooth(a, b, x) { var t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
  function isLight() { return document.documentElement.getAttribute('data-theme') === 'light'; }

  // "69% 50%" (the --scene-pos CSS variable) -> [0.69, 0.5]
  function scenePos() {
    try {
      var raw = getComputedStyle(document.documentElement).getPropertyValue('--scene-pos') || '50% 50%';
      var m = raw.match(/(-?[\d.]+)%\s+(-?[\d.]+)%/);
      if (m) return [parseFloat(m[1]) / 100, parseFloat(m[2]) / 100];
    } catch (e) {}
    return [0.5, 0.5];
  }

  // Where the picture lands on screen under `background-size: cover; background-position: <pos>`.
  function coverRect(iw, ih, W, H, pos) {
    var s = Math.max(W / iw, H / ih), w = iw * s, h = ih * s;
    return { s: s, w: w, h: h, x: (W - w) * pos[0], y: (H - h) * pos[1] };
  }

  /* -------------------------------------------------------------- shared canvas plumbing */
  var IMAGE_OF = { ocean: 'midnight-ocean', forest: 'misty-forest', gold: 'obsidian-gold', sakura: 'sakura-reverie-hd', abyss: 'abyssal-blue', nordic: 'nordic-kingdom', sands: 'endless-sands', italy: 'coastal-italy', biolum: 'bioluminescence' };

  function makeStage(kind, layer, drawFrame, setup, opts) {
    opts = opts || {};
    var canvas = document.createElement('canvas');
    canvas.className = 'sc-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.opacity = '0';
    var scrim = document.createElement('i');
    scrim.className = 'scrim';
    layer.insertBefore(scrim, layer.firstChild);
    layer.insertBefore(canvas, layer.firstChild);
    var ctx = opts.gl ? (canvas.getContext('webgl', { alpha: false, antialias: false, powerPreference: 'low-power' }) || null) : canvas.getContext('2d', { alpha: false });
    if (!ctx) { canvas.remove(); scrim.remove(); return null; }

    var st = { canvas: canvas, ctx: ctx, W: 0, H: 0, dpr: 1, img: null, ready: false, raf: 0, last: 0, t: (+window.__T58_SCENE_T0 || 0), dead: false,
      fps: 30, light: isLight(), small: window.innerWidth < 760 || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2) };

    function imageUrl() {
      return '/static/backgrounds/' + IMAGE_OF[kind] + (st.light ? '-light' : '-dark') + '.webp';
    }
    function loadImage(done) {
      var im = new Image();
      im.onload = function () { if (!st.dead) { st.img = im; done(); } };
      im.onerror = function () { destroy(); };
      im.src = imageUrl();
    }
    function resize() {
      var cssW = Math.max(1, window.innerWidth), cssH = Math.max(1, window.innerHeight);
      var dpr = Math.min(window.devicePixelRatio || 1, st.small ? 1 : 1.5);
      // cap the backing store: a 4K display needs no more than ~2300 px of this soft picture
      var cap = 2300 / cssW; if (cap < dpr) dpr = Math.max(0.5, cap);
      st.dpr = dpr; st.W = Math.round(cssW * dpr); st.H = Math.round(cssH * dpr);
      canvas.width = st.W; canvas.height = st.H;
      if (st.img) setup(st);
    }
    function frame(now) {
      if (st.dead) return;
      st.raf = requestAnimationFrame(frame);
      if (document.hidden || !st.ready) return;
      if (now - st.last < 1000 / st.fps) return;
      var dt = st.last ? Math.min(0.1, (now - st.last) / 1000) : 0.033;
      st.last = now; st.t += dt;
      try { drawFrame(st, dt); } catch (e) { destroy(); }
    }
    function onTheme() {
      var l = isLight();
      if (l === st.light) return;
      st.light = l; st.ready = false;
      loadImage(function () { setup(st); st.ready = true; });
    }
    var themeObs = window.MutationObserver ? new MutationObserver(onTheme) : null;
    if (themeObs) themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    var rzTimer = 0;
    function onResize() { clearTimeout(rzTimer); rzTimer = setTimeout(function () { st.ready = false; resize(); st.ready = !!st.img; }, 160); }
    window.addEventListener('resize', onResize);

    function destroy() {
      st.dead = true;
      cancelAnimationFrame(st.raf);
      window.removeEventListener('resize', onResize);
      if (themeObs) themeObs.disconnect();
      canvas.remove(); scrim.remove();
    }
    st.destroy = destroy;

    resize();
    loadImage(function () {
      try { resize(); setup(st); st.ready = true; drawFrame(st, 0.033); } catch (e) { destroy(); return; }
      canvas.style.transition = 'opacity .5s ease'; canvas.style.opacity = '1';
      st.raf = requestAnimationFrame(frame);
    });
    return st;
  }

  /* -------------------------------------------------------------- MIDNIGHT OCEAN */
  function oceanSetup(st) {
    var W = st.W, H = st.H, pos = scenePos();
    var r = coverRect(st.img.naturalWidth, st.img.naturalHeight, W, H, pos);
    // Wider-than-screen copy of the picture so a shifted strip never runs out of pixels at the edges.
    var M = Math.ceil(26 * st.dpr);
    var base = document.createElement('canvas'); base.width = W + 2 * M; base.height = H;
    var b = base.getContext('2d');
    b.imageSmoothingQuality = 'high';
    b.drawImage(st.img, r.x + M, r.y, r.w, r.h);
    // extend the left / right edges outward (cover can leave the picture exactly flush with the screen)
    b.drawImage(base, M, 0, 1, H, 0, 0, M, H);
    b.drawImage(base, W + M - 1, 0, 1, H, W + M, 0, M, H);
    st.base = base; st.M = M; st.rect = r;
    st.horizonY = Math.round(r.y + r.h * HORIZON);
    st.moon = { x: r.x + r.w * MOON.x, y: r.y + r.h * MOON.y, rad: r.w * 0.075 };
    st.strip = Math.max(2, Math.round(2 * st.dpr));
    // moon-glitter: sparkles that wink on and off inside the reflection column
    st.sparks = [];
    var ns = st.small ? 22 : 46;
    for (var k = 0; k < ns; k++) st.sparks.push({ dd: Math.pow(Math.random(), 0.8), u: rnd(-1, 1), ph: rnd(0, 6.28), sp: rnd(0.5, 1.4), sz: rnd(0.6, 1.5) });
    // soft clouds, pre-rendered once
    var skyH = Math.max(40, st.horizonY);
    st.clouds = [];
    var n = st.small ? 5 : 8;
    for (var i = 0; i < n; i++) {
      var depth = i / (n - 1);                                   // 0 = high & near, 1 = low & far
      var cw = Math.round(W * rnd(0.26, 0.5) * (1 - depth * 0.45)), ch = Math.round(cw * rnd(0.17, 0.26));
      st.clouds.push({
        spr: cloudSprite(cw, ch, st.light), w: cw, h: ch,
        x: rnd(-cw, W), y: skyH * (0.05 + 0.80 * Math.pow(depth, 0.85)) - ch * 0.5,
        v: (4 + 7 * (1 - depth) + rnd(0, 2)) * st.dpr,           // px / s: nearer (higher) clouds move faster
        a: (st.light ? 0.30 : 0.20) * (1 - depth * 0.35) * rnd(0.8, 1.1), flip: Math.random() < 0.5
      });
    }
  }

  // A cumulus bank built from fractal noise (so the top is bumpy and the edges wispy, never a smooth slab): an elliptical
  // envelope with a flattened underside, thresholded softly, shaded bright on top and dark underneath.
  function cloudSprite(w, h, light) {
    var nw = Math.max(48, Math.round(w / 3)), nh = Math.max(16, Math.round(h / 3));
    var noise = noiseTexture(nw, nh, 4, 3, 4, (Math.random() * 1e9) | 0);
    var c = document.createElement('canvas'); c.width = nw; c.height = nh;
    var g = c.getContext('2d'), img = g.createImageData(nw, nh), px = img.data;
    var top = light ? [255, 255, 255] : [196, 210, 236], bot = light ? [214, 224, 238] : [88, 106, 150];
    for (var y = 0; y < nh; y++) {
      var fy = y / (nh - 1);
      for (var x = 0; x < nw; x++) {
        var fx = x / (nw - 1);
        var ex = (fx - 0.5) / 0.5, ey = (fy - 0.64) / 0.52;
        var env = clamp(1 - ex * ex - ey * ey * (ey > 0 ? 3.2 : 1), 0, 1);   // flat, quickly-fading underside
        var n = noise[y * nw + x];
        var dens = smooth(0.30, 0.72, n * 0.72 + env * 0.62 - 0.12) * smooth(0.0, 0.18, env);
        var shade = clamp((fy - 0.18) / 0.62, 0, 1);                          // 0 = lit top, 1 = shadowed base
        var i = (y * nw + x) * 4;
        px[i] = top[0] + (bot[0] - top[0]) * shade; px[i + 1] = top[1] + (bot[1] - top[1]) * shade; px[i + 2] = top[2] + (bot[2] - top[2]) * shade;
        px[i + 3] = Math.round(255 * clamp(dens, 0, 1));
      }
    }
    g.putImageData(img, 0, 0);
    var out = document.createElement('canvas'); out.width = w; out.height = h;
    var o = out.getContext('2d'); o.imageSmoothingQuality = 'high'; o.drawImage(c, 0, 0, w, h);   // upscaling softens the edges
    return out;
  }

  function oceanFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, M = st.M, t = st.t, hy = st.horizonY, base = st.base, d = st.dpr;
    // sky: straight from the picture
    ctx.drawImage(base, M, 0, W, hy, 0, 0, W, hy);
    // clouds drifting over the sky, then catching moonlight near the moon
    var moon = st.moon;
    for (var i = 0; i < st.clouds.length; i++) {
      var c = st.clouds[i];
      c.x += c.v * (dt || 0.033);
      if (c.x > W + 20) c.x = -c.w - 20;
      var edge = Math.min(smooth(-c.w, -c.w * 0.5, c.x), smooth(W + 20, W - c.w * 0.4, c.x));
      var a = c.a * (0.35 + 0.65 * Math.max(edge, 0.0)) ;
      ctx.globalAlpha = a;
      ctx.drawImage(c.spr, c.x, c.y);
      var dx = c.x + c.w * 0.5 - moon.x, dy = c.y + c.h * 0.5 - moon.y;
      var prox = Math.exp(-(dx * dx + dy * dy) / (2 * Math.pow(moon.rad * 5.2, 2)));
      if (prox > 0.04 && !st.light) {
        ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = a * prox * 1.6;
        ctx.drawImage(c.spr, c.x, c.y); ctx.globalCompositeOperation = 'source-over';
      }
    }
    ctx.globalAlpha = 1;
    // water: the picture itself, strip by strip, each strip pushed by a travelling wave (perspective-scaled)
    var strip = st.strip, span = H - hy;
    for (var y = hy; y < H; y += strip) {
      var dd = (y - hy) / span;                                 // 0 at the horizon -> 1 at the bottom of the screen
      var z = 1 / (dd + 0.07);                                  // perspective: ripples bunch up toward the horizon
      var amp = (0.35 + 9.5 * Math.pow(dd, 1.25)) * d;
      var p1 = z * 9.5 - t * (0.55 + dd * 0.5);
      var p2 = z * 17.0 + t * (0.42 + dd * 0.35) + Math.sin(t * 0.21 + dd * 3) * 1.2;
      var p3 = z * 31.0 - t * 0.9;
      var dx2 = amp * (0.55 * Math.sin(p1) + 0.32 * Math.sin(p2) + 0.13 * Math.sin(p3));
      var dy2 = amp * 0.16 * Math.sin(z * 6.0 - t * (0.5 + dd * 0.3));         // slow rise and fall of the swell
      var sy = clamp(y + dy2, 0, H - strip);
      ctx.drawImage(base, M + dx2, sy, W, strip, 0, y, W, strip);
    }
    // glitter on the moon's path
    if (!st.light) {
      ctx.globalCompositeOperation = 'lighter';
      for (var k = 0; k < st.sparks.length; k++) {
        var sp = st.sparks[k], sdd = sp.dd, sy2 = hy + sdd * span * 0.92 + 2;
        var half = (0.012 + 0.10 * sdd) * W;                                       // the column widens toward the viewer
        var sx = moon.x + sp.u * half + Math.sin(t * 0.6 + sp.ph) * (2 + 8 * sdd) * d;
        var tw = Math.pow(Math.max(0, Math.sin(t * sp.sp * 2.2 + sp.ph)), 5);        // brief wink
        if (tw < 0.04) continue;
        var env2 = 1 - Math.abs(sp.u) * 0.55;
        ctx.globalAlpha = 0.85 * tw * env2;
        var wdt = (5 + 26 * sdd) * sp.sz * d, hgt = Math.max(1, (0.8 + 1.6 * sdd) * d);
        ctx.fillStyle = 'rgb(214,228,255)';
        ctx.fillRect(sx - wdt / 2, sy2, wdt, hgt);
      }
      ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
    }
  }

  /* -------------------------------------------------------------- MISTY FOREST */
  // Periodic value noise (tiles horizontally) -> a soft cloud texture, normalised 0..1
  function noiseTexture(w, h, cellsX, cellsY, octaves, seed) {
    var rng = (function (s) { return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; })(seed);
    function lattice(cx, cy) {
      var a = new Float32Array(cx * cy);
      for (var i = 0; i < a.length; i++) a[i] = rng();
      return a;
    }
    var layers = [];
    for (var o = 0; o < octaves; o++) { var cx = cellsX << o, cy = cellsY << o; layers.push({ cx: cx, cy: cy, a: lattice(cx, cy), amp: Math.pow(0.52, o) }); }
    var out = new Float32Array(w * h), max = 0, min = 1e9, x, y, k;
    for (y = 0; y < h; y++) {
      for (x = 0; x < w; x++) {
        var v = 0, tot = 0;
        for (k = 0; k < layers.length; k++) {
          var L = layers[k], fx = x / w * L.cx, fy = y / h * L.cy;
          var x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
          tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty);
          var xa = x0 % L.cx, xb = (x0 + 1) % L.cx, ya = Math.min(y0, L.cy - 1), yb = Math.min(y0 + 1, L.cy - 1);
          var n00 = L.a[ya * L.cx + xa], n10 = L.a[ya * L.cx + xb], n01 = L.a[yb * L.cx + xa], n11 = L.a[yb * L.cx + xb];
          v += L.amp * ((n00 * (1 - tx) + n10 * tx) * (1 - ty) + (n01 * (1 - tx) + n11 * tx) * ty);
          tot += L.amp;
        }
        v /= tot; out[y * w + x] = v; if (v > max) max = v; if (v < min) min = v;
      }
    }
    for (var i2 = 0; i2 < out.length; i2++) out[i2] = (out[i2] - min) / (max - min || 1);
    return out;
  }

  // One mist layer: white-ish RGBA texture whose alpha is billowy cloud noise shaped by a vertical band profile.
  function mistTexture(w, h, seed, opts) {
    var noise = noiseTexture(w, h, opts.cx, opts.cy, opts.oct, seed);
    var c = document.createElement('canvas'); c.width = w; c.height = h;
    var g = c.getContext('2d'), img = g.createImageData(w, h), px = img.data, rgb = opts.rgb;
    for (var y = 0; y < h; y++) {
      var fy = y / (h - 1);
      var band = opts.band(fy) * smooth(0, 0.16, fy) * smooth(1, 0.84, fy);   // feather the top/bottom so no layer ends in a straight edge
      for (var x = 0; x < w; x++) {
        var n = noise[y * w + x];
        var dens = smooth(opts.lo, opts.hi, n) * band;          // cut thin edges so there are clear gaps, not an even veil
        var i = (y * w + x) * 4;
        px[i] = rgb[0]; px[i + 1] = rgb[1]; px[i + 2] = rgb[2]; px[i + 3] = Math.round(255 * clamp(dens, 0, 1));
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  }

  function forestSetup(st) {
    var W = st.W, H = st.H, pos = scenePos();
    var r = coverRect(st.img.naturalWidth, st.img.naturalHeight, W, H, pos);
    var base = document.createElement('canvas'); base.width = W; base.height = H;
    var b = base.getContext('2d'); b.imageSmoothingQuality = 'high';
    b.drawImage(st.img, r.x, r.y, r.w, r.h);
    st.base = base; st.rect = r;
    var light = st.light, rgb = light ? [255, 255, 255] : [206, 224, 226];
    var tw = st.small ? 384 : 640, th = st.small ? 96 : 140;
    function band(centre, width, soft) { return function (f) { return Math.exp(-Math.pow((f - centre) / width, 2) * (soft || 1)); }; }
    // y: fraction of the picture height where the band sits; speed px/s (sign = direction); alpha = overall strength
    var defs = [
      { name: 'haze',  y0: 0.10, y1: 0.46, band: band(0.55, 0.42), cx: 3, cy: 2, oct: 4, lo: 0.30, hi: 0.78, sp: 3.2,  a: light ? 0.18 : 0.20, seed: 11 },
      { name: 'ridge', y0: 0.26, y1: 0.58, band: band(0.50, 0.30), cx: 4, cy: 3, oct: 4, lo: 0.40, hi: 0.80, sp: -5.5, a: light ? 0.22 : 0.24, seed: 23 },
      { name: 'trees', y0: 0.36, y1: 0.66, band: band(0.58, 0.26), cx: 5, cy: 3, oct: 5, lo: 0.46, hi: 0.82, sp: 8.5,  a: light ? 0.24 : 0.26, seed: 37 },
      { name: 'lake',  y0: 0.52, y1: 0.74, band: band(0.34, 0.30), cx: 6, cy: 3, oct: 5, lo: 0.44, hi: 0.80, sp: -11,  a: light ? 0.26 : 0.30, seed: 41, lakeOnly: true },
      { name: 'veil',  y0: 0.62, y1: 0.98, band: band(0.50, 0.46), cx: 4, cy: 3, oct: 4, lo: 0.50, hi: 0.86, sp: 15,   a: light ? 0.12 : 0.17, seed: 53, lakeOnly: true }
    ];
    st.mist = defs.map(function (d, i) {
      var tex = mistTexture(tw, th, d.seed, { cx: d.cx, cy: d.cy, oct: d.oct, lo: d.lo, hi: d.hi, band: d.band, rgb: rgb });
      var tex2 = mistTexture(tw, th, d.seed + 101, { cx: d.cx + 1, cy: d.cy, oct: d.oct, lo: d.lo, hi: d.hi, band: d.band, rgb: rgb });
      var top = r.y + r.h * d.y0, bot = r.y + r.h * d.y1;
      return { d: d, tex: tex, tex2: tex2, top: top, h: bot - top, scale: (r.w * (1.6 + i * 0.35)) / tw, ph: rnd(0, 1), off: rnd(0, tw) };
    });
  }

  function forestFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, t = st.t, r = st.rect;
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(st.base, 0, 0);
    var lakeY = r.y + r.h * LAKE_TOP;
    for (var i = 0; i < st.mist.length; i++) {
      var m = st.mist[i], d = m.d;
      m.off += d.sp * st.dpr * dt * 0.95;
      var tw = m.tex.width * m.scale;                           // on-screen width of one tile
      // density breathes slowly, each layer on its own clock, and the two copies cross-fade so shapes keep changing
      var breathe = 0.72 + 0.28 * Math.sin(t * (0.07 + i * 0.013) + m.ph * 6.28);
      var mix = 0.5 + 0.5 * Math.sin(t * 0.045 + m.ph * 6.28 + i);
      var vy = Math.sin(t * 0.05 + m.ph * 6.28) * m.h * 0.025;
      ctx.save();
      if (d.lakeOnly) {                                         // lake mist stays on the water; the foreground trunks stay clear
        ctx.beginPath(); ctx.rect(0, Math.max(0, lakeY - m.h * 0.15), W, H); ctx.clip();
      }
      var x0 = -(((m.off % tw) + tw) % tw);
      var copies = [[m.tex, 1 - mix, 1], [m.tex2, mix, -0.6]];
      for (var c = 0; c < copies.length; c++) {
        var cp = copies[c];
        ctx.globalAlpha = clamp(d.a * breathe * cp[1] * 2.6, 0, 1);
        var shift = c === 0 ? 0 : (m.off * cp[2]) - m.off;      // the second copy drifts at its own pace
        var start = x0 + (((shift % tw) + tw) % tw) - tw;
        for (var x = start; x < W; x += tw) ctx.drawImage(cp[0], x, m.top + vy, tw, m.h);
      }
      ctx.restore();
    }
    ctx.globalAlpha = 1;
    // keep the left foreground trees in front of the mist: re-draw that strip of the picture with a soft edge
    // (done with a mask canvas so the picture, not the mist, wins there)
    if (!st.keepMask) {
      var mc = document.createElement('canvas'); mc.width = W; mc.height = H;
      var mg = mc.getContext('2d');
      mg.drawImage(st.base, 0, 0);
      mg.globalCompositeOperation = 'destination-in';
      var gr = mg.createLinearGradient(0, 0, W * 0.30, 0);
      gr.addColorStop(0, 'rgba(0,0,0,0.78)'); gr.addColorStop(0.55, 'rgba(0,0,0,0.30)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
      mg.fillStyle = gr; mg.fillRect(0, 0, W, H);
      // only below the sky: tree trunks and rocks, not the left edge of the cloud cover
      var gy = mg.createLinearGradient(0, r.y + r.h * 0.12, 0, r.y + r.h * 0.34);
      gy.addColorStop(0, 'rgba(0,0,0,0)'); gy.addColorStop(1, 'rgba(0,0,0,1)');
      mg.globalCompositeOperation = 'destination-in'; mg.fillStyle = gy; mg.fillRect(0, 0, W, H);
      st.keepMask = mc;
    }
    ctx.drawImage(st.keepMask, 0, 0);
  }


  /* -------------------------------------------------------------- OBSIDIAN GOLD (WebGL)
     The marble itself flows: the picture is re-sampled through two layers of slow, curling domain-warp noise, so the black
     stone swirls like ink in water, and a soft pulse of light travels along the gold veins (found by colour: warm and
     bright). All on the GPU, one full-screen quad. Without WebGL the still picture is left alone. */
  var GOLD_VS = 'attribute vec2 p;varying vec2 v;void main(){v=vec2((p.x+1.0)*0.5,(1.0-p.y)*0.5);gl_Position=vec4(p,0.0,1.0);}';
  var GOLD_FS = [
    'precision mediump float;varying vec2 v;uniform sampler2D tex;uniform float t;uniform vec4 cov;uniform float lite;uniform vec2 asp;',
    'vec2 flow(vec2 q,float k){float a=sin(q.y*3.1+k*0.23)+sin(q.x*2.3-k*0.19+q.y*1.7);float b=cos(q.x*2.7+k*0.21)+cos(q.y*3.4-k*0.17-q.x*1.3);return vec2(a,b);}',
    'void main(){',
    ' vec2 uv=v*cov.xy+cov.zw; vec2 q=uv*asp;',
    ' vec2 w1=flow(q*1.6,t); vec2 w2=flow(q*3.4+w1*0.45,t*1.35);',
    ' uv+=w1*0.0048+w2*0.0026;',
    ' vec3 c=texture2D(tex,uv).rgb;',
    ' float gold=smoothstep(0.07,0.32,c.r-c.b)*smoothstep(0.22,0.6,c.r);',
    ' float pulse=0.5+0.5*sin(t*0.75+q.x*7.0+q.y*5.0+w1.x*1.6);',
    ' float pulse2=0.5+0.5*sin(t*0.43-q.x*4.0+q.y*8.0+w2.y*1.3);',
    ' c+=vec3(1.0,0.78,0.36)*gold*(0.10+0.26*pulse*pulse2)*(1.0-0.5*lite);',
    ' c*=0.965+0.05*sin(t*0.31+w2.y*1.2);',
    ' gl_FragColor=vec4(c,1.0);}'
  ].join('\n');

  function glShader(gl, type, src) {
    var sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error('shader');
    return sh;
  }
  function goldSetup(st) {
    var gl = st.ctx;
    if (!st.prog) {
      var pr = gl.createProgram();
      gl.attachShader(pr, glShader(gl, gl.VERTEX_SHADER, GOLD_VS));
      gl.attachShader(pr, glShader(gl, gl.FRAGMENT_SHADER, GOLD_FS));
      gl.linkProgram(pr);
      if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error('link');
      gl.useProgram(pr);
      var buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      var loc = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      st.prog = pr;
      st.u = { t: gl.getUniformLocation(pr, 't'), cov: gl.getUniformLocation(pr, 'cov'), lite: gl.getUniformLocation(pr, 'lite'), asp: gl.getUniformLocation(pr, 'asp'), tex: gl.getUniformLocation(pr, 'tex') };
      st.texture = gl.createTexture();
    }
    if (st.img.naturalWidth > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('size');
    gl.bindTexture(gl.TEXTURE_2D, st.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, st.img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.viewport(0, 0, st.W, st.H);
    var iw = st.img.naturalWidth, ih = st.img.naturalHeight, sc = Math.max(st.W / iw, st.H / ih);
    var fx = st.W / (iw * sc), fy = st.H / (ih * sc), pos = scenePos();
    st.cover = [fx, fy, (1 - fx) * pos[0], (1 - fy) * pos[1]];
    st.asp = [1, ih / iw];
  }
  function goldFrame(st) {
    var gl = st.ctx;
    gl.uniform1f(st.u.t, st.t); gl.uniform4f(st.u.cov, st.cover[0], st.cover[1], st.cover[2], st.cover[3]);
    gl.uniform1f(st.u.lite, st.light ? 1 : 0); gl.uniform2f(st.u.asp, st.asp[0], st.asp[1]); gl.uniform1i(st.u.tex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /* -------------------------------------------------------------- SAKURA REVERIE
     One slow day: the picture is the sunset; over about five minutes it eases through night (blue dusk, paper-lantern glow
     in the pagoda, fireflies), morning (soft pink light) and noon (bright, clear) and back to sunset. The lake ripples (the
     water in the picture is re-drawn in strips pushed by travelling waves, plus the odd widening ring), and petals drift
     diagonally across the scene from the blossom tree. */
  var SAKURA_LAKE = 0.60;                    // water starts here, as a fraction of the picture's height
  var PAGODA = { x: 0.317, y: 0.50 };        // paper-lantern windows
  // The picture is a sunset; every other time of day is a colour wash over it. A full day takes about five minutes, on the wall clock (so it never restarts when you change page):
  // sunrise -> morning -> noon -> afternoon -> sunset -> evening -> night -> (sunrise again).
  // [name, hold s, ease-in s, multiply top, multiply bottom, screen top, screen bottom, night amount (stars, lanterns, fireflies)]
  var SK_PHASES = [
    ['sunrise',   12, 20, [255, 226, 214], [255, 238, 228], [64, 30, 38],    [38, 20, 24],   0.10],
    ['morning',   16, 22, [255, 240, 246], [255, 246, 250], [72, 40, 62],    [40, 24, 34],   0],
    ['noon',      20, 22, [214, 232, 255], [246, 251, 255], [96, 136, 196],  [24, 36, 52],   0],
    ['afternoon', 16, 22, [255, 244, 224], [255, 249, 236], [70, 58, 22],    [30, 22, 8],    0],
    ['sunset',    24, 22, [255, 255, 255], [255, 255, 255], [0, 0, 0],       [0, 0, 0],      0],
    ['evening',   14, 22, [156, 132, 198], [128, 116, 170], [22, 14, 48],    [8, 6, 26],     0.45],
    ['night',     20, 24, [58, 74, 142],   [44, 58, 112],   [4, 8, 26],      [0, 0, 8],      1]
  ];
  var SK_START = 4;                          // start the day on the sunset (the picture as painted)
  var SK_TOTAL = SK_PHASES.reduce(function (a, p) { return a + p[1] + p[2]; }, 0);
  var SKX = function (x) { return x; };   // the picture is the full square painting, edge to edge (no padded wide canvas)
  function lerp(a, b, k) { return a + (b - a) * k; }
  function mix3(a, b, k) { return [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)]; }
  function rgb3(c) { return 'rgb(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ')'; }
  // -> { mt, mb, stp, sb, night } at time t of the day cycle
  function skyAt(t) {
    var off = 0; for (var q = 0; q < SK_START; q++) off += SK_PHASES[q][1] + SK_PHASES[q][2];
    off += SK_PHASES[SK_START][2];   // begin on the sunset hold
    var T = (((t + off) % SK_TOTAL) + SK_TOTAL) % SK_TOTAL, i, acc = 0, n = SK_PHASES.length;
    for (i = 0; i < n; i++) {
      var P = SK_PHASES[i], prev = SK_PHASES[(i + n - 1) % n], ease = P[2];
      // each phase = ease from the previous phase, then hold
      if (T < acc + ease) { var k = smooth(0, 1, (T - acc) / ease); return blend(prev, P, k); }
      acc += ease;
      if (T < acc + P[1]) return blend(P, P, 1);
      acc += P[1];
    }
    return blend(SK_PHASES[0], SK_PHASES[0], 1);
  }
  function blend(a, b, k) {
    return { mt: mix3(a[3], b[3], k), mb: mix3(a[4], b[4], k), stp: mix3(a[5], b[5], k), sb: mix3(a[6], b[6], k),
      night: lerp(a[7], b[7], k) };
  }
  function petalSprite(c1, c2) {
    var c = document.createElement('canvas'); c.width = 32; c.height = 20;
    var g = c.getContext('2d'), gr = g.createLinearGradient(0, 0, 32, 20);
    gr.addColorStop(0, c1); gr.addColorStop(1, c2);
    g.fillStyle = gr; g.beginPath(); g.moveTo(1, 10); g.bezierCurveTo(6, -2, 24, 0, 31, 10); g.bezierCurveTo(24, 21, 6, 21, 1, 10); g.fill();
    return c;
  }
  function sakuraSetup(st) {
    var W = st.W, H = st.H, r = coverRect(st.img.naturalWidth, st.img.naturalHeight, W, H, scenePos());
    var M = Math.ceil(20 * st.dpr);
    var base = document.createElement('canvas'); base.width = W + 2 * M; base.height = H;
    var b = base.getContext('2d'); b.imageSmoothingQuality = 'high';
    b.drawImage(st.img, r.x + M, r.y, r.w, r.h);
    b.drawImage(base, M, 0, 1, H, 0, 0, M, H); b.drawImage(base, W + M - 1, 0, 1, H, W + M, 0, M, H);
    st.base = base; st.M = M; st.rect = r;
    st.lakeY = Math.max(0, Math.round(r.y + r.h * SAKURA_LAKE)); st.strip = Math.max(2, Math.round(2 * st.dpr));
    st.lamp = { x: r.x + r.w * PAGODA.x, y: r.y + r.h * PAGODA.y, rad: r.w * 0.05 };
    st.petalSprites = st.light ? [petalSprite('#ffffff', '#f6b6cc'), petalSprite('#fff0f5', '#f08fb0')] : [petalSprite('#ffe3ec', '#f7a9c4'), petalSprite('#ffd3e1', '#f28fb3'), petalSprite('#fff1f5', '#f9b9cf')];
    var n = st.small ? 14 : 30;
    st.petals = []; for (var i = 0; i < n; i++) st.petals.push(newPetal(st, true));
    st.flies = []; for (var f = 0; f < (st.small ? 6 : 12); f++) st.flies.push({ x: rnd(0.14, 0.86), y: rnd(0.45, 0.85), ph: rnd(0, 6.28), sp: rnd(0.2, 0.5), rr: rnd(0.01, 0.035) });
    st.rings = []; st.nextRing = 2;
    st.mask = null;
    var wk = document.createElement('canvas'); wk.width = 2; wk.height = 2; st.tint = wk;
  }
  function newPetal(st, anywhere) {
    var r = st.rect, W = st.W, H = st.H, d = st.dpr;
    var fromTree = Math.random() < 0.7;
    var x = fromTree ? r.x + r.w * SKX(rnd(0.50, 0.90)) : rnd(0, W * 1.1);
    var y = fromTree ? r.y + r.h * rnd(0.26, 0.62) : rnd(-0.1 * H, 0.3 * H);
    if (anywhere) { x = rnd(0, W * 1.1); y = rnd(-0.05 * H, H); }
    return { x: x, y: y, vx: -rnd(14, 34) * d, vy: rnd(10, 24) * d, rot: rnd(0, 6.28), vr: rnd(-1.4, 1.4), flip: rnd(0, 6.28), vf: rnd(1.2, 3.2),
      sz: rnd(12, 22) * d, sw: rnd(0.5, 1.4), ph: rnd(0, 6.28), spr: (Math.random() * st.petalSprites.length) | 0, a: rnd(0.55, 0.9) };
  }
  function sakuraFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, M = st.M, t = st.t, d = st.dpr, ly = st.lakeY, base = st.base;
    // 2026-10-30: the day cycle runs on the WALL CLOCK, not the stage's own timer, so navigating to another page (a fresh stage, t=0) or
    // opening a second tab continues the same sunrise -> night sweep instead of restarting it.
    var sky = skyAt(Date.now() / 1000);
    // land + sky: the picture
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
    ctx.drawImage(base, M, 0, W, ly, 0, 0, W, ly);
    // lake: strips pushed sideways by travelling waves, stronger toward the viewer
    var span = Math.max(1, H - ly), strip = st.strip;
    for (var y = ly; y < H; y += strip) {
      var dd = (y - ly) / span, z = 1 / (dd + 0.12);
      var amp = (0.25 + 3.6 * Math.pow(dd, 1.1)) * d;
      var dx = amp * (0.6 * Math.sin(z * 7.5 - t * (0.7 + dd * 0.5)) + 0.4 * Math.sin(z * 15.0 + t * 0.55 + Math.sin(t * 0.2) * 2));
      ctx.drawImage(base, M + dx, y, W, strip, 0, y, W, strip);
    }
    // widening rings now and then
    st.nextRing -= dt;
    if (st.nextRing <= 0) { st.nextRing = rnd(1.6, 4.2); st.rings.push({ x: rnd(0.08, 0.92) * W, y: ly + rnd(0.12, 0.92) * span, age: 0, life: rnd(5, 8) }); if (st.rings.length > 7) st.rings.shift(); }
    ctx.lineWidth = 1.2 * d; ctx.strokeStyle = st.light ? '#ffffff' : '#ffd9e6';
    for (var q = st.rings.length - 1; q >= 0; q--) {
      var rg = st.rings[q]; rg.age += dt; var k = rg.age / rg.life;
      if (k >= 1) { st.rings.splice(q, 1); continue; }
      var rad = (6 + 70 * k) * d, persp = 0.18 + 0.2 * ((rg.y - ly) / span);
      ctx.globalAlpha = 0.28 * (1 - k) * (1 - k); ctx.beginPath(); ctx.ellipse(rg.x, rg.y, rad, rad * persp, 0, 0, 6.2832); ctx.stroke();
      if (k > 0.18) { ctx.globalAlpha *= 0.6; ctx.beginPath(); ctx.ellipse(rg.x, rg.y, rad * 0.62, rad * 0.62 * persp, 0, 0, 6.2832); ctx.stroke(); }
    }
    // time of day: a multiply wash then a screen wash, each a top->bottom gradient
    ctx.globalAlpha = 1;
    if (sky.mt[0] < 254.5 || sky.mt[1] < 254.5 || sky.mt[2] < 254.5 || sky.mb[0] < 254.5) {
      var g1 = ctx.createLinearGradient(0, 0, 0, H); g1.addColorStop(0, rgb3(sky.mt)); g1.addColorStop(1, rgb3(sky.mb));
      ctx.globalCompositeOperation = 'multiply'; ctx.fillStyle = g1; ctx.fillRect(0, 0, W, H);
    }
    if (sky.stp[0] + sky.stp[1] + sky.stp[2] + sky.sb[0] + sky.sb[1] + sky.sb[2] > 3) {
      var g2 = ctx.createLinearGradient(0, 0, 0, H); g2.addColorStop(0, rgb3(sky.stp)); g2.addColorStop(1, rgb3(sky.sb));
      ctx.globalCompositeOperation = 'screen'; ctx.fillStyle = g2; ctx.fillRect(0, 0, W, H);
    }
    ctx.globalCompositeOperation = 'source-over';
    var night = sky.night;
    // night: stars, lantern glow in the pagoda (and its reflection), fireflies
    if (night > 0.02) {
      ctx.globalCompositeOperation = 'lighter';
      if (!st.stars) { st.stars = []; for (var s = 0; s < (st.small ? 24 : 48); s++) st.stars.push({ x: rnd(0, 1), y: rnd(0, 0.16), ph: rnd(0, 6.28), sp: rnd(0.4, 1.3), sz: rnd(0.6, 1.6) }); }
      for (var si = 0; si < st.stars.length; si++) {
        var S = st.stars[si], tw = 0.45 + 0.55 * Math.sin(t * S.sp + S.ph);
        ctx.globalAlpha = night * 0.7 * Math.max(0, tw);
        ctx.fillStyle = '#e8f0ff'; var skyB = Math.min(H, st.rect.y + st.rect.h * 0.28), sx = st.rect.x + S.x * st.rect.w, sy = S.y / 0.16 * skyB * 0.92;
        if (skyB > 8 && sx >= 0 && sx <= W) ctx.fillRect(sx, sy, S.sz * d, S.sz * d);
      }
      var L = st.lamp, flick = 0.88 + 0.12 * Math.sin(t * 2.3) * Math.sin(t * 0.9 + 1);
      for (var pass = 0; pass < 2; pass++) {
        var ly2 = pass ? ly + (ly - L.y) * 0.95 : L.y;               // second pass = reflection in the lake
        var gl = ctx.createRadialGradient(L.x, ly2, 0, L.x, ly2, L.rad * (pass ? 1.9 : 1.5));
        gl.addColorStop(0, 'rgba(255,196,110,' + (0.62 * night * flick * (pass ? 0.55 : 1)) + ')'); gl.addColorStop(0.45, 'rgba(255,150,70,' + (0.22 * night * (pass ? 0.5 : 1)) + ')'); gl.addColorStop(1, 'rgba(255,140,60,0)');
        ctx.globalAlpha = 1; ctx.fillStyle = gl;
        ctx.fillRect(L.x - L.rad * 2.6, ly2 - L.rad * 2.6, L.rad * 5.2, L.rad * 5.2);
      }
      for (var fi = 0; fi < st.flies.length; fi++) {
        var F = st.flies[fi], fx = (F.x + Math.sin(t * F.sp + F.ph) * F.rr) * W, fy = (F.y + Math.cos(t * F.sp * 0.8 + F.ph) * F.rr) * H;
        var pulse = Math.pow(Math.max(0, Math.sin(t * 1.1 + F.ph * 3)), 3);
        var fg = ctx.createRadialGradient(fx, fy, 0, fx, fy, 9 * d);
        fg.addColorStop(0, 'rgba(255,240,150,' + (0.9 * pulse * night) + ')'); fg.addColorStop(1, 'rgba(255,240,150,0)');
        ctx.globalAlpha = 1; ctx.fillStyle = fg; ctx.fillRect(fx - 9 * d, fy - 9 * d, 18 * d, 18 * d);
      }
      ctx.globalCompositeOperation = 'source-over';
    }
    // petals, drifting diagonally (down and to the left) with a little flutter
    var dim = 1 - 0.45 * night;
    for (var pi = 0; pi < st.petals.length; pi++) {
      var P = st.petals[pi];
      P.x += (P.vx + Math.sin(t * P.sw + P.ph) * 9 * d) * dt; P.y += (P.vy + Math.cos(t * P.sw * 0.8 + P.ph) * 5 * d) * dt;
      P.rot += P.vr * dt; P.flip += P.vf * dt;
      if (P.x < -20 * d || P.y > H + 20 * d) { st.petals[pi] = newPetal(st, false); continue; }
      var edgeFade = Math.min(1, Math.max(0, (W - P.x + 10 * d) / (60 * d)), Math.max(0, P.y / (30 * d) + 1));
      ctx.globalAlpha = P.a * dim * Math.min(1, edgeFade);
      ctx.save(); ctx.translate(P.x, P.y); ctx.rotate(P.rot); ctx.scale(1, 0.25 + 0.75 * Math.abs(Math.cos(P.flip)));
      ctx.drawImage(st.petalSprites[P.spr], -P.sz / 2, -P.sz * 0.31, P.sz, P.sz * 0.62); ctx.restore();
    }
    ctx.globalAlpha = 1;
  }

  /* -------------------------------------------------------------- ABYSSAL BLUE
     Deep water, lit from above: soft turquoise light shafts that sway and breathe, caustic ripples near the surface, marine
     snow drifting UP, a distant whale that crosses the whole screen very slowly (its body undulates, tail beating), and
     schools of realistic painted fish (sardines, mackerel, glowing lanternfish) with the odd tuna gliding past. Everything is drawn behind a readable scrim and kept slow on purpose. */
  function causticTexture(n) {
    var c = document.createElement('canvas'); c.width = n; c.height = n;
    var g = c.getContext('2d'), img = g.createImageData(n, n), px = img.data, TAU = 6.2832;
    for (var y = 0; y < n; y++) {
      for (var x = 0; x < n; x++) {
        var u = x / n * TAU, v = y / n * TAU;
        // warped sine lattice -> thin bright filaments where waves cross (integer frequencies keep it tileable)
        var a = Math.sin(u * 2 + Math.sin(v * 3) * 0.9) + Math.sin(v * 2 - Math.sin(u * 3) * 0.9) + Math.sin((u + v) * 2 + 1.3 * Math.sin((u - v) * 3));
        var f = 1 - Math.abs(a) / 3;
        var val = Math.pow(clamp(f, 0, 1), 11);
        var i = (y * n + x) * 4;
        px[i] = 190; px[i + 1] = 255; px[i + 2] = 245; px[i + 3] = Math.round(255 * clamp(val * 1.1, 0, 1));
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  }
  function whaleSprite(w, h) {
    var c = document.createElement('canvas'); c.width = w; c.height = h;
    var g = c.getContext('2d'), Hm = h * 0.25, cy = h * 0.5, L = w * 0.82, x0 = w * 0.06;
    function prof(u) { return u < 0.36 ? Math.pow(Math.sin(u / 0.36 * 1.5708), 0.55) : Math.pow(Math.max(0, 1 - (u - 0.36) / 0.64), 1.35) * 0.96 + 0.045; }
    var grd = g.createLinearGradient(0, cy - Hm, 0, cy + Hm);
    grd.addColorStop(0, 'rgba(3,26,38,0.92)'); grd.addColorStop(0.62, 'rgba(6,38,52,0.88)'); grd.addColorStop(1, 'rgba(34,92,108,0.75)');
    g.fillStyle = grd; g.beginPath();
    var N = 60, k;
    for (k = 0; k <= N; k++) { var u = k / N; g.lineTo(x0 + u * L, cy - prof(u) * Hm * (u < 0.1 ? 0.9 : 1)); }
    for (k = N; k >= 0; k--) { var u2 = k / N; g.lineTo(x0 + u2 * L, cy + prof(u2) * Hm * 0.82); }
    g.closePath(); g.fill();
    // tail flukes (a swept Y) and the long pectoral fin
    g.fillStyle = 'rgba(4,30,42,0.9)'; var tx = x0 + L - 2, ty = cy;
    g.beginPath(); g.moveTo(tx - 6, ty); g.quadraticCurveTo(tx + w * 0.05, ty - h * 0.20, tx + w * 0.085, ty - h * 0.21); g.quadraticCurveTo(tx + w * 0.045, ty - h * 0.05, tx + 3, ty); g.quadraticCurveTo(tx + w * 0.045, ty + h * 0.05, tx + w * 0.085, ty + h * 0.21); g.quadraticCurveTo(tx + w * 0.05, ty + h * 0.20, tx - 6, ty); g.fill();
    g.fillStyle = 'rgba(8,48,62,0.85)'; g.beginPath(); g.moveTo(x0 + L * 0.30, cy + Hm * 0.55); g.quadraticCurveTo(x0 + L * 0.42, cy + Hm * 2.1, x0 + L * 0.52, cy + Hm * 2.35); g.quadraticCurveTo(x0 + L * 0.46, cy + Hm * 1.2, x0 + L * 0.40, cy + Hm * 0.7); g.fill();
    // dorsal fin
    g.beginPath(); g.moveTo(x0 + L * 0.62, cy - prof(0.62) * Hm); g.lineTo(x0 + L * 0.665, cy - prof(0.64) * Hm - h * 0.05); g.lineTo(x0 + L * 0.70, cy - prof(0.70) * Hm); g.fill();
    // blur it (once) so it reads as far away in the water
    var out = document.createElement('canvas'); out.width = w; out.height = h; var o = out.getContext('2d');
    try { o.filter = 'blur(' + Math.max(2, w / 110) + 'px)'; } catch (e) {}
    o.drawImage(c, 0, 0); o.filter = 'none';
    return out;
  }

  /* ---- realistic fish: each species is painted once onto a small canvas (countershaded body, scales, iridescent sheen, fins with
     rays, gill plate, eye with a catch-light) and then swims by swinging a separate tail around the tail-stock and rolling the body a
     little, so a school moves the way real fish do instead of sliding past as flat shapes. Always faces RIGHT; mirrored when swimming left. */
  var FISH_KINDS = {
    sardine:  { len: 1, depth: 0.27, back: ['#0a2c48', '#2d78a0'], flank: ['#b9dbe6', '#e8f6fa'], belly: '#ffffff', sheen: ['rgba(120,255,235,.38)', 'rgba(150,140,255,.30)'], stripe: 'rgba(40,110,150,.55)', bars: 0, eye: '#f4d27a', tail: 'fork', fin: 'rgba(170,210,225,.55)' },
    mackerel: { len: 1.15, depth: 0.25, back: ['#06324a', '#1f8a9a'], flank: ['#9fd0d8', '#e2f3f4'], belly: '#f6fdfd', sheen: ['rgba(90,255,200,.30)', 'rgba(80,170,255,.28)'], stripe: 'rgba(5,40,60,.0)', bars: 9, eye: '#e8c15c', tail: 'fork', fin: 'rgba(40,110,130,.6)' },
    lantern:  { len: 0.85, depth: 0.30, back: ['#0b1736', '#27407e'], flank: ['#3a5da0', '#7d9fd8'], belly: '#b7c9ee', sheen: ['rgba(140,170,255,.35)', 'rgba(200,120,255,.30)'], stripe: 'rgba(0,0,0,0)', bars: 0, eye: '#9ff5ff', tail: 'fan', fin: 'rgba(80,110,190,.55)', glow: true },
    tuna:     { len: 1.6, depth: 0.23, back: ['#04182c', '#14507a'], flank: ['#7fb6cf', '#dcecf2'], belly: '#f2f7f9', sheen: ['rgba(70,200,255,.30)', 'rgba(120,120,255,.22)'], stripe: 'rgba(240,200,60,.55)', bars: 0, eye: '#d9b44a', tail: 'lunate', fin: 'rgba(30,70,100,.8)', finlets: true }
  };
  function fishSprite(kind, Lpx) {
    var K = FISH_KINDS[kind], L = Math.max(24, Math.round(Lpx)), Hh = L * K.depth * 0.5, pad = Math.ceil(L * 0.28), cy = Hh + pad;
    var body = document.createElement('canvas'); body.width = L + pad; body.height = Math.ceil(2 * (Hh + pad));
    var g = body.getContext('2d'), ox = 0;         // x: 0 = tail-stock, L = snout
    function outline() {
      g.beginPath(); g.moveTo(L, cy + Hh * 0.10);
      g.bezierCurveTo(L * 0.93, cy - Hh * 0.55, L * 0.72, cy - Hh * 1.02, L * 0.46, cy - Hh * 1.0);
      g.bezierCurveTo(L * 0.24, cy - Hh * 0.97, L * 0.10, cy - Hh * 0.34, ox, cy - Hh * 0.17);
      g.lineTo(ox, cy + Hh * 0.17);
      g.bezierCurveTo(L * 0.11, cy + Hh * 0.40, L * 0.24, cy + Hh * 0.92, L * 0.46, cy + Hh * 0.94);
      g.bezierCurveTo(L * 0.74, cy + Hh * 0.96, L * 0.93, cy + Hh * 0.55, L, cy + Hh * 0.10); g.closePath();
    }
    // fins behind the body: dorsal, anal, pelvic (with rays)
    function fin(x1, y1, x2, y2, x3, y3, rays) {
      g.fillStyle = K.fin; g.beginPath(); g.moveTo(x1, y1); g.quadraticCurveTo(x2, y2, x3, y3); g.lineTo(x1, y1); g.fill();
      g.strokeStyle = 'rgba(255,255,255,.25)'; g.lineWidth = Math.max(0.5, L * 0.004);
      for (var i = 1; i < rays; i++) { var f = i / rays; g.beginPath(); g.moveTo(x1 + (x3 - x1) * f, y1 + (y3 - y1) * f); g.lineTo(x2 + (x3 - x2) * 0.2 * f + (x1 - x3) * 0.0, y2 + (y3 - y2) * 0.05 + (y2 - y1) * 0.0); g.stroke(); }
    }
    fin(L * 0.56, cy - Hh * 0.98, L * 0.46, cy - Hh * 1.65, L * 0.34, cy - Hh * 0.99, 7);            // dorsal
    if (kind === 'tuna' || kind === 'mackerel') fin(L * 0.27, cy - Hh * 0.97, L * 0.22, cy - Hh * 1.35, L * 0.16, cy - Hh * 0.62, 5);    // second dorsal
    fin(L * 0.36, cy + Hh * 0.92, L * 0.28, cy + Hh * 1.45, L * 0.18, cy + Hh * 0.6, 5);              // anal
    if (K.finlets) {                                                                                  // tuna finlets along the tail-stock
      g.fillStyle = 'rgba(240,200,60,.9)';
      for (var fl = 0; fl < 5; fl++) { var fx = L * (0.04 + fl * 0.035); g.beginPath(); g.moveTo(fx, cy - Hh * 0.2); g.lineTo(fx - L * 0.012, cy - Hh * 0.52); g.lineTo(fx + L * 0.02, cy - Hh * 0.2); g.fill(); g.beginPath(); g.moveTo(fx, cy + Hh * 0.2); g.lineTo(fx - L * 0.012, cy + Hh * 0.5); g.lineTo(fx + L * 0.02, cy + Hh * 0.2); g.fill(); }
    }
    // body: countershading (dark back -> silver flank -> pale belly)
    outline(); var cg = g.createLinearGradient(0, cy - Hh, 0, cy + Hh);
    cg.addColorStop(0, K.back[0]); cg.addColorStop(0.26, K.back[1]); cg.addColorStop(0.46, K.flank[0]); cg.addColorStop(0.68, K.flank[1]); cg.addColorStop(1, K.belly);
    g.fillStyle = cg; g.fill();
    g.save(); outline(); g.clip();
    // scales: overlapping arcs, faint dark edge + a bright upper lip, rows offset like real scale rows
    var sc = Math.max(2.4, L * 0.032), row = 0;
    g.lineWidth = Math.max(0.4, L * 0.0035);
    for (var yy = cy - Hh * 1.0; yy < cy + Hh; yy += sc * 0.72, row++) {
      for (var xx = L * 0.12 + (row % 2) * sc * 0.5; xx < L * 0.80; xx += sc) {
        g.strokeStyle = 'rgba(10,40,60,.16)'; g.beginPath(); g.arc(xx, yy, sc * 0.55, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke();
        g.strokeStyle = 'rgba(255,255,255,.10)'; g.beginPath(); g.arc(xx, yy - sc * 0.08, sc * 0.55, 1.15 * Math.PI, 1.85 * Math.PI); g.stroke();
      }
    }
    if (K.bars) {                                         // mackerel: wavy dark bars over the back
      g.strokeStyle = 'rgba(2,28,44,.7)'; g.lineWidth = Math.max(0.8, L * 0.012);
      for (var b = 0; b < K.bars; b++) { var bx = L * (0.14 + b * 0.075); g.beginPath(); g.moveTo(bx, cy - Hh * 1.0); for (var q = 1; q <= 5; q++) g.lineTo(bx + Math.sin(q * 1.7 + b) * L * 0.012, cy - Hh * 1.0 + q * Hh * 0.12); g.stroke(); }
    }
    if (K.stripe && K.stripe !== 'rgba(0,0,0,0)' && !/\.0\)$/.test(K.stripe)) {   // lateral line / tuna's yellow flank stripe
      g.strokeStyle = K.stripe; g.lineWidth = Math.max(0.8, L * (kind === 'tuna' ? 0.02 : 0.012)); g.beginPath(); g.moveTo(L * 0.05, cy - Hh * 0.05); g.quadraticCurveTo(L * 0.5, cy - Hh * 0.18, L * 0.82, cy - Hh * 0.06); g.stroke();
    }
    // iridescent sheen (cyan -> violet) and a soft top light
    var sh = g.createLinearGradient(0, cy - Hh, L, cy + Hh * 0.2); sh.addColorStop(0, K.sheen[0]); sh.addColorStop(0.55, 'rgba(255,255,255,0)'); sh.addColorStop(1, K.sheen[1]);
    g.globalCompositeOperation = 'lighter'; g.fillStyle = sh; g.fillRect(0, 0, L + pad, body.height);
    g.globalCompositeOperation = 'source-over';
    var hl = g.createLinearGradient(0, cy - Hh, 0, cy - Hh * 0.2); hl.addColorStop(0, 'rgba(255,255,255,.28)'); hl.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = hl; g.fillRect(0, cy - Hh, L, Hh * 0.8);
    var hd = g.createLinearGradient(L * 0.60, 0, L, 0); hd.addColorStop(0, 'rgba(0,10,24,0)'); hd.addColorStop(1, 'rgba(0,10,24,.28)');   // head a touch darker
    g.fillStyle = hd; g.fillRect(L * 0.6, cy - Hh, L * 0.4, Hh * 2);
    if (K.glow) {                                         // lanternfish photophores: rows of tiny lights along the belly and flank
      for (var ph = 0; ph < 11; ph++) { var px = L * (0.18 + ph * 0.057), py = cy + Hh * (0.38 + 0.12 * Math.sin(ph * 0.9));
        var gg = g.createRadialGradient(px, py, 0, px, py, L * 0.03); gg.addColorStop(0, 'rgba(190,255,255,.95)'); gg.addColorStop(1, 'rgba(60,200,255,0)'); g.fillStyle = gg; g.fillRect(px - L * 0.03, py - L * 0.03, L * 0.06, L * 0.06); }
    }
    g.restore();
    // gill plate, mouth, pectoral fin, eye
    g.strokeStyle = 'rgba(5,30,48,.45)'; g.lineWidth = Math.max(0.6, L * 0.008);
    g.beginPath(); g.arc(L * 0.70, cy - Hh * 0.02, Hh * 0.78, 1.25 * Math.PI, 0.75 * Math.PI, true); g.stroke();
    g.beginPath(); g.moveTo(L, cy + Hh * 0.10); g.lineTo(L * 0.935, cy + Hh * 0.16); g.stroke();
    g.fillStyle = K.fin; g.beginPath(); g.moveTo(L * 0.66, cy + Hh * 0.26); g.quadraticCurveTo(L * 0.54, cy + Hh * 0.78, L * 0.46, cy + Hh * 0.72); g.quadraticCurveTo(L * 0.54, cy + Hh * 0.40, L * 0.66, cy + Hh * 0.26); g.fill();
    var er = Math.max(1.3, Hh * 0.30), ex = L * 0.855, ey = cy - Hh * 0.16;
    g.fillStyle = K.eye; g.beginPath(); g.arc(ex, ey, er * 1.18, 0, 6.2832); g.fill();
    g.fillStyle = '#04080c'; g.beginPath(); g.arc(ex, ey, er * 0.82, 0, 6.2832); g.fill();
    g.fillStyle = 'rgba(255,255,255,.9)'; g.beginPath(); g.arc(ex - er * 0.28, ey - er * 0.30, er * 0.28, 0, 6.2832); g.fill();
    // tail: its own canvas, pivot at its left edge
    var tl = Math.round(L * (K.tail === 'lunate' ? 0.42 : 0.30)), th = Math.round(Hh * (K.tail === 'lunate' ? 3.1 : K.tail === 'fan' ? 1.9 : 2.5));
    var tail = document.createElement('canvas'); tail.width = tl + 2; tail.height = th + 2; var t2 = tail.getContext('2d'), tcy = (th + 2) / 2;
    t2.beginPath(); t2.moveTo(0, tcy - Hh * 0.16);
    if (K.tail === 'fan') { t2.bezierCurveTo(tl * 0.5, tcy - th * 0.46, tl * 0.95, tcy - th * 0.38, tl, tcy - th * 0.12); t2.bezierCurveTo(tl * 0.92, tcy, tl * 0.92, tcy, tl, tcy + th * 0.12); t2.bezierCurveTo(tl * 0.95, tcy + th * 0.38, tl * 0.5, tcy + th * 0.46, 0, tcy + Hh * 0.16); }
    else { t2.bezierCurveTo(tl * 0.35, tcy - th * 0.12, tl * 0.7, tcy - th * 0.46, tl, tcy - th * 0.5); t2.bezierCurveTo(tl * 0.72, tcy - th * 0.14, tl * 0.58, tcy - th * 0.04, tl * 0.56, tcy);
      t2.bezierCurveTo(tl * 0.58, tcy + th * 0.04, tl * 0.72, tcy + th * 0.14, tl, tcy + th * 0.5); t2.bezierCurveTo(tl * 0.7, tcy + th * 0.46, tl * 0.35, tcy + th * 0.12, 0, tcy + Hh * 0.16); }
    t2.closePath(); var tg = t2.createLinearGradient(0, 0, tl, 0); tg.addColorStop(0, K.back[1]); tg.addColorStop(1, K.fin); t2.fillStyle = tg; t2.fill();
    t2.save(); t2.clip(); t2.strokeStyle = 'rgba(255,255,255,.22)'; t2.lineWidth = Math.max(0.5, L * 0.004);
    for (var r = -5; r <= 5; r++) { t2.beginPath(); t2.moveTo(0, tcy + r * Hh * 0.03); t2.lineTo(tl, tcy + r * th * 0.09); t2.stroke(); }
    t2.restore();
    return { body: body, tail: tail, L: L, cy: cy, tcy: tcy, tl: tl };
  }
  // draws one fish centred on (x, y), heading right (dir=1) or left (dir=-1), tail beat phase `ph`, rolled by `roll`
  function drawFish(ctx, sp, x, y, dir, ph, roll, alpha, sc) {
    ctx.save(); ctx.translate(x, y); ctx.scale(dir * (sc || 1), sc || 1); ctx.rotate(roll); ctx.globalAlpha = alpha;
    var sw = Math.sin(ph) * 0.42, bx = -sp.L * 0.5, by = -sp.cy;
    ctx.drawImage(sp.body, bx, by);
    ctx.save(); ctx.translate(bx + 1, 0); ctx.scale(-1, 1); ctx.rotate(sw);   // the tail trails behind the tail-stock and swings about it
    ctx.drawImage(sp.tail, 0, -sp.tcy); ctx.restore();
    ctx.restore();
  }

  function abyssSetup(st) {
    var W = st.W, H = st.H, d = st.dpr, r = coverRect(st.img.naturalWidth, st.img.naturalHeight, W, H, scenePos());
    var base = document.createElement('canvas'); base.width = W; base.height = H;
    var b = base.getContext('2d'); b.imageSmoothingQuality = 'high'; b.drawImage(st.img, r.x, r.y, r.w, r.h);
    st.base = base; st.rect = r;
    st.src = { x: r.x + r.w * 0.50, y: r.y - r.h * 0.04 };                               // where the light enters at the surface
    st.rays = []; var nr = st.small ? 6 : 10;
    for (var i = 0; i < nr; i++) st.rays.push({ ang: rnd(-0.62, 0.62), w: rnd(0.035, 0.11), ph: rnd(0, 6.28), sp: rnd(0.05, 0.14), a: rnd(0.35, 0.9), sw: rnd(0.012, 0.035) });
    st.caus = causticTexture(st.small ? 128 : 192);
    st.cw = Math.max(64, Math.round(W / 2)); st.ch = Math.max(48, Math.round(H * 0.62 / 2));
    st.cc = document.createElement('canvas'); st.cc.width = st.cw; st.cc.height = st.ch; st.cctx = st.cc.getContext('2d');
    var fade = document.createElement('canvas'); fade.width = 4; fade.height = st.ch; var fg = fade.getContext('2d');
    var gr = fg.createLinearGradient(0, 0, 0, st.ch); gr.addColorStop(0, 'rgba(0,0,0,1)'); gr.addColorStop(0.55, 'rgba(0,0,0,0.45)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
    fg.fillStyle = gr; fg.fillRect(0, 0, 4, st.ch); st.cfade = fade;
    // the painted whale: region (picture fractions) that gets the swimming wave, plus a soft mask to blend it back in
    var rx = Math.round(r.x + r.w * 0.14), ry = Math.round(r.y + r.h * 0.33), rw = Math.round(r.w * 0.66), rh = Math.round(r.h * 0.30), pad = Math.round(16 * d);
    st.sw = { x: rx, y: ry, w: rw, h: rh, pad: pad };
    st.swc = document.createElement('canvas'); st.swc.width = rw; st.swc.height = rh + 2 * pad; st.swx = st.swc.getContext('2d');
    var mk = document.createElement('canvas'); mk.width = rw; mk.height = rh + 2 * pad; var mg = mk.getContext('2d');
    var rad = mg.createRadialGradient(rw / 2, mk.height / 2, 0, rw / 2, mk.height / 2, rw / 2);
    rad.addColorStop(0, 'rgba(0,0,0,1)'); rad.addColorStop(0.62, 'rgba(0,0,0,1)'); rad.addColorStop(1, 'rgba(0,0,0,0)');
    mg.setTransform(1, 0, 0, mk.height / rw, 0, 0); mg.fillStyle = rad; mg.fillRect(0, 0, rw, rw);   // ellipse feather
    st.swm = mk;
    st.motes = []; var nm = st.small ? 46 : 110;
    for (var m = 0; m < nm; m++) st.motes.push({ x: rnd(0, W), y: rnd(0, H), z: rnd(0.3, 1), ph: rnd(0, 6.28), tw: rnd(0.3, 1.1) });
    st.whale = { spr: whaleSprite(Math.round(W * 0.30), Math.round(W * 0.30 * 0.34)), x: 0, dir: Math.random() < 0.5 ? 1 : -1, wait: 1.5, active: false, y: 0.74, speed: 0 };
    var fs = st.small ? 0.8 : 1;
    st.fishSp = { sardine: fishSprite('sardine', 30 * d * fs), mackerel: fishSprite('mackerel', 40 * d * fs), lantern: fishSprite('lantern', 24 * d * fs), tuna: fishSprite('tuna', Math.max(90 * d, W * 0.085)) };
    st.schools = []; var kinds = ['sardine', 'mackerel', 'lantern'], ns = st.small ? 2 : 3;
    for (var s = 0; s < ns; s++) {
      var kind = kinds[s % kinds.length];
      var sc = { kind: kind, x: rnd(0, W), y: H * rnd(0.55, 0.9), dir: Math.random() < 0.5 ? 1 : -1, v: rnd(18, 34) * d, ph: rnd(0, 6.28), fish: [] };
      for (var f = 0; f < (st.small ? 7 : 13); f++) sc.fish.push({ ox: rnd(-80, 80) * d, oy: rnd(-30, 30) * d, ph: rnd(0, 6.28), sc: rnd(0.75, 1.2), beat: rnd(7, 10), z: rnd(0.6, 1) });
      st.schools.push(sc);
    }
    st.tuna = { active: false, wait: rnd(12, 25), x: 0, y: 0.5, dir: 1, v: 0, ph: 0 };
  }
  function abyssFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, t = st.t, d = st.dpr, light = st.light;
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.drawImage(st.base, 0, 0);
    // swimming: slices of the painted whale shifted by a travelling wave (larger toward the tail, on the left), blended back softly
    var Q = st.sw, sx = st.swx, NS = st.small ? 22 : 40, sl = Q.w / NS;
    sx.globalCompositeOperation = 'source-over'; sx.clearRect(0, 0, Q.w, Q.h + 2 * Q.pad);
    for (var kq = 0; kq < NS; kq++) {
      var uq = 1 - (kq + 0.5) / NS;                                         // 0 at the head (right) -> 1 at the tail (left)
      var wave = Math.sin(6.2832 * (uq * 0.75 - t * 0.13)) * Math.pow(uq, 1.6) * st.rect.h * 0.0105 + Math.sin(t * 0.09) * uq * st.rect.h * 0.003;
      sx.drawImage(st.base, Q.x + kq * sl, Q.y, sl + 1, Q.h, kq * sl, Q.pad + wave, sl + 1, Q.h);
    }
    sx.globalCompositeOperation = 'destination-in'; sx.drawImage(st.swm, 0, 0);
    ctx.drawImage(st.swc, Q.x, Q.y - Q.pad);
    // caustics (half-res, masked to fade with depth), two copies drifting opposite ways
    var cc = st.cctx, cw = st.cw, ch = st.ch, tw = st.caus.width * (st.small ? 1.1 : 0.9);
    cc.globalCompositeOperation = 'source-over'; cc.clearRect(0, 0, cw, ch);
    cc.globalCompositeOperation = 'lighter';
    for (var layer = 0; layer < 2; layer++) {
      var sc = layer ? 1.35 : 1, sz = tw * sc, ox = ((t * (layer ? -9 : 7) * 0.5) % sz + sz) % sz, oy = ((t * (layer ? 5 : -4) * 0.5) % sz + sz) % sz;
      cc.globalAlpha = (layer ? 0.55 : 0.8) * (0.8 + 0.2 * Math.sin(t * 0.4 + layer));
      for (var yy = -oy; yy < ch; yy += sz) for (var xx = -ox; xx < cw; xx += sz) cc.drawImage(st.caus, xx, yy, sz, sz);
    }
    cc.globalCompositeOperation = 'destination-in'; cc.globalAlpha = 1; cc.drawImage(st.cfade, 0, 0, cw, ch);
    ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = light ? 0.07 : 0.13; ctx.drawImage(st.cc, 0, 0, W, H * 0.62);
    // light shafts from the surface
    var S = st.src, len = H * 1.25;
    for (var i = 0; i < st.rays.length; i++) {
      var R = st.rays[i], a = R.ang + Math.sin(t * R.sp + R.ph) * R.sw, wdt = R.w * (0.85 + 0.15 * Math.sin(t * R.sp * 1.7 + R.ph));
      var breathe = 0.55 + 0.45 * Math.sin(t * R.sp * 2.3 + R.ph * 1.9);
      var ex = S.x + Math.sin(a) * len, ey = S.y + Math.cos(a) * len, nx = Math.cos(a), ny = -Math.sin(a);
      var half = len * wdt * 0.5;
      var gr = ctx.createLinearGradient(S.x, S.y, ex, ey);
      gr.addColorStop(0, 'rgba(170,255,240,' + (0.28 * R.a * breathe) + ')'); gr.addColorStop(0.55, 'rgba(90,220,215,' + (0.10 * R.a * breathe) + ')'); gr.addColorStop(1, 'rgba(60,190,200,0)');
      ctx.globalAlpha = light ? 0.55 : 1; ctx.fillStyle = gr;
      ctx.beginPath(); ctx.moveTo(S.x - nx * 6 * d, S.y - ny * 6 * d); ctx.lineTo(S.x + nx * 6 * d, S.y + ny * 6 * d);
      ctx.lineTo(ex + nx * half, ey + ny * half); ctx.lineTo(ex - nx * half, ey - ny * half); ctx.closePath(); ctx.fill();
    }
    // marine snow, rising
    ctx.fillStyle = '#d6f6ff';
    for (var m = 0; m < st.motes.length; m++) {
      var P = st.motes[m];
      P.y -= (5 + 17 * P.z) * d * dt; P.x += Math.sin(t * 0.35 * P.tw + P.ph) * 3.5 * d * dt;
      if (P.y < -6 * d) { P.y = H + 6 * d; P.x = rnd(0, W); }
      ctx.globalAlpha = (0.18 + 0.5 * P.z) * (0.55 + 0.45 * Math.sin(t * P.tw + P.ph)) * (light ? 0.7 : 1);
      var rr = (0.5 + 1.9 * P.z) * d; ctx.beginPath(); ctx.arc(P.x, P.y, rr, 0, 6.2832); ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
    // fish schools gliding: each fish is a painted species that beats its tail and rolls gently; they bunch, spread and bob as a school
    for (var s = 0; s < st.schools.length; s++) {
      var SC = st.schools[s], sp = st.fishSp[SC.kind]; SC.x += SC.dir * SC.v * dt;
      var cy = SC.y + Math.sin(t * 0.15 + SC.ph) * H * 0.045;
      if ((SC.dir > 0 && SC.x > W + 140 * d) || (SC.dir < 0 && SC.x < -140 * d)) { SC.dir = -SC.dir; SC.y = H * rnd(0.52, 0.9); SC.v = rnd(18, 34) * d; SC.x = SC.dir > 0 ? -120 * d : W + 120 * d; }
      var heading = Math.atan2(Math.cos(t * 0.15 + SC.ph) * H * 0.045 * 0.15, SC.v) * SC.dir;
      var spread = 1 + 0.35 * Math.sin(t * 0.21 + SC.ph * 2);
      for (var f = 0; f < SC.fish.length; f++) {
        var F = SC.fish[f], fx = SC.x + F.ox * spread + Math.sin(t * 0.7 + F.ph) * 10 * d, fy = cy + F.oy * spread + Math.cos(t * 0.6 + F.ph) * 7 * d;
        var glint = 0.5 + 0.5 * Math.sin(t * 0.9 + F.ph * 2);
        var al = (light ? 0.42 : 0.62) * F.z + 0.28 * glint * glint * F.z;
        drawFish(ctx, sp, fx, fy, SC.dir, t * F.beat + F.ph, Math.sin(t * 0.8 + F.ph) * 0.05 + heading, Math.min(0.95, al), F.sc);
      }
    }
    // now and then one big silver-blue tuna cruises through, well in front of the schools
    var TU = st.tuna, tsp = st.fishSp.tuna;
    if (!TU.active) {
      TU.wait -= dt;
      if (TU.wait <= 0) { TU.active = true; TU.dir = Math.random() < 0.5 ? 1 : -1; TU.y = rnd(0.42, 0.80); TU.v = (W + 2 * tsp.L) / rnd(45, 70); TU.x = TU.dir > 0 ? -tsp.L : W + tsp.L; TU.ph = rnd(0, 6.28); }
    } else {
      TU.x += TU.dir * TU.v * dt;
      if ((TU.dir > 0 && TU.x > W + tsp.L) || (TU.dir < 0 && TU.x < -tsp.L)) { TU.active = false; TU.wait = rnd(40, 90); }
      else {
        var ty = H * TU.y + Math.sin(t * 0.35 + TU.ph) * H * 0.02;
        drawFish(ctx, tsp, TU.x, ty, TU.dir, t * 4.2 + TU.ph, Math.sin(t * 0.35 + TU.ph) * 0.03, light ? 0.55 : 0.8, 1);
      }
    }
    ctx.globalAlpha = 1;
    // the whale: a long slow crossing, then a long quiet gap
    var Wh = st.whale, spr = Wh.spr, wl = spr.width;
    if (!Wh.active) {
      Wh.wait -= dt;
      if (Wh.wait <= 0) { Wh.active = true; Wh.dir = -Wh.dir; Wh.speed = (W + wl) / rnd(70, 95); Wh.x = Wh.dir > 0 ? -wl : W; Wh.y = rnd(0.26, 0.36); }
    } else {
      Wh.x += Wh.dir * Wh.speed * dt;
      if ((Wh.dir > 0 && Wh.x > W) || (Wh.dir < 0 && Wh.x < -wl)) { Wh.active = false; Wh.wait = rnd(6, 14); }
      var fadeIn = Math.min(smooth(-wl, wl * 0.2, Wh.x), smooth(W, W - wl * 1.2, Wh.x));
      var alpha = (light ? 0.55 : 0.78) * (Wh.dir > 0 ? smooth(-wl, wl * 0.3, Wh.x) * smooth(W + wl * 0.1, W - wl, Wh.x) : smooth(W, W - wl * 0.3, Wh.x) * smooth(-wl * 0.9, wl * 0.1, Wh.x));
      if (!st.whaleTmp) { st.whaleTmp = document.createElement('canvas'); st.whaleTmp.width = wl; st.whaleTmp.height = Math.round(spr.height * 1.5); }
      var wt = st.whaleTmp, wx = wt.getContext('2d'), N = 36, sw = wl / N, off = Math.round(spr.height * 0.25);
      wx.clearRect(0, 0, wt.width, wt.height);                 // slices are composed at full strength, THEN faded as one image (no seams)
      for (var k = 0; k < N; k++) {
        var u = k / N, bend = Math.sin(6.2832 * (u * 0.8 - t * 0.16)) * Math.pow(u, 1.5) * spr.height * 0.10 + Math.sin(t * 0.12) * spr.height * 0.03 * u;
        wx.drawImage(spr, k * sw, 0, sw + 1, spr.height, k * sw, off + bend, sw + 1, spr.height);
      }
      var baseY = H * Wh.y - spr.height * 0.5 - off;
      ctx.globalAlpha = Math.max(0, alpha) * fadeIn; ctx.save();
      if (Wh.dir < 0) { ctx.translate(Wh.x + wl, 0); ctx.scale(-1, 1); ctx.translate(-Wh.x, 0); }
      ctx.drawImage(wt, Wh.x, baseY);
      ctx.restore(); ctx.globalAlpha = 1;
    }
  }

  /* -------------------------------------------------------------- 2026-10-30 scenes: Nordic Kingdom, Endless Sands, Coastal Italy, Bioluminescence */
  function baseSetup(st) {
    var W = st.W, H = st.H, r = coverRect(st.img.naturalWidth, st.img.naturalHeight, W, H, scenePos());
    var b = document.createElement('canvas'); b.width = W; b.height = H;
    var g = b.getContext('2d'); g.imageSmoothingQuality = 'high'; g.drawImage(st.img, r.x, r.y, r.w, r.h);
    st.base = b; st.rect = r; st.blobs = {};
  }
  function PT(st, u, v) { return { x: st.rect.x + u * st.rect.w, y: st.rect.y + v * st.rect.h }; }
  function blobSprite(st, key, rgb) {
    if (st.blobs[key]) return st.blobs[key];
    var s = 128, c = document.createElement('canvas'); c.width = c.height = s; var g = c.getContext('2d');
    var gr = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    gr.addColorStop(0, 'rgba(' + rgb + ',1)'); gr.addColorStop(0.5, 'rgba(' + rgb + ',.42)'); gr.addColorStop(1, 'rgba(' + rgb + ',0)');
    g.fillStyle = gr; g.fillRect(0, 0, s, s); return (st.blobs[key] = c);
  }
  // re-draw a band of the picture in thin strips pushed sideways by travelling waves (water ripple / heat haze / swaying flowers)
  function waveBand(ctx, st, u0, u1, v0, v1, amp, wl, spd, t, ph) {
    var r = st.rect, x0 = Math.max(0, r.x + u0 * r.w), x1 = Math.min(st.W, r.x + u1 * r.w), y0 = Math.max(0, r.y + v0 * r.h), y1 = Math.min(st.H, r.y + v1 * r.h);
    var step = Math.max(2, Math.round(3 * st.dpr));
    if (x1 <= x0 || y1 <= y0) return;
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
    for (var y = y0; y < y1; y += step) {
      var k = (y - y0) / (y1 - y0), fade = smooth(0, 0.14, k) * (1 - smooth(0.88, 1, k));
      var dx = (Math.sin(y / wl + t * spd + (ph || 0)) + 0.5 * Math.sin(y / (wl * 0.43) - t * spd * 1.3)) * amp * fade * (0.5 + k);
      ctx.drawImage(st.base, x0, y, x1 - x0, step + 1, x0 + dx, y, x1 - x0, step + 1);
    }
    ctx.restore();
  }
  function driftBlobs(ctx, st, list, sprite, t, alphaMul) {
    var W = st.W;
    for (var i = 0; i < list.length; i++) {
      var c = list[i], x = (((c.u + t * c.sp) % 1.5) + 1.5) % 1.5 - 0.25, p = PT(st, 0, c.v), sz = c.s * st.rect.w;
      ctx.globalAlpha = c.a * alphaMul * (0.85 + 0.15 * Math.sin(t * 0.07 + c.ph));
      ctx.drawImage(sprite, x * W - sz / 2, p.y - sz * c.h / 2, sz, sz * c.h);
    }
  }
  function makeBlobs(n, v0, v1, s0, s1, sp0, sp1, a0, a1, h) {
    var o = []; for (var i = 0; i < n; i++) o.push({ u: rnd(0, 1.4), v: rnd(v0, v1), s: rnd(s0, s1), sp: rnd(sp0, sp1), a: rnd(a0, a1), ph: rnd(0, 6.28), h: h || 0.45 });
    return o;
  }
  function makeStars(n, v0, v1) { var o = []; for (var i = 0; i < n; i++) o.push({ u: rnd(0, 1), v: rnd(v0, v1), r: rnd(0.5, 1.5), ph: rnd(0, 6.28), sp: rnd(0.5, 2.2), a: rnd(0.35, 1) }); return o; }
  function bolt(x0, y0, x1, y1, rough, out, depth) {
    if (depth === 0) { out.push([x0, y0, x1, y1]); return; }
    var mx = (x0 + x1) / 2 + (Math.random() - 0.5) * rough, my = (y0 + y1) / 2 + (Math.random() - 0.5) * rough * 0.25;
    bolt(x0, y0, mx, my, rough / 2, out, depth - 1); bolt(mx, my, x1, y1, rough / 2, out, depth - 1);
  }
  function strike(st) {
    var W = st.W, H = st.H, d = st.dpr, segs = [], x0 = rnd(0.12, 0.88) * W, y0 = H * rnd(0.0, 0.1), x1 = x0 + rnd(-0.12, 0.12) * W, y1 = PT(st, 0, rnd(0.26, 0.36)).y;
    bolt(x0, y0, x1, y1, W * 0.07, segs, 6);
    var br = [], nb = 2 + ((Math.random() * 3) | 0);
    for (var i = 0; i < nb; i++) {
      var s = segs[(Math.random() * segs.length * 0.7) | 0], dir = Math.random() < 0.5 ? -1 : 1;
      bolt(s[0], s[1], s[0] + dir * rnd(0.04, 0.11) * W, s[1] + rnd(0.08, 0.2) * H, W * 0.04, br, 4);
    }
    return { main: segs, branches: br, t0: st.t, x: x0, d: d };
  }
  function drawStrike(ctx, st, L, k) {
    var d = st.dpr, passes = [[9 * d, 0.10], [4.5 * d, 0.22], [2 * d, 0.55], [1 * d, 1]];
    ctx.globalCompositeOperation = 'lighter'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (var p = 0; p < passes.length; p++) {
      ctx.lineWidth = passes[p][0]; ctx.strokeStyle = (p < 3 ? 'rgba(150,175,255,' : 'rgba(255,255,255,') + (passes[p][1] * k) + ')';
      ctx.beginPath(); for (var i = 0; i < L.main.length; i++) { ctx.moveTo(L.main[i][0], L.main[i][1]); ctx.lineTo(L.main[i][2], L.main[i][3]); } ctx.stroke();
      ctx.lineWidth = passes[p][0] * 0.55; ctx.beginPath();
      for (var j = 0; j < L.branches.length; j++) { ctx.moveTo(L.branches[j][0], L.branches[j][1]); ctx.lineTo(L.branches[j][2], L.branches[j][3]); } ctx.stroke();
    }
  }

  /* Nordic Kingdom: drifting fog, water ripples, flickering village lights, and lightning that strikes in and out */
  function nordicSetup(st) {
    baseSetup(st);
    st.fog = makeBlobs(st.small ? 5 : 8, 0.30, 0.56, 0.5, 0.95, 0.002, 0.006, 0.10, 0.22, 0.35);
    st.lights = []; for (var i = 0; i < (st.small ? 40 : 90); i++) st.lights.push({ u: rnd(0.50, 0.98), v: rnd(0.505, 0.56), ph: rnd(0, 6.28), sp: rnd(0.6, 2.4), a: rnd(0.4, 1) });
    st.nextStrike = st.t + rnd(3, 6); st.L = null;
  }
  function nordicFrame(st) {
    var ctx = st.ctx, W = st.W, H = st.H, t = st.t, d = st.dpr, lm = st.light ? 0.65 : 1;
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.drawImage(st.base, 0, 0);
    waveBand(ctx, st, 0, 1, 0.58, 0.84, 1.3 * d, 11 * d, 0.9, t, 0);
    ctx.globalCompositeOperation = 'lighter';
    var fog = blobSprite(st, 'fog', '170,190,215'); driftBlobs(ctx, st, st.fog, fog, t, lm);
    var warm = blobSprite(st, 'warm', '255,190,100');
    for (var i = 0; i < st.lights.length; i++) {
      var Lg = st.lights[i], p = PT(st, Lg.u, Lg.v), f = 0.55 + 0.3 * Math.sin(t * Lg.sp + Lg.ph) + 0.15 * Math.sin(t * Lg.sp * 3.1 + Lg.ph * 2), a = Lg.a * f * 0.55 * lm, s = 9 * d;
      ctx.globalAlpha = a; ctx.drawImage(warm, p.x - s, p.y - s, s * 2, s * 2);
      var ry = PT(st, 0, 0.51 + (Lg.v - 0.505) * 3.2 + 0.05).y;          // reflection: a wavering streak below the shoreline
      ctx.globalAlpha = a * 0.5; ctx.drawImage(warm, p.x - s * 0.6 + Math.sin(t * 1.4 + Lg.ph) * 2 * d, ry, s * 1.2, s * 2.6);
    }
    // lightning: a calm flicker every so often, striking in and out of the sky
    if (!st.L && t >= st.nextStrike) st.L = strike(st);
    if (st.L) {
      var e = t - st.L.t0, k = Math.max(Math.exp(-e * 4.5), e > 0.22 ? 0.7 * Math.exp(-(e - 0.22) * 5.5) : 0, e > 0.5 ? 0.35 * Math.exp(-(e - 0.5) * 7) : 0);
      if (e > 1.6) { st.L = null; st.nextStrike = t + rnd(7, 16); }
      else {
        var cl = ctx.createLinearGradient(0, 0, 0, H * 0.6);
        cl.addColorStop(0, 'rgba(140,165,235,' + (0.30 * k * lm) + ')'); cl.addColorStop(1, 'rgba(90,110,170,0)');
        ctx.globalAlpha = 1; ctx.fillStyle = cl; ctx.fillRect(0, 0, W, H * 0.6);
        if (e < 0.9) drawStrike(ctx, st, st.L, Math.min(1, k * 1.2) * lm);
        ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = 0.10 * k * lm; ctx.fillStyle = '#9fb4ff';
        ctx.fillRect(0, PT(st, 0, 0.58).y, W, H * 0.3);                  // a hint of the flash on the water
      }
    }
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  }

  /* Endless Sands: drifting sand, heat haze, moving clouds, and a traveller walking slowly across the dunes */
  function sandsSetup(st) {
    baseSetup(st);
    st.clouds = makeBlobs(st.small ? 4 : 7, 0.04, 0.30, 0.35, 0.7, 0.003, 0.008, 0.10, 0.22, 0.32);
    st.sand = []; for (var i = 0; i < (st.small ? 50 : 120); i++) st.sand.push({ x: rnd(0, st.W), y: rnd(st.H * 0.45, st.H), z: rnd(0.3, 1), ph: rnd(0, 6.28), l: rnd(5, 14) });
    st.walk = { u: 0.12, pause: 0 };
  }
  function drawWalker(ctx, st, t, x, y, h, alpha) {
    var ph = t * 3.0, sw = Math.sin(ph), bob = Math.abs(Math.cos(ph)) * h * 0.025, lw = Math.max(1, h * 0.07);
    ctx.save(); ctx.globalAlpha = alpha; ctx.lineCap = 'round';
    ctx.fillStyle = 'rgba(40,18,6,0.35)'; ctx.beginPath(); ctx.ellipse(x - h * 0.55, y + h * 0.02, h * 0.62, h * 0.06, 0, 0, 6.2832); ctx.fill();   // long low shadow
    ctx.strokeStyle = '#1b0e07'; ctx.fillStyle = '#1b0e07'; ctx.lineWidth = lw;
    var hip = y - h * 0.46 - bob, sh = hip - h * 0.34;
    ctx.beginPath(); ctx.moveTo(x, hip); ctx.lineTo(x + sw * h * 0.17, y); ctx.moveTo(x, hip); ctx.lineTo(x - sw * h * 0.17, y); ctx.stroke();   // legs
    ctx.lineWidth = lw * 1.5; ctx.beginPath(); ctx.moveTo(x, hip); ctx.lineTo(x + h * 0.03, sh); ctx.stroke();                                         // torso
    ctx.lineWidth = lw * 0.8; ctx.beginPath(); ctx.moveTo(x + h * 0.03, sh + h * 0.03); ctx.lineTo(x + h * 0.03 - sw * h * 0.12, sh + h * 0.22);
    ctx.moveTo(x + h * 0.03, sh + h * 0.03); ctx.lineTo(x + h * 0.03 + sw * h * 0.12, sh + h * 0.22); ctx.stroke();                                    // arms
    ctx.beginPath(); ctx.arc(x + h * 0.04, sh - h * 0.07, h * 0.075, 0, 6.2832); ctx.fill();                                                         // head
    ctx.restore();
  }
  function sandsFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, t = st.t, d = st.dpr, lm = st.light ? 0.7 : 1;
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.drawImage(st.base, 0, 0);
    waveBand(ctx, st, 0, 1, 0.34, 0.52, 1.7 * d, 8 * d, 1.5, t, 0);                // heat haze over the far dunes
    ctx.globalCompositeOperation = 'lighter';
    driftBlobs(ctx, st, st.clouds, blobSprite(st, 'cl', '255,170,90'), t, lm);
    var wk = st.walk;
    if (wk.pause > 0) wk.pause -= dt; else { wk.u += dt / 150; if (wk.u > 0.58) { wk.u = 0.12; wk.pause = 14; } }
    var p = PT(st, wk.u, 0.585 - (wk.u - 0.12) * 0.16 + Math.sin(wk.u * 25) * 0.003), h = st.rect.h * 0.052 * (1 - (wk.u - 0.12) * 0.25);
    var edge = smooth(0.12, 0.17, wk.u) * (1 - smooth(0.53, 0.58, wk.u)), moving = wk.pause <= 0;
    ctx.globalCompositeOperation = 'source-over';
    if (moving) drawWalker(ctx, st, t, p.x, p.y, h, 0.92 * edge);
    ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = '#f2cf96';
    for (var i = 0; i < st.sand.length; i++) {                                      // wind-blown sand skimming the dunes
      var S = st.sand[i]; S.x -= (22 + 60 * S.z) * d * dt; S.y += Math.sin(t * 0.8 + S.ph) * 6 * d * dt;
      if (S.x < -20 * d) { S.x = W + 20 * d; S.y = rnd(H * 0.45, H); }
      ctx.globalAlpha = (0.12 + 0.3 * S.z) * (0.6 + 0.4 * Math.sin(t * 1.3 + S.ph)) * lm;
      ctx.fillRect(S.x, S.y, S.l * d * (0.5 + S.z), Math.max(1, 1.1 * d * S.z));
    }
    ctx.globalAlpha = 1;
  }

  /* Coastal Italy: gentle waves, swaying flowers, drifting clouds, sunlight shimmer and boats gliding */
  function italySetup(st) {
    baseSetup(st);
    st.clouds = makeBlobs(st.small ? 4 : 7, 0.02, 0.26, 0.3, 0.6, 0.004, 0.01, 0.16, 0.34, 0.36);
    st.glint = []; for (var i = 0; i < (st.small ? 60 : 150); i++) st.glint.push({ u: rnd(0.32, 0.98), v: rnd(0.56, 0.92), ph: rnd(0, 6.28), sp: rnd(0.8, 3), s: rnd(1, 2.6) });
    st.boats = []; for (var b = 0; b < 4; b++) st.boats.push({ u: rnd(0.35, 0.95), v: rnd(0.63, 0.86), dir: Math.random() < 0.5 ? 1 : -1, sp: rnd(0.0012, 0.0026), s: rnd(0.9, 1.5) });
  }
  function drawBoat(ctx, st, B, t) {
    var p = PT(st, B.u, B.v), d = st.dpr * B.s, x = p.x, y = p.y + Math.sin(t * 1.1 + B.u * 40) * 0.8 * d, f = B.dir;
    ctx.save(); ctx.translate(x, y); ctx.scale(f, 1);
    ctx.globalAlpha = 0.5; ctx.fillStyle = 'rgba(255,255,255,0.8)';                // wake
    ctx.beginPath(); ctx.moveTo(-5 * d, 1.5 * d); ctx.lineTo(-30 * d, -1 * d); ctx.lineTo(-30 * d, 4 * d); ctx.closePath(); ctx.fill();
    ctx.globalAlpha = 1; ctx.fillStyle = '#f5f1e8'; ctx.beginPath(); ctx.moveTo(-9 * d, 0); ctx.lineTo(10 * d, 0); ctx.lineTo(7 * d, 3.2 * d); ctx.lineTo(-7 * d, 3.2 * d); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#d9dde3'; ctx.fillRect(-3 * d, -3 * d, 6 * d, 3 * d); ctx.fillStyle = '#2a6f9e'; ctx.fillRect(-9 * d, 0.2 * d, 19 * d, 0.9 * d);
    ctx.restore();
  }
  function italyFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, t = st.t, d = st.dpr, lm = st.light ? 0.7 : 1;
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.drawImage(st.base, 0, 0);
    waveBand(ctx, st, 0.30, 1, 0.52, 0.86, 1.6 * d, 12 * d, 1.0, t, 0);            // sea swell
    waveBand(ctx, st, 0.74, 1, 0.60, 1, 3.2 * d, 38 * d, 1.1, t, 1.2);             // blossoms and leaves swaying in the breeze
    for (var b = 0; b < st.boats.length; b++) {
      var B = st.boats[b]; B.u += B.dir * B.sp * dt; if (B.u > 1.02 || B.u < 0.30) { B.dir = -B.dir; B.v = rnd(0.63, 0.86); }
      drawBoat(ctx, st, B, t);
    }
    ctx.globalCompositeOperation = 'lighter';
    driftBlobs(ctx, st, st.clouds, blobSprite(st, 'cl', '255,255,255'), t, lm * 0.8);
    ctx.fillStyle = '#fff6d8';
    for (var i = 0; i < st.glint.length; i++) {                                     // sunlight shimmer on the water
      var G = st.glint[i], p = PT(st, G.u, G.v), a = Math.pow(Math.max(0, Math.sin(t * G.sp + G.ph)), 6);
      if (a < 0.02) continue; ctx.globalAlpha = a * 0.75 * lm; ctx.fillRect(p.x, p.y, G.s * d * 3, G.s * d * 0.9);
    }
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  }

  /* Bioluminescence: pulsing shoreline glow, rolling waves, twinkling stars, drifting glowing particles */
  var BIO_SHORE = [[-0.02, 0.60], [0.22, 0.69], [0.45, 0.80], [0.68, 0.90], [0.86, 0.97], [1.02, 1.0]];
  function bioSetup(st) {
    baseSetup(st); st.stars = makeStars(st.small ? 70 : 170, 0.0, 0.46);
    st.motes = []; for (var i = 0; i < (st.small ? 24 : 54); i++) st.motes.push({ u: rnd(0, 1), v: rnd(0.6, 1), ph: rnd(0, 6.28), sp: rnd(0.3, 0.9), z: rnd(0.4, 1) });
  }
  function bioFrame(st, dt) {
    var ctx = st.ctx, W = st.W, H = st.H, t = st.t, d = st.dpr, lm = st.light ? 0.7 : 1;
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; ctx.drawImage(st.base, 0, 0);
    waveBand(ctx, st, 0, 1, 0.58, 1, 2.2 * d, 14 * d, 0.8, t, 0);                  // waves rolling in
    ctx.globalCompositeOperation = 'lighter'; ctx.fillStyle = '#e9f1ff';
    for (var i = 0; i < st.stars.length; i++) {
      var S = st.stars[i], p = PT(st, S.u, S.v), a = S.a * (0.45 + 0.55 * Math.sin(t * S.sp + S.ph)) * lm;
      ctx.globalAlpha = Math.max(0, a) * 0.85; ctx.beginPath(); ctx.arc(p.x, p.y, S.r * d, 0, 6.2832); ctx.fill();
    }
    var pulse = 0.55 + 0.45 * Math.sin(t * 0.9) * Math.sin(t * 0.37 + 1);           // slow breathing glow along the surf line
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (var pass = 0; pass < 3; pass++) {
      ctx.lineWidth = [26, 12, 4][pass] * d; ctx.strokeStyle = 'rgba(' + ['30,110,255', '60,170,255', '170,235,255'][pass] + ',' + ([0.10, 0.16, 0.26][pass] * pulse * lm) + ')';
      ctx.beginPath(); for (var k = 0; k < BIO_SHORE.length; k++) { var q = PT(st, BIO_SHORE[k][0], BIO_SHORE[k][1] + Math.sin(t * 0.6 + k) * 0.004); k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); }
      ctx.stroke();
    }
    var glow = blobSprite(st, 'bio', '90,200,255');
    for (var m = 0; m < st.motes.length; m++) {
      var M = st.motes[m]; M.v -= 0.006 * M.sp * dt; M.u += Math.sin(t * 0.4 + M.ph) * 0.0006; if (M.v < 0.55) { M.v = rnd(0.9, 1); M.u = rnd(0, 1); }
      var mp = PT(st, M.u, M.v), s = (5 + 7 * M.z) * d; ctx.globalAlpha = (0.2 + 0.5 * M.z) * (0.5 + 0.5 * Math.sin(t * 1.3 * M.sp + M.ph)) * lm;
      ctx.drawImage(glow, mp.x - s, mp.y - s, s * 2, s * 2);
    }
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  }

  /* -------------------------------------------------------------- public API */
  window.T58SceneCanvas = {
    mount: function (kind, layer) {
      try {
        if (!document.createElement('canvas').getContext) return null;
        if (kind === 'ocean') return makeStage('ocean', layer, oceanFrame, oceanSetup);
        if (kind === 'forest') return makeStage('forest', layer, forestFrame, function (st) { st.keepMask = null; forestSetup(st); });
        if (kind === 'gold') return makeStage('gold', layer, goldFrame, goldSetup, { gl: true });
        if (kind === 'sakura') return makeStage('sakura', layer, sakuraFrame, sakuraSetup);
        if (kind === 'abyss') return makeStage('abyss', layer, abyssFrame, abyssSetup);
        if (kind === 'nordic') return makeStage('nordic', layer, nordicFrame, nordicSetup);
        if (kind === 'sands') return makeStage('sands', layer, sandsFrame, sandsSetup);
        if (kind === 'italy') return makeStage('italy', layer, italyFrame, italySetup);
        if (kind === 'biolum') return makeStage('biolum', layer, bioFrame, bioSetup);
      } catch (e) {}
      return null;
    }
  };
})();
