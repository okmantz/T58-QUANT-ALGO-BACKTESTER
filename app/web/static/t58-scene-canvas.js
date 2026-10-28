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
  function makeStage(kind, layer, drawFrame, setup) {
    var canvas = document.createElement('canvas');
    canvas.className = 'sc-canvas';
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.opacity = '0';
    var scrim = document.createElement('i');
    scrim.className = 'scrim';
    layer.insertBefore(scrim, layer.firstChild);
    layer.insertBefore(canvas, layer.firstChild);
    var ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) { canvas.remove(); scrim.remove(); return null; }

    var st = { canvas: canvas, ctx: ctx, W: 0, H: 0, dpr: 1, img: null, ready: false, raf: 0, last: 0, t: 0, dead: false,
      fps: 30, light: isLight(), small: window.innerWidth < 760 || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2) };

    function imageUrl() {
      return '/static/backgrounds/' + (kind === 'ocean' ? 'midnight-ocean' : 'misty-forest') + (st.light ? '-light' : '-dark') + '.webp';
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
      resize();
      setup(st);
      st.ready = true;
      drawFrame(st, 0.033);
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

  /* -------------------------------------------------------------- public API */
  window.T58SceneCanvas = {
    mount: function (kind, layer) {
      try {
        if (!document.createElement('canvas').getContext) return null;
        if (kind === 'ocean') return makeStage('ocean', layer, oceanFrame, oceanSetup);
        if (kind === 'forest') return makeStage('forest', layer, forestFrame, function (st) { st.keepMask = null; forestSetup(st); });
      } catch (e) {}
      return null;
    }
  };
})();
