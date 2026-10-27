/* Personalize: seasonal page backdrop (+ quiet falling motion), accent colour, optional Bible verse bar.
 * Per-browser preferences live in localStorage (t58-bg, t58-fx, t58-accent, t58-accent-hex, t58-accent-vars, t58-verse);
 * everything degrades to the default mountains / teal / no verse bar if storage is unavailable.
 *
 * Verses: the reference list below is the source of truth. When /static/esv-verses.json exists (built once by
 * scripts/fetch_esv_verses.py from the official ESV API) the bar shows the ESV text with Crossway's required notice;
 * until then it falls back to the public-domain KJV text below and labels it (KJV) so nothing is ever mislabelled.
 * The verse is chosen from the clock (30-minute slots), so it is the same on every page and tab without server state.
 *
 * Motion (snow / leaves / petals / fireflies): a handful of CSS-animated <i> elements in one fixed, pointer-events:none
 * layer. Only transform + opacity animate (compositor-only: no layout, no paint, no JS per frame). It is skipped for
 * prefers-reduced-motion, paused while the tab is hidden, thinned on small / low-core devices, and can be switched off
 * in Account settings -> Personalize. */
(function () {
  'use strict';
  var SEASON_BGS = ['spring', 'summer', 'autumn', 'winter'];
  // Animated scenes: name -> CSS class of its motion layer (see "Scene motion" in theme.css)
  var SCENES = { 'cosmic-drift': 'cosmic', 'obsidian': 'obsidian', 'midnight-ocean': 'ocean', 'rainfall-district': 'rain', 'silent-sanctuary': 'sanctuary', 'misty-forest': 'forest' };
  var BGS = ['mountains'].concat(SEASON_BGS, Object.keys(SCENES));
  var SLOT_MS = 30 * 60 * 1000;
  var VERSES = [["Psalm 23:1", "The LORD is my shepherd; I shall not want."], ["Proverbs 3:5", "Trust in the LORD with all thine heart; and lean not unto thine own understanding."], ["Philippians 4:13", "I can do all things through Christ which strengtheneth me."], ["Joshua 1:9", "Have not I commanded thee? Be strong and of a good courage; be not afraid, neither be thou dismayed: for the LORD thy God is with thee whithersoever thou goest."], ["Isaiah 41:10", "Fear thou not; for I am with thee: be not dismayed; for I am thy God: I will strengthen thee; yea, I will help thee; yea, I will uphold thee with the right hand of my righteousness."], ["Jeremiah 29:11", "For I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end."], ["Romans 8:28", "And we know that all things work together for good to them that love God, to them who are the called according to his purpose."], ["Psalm 46:1", "God is our refuge and strength, a very present help in trouble."], ["Proverbs 16:3", "Commit thy works unto the LORD, and thy thoughts shall be established."], ["Matthew 6:33", "But seek ye first the kingdom of God, and his righteousness; and all these things shall be added unto you."], ["Psalm 37:5", "Commit thy way unto the LORD; trust also in him; and he shall bring it to pass."], ["Proverbs 21:5", "The thoughts of the diligent tend only to plenteousness; but of every one that is hasty only to want."], ["Colossians 3:23", "And whatsoever ye do, do it heartily, as to the Lord, and not unto men."], ["Proverbs 22:3", "A prudent man foreseeth the evil, and hideth himself: but the simple pass on, and are punished."], ["Psalm 118:24", "This is the day which the LORD hath made; we will rejoice and be glad in it."], ["Isaiah 40:31", "But they that wait upon the LORD shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary; and they shall walk, and not faint."], ["Galatians 6:9", "And let us not be weary in well doing: for in due season we shall reap, if we faint not."], ["2 Timothy 1:7", "For God hath not given us the spirit of fear; but of power, and of love, and of a sound mind."], ["Psalm 121:1-2", "I will lift up mine eyes unto the hills, from whence cometh my help. My help cometh from the LORD, which made heaven and earth."], ["Proverbs 27:17", "Iron sharpeneth iron; so a man sharpeneth the countenance of his friend."], ["Matthew 11:28", "Come unto me, all ye that labour and are heavy laden, and I will give you rest."], ["John 14:27", "Peace I leave with you, my peace I give unto you: not as the world giveth, give I unto you. Let not your heart be troubled, neither let it be afraid."], ["Psalm 27:1", "The LORD is my light and my salvation; whom shall I fear? the LORD is the strength of my life; of whom shall I be afraid?"], ["Proverbs 10:4", "He becometh poor that dealeth with a slack hand: but the hand of the diligent maketh rich."], ["Lamentations 3:22-23", "It is of the LORD's mercies that we are not consumed, because his compassions fail not. They are new every morning: great is thy faithfulness."], ["Psalm 19:14", "Let the words of my mouth, and the meditation of my heart, be acceptable in thy sight, O LORD, my strength, and my redeemer."], ["Romans 12:12", "Rejoicing in hope; patient in tribulation; continuing instant in prayer."], ["James 1:5", "If any of you lack wisdom, let him ask of God, that giveth to all men liberally, and upbraideth not; and it shall be given him."], ["Proverbs 4:7", "Wisdom is the principal thing; therefore get wisdom: and with all thy getting get understanding."], ["Psalm 34:8", "O taste and see that the LORD is good: blessed is the man that trusteth in him."], ["Hebrews 11:1", "Now faith is the substance of things hoped for, the evidence of things not seen."], ["1 Corinthians 16:13", "Watch ye, stand fast in the faith, quit you like men, be strong."], ["Psalm 56:3", "What time I am afraid, I will trust in thee."], ["Proverbs 15:22", "Without counsel purposes are disappointed: but in the multitude of counsellors they are established."], ["Numbers 6:24-26", "The LORD bless thee, and keep thee: The LORD make his face shine upon thee, and be gracious unto thee: The LORD lift up his countenance upon thee, and give thee peace."], ["Psalm 90:17", "And let the beauty of the LORD our God be upon us: and establish thou the work of our hands upon us; yea, the work of our hands establish thou it."], ["Micah 6:8", "He hath shewed thee, O man, what is good; and what doth the LORD require of thee, but to do justly, and to love mercy, and to walk humbly with thy God?"], ["Proverbs 11:3", "The integrity of the upright shall guide them: but the perverseness of transgressors shall destroy them."], ["Romans 15:13", "Now the God of hope fill you with all joy and peace in believing, that ye may abound in hope, through the power of the Holy Ghost."], ["Psalm 32:8", "I will instruct thee and teach thee in the way which thou shalt go: I will guide thee with mine eye."], ["Deuteronomy 31:6", "Be strong and of a good courage, fear not, nor be afraid of them: for the LORD thy God, he it is that doth go with thee; he will not fail thee, nor forsake thee."], ["Proverbs 14:23", "In all labour there is profit: but the talk of the lips tendeth only to penury."], ["Psalm 62:8", "Trust in him at all times; ye people, pour out your heart before him: God is a refuge for us."], ["1 Peter 5:7", "Casting all your care upon him; for he careth for you."], ["Philippians 4:6", "Be careful for nothing; but in every thing by prayer and supplication with thanksgiving let your requests be made known unto God."], ["Proverbs 13:11", "Wealth gotten by vanity shall be diminished: but he that gathereth by labour shall increase."], ["Isaiah 26:3", "Thou wilt keep him in perfect peace, whose mind is stayed on thee: because he trusteth in thee."], ["Psalm 16:8", "I have set the LORD always before me: because he is at my right hand, I shall not be moved."]];

  function get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function bg() { var v = get('t58-bg'); return BGS.indexOf(v) >= 0 ? v : 'mountains'; }
  function verseOn() { return get('t58-verse') === '1'; }
  function fxOn() { return get('t58-fx') !== '0'; }
  function isLight() { return document.documentElement.getAttribute('data-theme') === 'light'; }
  function applyBg() { document.documentElement.setAttribute('data-bg', bg()); }

  // Ease the new scene in (theme.css: html.t58-bg-swap) when the backdrop or the theme changes -- never on a plain page load.
  var swapTimer = 0;
  function swap() {
    var el = document.documentElement;
    el.classList.remove('t58-bg-swap');
    void el.offsetWidth; // restart the animation if it is already running
    el.classList.add('t58-bg-swap');
    clearTimeout(swapTimer);
    swapTimer = setTimeout(function () { el.classList.remove('t58-bg-swap'); }, 1000);
  }

  /* ------------------------------------------------------------------ accent colour */
  var ACCENTS = {
    teal:    { label: 'Teal',          dark: '#35e0b0', light: '#0a6f54' },
    blue:    { label: 'Electric blue', dark: '#4d90ff', light: '#1d5fd6' },
    violet:  { label: 'Violet',        dark: '#9b8cff', light: '#5a4bd1' },
    emerald: { label: 'Emerald',       dark: '#35d878', light: '#107a42' },
    amber:   { label: 'Amber',         dark: '#f5b83d', light: '#9a6208' },
    crimson: { label: 'Crimson',       dark: '#ff5d73', light: '#c02438' }
  };
  var ACCENT_KEYS = Object.keys(ACCENTS).concat(['custom']);

  function normHex(s) {
    s = String(s || '').trim().replace(/^#/, '');
    if (/^[0-9a-f]{3}$/i.test(s)) s = s.replace(/(.)/g, '$1$1');
    return /^[0-9a-f]{6}$/i.test(s) ? '#' + s.toLowerCase() : null;
  }
  function hexToRgb(h) { var n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function rgbToHex(c) { return '#' + c.map(function (v) { v = Math.max(0, Math.min(255, Math.round(v))); return (v < 16 ? '0' : '') + v.toString(16); }).join(''); }
  function rgbToHsl(c) {
    var r = c[0] / 255, g = c[1] / 255, b = c[2] / 255, mx = Math.max(r, g, b), mn = Math.min(r, g, b), h = 0, s = 0, l = (mx + mn) / 2;
    if (mx !== mn) {
      var d = mx - mn; s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
      h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; h /= 6;
    }
    return [h, s, l];
  }
  function hslToRgb(a) {
    var h = a[0], s = a[1], l = a[2];
    function f(p, q, t) { if (t < 0) t += 1; if (t > 1) t -= 1; if (t < 1 / 6) return p + (q - p) * 6 * t; if (t < 1 / 2) return q; if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6; return p; }
    if (s === 0) { var v = l * 255; return [v, v, v]; }
    var q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
    return [f(p, q, h + 1 / 3) * 255, f(p, q, h) * 255, f(p, q, h - 1 / 3) * 255];
  }
  function lum(c) {
    var x = c.map(function (v) { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
    return 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2];
  }
  function contrast(a, b) { var la = lum(a), lb = lum(b); return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05); }
  function ink(c) { return contrast(c, [4, 18, 14]) >= contrast(c, [255, 255, 255]) ? '#04120e' : '#ffffff'; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // One picked colour -> the two tuned variants (bright on the dark theme, deep on the light theme).
  function variantsFor(hex) {
    var hsl = rgbToHsl(hexToRgb(hex));
    var dark = rgbToHex(hslToRgb([hsl[0], clamp(hsl[1], 0.45, 0.95), clamp(hsl[2], 0.56, 0.70)]));
    var light = rgbToHex(hslToRgb([hsl[0], clamp(hsl[1], 0.45, 0.90), clamp(hsl[2], 0.22, 0.36)]));
    return { dark: dark, light: light };
  }
  function varsFor(hex, mode) {
    var rgb = hexToRgb(hex), hsl = rgbToHsl(rgb);
    var dim = rgbToHex(hslToRgb([hsl[0], hsl[1], mode === 'light' ? hsl[2] * 0.85 : hsl[2] * 0.45]));
    var on = ink(rgb), rgbStr = rgb.join(', ');
    return {
      '--teal': hex, '--teal-dim': dim, '--teal-rgb': rgbStr,
      '--on-teal': on, '--on-teal2': on, '--on-teal3': on,
      '--glow-teal': '0 0 0 1px rgba(' + rgbStr + ', 0.14), 0 0 26px rgba(' + rgbStr + ', 0.08)'
    };
  }
  function accentName() { var v = get('t58-accent'); return ACCENT_KEYS.indexOf(v) >= 0 ? v : 'teal'; }
  function accentBase() {   // {dark, light} hex pair for the current choice
    var n = accentName();
    if (n === 'custom') { var h = normHex(get('t58-accent-hex')); return h ? variantsFor(h) : ACCENTS.teal; }
    return ACCENTS[n];
  }
  var ACCENT_PROPS = ['--teal', '--teal-dim', '--teal-rgb', '--on-teal', '--on-teal2', '--on-teal3', '--glow-teal'];
  function applyAccent() {
    var root = document.documentElement, n = accentName();
    if (n === 'teal') {   // the default look: remove every override so the stylesheet's own values apply
      ACCENT_PROPS.forEach(function (p) { root.style.removeProperty(p); });
      root.removeAttribute('data-ui-accent');
      return;
    }
    var mode = isLight() ? 'light' : 'dark', v = varsFor(accentBase()[mode], mode);
    Object.keys(v).forEach(function (k) { root.style.setProperty(k, v[k]); });
    root.setAttribute('data-ui-accent', n);
  }
  function cacheAccentVars() {   // the tiny inline script in _head_common.html reads this to paint the accent before first paint
    var n = accentName();
    if (n === 'teal') { try { localStorage.removeItem('t58-accent-vars'); } catch (e) {} return; }
    var b = accentBase();
    set('t58-accent-vars', JSON.stringify({ dark: varsFor(b.dark, 'dark'), light: varsFor(b.light, 'light') }));
  }
  function setAccent(name, hex) {
    if (ACCENT_KEYS.indexOf(name) < 0) name = 'teal';
    if (name === 'custom') {
      var h = normHex(hex);
      if (!h) return false;
      set('t58-accent-hex', h);
    }
    set('t58-accent', name);
    cacheAccentVars();
    applyAccent();
    try { window.dispatchEvent(new CustomEvent('t58-accent-change', { detail: { name: name, hex: accentHex() } })); } catch (e) {}
    return true;
  }
  function accentHex() { return accentBase()[isLight() ? 'light' : 'dark']; }

  /* ------------------------------------------------------------------ verse bar */
  var esv = null, esvState = 'idle';   // idle | loading | done | none
  function loadEsv(cb) {
    if (esvState === 'done' || esvState === 'none') { cb(); return; }
    if (esvState === 'loading') return;
    esvState = 'loading';
    try {
      fetch('/static/esv-verses.json', { cache: 'force-cache' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (j) { esv = (j && j.verses) ? j : null; esvState = esv ? 'done' : 'none'; cb(); })
        .catch(function () { esvState = 'none'; cb(); });
    } catch (e) { esvState = 'none'; cb(); }
  }
  function slotVerse() { return VERSES[Math.floor(Date.now() / SLOT_MS) % VERSES.length]; }

  function renderVerse() {
    var existing = document.getElementById('t58-verse');
    if (!verseOn()) { if (existing) existing.remove(); return; }
    var host = document.querySelector('.t58-main');
    if (!host) return;
    if (esvState === 'idle') { loadEsv(renderVerse); return; }
    if (esvState === 'loading') return;
    var v = slotVerse(), text = v[1], tag = 'KJV', title = '';
    if (esv && esv.verses[v[0]]) {
      text = esv.verses[v[0]]; tag = 'ESV';
      title = esv.copyright || '';
    }
    if (!existing) {
      existing = document.createElement('div');
      existing.id = 't58-verse'; existing.className = 't58-verse'; existing.setAttribute('role', 'note');
      host.insertBefore(existing, host.firstChild);
    }
    var key = v[0] + '|' + tag;
    if (existing.dataset.ref === key) return;
    existing.dataset.ref = key;
    existing.textContent = '';
    if (title) existing.title = title; else existing.removeAttribute('title');
    var q = document.createElement('span'); q.className = 't58-verse-text'; q.textContent = '“' + text + '”';
    var r = document.createElement('span'); r.className = 't58-verse-ref'; r.textContent = '— ' + v[0] + ' (' + tag + ')';
    existing.appendChild(q); existing.appendChild(r);
  }

  /* ------------------------------------------------------------------ seasonal motion */
  var AMBIENT_ID = 't58-ambient';
  var KIND = { winter: 'snow', autumn: 'leaf', spring: 'petal', summer: 'firefly' };
  var BASE_COUNT = { snow: 16, leaf: 9, petal: 11, firefly: 9 };
  function rnd(a, b) { return a + Math.random() * (b - a); }

  function removeAmbient() { var el = document.getElementById(AMBIENT_ID); if (el) el.remove(); }
  function buildAmbient() {
    removeAmbient();
    var kind = KIND[bg()];
    if (!kind || !fxOn()) return;
    try { if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return; } catch (e) {}
    try { if (navigator.connection && navigator.connection.saveData) return; } catch (e) {}
    var n = BASE_COUNT[kind];
    var small = window.innerWidth < 760 || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2);
    if (small) n = Math.max(4, Math.round(n * 0.55));
    var layer = document.createElement('div');
    layer.id = AMBIENT_ID; layer.className = 't58-ambient t58-ambient-' + kind; layer.setAttribute('aria-hidden', 'true');
    for (var i = 0; i < n; i++) {
      var p = document.createElement('i'), inner = document.createElement('b'), s = p.style;
      if (kind === 'firefly') {
        s.setProperty('--x', rnd(4, 96).toFixed(1) + '%');
        s.setProperty('--y', rnd(38, 90).toFixed(1) + '%');
        s.setProperty('--dx', rnd(-70, 70).toFixed(0) + 'px');
        s.setProperty('--dy', rnd(-60, 40).toFixed(0) + 'px');
        s.setProperty('--dur', rnd(16, 30).toFixed(1) + 's');
        s.setProperty('--glow', rnd(3.2, 6.5).toFixed(1) + 's');
        s.setProperty('--delay', (-rnd(0, 24)).toFixed(1) + 's');
        s.setProperty('--sz', rnd(3, 4.6).toFixed(1) + 'px');
      } else {
        // petals start from the blossom tree on the left; everything else from anywhere along the top
        var x = kind === 'petal' ? (Math.random() < 0.65 ? rnd(0, 38) : rnd(0, 100)) : rnd(0, 100);
        var sz = kind === 'snow' ? rnd(2.6, 6.2) : kind === 'leaf' ? rnd(9, 15) : rnd(7, 11);
        s.setProperty('--x', x.toFixed(1) + '%');
        s.setProperty('--sz', sz.toFixed(1) + 'px');
        s.setProperty('--dur', (kind === 'snow' ? rnd(20, 34) : kind === 'leaf' ? rnd(24, 38) : rnd(22, 36)).toFixed(1) + 's');
        s.setProperty('--delay', (-rnd(0, 34)).toFixed(1) + 's');
        s.setProperty('--sway', rnd(16, 46).toFixed(0) + 'px');
        s.setProperty('--swaydur', rnd(4.5, 8.5).toFixed(1) + 's');
        s.setProperty('--spin', rnd(180, 520).toFixed(0) + 'deg');
        s.setProperty('--o', (kind === 'snow' ? rnd(0.5, 0.9) : rnd(0.55, 0.85)).toFixed(2));
        if (kind === 'leaf') inner.className = 'lf' + (i % 4);
        if (kind === 'petal') inner.className = 'pt' + (i % 3);
      }
      p.appendChild(inner); layer.appendChild(p);
    }
    document.body.appendChild(layer);
  }

  /* ------------------------------------------------------------------ scene motion (#t58-scene-fx) */
  var SCENE_ID = 't58-scene-fx';
  function mk(cls, vars) {
    var e = document.createElement('i'); if (cls) e.className = cls;
    if (vars) Object.keys(vars).forEach(function (k) { e.style.setProperty(k, vars[k]); });
    return e;
  }
  function f1(v) { return (+v).toFixed(1); }
  function removeScene() { var el = document.getElementById(SCENE_ID); if (el) el.remove(); }
  function buildScene() {
    removeScene();
    var kind = SCENES[bg()];
    if (!kind || !fxOn()) return;
    try { if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return; } catch (e) {}
    try { if (navigator.connection && navigator.connection.saveData) return; } catch (e) {}
    var small = window.innerWidth < 760 || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2);
    function cnt(n) { return small ? Math.max(3, Math.round(n * 0.55)) : n; }
    var layer = document.createElement('div');
    layer.id = SCENE_ID; layer.className = 't58-scene-fx t58-fx-' + kind; layer.setAttribute('aria-hidden', 'true');
    var i, e;
    if (kind === 'cosmic') {
      layer.appendChild(mk('neb a')); layer.appendChild(mk('neb b'));
      var tints = ['255,255,255', '200,190,255', '150,235,225'];
      for (i = 0; i < cnt(26); i++) layer.appendChild(mk('star', {
        '--x': f1(rnd(1, 99)) + '%', '--y': f1(rnd(1, 99)) + '%', '--sz': f1(rnd(1.4, 3)) + 'px', '--o': f1(rnd(.45, .95)),
        '--dur': f1(rnd(3.5, 9)) + 's', '--delay': f1(-rnd(0, 9)) + 's', '--c': tints[i % 3]
      }));
    } else if (kind === 'obsidian') {
      layer.appendChild(mk('sheen')); layer.appendChild(mk('sheen b'));
    } else if (kind === 'ocean') {
      var ys = [58, 66, 74, 82, 90];
      ys.forEach(function (y, k) {
        layer.appendChild(mk('swell', { top: y + '%', '--o': f1(.07 + k * .035), '--dur': f1(46 - k * 6) + 's', '--bob': f1(rnd(5, 8)) + 's' }));
      });
      layer.appendChild(mk('moonpath'));
      for (i = 0; i < cnt(14); i++) layer.appendChild(mk('glint', {
        '--x': f1(rnd(58, 82)) + '%', '--y': f1(rnd(54, 90)) + '%', '--sz': f1(rnd(10, 30)) + 'px', '--o': f1(rnd(.35, .8)),
        '--dur': f1(rnd(3, 7)) + 's', '--delay': f1(-rnd(0, 7)) + 's'
      }));
    } else if (kind === 'rain') {
      for (i = 0; i < cnt(46); i++) layer.appendChild(mk('drop', {
        '--x': f1(rnd(0, 108)) + '%', '--len': f1(rnd(22, 58)) + 'px', '--o': f1(rnd(.18, .5)),
        '--dur': f1(rnd(.9, 1.7)) + 's', '--delay': f1(-rnd(0, 2)) + 's'
      }));
      var cols = ['64,224,240', '255,176,72', '64,224,240', '255,120,150'];
      for (i = 0; i < cnt(7); i++) layer.appendChild(mk('refl', {
        '--x': f1(rnd(4, 90)) + '%', '--w': f1(rnd(30, 90)) + 'px', '--rc': cols[i % 4],
        '--dur': f1(rnd(7, 14)) + 's', '--delay': f1(-rnd(0, 10)) + 's'
      }));
      for (i = 0; i < cnt(8); i++) layer.appendChild(mk('light', {
        '--x': f1(rnd(30, 74)) + '%', '--y': f1(rnd(40, 58)) + '%', '--sz': f1(rnd(2, 4)) + 'px', '--rc': cols[i % 4],
        '--dur': f1(rnd(3, 7)) + 's', '--delay': f1(-rnd(0, 7)) + 's'
      }));
      for (i = 0; i < cnt(4); i++) layer.appendChild(mk('car', {
        '--y': f1(rnd(46, 60)) + '%', '--rc': i % 2 ? '255,176,72' : '255,90,90',
        '--dur': f1(rnd(14, 24)) + 's', '--delay': f1(-rnd(0, 20)) + 's'
      }));
    } else if (kind === 'sanctuary') {
      layer.appendChild(mk('mist', { '--y': '46%', '--dur': '44s' })); layer.appendChild(mk('mist', { '--y': '66%', '--dur': '62s' }));
      for (i = 0; i < cnt(16); i++) layer.appendChild(mk('mote', {
        '--x': f1(rnd(4, 96)) + '%', '--y': f1(rnd(35, 92)) + '%', '--dx': f1(rnd(-70, 70)) + 'px', '--dy': f1(rnd(-60, 40)) + 'px',
        '--dur': f1(rnd(16, 30)) + 's', '--glow': f1(rnd(3.2, 6.5)) + 's', '--delay': f1(-rnd(0, 24)) + 's', '--sz': f1(rnd(2.6, 4.2)) + 'px'
      }));
      for (i = 0; i < cnt(7); i++) {
        e = mk('leaf', { '--x': f1(rnd(0, 100)) + '%', '--sz': f1(rnd(9, 14)) + 'px', '--dur': f1(rnd(26, 40)) + 's', '--delay': f1(-rnd(0, 36)) + 's', '--sway': f1(rnd(16, 40)) + 'px', '--swaydur': f1(rnd(4.5, 8)) + 's', '--spin': f1(rnd(180, 460)) + 'deg', '--o': f1(rnd(.5, .8)) });
        var b = document.createElement('b'); b.className = 'l' + (i % 2); e.appendChild(b); layer.appendChild(e);
      }
    } else if (kind === 'forest') {
      [[8, 30, .85, 70], [30, 26, .7, 96], [52, 30, .8, 80], [70, 24, .55, 120]].forEach(function (c) {
        layer.appendChild(mk('fog', { '--y': c[0] + '%', '--h': c[1] + '%', '--o': c[2], '--dur': c[3] + 's' }));
      });
      for (i = 0; i < cnt(7); i++) layer.appendChild(mk('ripple', {
        '--x': f1(rnd(22, 80)) + '%', '--y': f1(rnd(70, 92)) + '%', '--w': f1(rnd(90, 190)) + 'px',
        '--dur': f1(rnd(7, 12)) + 's', '--delay': f1(-rnd(0, 12)) + 's'
      }));
    }
    document.body.insertBefore(layer, document.body.firstChild);
  }
  function syncAmbientPause() {
    [AMBIENT_ID, SCENE_ID].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.classList.toggle('paused', !!document.hidden);
    });
  }

  window.T58Personalize = {
    backgrounds: BGS, seasons: SEASON_BGS, scenes: Object.keys(SCENES), bg: bg, verse: verseOn, fx: fxOn,
    setBg: function (name) { if (BGS.indexOf(name) < 0) name = 'mountains'; set('t58-bg', name); applyBg(); swap(); buildAmbient(); buildScene(); syncAmbientPause(); },
    setVerse: function (on) { set('t58-verse', on ? '1' : '0'); renderVerse(); },
    setFx: function (on) { set('t58-fx', on ? '1' : '0'); buildAmbient(); buildScene(); },
    accents: ACCENTS, accentName: accentName, accentHex: accentHex, setAccent: setAccent, normHex: normHex,
    verses: VERSES.length
  };

  applyBg();
  applyAccent();
  cacheAccentVars();
  // The theme toggle (t58-chrome.js) flips data-theme: swap night <-> day scenes (fade that too) and retune the accent.
  if (window.MutationObserver) {
    new MutationObserver(function () { swap(); applyAccent(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
  // Another tab changed a preference: follow it.
  window.addEventListener('storage', function (e) {
    if (e.key === 't58-bg') { applyBg(); swap(); buildAmbient(); buildScene(); }
    else if (e.key === 't58-fx') { buildAmbient(); buildScene(); }
    else if (e.key === 't58-accent' || e.key === 't58-accent-hex') { applyAccent(); }
    else if (e.key === 't58-verse') renderVerse();
  });
  document.addEventListener('visibilitychange', syncAmbientPause);
  function boot() { renderVerse(); setInterval(renderVerse, 60 * 1000); buildAmbient(); buildScene(); syncAmbientPause(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
