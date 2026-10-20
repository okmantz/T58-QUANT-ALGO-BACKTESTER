/* Market data from a data provider -- on every page that picks market data.
 *
 * Every test, optimize and validation page lists the files in the data library. This adds the other way in: use data
 * straight from a provider whose key is saved under Settings > API Keys (Alpaca, London Strategic Edge, Twelve Data,
 * Polygon, Tiingo, Alpha Vantage, Databento) or one that needs no key (Dukascopy).
 *
 *  - Single pickers (<select name="existing_dataset">, "..._existing_dataset", "dataset_label", a one-market
 *    <select name="datasets">): a "Data from a provider" group appears at the top of the list. Pick a provider, type the
 *    instrument, bar size and dates, and press Run -- the bars are downloaded first (a recent identical download is
 *    reused, so your API quota is not spent twice) and the run then uses that data. "Download now" previews it.
 *  - Multi pickers (checkbox lists named "datasets", multi-selects "datasets" / "ds_datasets"): an "Add data from a
 *    provider" panel takes one or more instruments (comma separated), downloads each and adds it, ticked, to the run.
 *
 * Downloads are saved into the data library like any imported file, so nothing downstream needs to know an API was
 * involved -- and the server also understands the pick itself (value "__api__:provider|SYMBOL|bar|from|to"), which is
 * the safety net for flows that never fire a normal submit.  Keys never reach the browser: the page only learns
 * whether one is saved.
 */
(function () {
  'use strict';
  if (window.T58ApiSource) return;

  var PREFIX = '__api__:';
  var MEM_KEY = 't58-api-source-v2';
  var providersPromise = null;

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'style') e.style.cssText = attrs[k]; else e.setAttribute(k, attrs[k]);
    });
    if (text != null) e.textContent = text;
    return e;
  }
  function makeToken(provider, symbol, tf, start, end) {
    return PREFIX + [provider, (symbol || '').trim(), tf || '', start || '', end || ''].join('|');
  }
  function parseToken(v) {
    v = v || '';
    if (v.indexOf(PREFIX) !== 0) return null;
    var p = v.slice(PREFIX.length).split('|');
    return { provider: p[0] || '', symbol: (p[1] || '').trim(), tf: p[2] || '', start: p[3] || '', end: p[4] || '' };
  }
  function splitSymbols(text) {
    return (text || '').split(/[\s,;]+/).map(function (s) { return s.trim(); }).filter(Boolean)
      .filter(function (s, i, a) { return a.indexOf(s) === i; });
  }
  function mem() { try { return JSON.parse(localStorage.getItem(MEM_KEY) || '{}') || {}; } catch (e) { return {}; } }
  function remember(patch) {
    try { var m = mem(); Object.keys(patch).forEach(function (k) { m[k] = patch[k]; }); localStorage.setItem(MEM_KEY, JSON.stringify(m)); } catch (e) { /* private mode */ }
  }
  function getProviders() {
    if (providersPromise) return providersPromise;
    providersPromise = fetch('/data/api/providers', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : { providers: [] }; })
      .catch(function () { return { providers: [] }; });
    return providersPromise;
  }
  function fmtRange(res) {
    return (res.rows || 0).toLocaleString() + ' bars, ' + String(res.first || '').slice(0, 10) + ' → ' + String(res.last || '').slice(0, 10) +
      (res.cached ? ' (reused a recent download)' : '');
  }
  function fetchOne(provider, symbol, tf, start, end) {
    return fetch('/data/api/fetch', {
      method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: provider, symbol: symbol, timeframe: tf, start: start || '', end: end || '' })
    }).then(function (r) { return r.json().catch(function () { return { ok: false, error: 'Unexpected response from the server.' }; }); })
      .then(function (res) { if (!res.ok) throw new Error(res.error || 'Download failed.'); return res; });
  }
  function injectStyle() {
    if (document.getElementById('t58-api-style')) return;
    var s = el('style', { id: 't58-api-style' });
    s.textContent = [
      '.t58-api-row{margin:8px 0 4px;padding:10px;border:1px solid var(--border-light,#2a3242);border-radius:10px;background:var(--panel-3,#161b22);',
      '  display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;font-size:13px}',
      '.t58-api-row[hidden]{display:none}',
      '.t58-api-f{display:flex;flex-direction:column;gap:3px;min-width:0}',
      '.t58-api-f>span{font-size:11px;color:var(--text-muted,#9aa4b2);text-transform:uppercase;letter-spacing:.04em}',
      '.t58-api-row input,.t58-api-row select{margin:0!important;width:auto!important;min-width:0;padding:8px 9px!important;font-size:13px!important}',
      '.t58-api-sym{width:190px!important}.t58-api-sym.wide{width:300px!important}',
      '.t58-api-go{padding:9px 14px;border-radius:8px;border:1px solid var(--teal,#35e0b0);background:var(--teal,#35e0b0);color:var(--on-teal,#04120e);',
      '  font-weight:700;font-size:13px;cursor:pointer;margin:0}',
      '.t58-api-go:disabled{opacity:.55;cursor:progress}',
      '.t58-api-go.ghost{background:none;color:var(--teal,#35e0b0)}',
      '.t58-api-msg{flex:1 1 100%;font-size:12px;color:var(--text-muted,#9aa4b2);line-height:1.5}',
      '.t58-api-msg.err{color:var(--coral,#ff7a7a)}.t58-api-msg.ok{color:var(--teal,#35e0b0)}',
      '.t58-api-msg a{color:var(--teal,#35e0b0)}',
      '.t58-api-chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0 2px;font-size:12px}',
      '.t58-api-chips-l{color:var(--text-muted,#9aa4b2);margin-right:2px}',
      '.t58-api-chip{padding:4px 10px;border-radius:99px;border:1px solid var(--border-light,#2a3242);background:var(--panel-3,#161b22);color:var(--text,#e7ebf2);font-size:12px;cursor:pointer;margin:0;width:auto}',
      '.t58-api-chip:hover{border-color:var(--teal,#35e0b0)}',
      '.t58-api-chip.off{opacity:.55;border-style:dashed}',
      '.t58-api-note{margin:6px 0 2px;font-size:12px;color:var(--text-muted,#9aa4b2)}',
      '.t58-api-note a{color:var(--teal,#35e0b0)}',
      '.t58-api-multi{margin:10px 0 4px;border:1px dashed var(--border-light,#2a3242);border-radius:10px;padding:2px 10px 10px}',
      '.t58-api-multi>summary{cursor:pointer;padding:8px 0;font-size:13px;font-weight:600;color:var(--teal,#35e0b0)}',
      '.t58-api-added{margin:6px 0 0}',
      '.t58-api-added .dataset-row{display:flex;gap:9px;align-items:flex-start;margin:0 0 4px;cursor:pointer}',
      '.t58-api-added .dataset-row input{margin-top:3px}',
      '.t58-api-added-h{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted,#9aa4b2);margin:8px 0 4px}'
    ].join('\n');
    document.head.appendChild(s);
  }

  // ---------------------------------------------------------------- shared field row
  function buildFields(info, opts) {
    opts = opts || {};
    var m = mem();
    var f = {};
    f.root = el('div', { 'class': 't58-api-row' });
    function field(label, node) { var w = el('label', { 'class': 't58-api-f' }); w.appendChild(el('span', null, label)); w.appendChild(node); f.root.appendChild(w); return w; }

    if (opts.pickProvider) {
      f.provider = el('select', { 'aria-label': 'Data provider' });
      info.providers.forEach(function (p) {
        var o = el('option', { value: p.id }, p.label + (p.ready ? '' : ' — no key saved'));
        if (!p.ready) o.disabled = true;
        f.provider.appendChild(o);
      });
      var firstReady = info.providers.filter(function (p) { return p.ready; })[0];
      var want = info.providers.filter(function (p) { return p.id === m.provider && p.ready; })[0] || firstReady;
      if (want) f.provider.value = want.id;
      field('Provider', f.provider);
    }
    f.symbol = el('input', { type: 'text', autocomplete: 'off', spellcheck: 'false', 'aria-label': opts.multi ? 'Instruments' : 'Instrument',
                             'class': 't58-api-sym' + (opts.multi ? ' wide' : '') });
    field(opts.multi ? 'Instruments (comma separated)' : 'Instrument', f.symbol);
    f.tf = el('select', { 'aria-label': 'Bar size' });
    field('Bar size', f.tf);
    f.start = el('input', { type: 'date', 'aria-label': 'From (optional)', title: 'From (leave empty for the default recent window)' });
    field('From', f.start);
    f.end = el('input', { type: 'date', 'aria-label': 'To (optional)', title: 'To (leave empty for today)' });
    field('To', f.end);
    f.go = el('button', { type: 'button', 'class': 't58-api-go' + (opts.multi ? '' : ' ghost') }, opts.multi ? 'Download & add' : 'Download now');
    f.root.appendChild(f.go);
    f.msg = el('div', { 'class': 't58-api-msg', role: 'status', 'aria-live': 'polite' });
    f.root.appendChild(f.msg);
    f.start.value = m.start || '';
    f.end.value = m.end || '';

    f.currentProvider = function () {
      var id = opts.pickProvider ? f.provider.value : (opts.fixedProvider || '');
      return info.providers.filter(function (p) { return p.id === id; })[0] || null;
    };
    f.refreshProvider = function () {
      var p = f.currentProvider();
      var keep = f.tf.value || m.tf || '5m';
      f.tf.innerHTML = '';
      ((p && p.timeframes) || ['1m', '5m', '15m', '30m', '1h', '4h', '1d']).forEach(function (t) { f.tf.appendChild(el('option', { value: t }, t)); });
      f.tf.value = Array.prototype.some.call(f.tf.options, function (o) { return o.value === keep; }) ? keep : (Array.prototype.some.call(f.tf.options, function (o) { return o.value === '5m'; }) ? '5m' : f.tf.options[0].value);
      f.symbol.placeholder = p ? (opts.multi ? 'e.g. ' + p.hint : 'e.g. ' + p.hint) : '';
      var last = (mem()['sym_' + (p ? p.id : '')] || '');
      if (!f.symbol.value && last) f.symbol.value = last;
      f.setMsg(p && p.note ? p.note : '', '');
    };
    f.setMsg = function (text, kind) { f.msg.textContent = text || ''; f.msg.className = 't58-api-msg' + (kind ? ' ' + kind : ''); };
    f.save = function () {
      var p = f.currentProvider(), patch = { tf: f.tf.value, start: f.start.value, end: f.end.value };
      if (p) { patch.provider = p.id; patch['sym_' + p.id] = f.symbol.value; }
      remember(patch);
    };
    if (opts.pickProvider) f.provider.addEventListener('change', function () { f.symbol.value = ''; f.refreshProvider(); });
    f.refreshProvider();
    return f;
  }

  // ---------------------------------------------------------------- form submit hook (download first, then run)
  function hookForm(form, handler) {
    if (!form) return;
    if (!form.__t58api) {
      form.__t58api = [];
      form.addEventListener('submit', function (ev) {
        if (form.__t58apiBusy) { ev.preventDefault(); ev.stopImmediatePropagation(); return; }
        var need = form.__t58api.filter(function (h) { return h.needs(); });
        if (!need.length) return;
        ev.preventDefault(); ev.stopImmediatePropagation();
        var submitter = ev.submitter || null;
        form.__t58apiBusy = true;
        need.reduce(function (p, h) { return p.then(function () { return h.run(); }); }, Promise.resolve())
          .then(function () {
            form.__t58apiBusy = false;
            try { if (form.requestSubmit) { form.requestSubmit(submitter || undefined); return; } } catch (e) { /* fall through */ }
            form.submit();
          }).catch(function () { form.__t58apiBusy = false; });
      }, true);
    }
    form.__t58api.push(handler);
  }

  // ---------------------------------------------------------------- single pickers
  function enhanceSingle(sel, info) {
    if (sel.dataset.t58Api || !info.providers.length) return;
    sel.dataset.t58Api = '1';
    var isSelect1 = !sel.multiple;
    if (!isSelect1) return;

    var group = el('optgroup', { label: '☁ Data from a provider (your saved API keys)' });
    info.providers.forEach(function (p) {
      var o = el('option', { value: PREFIX + p.id }, p.label + (p.ready ? '' : ' — add a key in Settings › API Keys'));
      if (!p.ready) o.disabled = true;
      group.appendChild(o);
    });
    var firstGroup = sel.querySelector('optgroup');
    if (firstGroup) sel.insertBefore(group, firstGroup); else sel.appendChild(group);

    var f = buildFields(info, { fixedProvider: '' });
    f.root.hidden = true;
    // when the user switches providers the dropdown itself is the provider switch
    var dl = el('div', { 'class': 't58-api-note' });
    sel.insertAdjacentElement('afterend', f.root);
    f.root.insertAdjacentElement('afterend', dl);

    // Visible one-click chips (the dropdown group alone was easy to miss): "☁ Download from  [Alpaca] [London Strategic Edge] ..."
    var chips = el('div', { 'class': 't58-api-chips' });
    chips.appendChild(el('span', { 'class': 't58-api-chips-l' }, '☁ Download from a data provider:'));
    info.providers.forEach(function (p) {
      var b = el('button', { type: 'button', 'class': 't58-api-chip' + (p.ready ? '' : ' off'),
                             title: p.ready ? 'Use ' + p.label + ' (key saved)' : 'No key saved — add one under Account › API Keys' }, p.label);
      b.addEventListener('click', function () {
        if (!p.ready) { window.location.href = '/settings/api-keys'; return; }
        var o = Array.prototype.filter.call(sel.options, function (x) { return x.value.indexOf(PREFIX) === 0 && x.value.slice(PREFIX.length).split('|')[0] === p.id; })[0];
        if (o) { sel.value = o.value; sel.dispatchEvent(new Event('change', { bubbles: true })); }
      });
      chips.appendChild(b);
    });
    sel.insertAdjacentElement('afterend', chips);

    var lastValid = (sel.value && sel.value.indexOf(PREFIX) !== 0) ? sel.value : '';
    function providerId() { var t = parseToken(sel.value); return t ? t.provider : ''; }
    function providerOpt() { return sel.options[sel.selectedIndex] && sel.options[sel.selectedIndex].value.indexOf(PREFIX) === 0 ? sel.options[sel.selectedIndex] : null; }
    function currentP() { var id = providerId(); return info.providers.filter(function (p) { return p.id === id; })[0] || null; }
    f.currentProvider = currentP;

    function syncToken() {
      var o = providerOpt(); if (!o) return;
      var id = providerId();
      o.value = f.symbol.value.trim() ? makeToken(id, f.symbol.value, f.tf.value, f.start.value, f.end.value) : PREFIX + id;
    }
    var shown = '';
    function show() {
      var id = providerId();
      if (!id) { f.root.hidden = true; shown = ''; return; }
      f.root.hidden = false; dl.textContent = '';
      if (shown !== id) { f.symbol.value = ''; shown = id; }   // each provider names instruments its own way
      f.refreshProvider();
      syncToken();
    }
    ['input', 'change'].forEach(function (evn) {
      [f.symbol, f.tf, f.start, f.end].forEach(function (n) { n.addEventListener(evn, function () { syncToken(); f.save(); }); });
    });

    function addFile(res) {
      var opt = Array.prototype.filter.call(sel.querySelectorAll('option'), function (o) { return o.value === res.full_name; })[0];
      if (!opt) {
        var g = Array.prototype.filter.call(sel.querySelectorAll('optgroup'), function (o) { return o.label === res.instrument; })[0];
        if (!g) { g = el('optgroup', { label: res.instrument }); sel.insertBefore(g, group.nextSibling); }
        opt = el('option', { value: res.full_name }, res.name);
        g.appendChild(opt);
      }
      sel.value = res.full_name;
    }
    function run() {
      var p = currentP();
      if (!p) return Promise.resolve();
      if (!f.symbol.value.trim()) { f.setMsg('Type the instrument to download (e.g. ' + p.hint + ').', 'err'); f.symbol.focus(); return Promise.reject(new Error('no instrument')); }
      f.go.disabled = true; f.setMsg('Downloading ' + f.symbol.value.trim() + ' from ' + p.label + '…', '');
      f.save();
      return fetchOne(p.id, f.symbol.value.trim(), f.tf.value, f.start.value, f.end.value).then(function (res) {
        f.go.disabled = false;
        addFile(res); lastValid = res.full_name;
        f.root.hidden = true;
        dl.innerHTML = '';
        dl.appendChild(document.createTextNode('☁ ' + p.label + ': ' + res.name + ' — ' + fmtRange(res) + '. '));
        var again = el('a', { href: '#' }, 'Use a different instrument or range');
        again.addEventListener('click', function (e) {
          e.preventDefault();
          var o = Array.prototype.filter.call(sel.options, function (x) { return x.value.indexOf(PREFIX) === 0 && x.value.slice(PREFIX.length).split('|')[0] === p.id; })[0];
          if (o) { sel.value = o.value; sel.dispatchEvent(new Event('change', { bubbles: true })); }
        });
        dl.appendChild(again);
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        return res;
      }).catch(function (e) { f.go.disabled = false; f.setMsg(e.message || String(e), 'err'); throw e; });
    }
    f.go.addEventListener('click', function () { run().catch(function () {}); });
    sel.addEventListener('change', show);
    if (sel.form) hookForm(sel.form, { needs: function () { return !!providerId() && !sel.disabled; }, run: run });
    show();
  }

  // ---------------------------------------------------------------- multi pickers
  function addCheckboxRow(host, res, p) {
    var existing = Array.prototype.filter.call(document.querySelectorAll('input[name="datasets"]'), function (c) { return c.value === res.full_name; })[0];
    if (existing) { if (!existing.checked) { existing.checked = true; existing.dispatchEvent(new Event('change', { bubbles: true })); } return; }
    var row = el('label', { 'class': 'dataset-row' });
    var cb = el('input', { type: 'checkbox', name: 'datasets', value: res.full_name });
    cb.checked = true;
    row.appendChild(cb);
    row.appendChild(el('span', { 'class': 'dataset-name' }, res.name));
    row.appendChild(el('span', { 'class': 'dataset-size' }, (p ? p.label + ' · ' : '') + (res.rows || 0).toLocaleString() + ' bars'));
    host.appendChild(row);
    cb.dispatchEvent(new Event('change', { bubbles: true }));
  }
  function addSelectOption(sel, res) {
    var opt = Array.prototype.filter.call(sel.querySelectorAll('option'), function (o) { return o.value === res.full_name; })[0];
    if (!opt) {
      var g = Array.prototype.filter.call(sel.querySelectorAll('optgroup'), function (o) { return o.label === '☁ From data providers'; })[0];
      if (!g) { g = el('optgroup', { label: '☁ From data providers' }); sel.insertBefore(g, sel.firstChild); }
      opt = el('option', { value: res.full_name }, res.name);
      g.appendChild(opt);
    }
    opt.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function buildMultiPanel(info, anchor, mode, sel) {
    var det = el('details', { 'class': 't58-api-multi' });
    det.appendChild(el('summary', null, '☁ Add data from a provider (Alpaca, London Strategic Edge, …)'));
    var f = buildFields(info, { pickProvider: true, multi: true });
    det.appendChild(f.root);
    var addedHead = el('div', { 'class': 't58-api-added-h' }, 'Added from providers (ticked = included in the run)');
    var added = el('div', { 'class': 't58-api-added' });
    added.hidden = true; addedHead.hidden = true;
    det.appendChild(addedHead); det.appendChild(added);
    anchor.insertAdjacentElement('afterend', det);
    if (!info.providers.some(function (p) { return p.ready; })) {
      f.setMsg('No provider keys are saved yet. Add one under Settings › API Keys (Dukascopy needs none).', 'err');
    }

    function run() {
      var syms = splitSymbols(f.symbol.value);
      var p = f.currentProvider();
      if (!p) { f.setMsg('Choose a provider with a saved key.', 'err'); return Promise.reject(new Error('no provider')); }
      if (!syms.length) { f.setMsg('Type at least one instrument (e.g. ' + p.hint + ').', 'err'); f.symbol.focus(); return Promise.reject(new Error('no instrument')); }
      f.go.disabled = true; f.save();
      var done = 0, failed = [];
      return syms.reduce(function (chain, s) {
        return chain.then(function () {
          f.setMsg('Downloading ' + s + ' (' + (done + failed.length + 1) + ' of ' + syms.length + ')…', '');
          return fetchOne(p.id, s, f.tf.value, f.start.value, f.end.value).then(function (res) {
            done++;
            if (mode === 'select') addSelectOption(sel, res); else { added.hidden = false; addedHead.hidden = false; addCheckboxRow(added, res, p); }
          }).catch(function (e) { failed.push(s + ': ' + (e.message || e)); });
        });
      }, Promise.resolve()).then(function () {
        f.go.disabled = false;
        if (failed.length) f.setMsg((done ? 'Added ' + done + '. ' : '') + 'Failed — ' + failed.join(' | '), 'err');
        else { f.setMsg('Added ' + done + ' dataset' + (done === 1 ? '' : 's') + '. They are ticked and will be used in the run.', 'ok'); f.symbol.value = ''; }
        if (failed.length) throw new Error(failed.join(' | '));
      }).catch(function (e) { f.go.disabled = false; throw e; });
    }
    f.go.addEventListener('click', function () { run().catch(function () {}); });
    var form = (sel && sel.form) || anchor.closest('form');
    // typed but not added yet -> add it when the form is submitted, so "type it and press Run" just works
    if (form) hookForm(form, { needs: function () { return splitSymbols(f.symbol.value).length > 0 && f.currentProvider(); }, run: run });
    return det;
  }

  // ---------------------------------------------------------------- discovery
  function enhanceMulti(info) {
    // checkbox lists (multi-market, multi-instrument search / evolution / speed run, cross-instrument)
    var boxes = document.querySelectorAll('input[type="checkbox"][name="datasets"]');
    var done = new Set();
    Array.prototype.forEach.call(boxes, function (b) {
      var form = b.form || b.closest('form');
      if (!form || done.has(form) || form.dataset.t58ApiMulti) return;
      done.add(form); form.dataset.t58ApiMulti = '1';
      var list = b.closest('.dataset-list') || b.closest('.t58-ms') || b.parentNode;
      var anchor = list.closest('.t58-ms') || list;
      buildMultiPanel(info, anchor, 'checkbox', null);
    });
    // pages with NO stored datasets show an empty-note instead of a list; give them the panel too
    Array.prototype.forEach.call(document.querySelectorAll('form .empty-note'), function (note) {
      var form = note.closest('form');
      if (!form || form.dataset.t58ApiMulti) return;
      if (!/dataset/i.test(note.textContent) || !/(at least two|2 or more|pick 2)/i.test(note.textContent + (note.parentNode.textContent || ''))) return;
      form.dataset.t58ApiMulti = '1';
      buildMultiPanel(info, note, 'checkbox', null);
    });
    // multi-selects
    Array.prototype.forEach.call(document.querySelectorAll('select[multiple][name="datasets"], select[multiple][name="ds_datasets"]'), function (s) {
      if (s.dataset.t58ApiMulti) return;
      s.dataset.t58ApiMulti = '1';
      buildMultiPanel(info, s, 'select', s);
    });
  }

  var SINGLE = 'select[name$="existing_dataset"], select[name="dataset_label"], select[name="datasets"]:not([multiple])';

  // A page whose data library is still empty used to render NO dataset dropdown at all (just the CSV upload), which left
  // nothing for the provider picker to attach to -- so a saved Alpaca / London Strategic Edge key could not be used.
  // Give every single-dataset form a real (initially empty) dropdown so the provider group always has a home.
  function ensureDatasetSelects() {
    Array.prototype.forEach.call(document.querySelectorAll('input[type="file"][name="csv_file"]'), function (fi) {
      var form = fi.form || fi.closest('form');
      if (!form) return;
      if (form.querySelector(SINGLE + ', input[type="checkbox"][name="datasets"], select[multiple][name="datasets"]')) return;
      var sel = el('select', { name: 'existing_dataset', 'class': 'inp', id: 't58-synth-dataset-pick' });
      sel.appendChild(el('option', { value: '' }, '— none, I\'m uploading a file or using a data provider —'));
      var anchor = fi.previousElementSibling && fi.previousElementSibling.tagName === 'LABEL' ? fi.previousElementSibling : fi;
      var lab = el('label', { 'class': 'flabel', 'for': 't58-synth-dataset-pick' }, 'Market data');
      anchor.parentNode.insertBefore(lab, anchor);
      anchor.parentNode.insertBefore(sel, anchor);
    });
  }

  function init() {
    if (!document.querySelector('input[type="file"][name="csv_file"], ' + SINGLE + ', input[type="checkbox"][name="datasets"], select[multiple][name="datasets"], select[multiple][name="ds_datasets"], form .empty-note')) return;
    getProviders().then(function (info) {
      if (!info || !info.providers || !info.providers.length) return;
      injectStyle();
      ensureDatasetSelects();
      Array.prototype.forEach.call(document.querySelectorAll(SINGLE), function (s) { enhanceSingle(s, info); });
      enhanceMulti(info);
    });
  }

  window.T58ApiSource = { PREFIX: PREFIX, makeToken: makeToken, parseToken: parseToken, splitSymbols: splitSymbols, init: init };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
