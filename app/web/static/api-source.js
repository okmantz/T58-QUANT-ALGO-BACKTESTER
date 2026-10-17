/* Market-data dropdown + data providers.
 *
 * Adds a "Download from a data provider" group to every <select name="existing_dataset">.
 * Pick a provider -> one compact row appears (instrument, bar size, dates, Download).
 * Download saves the bars into the data library using the key already saved on
 * Settings > API Keys, then selects the new file -- so the rest of the page just sees
 * an ordinary stored file. Pages without a provider pick cost no extra space.
 */
(function () {
  'use strict';
  var PREFIX = '__api__:';
  var cache = null;

  function getProviders() {
    if (cache) return cache;
    cache = fetch('/data/api/providers', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : { providers: [], timeframes: [] }; })
      .catch(function () { return { providers: [], timeframes: [] }; });
    return cache;
  }

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text) e.textContent = text;
    return e;
  }

  function enhance(sel, info) {
    if (sel.dataset.apiSource || !info.providers.length) return;
    sel.dataset.apiSource = '1';

    var group = el('optgroup', { label: 'Download from a data provider' });
    var ready = info.providers.filter(function (p) { return p.ready; });
    ready.forEach(function (p) { group.appendChild(el('option', { value: PREFIX + p.id }, '⬇ ' + p.label)); });
    if (!ready.length) group.appendChild(el('option', { value: PREFIX + '_none' }, '⬇ Add a data-provider key in Settings › API Keys'));
    sel.appendChild(group);

    var box = el('div', { 'class': 'api-src-row', hidden: '' });
    box.style.cssText = 'margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px;';
    var sym = el('input', { type: 'text', placeholder: 'Instrument', autocomplete: 'off', 'aria-label': 'Instrument' });
    sym.style.cssText = 'width:150px;';
    var tf = el('select', { 'aria-label': 'Bar size' });
    var start = el('input', { type: 'date', 'aria-label': 'From (optional)', title: 'From (optional)' });
    var end = el('input', { type: 'date', 'aria-label': 'To (optional)', title: 'To (optional)' });
    var go = el('button', { type: 'button', 'class': 'lc-btn small' }, 'Download');
    var msg = el('span', { 'class': 'lc-hint', role: 'status' });
    var link = el('a', { href: '/settings/api-keys', target: '_blank', rel: 'noopener' }, 'Settings › API Keys');
    [sym, tf, start, end, go, msg].forEach(function (n) { box.appendChild(n); });
    sel.insertAdjacentElement('afterend', box);

    var lastValid = sel.value && sel.value.indexOf(PREFIX) !== 0 ? sel.value : '';

    function providerId() { return sel.value.indexOf(PREFIX) === 0 ? sel.value.slice(PREFIX.length) : ''; }
    function provider(id) { return info.providers.filter(function (p) { return p.id === id; })[0]; }

    function show() {
      var id = providerId();
      if (!id) { box.hidden = true; return; }
      var p = provider(id);
      box.hidden = false;
      msg.textContent = '';
      if (!p) {  // "add a key" placeholder
        [sym, tf, start, end, go].forEach(function (n) { n.style.display = 'none'; });
        msg.textContent = 'No provider keys saved yet: add one in ';
        msg.appendChild(link);
        return;
      }
      [sym, tf, start, end, go].forEach(function (n) { n.style.display = ''; });
      sym.placeholder = 'Instrument, e.g. ' + p.hint;
      var keep = tf.value;
      tf.innerHTML = '';
      p.timeframes.forEach(function (t) { tf.appendChild(el('option', { value: t }, t)); });
      tf.value = p.timeframes.indexOf(keep) >= 0 ? keep : (p.timeframes.indexOf('5m') >= 0 ? '5m' : p.timeframes[0]);
    }

    function addFile(res) {
      var opt = Array.prototype.filter.call(sel.querySelectorAll('option'), function (o) { return o.value === res.full_name; })[0];
      if (!opt) {
        var g = Array.prototype.filter.call(sel.querySelectorAll('optgroup'), function (o) { return o.label === res.instrument; })[0];
        if (!g) { g = el('optgroup', { label: res.instrument }); sel.insertBefore(g, group); }
        opt = el('option', { value: res.full_name }, res.name);
        g.appendChild(opt);
      }
      sel.value = res.full_name;
    }

    go.addEventListener('click', function () {
      var id = providerId();
      if (!id) return;
      if (!sym.value.trim()) { msg.textContent = 'Enter an instrument.'; sym.focus(); return; }
      go.disabled = true;
      msg.textContent = 'Downloading…';
      fetch('/data/api/fetch', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: id, symbol: sym.value.trim(), timeframe: tf.value, start: start.value, end: end.value })
      }).then(function (r) { return r.json().catch(function () { return { ok: false, error: 'Unexpected response.' }; }); })
        .then(function (res) {
          go.disabled = false;
          if (!res.ok) { msg.textContent = res.error || 'Download failed.'; return; }
          addFile(res);
          lastValid = res.full_name;
          box.hidden = true;
          sel.dispatchEvent(new Event('change', { bubbles: true }));
          var note = sel.parentNode.querySelector('.api-src-done') || sel.insertAdjacentElement('afterend', el('div', { 'class': 'lc-hint api-src-done' }));
          note.textContent = 'Saved ' + res.rows.toLocaleString() + ' bars (' + res.first.slice(0, 10) + ' → ' + res.last.slice(0, 10) + ') as ' + res.name + '.';
        }).catch(function (e) { go.disabled = false; msg.textContent = 'Download failed: ' + e; });
    });

    sel.addEventListener('change', function () {
      var done = sel.parentNode.querySelector('.api-src-done');
      if (done && providerId()) done.textContent = '';
      show();
    });

    if (sel.form) {
      sel.form.addEventListener('submit', function (ev) {
        if (providerId()) {
          ev.preventDefault();
          ev.stopImmediatePropagation();
          box.hidden = false;
          if (provider(providerId())) msg.textContent = 'Click Download first, then run.';
        }
      }, true);
    }
  }

  function init() {
    var sels = document.querySelectorAll('select[name$="existing_dataset"]');
    if (!sels.length) return;
    getProviders().then(function (info) {
      Array.prototype.forEach.call(document.querySelectorAll('select[name$="existing_dataset"]'), function (s) { enhance(s, info); });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
