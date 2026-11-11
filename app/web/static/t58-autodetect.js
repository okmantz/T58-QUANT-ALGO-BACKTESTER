/* T58 auto-detect: the engine reads the instrument settings (pip size, $ per point, commission) straight from the market
 * data the person picked or uploaded, so no "pick a known instrument" step exists anywhere.
 *
 * Works on any <form> that has a pip_size field. Watching (delegated, so rows added later still work):
 *   select[name$="existing_dataset"]  - a stored dataset      input[type=file][name="csv_file"] - an upload
 *   input[name="datasets"]            - multi-instrument checklists (first ticked row is used)
 * It POSTs the same data the run would use to /data/detect-pip-size and fills pip_size, contract_size, commission and the
 * hidden acc_instrument, but never overwrites a value the person typed. A label/data scale clash fills nothing and says so.
 */
(function () {
  'use strict';
  var timers = new WeakMap(), seq = 0;

  function formOf(t) { return t && (t.form || (t.closest && t.closest('form'))); }
  function pipField(f) { return f && f.querySelector('[name="pip_size"]'); }

  function sourceFor(form) {
    var sel = form.querySelector('select[name$="existing_dataset"]');
    if (sel && sel.value && !(sel.value.indexOf('__api__:') === 0 && sel.value.indexOf('|') < 0)) return { kind: 'dataset', value: sel.value, anchor: sel };
    var fi = form.querySelector('input[type="file"][name="csv_file"]');
    if (fi && fi.files && fi.files.length) return { kind: 'file', input: fi, anchor: fi };
    var ck = form.querySelector('input[name="datasets"]:checked');
    if (ck) return { kind: 'dataset', value: ck.value, anchor: ck.closest('.dataset-list') || ck };
    return null;
  }

  function pill(form, anchor) {
    var p = form.querySelector('.t58-ad');
    if (!p) {
      p = document.createElement('div'); p.className = 't58-ad'; p.setAttribute('aria-live', 'polite');
      var host = anchor && anchor.parentNode ? anchor : null;
      var at = host ? (host.closest('.t58c') || host) : null;      // sit under the combo popover button when present
      if (at && at.parentNode) at.parentNode.insertBefore(p, at.nextSibling); else form.insertBefore(p, form.firstChild);
    }
    return p;
  }
  function say(form, anchor, kind, html) {
    var p = pill(form, anchor); p.className = 't58-ad on ' + kind; p.innerHTML = html;
  }
  function esc(s) { var d = document.createElement('div'); d.textContent = String(s == null ? '' : s); return d.innerHTML; }

  function untouched(el, defaults) {
    if (!el) return false;
    if (el.dataset.t58Touched === '1') return false;
    var v = (el.value || '').trim();
    return v === '' || defaults.indexOf(parseFloat(v)) >= 0 || el.dataset.t58Auto === '1';
  }
  function put(el, v, defaults) {
    if (!el || v == null) return false;
    if (!untouched(el, defaults)) return false;
    el.value = v; el.dataset.t58Auto = '1';
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }

  function run(form) {
    var pip = pipField(form); if (!pip) return;
    var src = sourceFor(form); if (!src) return;
    var fd = new FormData();
    if (src.kind === 'file') { for (var i = 0; i < src.input.files.length; i++) fd.append('csv_file', src.input.files[i]); }
    else fd.append('existing_dataset', src.value);
    var my = ++seq; form.__t58adSeq = my;
    say(form, src.anchor, 'busy', 'Reading your market data to set pip size, $ per point and commission…');
    fetch('/data/detect-pip-size', { method: 'POST', body: fd })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d || {} }; }); })
      .then(function (res) {
        if (form.__t58adSeq !== my) return;                              // a newer pick superseded this one
        var d = res.d;
        if (!res.ok || d.error) { say(form, src.anchor, 'warn', 'Could not read the instrument from this data (' + esc(d.error || 'unknown error') + '). Defaults are in use — expand Advanced to set them.'); return; }
        if (d.warning || d.pip_size == null) { say(form, src.anchor, 'warn', esc(d.warning || 'Could not infer the instrument.')); return; }
        put(pip, d.pip_size, [0.0001]);
        put(form.querySelector('[name="contract_size"]'), d.contract_size, []);
        put(form.querySelector('[name="commission"]'), d.commission, [0, 3]);
        var acc = form.querySelector('#acc_instrument'); if (acc && d.symbol && !acc.value) acc.value = d.symbol;
        var bits = [];
        if (d.symbol) bits.push('<b>' + esc(d.symbol) + '</b>');
        bits.push('pip ' + esc(d.pip_size));
        if (d.contract_size != null) bits.push('$' + esc(d.contract_size) + '/pt');
        if (d.commission != null) bits.push('$' + esc(d.commission) + ' commission');
        say(form, src.anchor, 'ok', '<span class="t58-ad-ic">✓</span> Auto-detected from your data: ' + bits.join(' · '));
      })
      .catch(function () { if (form.__t58adSeq === my) say(form, src.anchor, 'warn', 'Could not reach the detector. Defaults are in use — expand Advanced to set them.'); });
  }

  function schedule(form) {
    clearTimeout(timers.get(form));
    timers.set(form, setTimeout(function () { run(form); }, 150));
  }

  document.addEventListener('change', function (e) {
    var t = e.target; if (!t || !t.matches) return;
    var form = formOf(t); if (!form || !pipField(form)) return;
    if (t.matches('[name="pip_size"],[name="contract_size"],[name="commission"]') && e.isTrusted) { t.dataset.t58Touched = '1'; t.dataset.t58Auto = ''; return; }
    if (t.matches('select[name$="existing_dataset"],input[type="file"][name="csv_file"],input[name="datasets"]')) schedule(form);
  });

  function boot() {
    document.querySelectorAll('form').forEach(function (f) {
      if (!pipField(f)) return;
      var row = f.querySelector('#acc_instrument'); if (row && row.closest('div')) row.closest('div').style.display = 'none';
      if (sourceFor(f)) schedule(f);
    });
    setTimeout(function () { document.querySelectorAll('form').forEach(function (f) { if (pipField(f) && !f.querySelector('.t58-ad.on') && sourceFor(f)) schedule(f); }); }, 900);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
