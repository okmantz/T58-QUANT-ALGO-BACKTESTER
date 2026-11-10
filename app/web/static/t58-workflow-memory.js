/* Workflow helpers for every run form (2026-11-05):
 *  1. pre-fills a run form from the last values submitted to that workflow (kept server-side), then from the trader profile;
 *  2. saves the form's values when it is submitted;
 *  3. shows a verification badge next to a prop-firm preset picker (verified / older than 30 days / never verified);
 *  4. shows a pre-flight + time-to-answer line above the submit button.
 * Everything is best-effort: any failure leaves the form exactly as the server rendered it. */
(function () {
  'use strict';
  var WORKFLOWS = {
    '/full-pipeline/start': 'full_pipeline', '/quick-optimize/start': 'quick_optimize', '/discovery-engine/start': 'discovery_engine',
    '/evolution/start': 'evolution', '/search/start': 'search', '/multi-market/start': 'multi_market', '/refine/start': 'refine'
  };
  var SKIP = /^(csrf|_token|file|upload|strategy_file|password|api|key|token|secret|webhook|email)/i;

  function workflowOf(form) {
    var a = (form.getAttribute('action') || '').split('?')[0];
    if (form.dataset && form.dataset.workflow) return form.dataset.workflow;
    return WORKFLOWS[a] || null;
  }
  function fieldsOf(form) {
    return Array.prototype.filter.call(form.elements, function (el) { return el.name && !el.disabled && el.type !== 'file' && el.type !== 'submit' && el.type !== 'button' && !SKIP.test(el.name); });
  }
  function setValue(el, val) {
    if (el.type === 'checkbox') { el.checked = (val === 'on' || val === 'true' || val === '1'); return; }
    if (el.type === 'radio') { el.checked = (el.value === val); return; }
    if (el.tagName === 'SELECT') { for (var i = 0; i < el.options.length; i++) if (el.options[i].value === val) { el.value = val; break; } }
    else el.value = val;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
  function getJSON(url) { return fetch(url, { credentials: 'same-origin' }).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }); }

  function prefill(form, wf) {
    getJSON('/api/workflow-memory/' + wf).then(function (mem) {
      var vals = (mem && mem.values) || null;
      var touched = false;
      if (vals) {
        fieldsOf(form).forEach(function (el) {
          if (el.type === 'checkbox') { if (el.name in vals) { setValue(el, vals[el.name]); touched = true; } else if (Object.keys(vals).length) { /* unchecked boxes are not submitted */ } return; }
          if (el.name in vals && vals[el.name] !== '') { setValue(el, vals[el.name]); touched = true; }
        });
        /* adaptive risk is ON by default everywhere: an unchecked remembered box must not silently switch it off */
        var ad = form.querySelector('[name="adaptive_risk_enabled"]');
        if (ad && vals && !('adaptive_risk_enabled' in vals)) ad.checked = true;
      }
      if (!touched) {
        getJSON('/api/profile/defaults').then(function (p) {
          if (!p || !p.defaults) return;
          Object.keys(p.defaults).forEach(function (k) { var el = form.querySelector('[name="' + k + '"]'); if (el && (el.value === '' || el.value === undefined)) setValue(el, String(p.defaults[k])); });
        });
      } else {
        showNote(form, 'Opened with the settings from your last run.');
      }
    });
  }
  function showNote(form, html) {
    var n = document.createElement('div');
    n.className = 't58-wf-note'; n.style.cssText = 'font-size:12px;opacity:.75;margin:6px 0;';
    n.innerHTML = html;
    form.insertBefore(n, form.firstChild);
  }
  function remember(form, wf) {
    var data = {};
    fieldsOf(form).forEach(function (el) {
      if (el.type === 'checkbox') { if (el.checked) data[el.name] = 'on'; return; }
      if (el.type === 'radio' && !el.checked) return;
      data[el.name] = el.value;
    });
    var blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    if (navigator.sendBeacon) navigator.sendBeacon('/api/workflow-memory/' + wf, blob);
    else fetch('/api/workflow-memory/' + wf, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data), keepalive: true });
  }

  function badge(sel, presets) {
    var span = document.createElement('span');
    span.className = 't58-preset-badge'; span.style.cssText = 'font-size:12px;margin-left:8px;';
    sel.parentNode.appendChild(span);
    function render() {
      var p = null;
      try { p = sel.value ? JSON.parse(sel.value) : null; } catch (e) { p = null; }
      if (!p || !p.key) { span.textContent = ''; return; }
      var st = p.rules_state, age = p.rules_age_days;
      if (st === 'ok') { span.style.color = 'var(--lc-acc, #2fb67c)'; span.textContent = '✓ rules verified ' + age + ' day(s) ago'; span.title = 'Verified against the firm’s own page; valid for ' + (p.rules_days_left) + ' more day(s).'; }
      else if (age != null) { span.style.color = '#d9a441'; span.innerHTML = '⚠ rules last verified ' + age + ' days ago (older than 30) — a READY verdict needs a re-check <a href="' + (p.rules_page_url || '#') + '" target="_blank" rel="noopener">firm rules page</a>'; }
      else { span.style.color = '#d9a441'; span.innerHTML = '⚠ rules never verified — verdict is capped at MARGINAL <a href="' + (p.rules_page_url || '#') + '" target="_blank" rel="noopener">firm rules page</a>'; }
    }
    sel.addEventListener('change', render); render();
  }

  function preflight(form, wf) {
    var btn = form.querySelector('button[type=submit],input[type=submit]');
    if (!btn) return;
    var box = document.createElement('div');
    box.className = 't58-preflight'; box.style.cssText = 'font-size:12.5px;margin:10px 0;line-height:1.5;';
    btn.parentNode.insertBefore(box, btn);
    var timer = null;
    function run() {
      var q = new URLSearchParams({ workflow: wf });
      var ds = form.querySelector('[name="dataset"]'); if (ds && ds.value) q.set('dataset', ds.value);
      var pk = form.querySelector('[name="prop_preset_key"],#fp_prop_preset_key'); if (pk && pk.value) q.set('preset', pk.value);
      var ad = form.querySelector('[name="adaptive_risk_enabled"]'); if (ad) q.set('adaptive', ad.checked ? '1' : '0');
      var om = form.querySelector('[name="optimizer_mode"]'); if (om && om.value) q.set('optimizer_mode', om.value);
      getJSON('/api/preflight?' + q.toString()).then(function (d) {
        if (!d || !d.checks) { box.textContent = ''; return; }
        box.innerHTML = d.checks.map(function (c) {
          var icon = c.status === 'ok' ? '✓' : (c.status === 'warn' ? '⚠' : '✗');
          var col = c.status === 'ok' ? 'inherit' : (c.status === 'warn' ? '#d9a441' : '#e0564b');
          return '<div style="color:' + col + '">' + icon + ' ' + c.text.replace(/</g, '&lt;') + '</div>';
        }).join('');
      });
    }
    function later() { clearTimeout(timer); timer = setTimeout(run, 350); }
    form.addEventListener('change', later); run();
  }

  document.addEventListener('DOMContentLoaded', function () {
    Array.prototype.forEach.call(document.forms, function (form) {
      var wf = workflowOf(form);
      if (!wf) return;
      prefill(form, wf);
      preflight(form, wf);
      form.addEventListener('submit', function () { try { remember(form, wf); } catch (e) { /* never block a launch */ } });
    });
    Array.prototype.forEach.call(document.querySelectorAll('select[id$="prop_preset_select"]'), function (sel) {
      var tries = 0;
      (function wait() { if (sel.options.length > 1 || ++tries > 20) badge(sel); else setTimeout(wait, 150); })();
    });
  });
})();
