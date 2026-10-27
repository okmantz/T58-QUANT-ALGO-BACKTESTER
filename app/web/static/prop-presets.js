/*
 * Shared "prop-firm preset" dropdown + account/balance field sync helper.
 *
 * Why this exists: every tab that has a Prop-Firm Rules section (account
 * size, profit target, daily loss limit, max drawdown, ...) used to need
 * its own copy-pasted <select> + populate/apply JS, and the Flask route
 * behind that page had to remember to pass prop_presets_json into every
 * render_template() call (easy to forget on an error-path re-render, and
 * several tabs never got the dropdown built at all). This file fetches
 * the preset list once from GET /api/prop-firm-presets and gives every
 * page one shared way to wire up the dropdown, so adding it to a new tab
 * is a two-line call instead of a template-specific script block.
 *
 * Usage (in a page's own <script> block, after including this file):
 *
 *   initPropFirmPresetDropdown('my_preset_select', {
 *     account_size: 'account_size',        // form field name (or #id)
 *     profit_target: 'profit_target',
 *     daily_loss: 'daily_loss',
 *     max_dd: 'max_dd',
 *   });
 *
 * Any of the four target keys can be omitted if that page doesn't have
 * the corresponding field. Target values may be a form field `name`
 * (resolved via the <select>'s closest <form>) or an element `id`
 * (resolved via document.getElementById) -- whichever the field already
 * uses; this tries name first, then id, so existing pages don't need to
 * rename anything.
 *
 * syncAccountAndBalanceFields(idOrNameA, idOrNameB) is the companion
 * RISK-001 fix: keeps two fields (typically "Account size" and "Initial
 * balance", which the backend already forces to agree -- see
 * app.backtest.risk.with_prop_safety_defaults) visually in sync so the
 * person never sees them silently disagree.
 *
 * v8: algo-trading badge + filter -- every preset dict carries
 * `algo_trading_allowed` (see app.prop.presets), and every preset picker
 * now shows a per-option badge plus an "Only show firms that allow algo
 * trading" toggle (default ON, so nobody accidentally optimizes for a
 * firm that bans automation; the toggle is always visible so every
 * preset stays one click away). renderPropPresetOptions(select, presets,
 * valueFn) renders the options with badges; addAlgoTradingFilter(selectId,
 * presets, valueFn) adds the toggle on top. Pages that build their own
 * option values (e.g. an index or a preset key instead of the JSON blob)
 * pass valueFn; the default is JSON.stringify(p) like the original
 * inline populators. initPropFirmPresetDropdown() below uses both
 * automatically.
 */

// `var` + typeof guard (not `let`): this file may be included twice on
// pages that both load it directly and include the
// _prop_rules_extra_fields.html fragment -- re-execution must be a no-op,
// not a "has already been declared" SyntaxError.
var _PROP_FIRM_PRESETS_CACHE = (typeof _PROP_FIRM_PRESETS_CACHE !== "undefined") ? _PROP_FIRM_PRESETS_CACHE : null;
var _PROP_FIRM_PRESETS_PROMISE = (typeof _PROP_FIRM_PRESETS_PROMISE !== "undefined") ? _PROP_FIRM_PRESETS_PROMISE : null;

function _fetchPropFirmPresets() {
  if (_PROP_FIRM_PRESETS_CACHE) return Promise.resolve(_PROP_FIRM_PRESETS_CACHE);
  if (_PROP_FIRM_PRESETS_PROMISE) return _PROP_FIRM_PRESETS_PROMISE;
  _PROP_FIRM_PRESETS_PROMISE = fetch('/api/prop-firm-presets')
    .then((r) => (r.ok ? r.json() : []))
    .then((data) => {
      _PROP_FIRM_PRESETS_CACHE = Array.isArray(data) ? data : [];
      return _PROP_FIRM_PRESETS_CACHE;
    })
    .catch(() => []);
  return _PROP_FIRM_PRESETS_PROMISE;
}

function _resolveFieldEl(form, nameOrId) {
  if (!nameOrId) return null;
  let el = form ? form.querySelector(`[name="${nameOrId}"]`) : null;
  if (!el) el = document.getElementById(nameOrId);
  return el;
}

/* v9: tidy picker. Presets are grouped by firm (<optgroup>), each row reads "$100k Challenge -- algo OK" instead of a
   long sentence, and a small summary strip under the dropdown shows the selected preset's key rules as chips. */
function _ppInjectCss() {
  if (document.getElementById("t58-pp-css")) return;
  const st = document.createElement("style");
  st.id = "t58-pp-css";
  st.textContent =
    ".t58-pp-filter{display:flex;align-items:center;gap:8px;font-size:12.5px;margin:2px 0 7px;cursor:pointer;color:var(--text-muted,#8a93a6);user-select:none}" +
    ".t58-pp-filter input{width:auto;margin:0}" +
    "select.t58-pp-select optgroup{font-weight:700;font-style:normal;color:var(--teal,#35e0b0)}" +
    "select.t58-pp-select option{font-weight:400;color:var(--text,#e8ecf3);background:var(--pop-bg,#0c1c22)}" +
    ".t58-pp-sum{display:none;flex-wrap:wrap;gap:6px;margin:8px 0 2px;font-size:12px}" +
    ".t58-pp-sum.on{display:flex}" +
    ".t58-pp-chip{display:inline-flex;gap:5px;align-items:baseline;padding:3px 9px;border-radius:999px;border:1px solid var(--border-light,#2a3142);background:var(--panel-3,#151a24);color:var(--text,#e8ecf3);white-space:nowrap}" +
    ".t58-pp-chip b{font-weight:700}.t58-pp-chip span{color:var(--text-muted,#8a93a6);font-size:11px}" +
    ".t58-pp-chip.ok{border-color:rgba(var(--ok-rgb,53,224,176),.5);color:var(--ok,#35e0b0)}" +
    ".t58-pp-chip.warn{border-color:rgba(245,184,61,.55);color:var(--amber,#f5b83d)}" +
    ".t58-pp-note{flex-basis:100%;font-size:11.5px;color:var(--text-muted,#8a93a6);line-height:1.4}";
  document.head.appendChild(st);
}

function _ppMoney(n) {
  n = Number(n);
  if (!isFinite(n)) return "";
  return n >= 1000 ? "$" + (n / 1000) + "k" : "$" + n;
}

/* "FTMO - $100k Challenge" -> "$100k Challenge" (the firm name is already the group heading). */
function _ppShortLabel(p) {
  let l = String(p.label || "");
  const firm = String(p.firm || "");
  if (firm && l.toLowerCase().indexOf(firm.toLowerCase()) === 0) l = l.slice(firm.length).replace(/^\s*[-–—:]\s*/, "");
  return l || String(p.label || "");
}

function _ppSummary(select) {
  let box = select._ppSummary;
  if (!box) {
    box = document.createElement("div");
    box.className = "t58-pp-sum";
    box.setAttribute("aria-live", "polite");
    select.parentNode.insertBefore(box, select.nextSibling);
    select._ppSummary = box;
  }
  return box;
}

function _ppUpdateSummary(select) {
  const box = _ppSummary(select);
  const opt = select.options[select.selectedIndex];
  const p = opt && opt._preset;
  if (!p) { box.className = "t58-pp-sum"; box.innerHTML = ""; return; }
  const chip = (cls, label, val) => {
    if (val === undefined || val === null || val === "") return "";
    return '<span class="t58-pp-chip ' + cls + '"><span>' + label + "</span><b>" + val + "</b></span>";
  };
  const pct = (v) => (v === undefined || v === null ? "" : v + "%");
  const verified = p.verified_on && p.verification_tier === "primary";
  let html = "";
  html += chip("", "Account", _ppMoney(p.account_size));
  html += chip("", "Profit target", pct(p.evaluation_profit_target_pct));
  html += chip("", "Daily loss", pct(p.daily_loss_limit_pct));
  html += chip("", "Max drawdown", pct(p.max_drawdown_pct));
  html += chip("", "Drawdown", p.drawdown_type ? String(p.drawdown_type).replace(/_/g, " ") : "");
  html += chip(p.algo_trading_allowed ? "ok" : "warn", "Algo trading", p.algo_trading_allowed ? "allowed" : "not allowed");
  html += chip(verified ? "ok" : "warn", "Rules", verified ? "verified " + p.verified_on : (p.verified_on ? "checked " + p.verified_on : "unverified"));
  if (!p.algo_trading_allowed && p.algo_policy_note) html += '<span class="t58-pp-note">' + String(p.algo_policy_note).replace(/</g, "&lt;") + "</span>";
  box.innerHTML = html;
  box.className = "t58-pp-sum on";
}

/**
 * Renders preset <option>s grouped by firm. `valueFn(p, i)` builds the option value (default: JSON.stringify(p), matching
 * the original inline populators). The select's existing placeholder option (value="") is preserved. Each option keeps its
 * preset on `option._preset` so the summary strip can describe whichever one is selected.
 */
function renderPropPresetOptions(select, presets, valueFn) {
  if (!select) return;
  _ppInjectCss();
  select.classList.add("t58-pp-select");
  const valueOf = valueFn || ((p) => JSON.stringify(p));
  const prevValue = select.value;
  const placeholder = select.querySelector('option[value=""]');
  select.innerHTML = "";
  if (placeholder) select.appendChild(placeholder);
  const groups = new Map();
  (presets || []).forEach((p, i) => {
    const key = p.firm || "Other";
    if (!groups.has(key)) {
      const g = document.createElement("optgroup");
      g.label = key;
      groups.set(key, g);
      select.appendChild(g);
    }
    const opt = document.createElement("option");
    opt.value = valueOf(p, i);
    // Verification detail lives in the tooltip + summary strip: the dropdown row stays short enough to scan.
    opt.textContent = `${_ppShortLabel(p)}${p.algo_trading_allowed ? "" : "  —  no algo trading"}`;
    opt.title = p.verified_on
      ? `Rules last checked ${p.verified_on}${p.source_url ? " against " + p.source_url : ""}. Always confirm at the firm's site before paying for a challenge.`
      : "These rules have not been confirmed against the firm's own page recently -- check the firm's site before relying on them.";
    opt.dataset.algoAllowed = p.algo_trading_allowed ? "1" : "0";
    opt._preset = p;
    groups.get(key).appendChild(opt);
  });
  if (prevValue && [].some.call(select.options, (o) => o.value === prevValue)) select.value = prevValue;
  if (!select._ppBound) {
    select._ppBound = true;
    select.addEventListener("change", () => _ppUpdateSummary(select));
  }
  _ppUpdateSummary(select);
}

/**
 * Inserts an "Only show firms that allow algo trading" switch just before the preset <select> and re-renders its options
 * whenever it changes. Default ON: nobody should optimise for (or quick-fill rules from) a firm that bans automation; the
 * switch stays visible so every preset is one click away. `valueFn` is passed through to renderPropPresetOptions.
 */
function addAlgoTradingFilter(selectId, presets, valueFn) {
  const select = document.getElementById(selectId);
  if (!select || !presets) return;
  _ppInjectCss();
  if (document.getElementById(selectId + "_algo_only")) return;
  const wrap = document.createElement("label");
  wrap.className = "t58-pp-filter";
  wrap.title = "Untick to also list firms whose rules prohibit algorithmic/automated trading.";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = true;
  cb.id = selectId + "_algo_only";
  wrap.appendChild(cb);
  wrap.appendChild(document.createTextNode("Only firms that allow algo trading"));
  select.parentNode.insertBefore(wrap, select);
  const render = () => {
    // Preserve each preset's ORIGINAL index when the toggle hides some firms: pages like payout_probability.html build
    // option values from the index (PROP_PRESETS[parseInt(value)]), so a filtered list must not renumber them.
    const kept = [];
    const keptIdx = [];
    presets.forEach((p, i) => {
      if (!cb.checked || p.algo_trading_allowed) { kept.push(p); keptIdx.push(i); }
    });
    renderPropPresetOptions(
      select, kept,
      valueFn ? (p, j) => valueFn(p, keptIdx[j]) : undefined,
    );
  };
  cb.addEventListener("change", render);
  render();
}

/**
 * Populates `selectId` with every known prop-firm preset and wires its
 * onchange to fill the target fields described in `fieldMap`.
 * `fieldMap` keys: account_size, profit_target, daily_loss, max_dd,
 * drawdown_type, drawdown_check_mode, consistency, min_days,
 * payout_threshold, payout_cap, payout_freq, buffer -- each value is a
 * form field name or element id already on the page. Any key not
 * present in `fieldMap` (or not found on the page) is silently skipped.
 */
function initPropFirmPresetDropdown(selectId, fieldMap) {
  const select = document.getElementById(selectId);
  if (!select) return;

  _fetchPropFirmPresets().then((presets) => {
    // v8: badges + the "only algo-allowed" filter toggle (default ON)
    // render here, once the async preset list arrives.
    addAlgoTradingFilter(selectId, presets);
  });

  const PRESET_TO_FIELD_KEY = {
    account_size: 'account_size',
    profit_target: 'evaluation_profit_target_pct',
    daily_loss: 'daily_loss_limit_pct',
    max_dd: 'max_drawdown_pct',
    drawdown_type: 'drawdown_type',
    drawdown_check_mode: 'drawdown_check_mode',
    consistency: 'consistency_rule_pct',
    min_days: 'min_trading_days',
    payout_threshold: 'payout_threshold_pct',
    payout_cap: 'payout_cap_pct',
    payout_freq: 'payout_frequency_days',
    buffer: 'required_buffer_pct',
  };

  select.addEventListener('change', () => {
    if (!select.value) return;
    let preset;
    try {
      preset = JSON.parse(select.value);
    } catch (e) {
      return;
    }
    const form = select.closest('form');
    Object.keys(fieldMap || {}).forEach((targetKey) => {
      const presetKey = PRESET_TO_FIELD_KEY[targetKey];
      if (!presetKey || preset[presetKey] === undefined || preset[presetKey] === null) return;
      const el = _resolveFieldEl(form, fieldMap[targetKey]);
      if (!el) return;
      el.value = preset[presetKey];
      // Fire a real input event so any live-sync listener (e.g.
      // syncAccountAndBalanceFields below) picks up the new value too.
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
  });
}

/**
 * RISK-001 fix: keeps two fields showing "the same dollar account" in
 * sync live, so a person editing one immediately sees the other follow
 * -- rather than discovering later (or never) that the backend silently
 * preferred one of two values it was given. Accepts element ids.
 */
function syncAccountAndBalanceFields(idA, idB) {
  const a = document.getElementById(idA);
  const b = document.getElementById(idB);
  if (!a || !b) return;
  a.addEventListener('input', () => { b.value = a.value; });
  b.addEventListener('input', () => { a.value = b.value; });
}

/*
 * v9 auto-tidy: many pages build (or server-render) their own preset <option>s from the preset's full label. Rather than
 * touch ~35 templates, this regroups any such dropdown by firm and shortens the rows -- same <option> elements (values,
 * selected state and page handlers are untouched), only moved into <optgroup>s with the firm prefix trimmed.
 */
var _PP_SELECTOR = 'select[id$="prop_preset_select"], select[name="prop_preset"], select#lc-prop-preset, select#search_multi_preset_select';

function _ppRegroup(select, presets) {
  if (!select || select.querySelector("optgroup") || !presets || !presets.length) return;
  const byLabel = new Map();
  presets.forEach((p) => byLabel.set(String(p.label), p));
  const items = [];
  [].forEach.call(select.options, (o) => {
    if (!o.value) return;
    const text = (o.textContent || "").trim();
    let hit = byLabel.get(text);
    if (!hit) { for (const p of presets) { if (text.indexOf(String(p.label)) === 0) { hit = p; break; } } }
    if (hit) items.push([o, hit]);
  });
  if (!items.length) return;
  _ppInjectCss();
  select.classList.add("t58-pp-select");
  const groups = new Map();
  items.forEach(([o, p]) => {
    const key = p.firm || "Other";
    if (!groups.has(key)) {
      const g = document.createElement("optgroup");
      g.label = key;
      groups.set(key, g);
      select.appendChild(g);
    }
    o.textContent = _ppShortLabel(p) + (p.algo_trading_allowed ? "" : "  —  no algo trading");
    o._preset = p;
    groups.get(key).appendChild(o);
  });
  if (!select._ppBound) {
    select._ppBound = true;
    select.addEventListener("change", () => _ppUpdateSummary(select));
  }
  if (!select.multiple) _ppUpdateSummary(select);
}

function _ppScan() {
  _fetchPropFirmPresets().then((presets) => {
    document.querySelectorAll(_PP_SELECTOR).forEach((s) => _ppRegroup(s, presets));
  });
}

(function () {
  if (typeof document === "undefined" || window._ppAutoTidy) return;
  window._ppAutoTidy = true;
  let t = null;
  const kick = () => { clearTimeout(t); t = setTimeout(_ppScan, 60); };
  const start = () => {
    kick();
    try {
      new MutationObserver(kick).observe(document.body, { childList: true, subtree: true });
    } catch (e) { /* older browser: the initial scan still runs */ }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
