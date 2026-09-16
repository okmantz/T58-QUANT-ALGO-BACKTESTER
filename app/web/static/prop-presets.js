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
 */

let _PROP_FIRM_PRESETS_CACHE = null;
let _PROP_FIRM_PRESETS_PROMISE = null;

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
    presets.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = JSON.stringify(p);
      opt.textContent = p.label;
      select.appendChild(opt);
    });
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
