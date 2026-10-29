/* T58 shared page chrome.
   Loaded once by _sidebar.html (included on every page except the bare
   "_job" progress pages), so this runs on every screen without having to
   touch each template individually.

   Responsibilities:
   - build + inject the persistent stage stepper
   - remember which sidebar groups the person had open/closed
   - a small animateNumber() helper other pages can call for counted-up KPIs
   - light/dark theme toggle (see initThemeToggle below) -- the light
     theme's full CSS palette already existed in theme.css
     (html[data-theme="light"]), it just had no UI control anywhere; this
     re-adds that control as a small icon inside the sidebar itself,
     exactly where the removal note above used to suggest putting it back.
*/
(function () {
  "use strict";

  var GROUP_KEY_PREFIX = "t58-navgroup:";
  var THEME_KEY = "t58-theme";

  function initThemeToggle() {
    var btn = document.getElementById("t58-theme-toggle");
    if (!btn) return;
    var icon = document.getElementById("t58-theme-icon");
    var label = document.getElementById("t58-theme-label");

    function apply(theme) {
      if (theme === "light") {
        document.documentElement.setAttribute("data-theme", "light");
        if (icon) icon.innerHTML = '<svg class="lu"><use href="#lu-sun"/></svg>'; // sun
        if (label) label.textContent = "Light theme";
      } else {
        document.documentElement.removeAttribute("data-theme");
        if (icon) icon.innerHTML = '<svg class="lu"><use href="#lu-moon"/></svg>'; // moon
        if (label) label.textContent = "Dark theme";
      }
    }

    var current = "dark";
    try { current = localStorage.getItem(THEME_KEY) || "dark"; } catch (e) {}
    apply(current);

    btn.addEventListener("click", function () {
      current = current === "light" ? "dark" : "light";
      apply(current);
      try { localStorage.setItem(THEME_KEY, current); } catch (e) {}
    });
  }


  /* Profile avatar (top right, every page). Click opens a small menu: Account settings / Change photo / Remove photo.
     Photo upload and removal reuse the Account page's existing routes (same-origin POSTs carrying the CSRF header),
     so the picture saved here is the same one the Account page shows. */
  function initProfileAvatar() {
    var root = document.getElementById("t58-profile");
    var btn = document.getElementById("t58-avatar-btn");
    var menu = document.getElementById("t58-profile-menu");
    if (!root || !btn || !menu) return;
    var csrf = root.getAttribute("data-csrf") || "";
    var fileInput = document.getElementById("t58-avatar-file");
    var removeBtn = document.getElementById("t58-avatar-remove");

    function setOpen(open) {
      menu.hidden = !open;
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.addEventListener("click", function (e) { e.stopPropagation(); setOpen(menu.hidden); });
    document.addEventListener("click", function (e) { if (!root.contains(e.target)) setOpen(false); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") { setOpen(false); btn.blur(); } });

    function showPicture(url) {
      btn.innerHTML = "";
      if (url) {
        var img = document.createElement("img");
        img.id = "t58-avatar-img";
        img.alt = btn.getAttribute("title") || "Profile";
        img.src = url;
        btn.appendChild(img);
      }
      if (removeBtn) removeBtn.hidden = !url;
    }
    function post(url, body) {
      var opts = { method: "POST", headers: { "X-CSRF-Token": csrf }, credentials: "same-origin" };
      if (body) opts.body = body;
      return fetch(url, opts);
    }

    if (fileInput) {
      fileInput.addEventListener("change", function () {
        var f = fileInput.files && fileInput.files[0];
        if (!f) return;
        var fd = new FormData();
        fd.append("profile_picture", f);
        post("/settings/account/profile-picture", fd).then(function (r) {
          if (!r.ok) throw new Error("upload failed");
          showPicture("/settings/account/profile-picture/file?v=" + Date.now());
          setOpen(false);
        }).catch(function () { alert("That image could not be saved. Use a PNG, JPG, GIF or WEBP file."); });
        fileInput.value = "";
      });
    }
    var label = fileInput && fileInput.parentElement;
    if (label) label.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
    if (removeBtn) {
      removeBtn.addEventListener("click", function () {
        post("/settings/account/profile-picture/remove").then(function () {
          var initials = (btn.getAttribute("data-initials") || "");
          btn.innerHTML = "";
          var span = document.createElement("span");
          span.id = "t58-avatar-initials";
          span.textContent = initials || (btn.getAttribute("title") || "T").slice(0, 1).toUpperCase();
          btn.appendChild(span);
          removeBtn.hidden = true;
          setOpen(false);
        });
      });
    }
  }

  function restoreNavGroups() {
    document.querySelectorAll(".t58-nav-group").forEach(function (group, idx) {
      var summary = group.querySelector("summary");
      var key = GROUP_KEY_PREFIX + (summary ? summary.textContent.trim() : idx);
      var saved = null;
      try { saved = localStorage.getItem(key); } catch (e) {}
      // Only override the server-rendered default (open on the active
      // section) if the person has explicitly toggled this group before.
      if (saved === "open") group.setAttribute("open", "");
      else if (saved === "closed") group.removeAttribute("open");
      group.addEventListener("toggle", function () {
        try { localStorage.setItem(key, group.open ? "open" : "closed"); } catch (e) {}
      });
    });
  }

  /* The 8-stage journey: Create -> Test -> Optimize -> Validate ->
     Champion -> Forward Test -> Deploy -> Monitor. Keyed by URL path so it
     works with zero coupling to how each template names active_page. This
     is a navigation aid only -- it marks which stage the current page
     belongs to, not whether that stage is "done" for any given strategy
     (the app has no persisted "current strategy" to make that claim about
     honestly). */
  var STAGES = [
    // Every stage links to the SAME page its sidebar entry opens (Create -> the Create page, not Speed Run).
    { name: "Create", href: "/start-here/create", paths: ["/start-here/create", "/speed-run", "/speed-run-multi", "/generate-strategies", "/research-agent", "/research", "/research-loop", "/forge", "/discovery-engine", "/discover", "/overnight-autopilot"] },
    { name: "Test", href: "/full-pipeline", paths: ["/full-pipeline", "/", "/run", "/payout-probability"] },
    { name: "Optimize", href: "/optimize-simple", paths: ["/optimize-simple", "/optimize", "/search", "/refine", "/quick-optimize", "/multi-objective", "/evolution", "/multi-market", "/risk-sweep", "/cross-instrument"] },
    { name: "Validate", href: "/validate-simple", paths: ["/validate-simple", "/validate", "/walk-forward-opt", "/walk-forward-ga", "/cpcv", "/pbo", "/sensitivity", "/parameter-robustness", "/regime-matrix"] },
    { name: "Champion", href: "/champion-simple", paths: ["/champion-simple", "/portfolio", "/ensemble", "/family-diversity", "/universe", "/compare"] },
    { name: "Forward Test", href: "/forward-test", paths: ["/forward-test", "/replay"] },
    { name: "Deploy", href: "/deploy-live", paths: ["/deploy-live", "/webhook-relay"] },
    { name: "Monitor", href: "/live-market", paths: ["/live-market"] }
  ];


  function currentStageIndex() {
    var path = window.location.pathname.replace(/\/+$/, "") || "/";
    for (var i = 0; i < STAGES.length; i++) {
      if (STAGES[i].paths.indexOf(path) !== -1) return i;
    }
    return -1;
  }

  function buildStepper() {
    var main = document.querySelector(".t58-main");
    if (!main || document.querySelector(".t58-stepper")) return;

    var current = currentStageIndex();
    var bar = document.createElement("div");
    bar.className = "t58-stepper";
    STAGES.forEach(function (stage, i) {
      var a = document.createElement("a");
      a.href = stage.href;
      a.textContent = stage.name;
      if (i === current) a.className = "current";
      bar.appendChild(a);
      if (i < STAGES.length - 1) {
        var sep = document.createElement("span");
        sep.className = "sep";
        sep.textContent = "\u2192";
        bar.appendChild(sep);
      }
    });

    main.insertBefore(bar, main.firstChild);
  }

  /* Simple counted-up number animation for hero KPIs. Any element with
     [data-animate-to] gets counted from 0 (or its current text) up to the
     target on load. Safe no-op if nothing on the page uses it. */
  function animateNumbers() {
    document.querySelectorAll("[data-animate-to]").forEach(function (el) {
      var target = parseFloat(el.getAttribute("data-animate-to"));
      var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      var suffix = el.getAttribute("data-suffix") || "";
      if (isNaN(target)) return;
      var start = 0;
      var duration = 700;
      var startTime = null;
      function step(ts) {
        if (startTime === null) startTime = ts;
        var progress = Math.min(1, (ts - startTime) / duration);
        var eased = 1 - Math.pow(1 - progress, 3);
        var val = start + (target - start) * eased;
        el.textContent = val.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

  /* Mobile hamburger: toggles the off-canvas sidebar drawer open/closed,
     closes on backdrop click, on Escape, or after tapping a nav link. */
  function initMobileMenu() {
    var btn = document.getElementById("t58-mobile-menu-btn");
    var sidebar = document.getElementById("t58-sidebar");
    var backdrop = document.getElementById("t58-sidebar-backdrop");
    if (!btn || !sidebar || !backdrop) return;

    function closeMenu() {
      sidebar.classList.remove("open");
      backdrop.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }
    function openMenu() {
      sidebar.classList.add("open");
      backdrop.classList.add("open");
      btn.setAttribute("aria-expanded", "true");
    }

    btn.addEventListener("click", function () {
      if (sidebar.classList.contains("open")) closeMenu();
      else openMenu();
    });
    backdrop.addEventListener("click", closeMenu);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeMenu();
    });
    sidebar.querySelectorAll("a.t58-nav-item").forEach(function (a) {
      a.addEventListener("click", closeMenu);
    });
  }

  /* Instant navigation (2026-10-17 perf pass): when the pointer rests on a sidebar link (or a finger touches
     it) the browser quietly prefetches that page, so the click that follows opens it from cache instead of
     waiting on the server. Sidebar page links only -- never APIs, downloads, static files or reports -- and
     skipped on Data-Saver / 2G connections. */
  function initInstantNav() {
    var conn = navigator.connection || {};
    if (conn.saveData || /(^|-)2g$/.test(conn.effectiveType || "")) return;
    var seen = {};
    var hoverTimer = null;
    var SKIP = /^\/(static|api|reports|webhook|logout|lock|activate|settings\/account)(\/|$)/;

    function prefetch(a) {
      var href = a.getAttribute("href") || "";
      if (href.charAt(0) !== "/" || href.indexOf("//") === 0 || href.indexOf("#") !== -1) return;
      if ((a.target && a.target !== "_self") || a.hasAttribute("download")) return;
      if (SKIP.test(href.split("?")[0]) || seen[href]) return;
      seen[href] = true;
      var link = document.createElement("link");
      link.rel = "prefetch";
      link.as = "document";
      link.href = href;
      document.head.appendChild(link);
    }

    var sidebar = document.getElementById("t58-sidebar");
    if (!sidebar) return;
    sidebar.addEventListener("mouseover", function (e) {
      var a = e.target.closest && e.target.closest("a.t58-nav-item[href]");
      if (!a) return;
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(function () { prefetch(a); }, 90);
    });
    sidebar.addEventListener("mouseout", function () { clearTimeout(hoverTimer); });
    sidebar.addEventListener("touchstart", function (e) {
      var a = e.target.closest && e.target.closest("a.t58-nav-item[href]");
      if (a) prefetch(a);
    }, { passive: true });
  }

  window.T58Chrome = { animateNumbers: animateNumbers };

  document.addEventListener("DOMContentLoaded", function () {
    buildStepper();
    restoreNavGroups();
    animateNumbers();
    initThemeToggle();
    initProfileAvatar();
    initGlobalSearch();
    initMobileMenu();
    initInstantNav();
  });
})();

/* Global search box (top of sidebar) -- fans out across strategies,
   datasets, reports, and runs via GET /api/global-search?q=..., the
   same app.search.global_search backend the desktop app's own top-bar
   search box already uses. Debounced, and closes on click-away or
   Escape. Purely a convenience layer: a fetch failure just shows "no
   matches" rather than an error dialog. */
function initGlobalSearch() {
  var input = document.getElementById("t58-global-search");
  var resultsBox = document.getElementById("t58-global-search-results");
  if (!input || !resultsBox) return;

  var KIND_LABELS = { strategy: "STRATEGY", dataset: "DATASET", report: "REPORT", run: "RUN" };
  var debounceTimer = null;
  var currentRequestId = 0;

  function hideResults() {
    resultsBox.style.display = "none";
    resultsBox.innerHTML = "";
  }

  function renderResults(query, results) {
    resultsBox.innerHTML = "";
    if (!results.length) {
      var empty = document.createElement("div");
      empty.style.cssText = "padding:12px 14px;font-size:12.5px;color:var(--text-dim);";
      empty.textContent = "No strategies, reports, datasets, or runs matched \"" + query + "\".";
      resultsBox.appendChild(empty);
      resultsBox.style.display = "block";
      return;
    }
    results.forEach(function (r) {
      var row = document.createElement(r.url ? "a" : "div");
      if (r.url) row.href = r.url, row.target = "_blank", row.rel = "noopener";
      row.style.cssText = "display:block;padding:9px 14px;font-size:12.5px;color:var(--text);" +
        "text-decoration:none;border-bottom:1px solid var(--border);cursor:" + (r.url ? "pointer" : "default") + ";";
      row.onmouseenter = function () { row.style.background = "var(--panel-3)"; };
      row.onmouseleave = function () { row.style.background = "transparent"; };
      var kindSpan = document.createElement("span");
      kindSpan.textContent = (KIND_LABELS[r.kind] || r.kind.toUpperCase()) + "  ";
      kindSpan.style.cssText = "font-size:10px;font-weight:700;letter-spacing:.04em;color:var(--teal);";
      var titleSpan = document.createElement("span");
      titleSpan.textContent = r.title;
      titleSpan.style.fontWeight = "600";
      var sub = document.createElement("div");
      sub.textContent = r.subtitle || "";
      sub.style.cssText = "font-size:11px;color:var(--text-muted);margin-top:2px;";
      row.appendChild(kindSpan);
      row.appendChild(titleSpan);
      row.appendChild(sub);
      resultsBox.appendChild(row);
    });
    resultsBox.style.display = "block";
  }

  input.addEventListener("input", function () {
    var query = input.value.trim();
    clearTimeout(debounceTimer);
    if (!query) { hideResults(); return; }
    debounceTimer = setTimeout(function () {
      var requestId = ++currentRequestId;
      fetch("/api/global-search?q=" + encodeURIComponent(query))
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (requestId !== currentRequestId) return; // a newer keystroke already superseded this request
          renderResults(query, data.results || []);
        })
        .catch(function () {
          if (requestId !== currentRequestId) return;
          renderResults(query, []);
        });
    }, 250);
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { input.blur(); hideResults(); }
  });

  document.addEventListener("click", function (e) {
    if (e.target !== input && !resultsBox.contains(e.target)) hideResults();
  });
}

/* Submits a ".mini-form" div (data-action="/some/route", optional
   data-confirm="...") as a real full-page POST navigation, by building a
   throwaway <form> directly on document.body and calling .submit() on it.
   Kept as a plain global (not inside the T58Chrome IIFE above) so it's
   reachable from inline onclick="" attributes on any page. See the
   T58SubmitMiniForm block comment in index.html/search.html for why these
   controls are plain divs instead of nested <form> elements. */
function t58SubmitMiniForm(wrap) {
  if (!wrap) return;
  var action = wrap.getAttribute("data-action");
  if (!action) return;
  var confirmMsg = wrap.getAttribute("data-confirm");
  if (confirmMsg && !window.confirm(confirmMsg)) return;

  var form = document.createElement("form");
  form.method = "post";
  form.action = action;
  form.style.display = "none";

  wrap.querySelectorAll("input[name], select[name], textarea[name]").forEach(function (el) {
    if (el.type === "checkbox" || el.type === "radio") {
      if (!el.checked) return;
      var hidden = document.createElement("input");
      hidden.type = "hidden";
      hidden.name = el.name;
      hidden.value = el.value || "1";
      form.appendChild(hidden);
      return;
    }
    if (el.tagName === "SELECT" && el.multiple) {
      Array.prototype.forEach.call(el.selectedOptions, function (opt) {
        var hidden = document.createElement("input");
        hidden.type = "hidden";
        hidden.name = el.name;
        hidden.value = opt.value;
        form.appendChild(hidden);
      });
      return;
    }
    var hidden = document.createElement("input");
    hidden.type = "hidden";
    hidden.name = el.name;
    hidden.value = el.value;
    form.appendChild(hidden);
  });

  document.body.appendChild(form);
  form.submit();
}

/* Shared "Detect pip size from data" handler -- one implementation for
   every page's own button instead of each page (Speed Run, WFO, WFGA,
   CPCV, PBO, Sensitivity, Parameter Robustness, Regime Matrix,
   Multi-Objective, Payout Probability, Overnight Autopilot, Prop-Firm
   Recommender, and the three multi-instrument pages) re-implementing the
   same fetch('/data/detect-pip-size') call (search.html and
   evolution.html already had their own copy of this before this pass --
   see t58DetectPipSize below, which is exactly their logic, generalized).

   opts:
     statusId   - id of a <p>/<span> to show progress/result/error in
     pipFieldId - id of the numeric <input name="pip_size"> to fill in
     fileSelector    - CSS selector for the page's csv_file <input type=file> (optional)
     datasetSelector - CSS selector for a single <select name="existing_dataset"> (optional)
     datasetValue    - a literal dataset name/path to use instead of reading a <select>
                        (e.g. the first checked box in a multi-instrument checklist) (optional)
   At least one dataset source must resolve to something at click time. */
function t58DetectPipSize(opts) {
  var statusEl = document.getElementById(opts.statusId);
  var pipField = document.getElementById(opts.pipFieldId);
  if (!statusEl || !pipField) return;

  var fd = new FormData();
  var haveSource = false;

  var fileInput = opts.fileSelector ? document.querySelector(opts.fileSelector) : null;
  if (fileInput && fileInput.files && fileInput.files.length) {
    for (var i = 0; i < fileInput.files.length; i++) fd.append('csv_file', fileInput.files[i]);
    haveSource = true;
  }

  var datasetValue = opts.datasetValue;
  if (!datasetValue && opts.datasetSelector) {
    var datasetSelect = document.querySelector(opts.datasetSelector);
    if (datasetSelect && datasetSelect.value) datasetValue = datasetSelect.value;
  }
  if (datasetValue) {
    fd.append('existing_dataset', datasetValue);
    haveSource = true;
  }

  if (!haveSource) {
    statusEl.style.color = '#ffcf7a';
    statusEl.textContent = 'Select a market data CSV or a stored dataset above first.';
    return;
  }

  statusEl.style.color = '#888';
  statusEl.textContent = 'Detecting...';
  fetch('/data/detect-pip-size', { method: 'POST', body: fd })
    .then(function (r) { return r.json().then(function (data) { return { ok: r.ok, data: data }; }); })
    .then(function (res) {
      if (!res.ok || res.data.error) {
        statusEl.style.color = '#ff9d9d';
        statusEl.textContent = (res.data && res.data.error) || "Couldn't detect pip size.";
        return;
      }
      pipField.value = res.data.pip_size;
      statusEl.style.color = '#b4ffcb';
      statusEl.textContent = res.data.message;
    })
    .catch(function (err) {
      statusEl.style.color = '#ff9d9d';
      statusEl.textContent = "Couldn't detect: " + err;
    });
}

/* UPGRADE (instrument-metadata-had-no-ui, web app): web-side counterpart
   to the desktop app's "Or pick a known instrument (auto-fills pip size +
   $/point)" dropdown (see app.ui.main_window.MainWindow/RunContextPanel's
   _apply_instrument_spec). Fetches app.data.instrument_specs.
   KNOWN_INSTRUMENTS once (via /data/instrument-specs) and caches it on
   window, since it only changes with a new app release -- every picker on
   every page shares the one fetch rather than each re-requesting it. */
var _t58InstrumentSpecsCache = null;
function _t58FetchInstrumentSpecs(cb) {
  if (_t58InstrumentSpecsCache) { cb(_t58InstrumentSpecsCache); return; }
  fetch('/data/instrument-specs')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      _t58InstrumentSpecsCache = data.instruments || [];
      cb(_t58InstrumentSpecsCache);
    })
    .catch(function () { cb([]); });
}

/* Populates a <select> (by id) with the known instruments, once, on page load. Instruments are grouped (stock index,
   micro stock index, metals & energy, currencies, rates), each row reads "ES  E-mini S&P 500", and a small strip under
   the dropdown shows the picked contract's exchange, $ per point, pip size and tick. Keeps the first "(none -- set
   manually)" option in place if the select already has one. */
function _t58InstCss() {
  if (document.getElementById('t58-inst-css')) return;
  var st = document.createElement('style');
  st.id = 't58-inst-css';
  st.textContent =
    'select.t58-inst-select optgroup{font-weight:700;font-style:normal;color:var(--teal,#35e0b0)}' +
    'select.t58-inst-select option{font-weight:400;color:var(--text,#e8ecf3);background:var(--pop-bg,#0c1c22)}' +
    '.t58-inst-sum{display:none;flex-wrap:wrap;gap:6px;margin:8px 0 2px;font-size:12px}.t58-inst-sum.on{display:flex}' +
    '.t58-inst-chip{display:inline-flex;gap:5px;align-items:baseline;padding:3px 9px;border-radius:999px;border:1px solid var(--border-light,#2a3142);background:var(--panel-3,#151a24);color:var(--text,#e8ecf3);white-space:nowrap}' +
    '.t58-inst-chip b{font-weight:700}.t58-inst-chip span{color:var(--text-muted,#8a93a6);font-size:11px}';
  document.head.appendChild(st);
}

function t58PopulateInstrumentPicker(selectId) {
  var select = document.getElementById(selectId);
  if (!select) return;
  _t58InstCss();
  _t58FetchInstrumentSpecs(function (instruments) {
    if (select.querySelector('optgroup')) return;
    select.classList.add('t58-inst-select');
    var order = ['Stock index futures', 'Micro stock index futures', 'Metals & energy', 'Currencies', 'Interest rates', 'Other'];
    var groups = {};
    instruments.forEach(function (inst) {
      var cat = inst.category || 'Other';
      if (!groups[cat]) { groups[cat] = document.createElement('optgroup'); groups[cat].label = cat; }
      var opt = document.createElement('option');
      opt.value = inst.symbol;
      opt.textContent = inst.symbol + '  \u2014  ' + inst.description.replace(/ futures$/i, '');
      opt.title = inst.description + ' (' + inst.exchange + ')';
      groups[cat].appendChild(opt);
    });
    order.concat(Object.keys(groups)).forEach(function (cat) {
      if (groups[cat] && !groups[cat].parentNode) select.appendChild(groups[cat]);
    });
    var box = document.createElement('div');
    box.className = 't58-inst-sum';
    box.setAttribute('aria-live', 'polite');
    select.parentNode.insertBefore(box, select.nextSibling);
    function chip(label, val) { return '<span class="t58-inst-chip"><span>' + label + '</span><b>' + val + '</b></span>'; }
    select.addEventListener('change', function () {
      var spec = null;
      for (var i = 0; i < instruments.length; i++) if (instruments[i].symbol === select.value) { spec = instruments[i]; break; }
      if (!spec) { box.className = 't58-inst-sum'; box.innerHTML = ''; return; }
      box.innerHTML = chip('Exchange', spec.exchange) + chip('Per point', '$' + spec.contract_size) +
        chip('Pip size', spec.pip_size) + chip('Tick', spec.tick_size + ' = $' + spec.tick_value) +
        chip('Commission', '$' + spec.default_commission_round_turn + ' round-turn');
      box.className = 't58-inst-sum on';
    });
  });
}

/* opts:
     selectEl          - the <select> element the person just changed (this)
     pipFieldId        - id of the numeric <input name="pip_size"> to fill in
     contractFieldId   - id of the numeric <input name="contract_size"> to fill in
     commissionFieldId - optional id of the numeric <input name="commission"> to fill in
     statusId          - optional id of a <p>/<span> to show a confirmation in
   Mirrors MainWindow._apply_instrument_spec exactly: sets pip_size AND
   contract_size ($/point) from the picked symbol's spec, leaving every
   other risk field untouched. A blank/"(none)" selection does nothing.
   COMMISSION-DEFAULT-UPGRADE (2026-09-26): when commissionFieldId is
   given, also fills a realistic round-turn commission -- but ONLY when
   that field is still at its 0/blank default, exactly like the desktop
   app's own _apply_instrument_spec -- never overwrites a rate the user
   already typed in. This closes the "$0 commission on a known futures
   instrument" gap for every web page that offers this dropdown. */
function t58ApplyInstrumentSpec(opts) {
  var symbol = opts.selectEl && opts.selectEl.value;
  if (!symbol) return;
  var pipField = document.getElementById(opts.pipFieldId);
  var contractField = opts.contractFieldId ? document.getElementById(opts.contractFieldId) : null;
  var commissionField = opts.commissionFieldId ? document.getElementById(opts.commissionFieldId) : null;
  var statusEl = opts.statusId ? document.getElementById(opts.statusId) : null;
  _t58FetchInstrumentSpecs(function (instruments) {
    var spec = null;
    for (var i = 0; i < instruments.length; i++) {
      if (instruments[i].symbol === symbol) { spec = instruments[i]; break; }
    }
    if (!spec) return;
    if (pipField) pipField.value = spec.pip_size;
    if (contractField) contractField.value = spec.contract_size;
    var commissionNote = '';
    if (commissionField && [0, 3].indexOf(parseFloat(commissionField.value) || 0) >= 0) {
      commissionField.value = spec.default_commission_round_turn;
      commissionNote = ', commission=$' + spec.default_commission_round_turn + '/trade';
    }
    if (statusEl) {
      statusEl.style.color = '#b4ffcb';
      statusEl.textContent = 'Applied ' + spec.symbol + ' (' + spec.description + ', ' + spec.exchange + '): '
        + 'pip_size=' + spec.pip_size + ', $' + spec.contract_size + '/point per contract' + commissionNote + '.';
    }
  });
}

/* Checks or unchecks every checkbox inside the given container id --
   backs the "Select All" / "Clear All" buttons added to the Evolution
   Lab (and multi-instrument Evolution Lab) family checklists. */
function t58SetCheckboxes(containerId, checked) {
  var container = document.getElementById(containerId);
  if (!container) return;
  container.querySelectorAll('input[type=checkbox]').forEach(function (cb) {
    cb.checked = checked;
  });
}

/* ---- Verdict "why?" popup -------------------------------------------------
 * Any element carrying data-ve='{"verdict":..,"title":..,"summary":..,"points":[..]}'
 * (see app/scoring/verdict_explain.py) opens a small plain-English box on
 * click / Enter / Space. Esc, outside-click or the x closes it. Event
 * delegation, so it also works for rows repainted by the dashboard poll. */
(function () {
  var box = null, lastFocus = null;
  function css() {
    if (document.getElementById("t58-ve-css")) return;
    var s = document.createElement("style");
    s.id = "t58-ve-css";
    s.textContent =
      "[data-ve]{cursor:pointer}[data-ve]:hover{filter:brightness(1.2)}" +
      "[data-ve]:focus-visible{outline:2px solid var(--teal,#35e0b0);outline-offset:2px}" +
      ".t58-ve{position:absolute;z-index:9999;width:min(320px,calc(100vw - 24px));padding:14px 16px;" +
      "background:var(--pop-bg,#0c1c22);color:var(--text,#e8ecf3);border:1px solid var(--pop-bd,#2a3142);" +
      "border-radius:10px;box-shadow:0 14px 36px rgba(0,0,0,.6);font-size:13px;line-height:1.45}" +
      ".t58-ve h4{margin:0 22px 6px 0;font-size:14px}.t58-ve p{margin:0 0 8px;color:var(--text-dim,#9aa4b8)}" +
      ".t58-ve ul{margin:0;padding-left:18px}.t58-ve li{margin:0 0 5px}" +
      ".t58-ve .x{position:absolute;top:6px;right:8px;background:none;border:0;color:inherit;font-size:18px;cursor:pointer}" +
      ".t58-ve.READY h4{color:var(--ok,#35e0b0)}.t58-ve.MARGINAL h4{color:var(--amber,#f5b83d)}" +
      ".t58-ve.NOT h4{color:var(--coral,#ff6f6f)}";
    document.head.appendChild(s);
  }
  function close() {
    if (box) { box.remove(); box = null; }
    if (lastFocus && lastFocus.focus) { try { lastFocus.focus(); } catch (e) {} }
    lastFocus = null;
  }
  function esc(t) { var d = document.createElement("div"); d.textContent = String(t == null ? "" : t); return d.innerHTML; }
  function open(el) {
    var d;
    try { d = JSON.parse(el.getAttribute("data-ve")); } catch (e) { return; }
    if (!d) return;
    var wasSame = box && box._src === el;
    close();
    if (wasSame) return;
    css();
    lastFocus = el;
    box = document.createElement("div");
    box._src = el;
    box.className = "t58-ve " + (d.verdict === "NOT READY" ? "NOT" : d.verdict);
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-label", d.title || "Verdict explanation");
    var pts = (d.points || []).map(function (p) { return "<li>" + esc(p) + "</li>"; }).join("");
    box.innerHTML = '<button class="x" aria-label="Close">×</button><h4>' + esc(d.title) + "</h4><p>" +
      esc(d.summary) + "</p>" + (pts ? "<ul>" + pts + "</ul>" : "");
    document.body.appendChild(box);
    var r = el.getBoundingClientRect(), w = box.offsetWidth;
    var left = Math.max(12, Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - w - 12));
    box.style.left = left + "px";
    box.style.top = (window.scrollY + r.bottom + 8) + "px";
    box.querySelector(".x").addEventListener("click", function (e) { e.stopPropagation(); close(); });
  }
  document.addEventListener("click", function (e) {
    var el = e.target.closest && e.target.closest("[data-ve]");
    if (el) { e.preventDefault(); e.stopPropagation(); open(el); return; }
    if (box && !box.contains(e.target)) close();
  }, true);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { close(); return; }
    if ((e.key === "Enter" || e.key === " ") && e.target.closest && e.target.closest("[data-ve]")) {
      e.preventDefault(); open(e.target.closest("[data-ve]"));
    }
  });
  window.addEventListener("resize", close);
})();


/* One instrument picker per form (2026-10-28).
 * A page can end up with up to three ways to choose an instrument and auto-fill pip size: its own
 * "*_instrument_select" (next to the Detect pip size button), the shared Accuracy-settings "Instrument (pick first)"
 * box, and the prop-firm block's injected "#lc-instrument-select". They did the same job, so users saw it twice.
 * Keep exactly one per form, in this order of preference: page picker > prop-firm picker > accuracy box.
 * Hidden controls stay in the DOM (the server still reads acc_instrument when it is filled in). */
function t58DedupeInstrumentPickers() {
  document.querySelectorAll('form').forEach(function (form) {
    var page = form.querySelector('select[id$="instrument_select"]:not(#lc-instrument-select)');
    var lc = form.querySelector('#lc-instrument-select');
    var acc = form.querySelector('#acc_instrument');
    function hideAcc() {
      if (!acc) return;
      var box = acc.closest('div');
      if (box && !box.getAttribute('data-t58-dedup')) { box.setAttribute('data-t58-dedup', '1'); box.style.display = 'none'; }
    }
    function hideLc() {
      form.querySelectorAll('.lc-instr-part, #lc-instrument-select, label[for="lc-instrument-select"]').forEach(function (e) { e.style.display = 'none'; });
    }
    if (page) { hideAcc(); if (lc) hideLc(); }
    else if (lc) { hideAcc(); }
    /* keep the hidden accuracy box in step with whichever picker is visible, so the server's mismatch check still works */
    var keep = page || lc;
    if (keep && acc && !keep._t58AccSync) {
      keep._t58AccSync = true;
      keep.addEventListener('change', function () {
        var opt = keep.options && keep.options[keep.selectedIndex];
        var sym = opt ? (opt.value || '').split(/[\s(]/)[0] : '';
        if (sym && /^[A-Za-z0-9_.!-]{1,12}$/.test(sym)) acc.value = sym;
      });
    }
  });
}
window.addEventListener('load', function () { t58DedupeInstrumentPickers(); setTimeout(t58DedupeInstrumentPickers, 600); });
