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
        if (icon) icon.innerHTML = "&#9728;"; // sun
        if (label) label.textContent = "Light theme";
      } else {
        document.documentElement.removeAttribute("data-theme");
        if (icon) icon.innerHTML = "&#9789;"; // moon
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
    { name: "Create", href: "/speed-run", paths: ["/speed-run", "/generate-strategies", "/research-agent"] },
    { name: "Test", href: "/", paths: ["/", "/payout-probability"] },
    { name: "Optimize", href: "/optimize", paths: ["/search", "/refine", "/full-pipeline", "/quick-optimize", "/multi-objective", "/evolution", "/optimize"] },
    { name: "Validate", href: "/validate", paths: ["/walk-forward-opt", "/walk-forward-ga", "/cpcv", "/pbo", "/sensitivity", "/parameter-robustness", "/regime-matrix", "/validate"] },
    { name: "Champion", href: "/family-diversity", paths: ["/portfolio", "/ensemble", "/family-diversity"] },
    { name: "Forward Test", href: "/forward-test", paths: ["/forward-test"] },
    { name: "Deploy", href: "/deploy-live", paths: ["/deploy-live"] },
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

  window.T58Chrome = { animateNumbers: animateNumbers };

  document.addEventListener("DOMContentLoaded", function () {
    buildStepper();
    restoreNavGroups();
    animateNumbers();
    initThemeToggle();
    initGlobalSearch();
    initMobileMenu();
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
