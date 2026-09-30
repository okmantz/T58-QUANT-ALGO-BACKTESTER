/* ==========================================================================
   T58 first-run guided tour (web)

   A short, skippable, 7-step walkthrough shown once, the first time someone
   lands on the Dashboard after activating their license. It spotlights the
   real sidebar items ("here is where you add data", "here is where you
   test") and explains the order to do things in.

   - Self-contained: injects its own CSS, reads the app's theme variables
     (--panel-2, --teal, ...) so it follows the light/dark toggle.
   - Shows once per browser (localStorage key below). Finishing, skipping,
     or closing it all count as "seen". If localStorage is blocked it never
     auto-starts, so nobody gets nagged on every page load.
   - Replay anytime: any element with [data-t58-tour-start], the URL
     /dashboard?tour=1, or window.T58Tour.start() in the console.
   - No innerHTML anywhere: all text is set via textContent, so step copy
     can never inject markup. "**bold**" in step text renders as <strong>.
   - Keyboard: Right/Left arrows = next/back, Esc = skip.
   ========================================================================== */
(function () {
  "use strict";

  var STORE_KEY = "t58-tour-v1";
  var Z = 10050;
  var MOBILE_QUERY = "(max-width: 760px)";

  /* ---- helpers to find real sidebar elements by their link ------------- */
  function navLink(href) {
    return document.querySelector('.t58-sidebar a.t58-nav-item[href="' + href + '"]');
  }
  function navGroup(href) {
    var a = navLink(href);
    return a ? a.closest("details.t58-nav-group") : null;
  }

  /* ---- the steps ------------------------------------------------------- */
  // target: () => element to spotlight (null/undefined = centered card, no spotlight)
  // group:  () => sidebar <details> that must be open for the target to be visible
  var STEPS = [
    {
      title: "Welcome to T58 \uD83D\uDC4B",
      text: [
        "T58 answers one question: **if you trade this strategy under your prop firm's rules, what are your odds of passing the evaluation and reaching a payout?**",
        "Getting to that answer takes five simple moves. This quick tour shows you where each one lives."
      ],
      chips: ["Data", "Strategy", "Test", "Full Pipeline", "Verdict"]
    },
    {
      title: "1 \u00B7 Add your market data",
      text: [
        "Every test needs price history. Open **Data Center** to import a file (CSV, TSV or Parquet). Or skip ahead and upload straight into box 1 on Run & Report.",
        "Either way it's saved, so you only upload once. Columns needed: **timestamp, open, high, low, close, volume**."
      ],
      target: function () { return navLink("/data-center"); },
      group: function () { return navGroup("/data-center"); },
      action: { label: "Open Data Center", href: "/data-center" }
    },
    {
      title: "2 \u00B7 Create or bring a strategy",
      text: [
        "**No idea yet?** Let T58 hunt for one: Forge Strategy or Speed Run, one button each.",
        "**Have an idea?** Describe it in plain English with Generate Strategies (AI, optional), or build it yourself. Run & Report takes a no-code Manual builder, Python, PineScript or MQL5.",
        "Everything you save lands in the **Strategy Library**."
      ],
      target: function () { return navGroup("/forge"); },
      group: function () { return navGroup("/forge"); },
      action: { label: "Open Forge Strategy", href: "/forge" }
    },
    {
      title: "3 \u00B7 Run your first test",
      text: [
        "**Run & Report** is your first stop. Pick your data, pick your strategy, then set your prop-firm rules. FTMO, Apex, TopStep, The5%ers, FundedNext and Lucid are built in as quick-fill presets.",
        "Set your risk, hit Run, and you get a report with your **probability of passing and reaching a payout**."
      ],
      target: function () { return navLink("/"); },
      group: function () { return navGroup("/"); },
      action: { label: "Open Run & Report", href: "/" }
    },
    {
      title: "4 \u00B7 Full Pipeline: the deep test",
      text: [
        "Same inputs, but **one button does everything**: baseline backtest, lookahead-bias check, a search for a version that holds up out-of-sample, Monte Carlo, then a plain verdict: **READY, MARGINAL or NOT READY**, with reasons.",
        "It's the slowest tool in the app, so try a quick Run & Report first. Winners are saved to your Strategy Library automatically."
      ],
      target: function () { return navLink("/full-pipeline"); },
      group: function () { return navGroup("/full-pipeline"); },
      action: { label: "Open Full Pipeline", href: "/full-pipeline" }
    },
    {
      title: "Your home base: what's next?",
      text: [
        "This panel is the Dashboard's headline. After every run it tells you **what you're working on, whether it's working, why, and the exact next step**, so you never have to guess the order.",
        "Your saved strategies live in Create \u2192 Strategy Library."
      ],
      target: function () {
        return document.querySelector(".t58-champion") || document.querySelector(".t58-page-header");
      },
      scroll: "start" // big panel in the main area: park it at the top so the card fits below
    },
    {
      title: "Help is always one click away",
      text: [
        "The **User Manual** is the full walkthrough. Every section in the sidebar also has its own \uD83D\uDCA1 **Start Here** page, and Education and Resources cover the trading basics.",
        "Want this tour again? Use the **Take the tour** button at the top of the Dashboard."
      ],
      target: function () { return navLink("/user-manual"); },
      action: { label: "Start my first test \u2192", href: "/", primary: true }
    }
  ];

  /* ---- state ----------------------------------------------------------- */
  var state = null; // { i, opened:[details], cleanup:[fn], els:{}, prevFocus, openedDrawer }

  function seen() {
    try { return !!localStorage.getItem(STORE_KEY); } catch (e) { return true; }
  }
  function markSeen(how) {
    try { localStorage.setItem(STORE_KEY, how || "done"); } catch (e) {}
  }
  function isMobile() {
    return !!(window.matchMedia && window.matchMedia(MOBILE_QUERY).matches);
  }

  /* ---- CSS (injected once) --------------------------------------------- */
  function injectCss() {
    if (document.getElementById("t58-tour-css")) return;
    var css = [
      ".t58-tour-block{position:fixed;inset:0;z-index:" + (Z - 2) + ";background:transparent;}",
      ".t58-tour-spot{position:fixed;z-index:" + (Z - 1) + ";pointer-events:none;border-radius:10px;",
      "  box-shadow:0 0 0 9999px rgba(3,6,10,.74),0 0 0 2px var(--teal,#35e0b0),0 0 26px rgba(var(--teal-rgb,53,224,176),.55);",
      "  transition:top .28s ease,left .28s ease,width .28s ease,height .28s ease,opacity .2s ease;}",
      ".t58-tour-spot.t58-tour-nospot{box-shadow:0 0 0 9999px rgba(3,6,10,.74);width:0;height:0;top:50%;left:50%;}",
      ".t58-tour-card{position:fixed;z-index:" + Z + ";width:min(380px,calc(100vw - 24px));box-sizing:border-box;",
      "  background:var(--panel-2,#10141c);color:var(--text,#e7ebf2);border:1px solid var(--teal,#35e0b0);",
      "  border-radius:14px;padding:16px 18px 14px;box-shadow:0 18px 50px rgba(0,0,0,.55);",
      "  font-size:13.5px;line-height:1.55;transition:top .28s ease,left .28s ease;}",
      ".t58-tour-card *{box-sizing:border-box;}",
      ".t58-tour-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}",
      ".t58-tour-step{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--teal,#35e0b0);}",
      ".t58-tour-x{background:none;border:0;color:var(--text-muted,#8a93a6);font-size:20px;line-height:1;cursor:pointer;padding:0 2px;}",
      ".t58-tour-x:hover{color:var(--text,#e7ebf2);}",
      ".t58-tour-title{margin:0 0 8px;font-size:16px;font-weight:700;color:var(--text,#e7ebf2);}",
      ".t58-tour-card p{margin:0 0 9px;color:var(--text,#e7ebf2);}",
      ".t58-tour-card strong{color:var(--teal,#35e0b0);font-weight:700;}",
      ".t58-tour-chips{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:4px 0 10px;}",
      ".t58-tour-chip{padding:4px 10px;border-radius:999px;border:1px solid var(--border-light,#2a3242);",
      "  background:var(--panel-3,#151a24);font-size:12px;font-weight:600;}",
      ".t58-tour-arrow{color:var(--text-muted,#8a93a6);font-size:12px;}",
      ".t58-tour-dots{display:flex;gap:6px;margin:12px 0 12px;}",
      ".t58-tour-dot{width:8px;height:8px;border-radius:50%;padding:0;border:0;cursor:pointer;background:var(--border-light,#2a3242);}",
      ".t58-tour-dot.on{background:var(--teal,#35e0b0);box-shadow:0 0 8px rgba(var(--teal-rgb,53,224,176),.7);}",
      ".t58-tour-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}",
      ".t58-tour-actions .t58-tour-spacer{flex:1;}",
      ".t58-tour-btn{font:inherit;font-size:12.5px;font-weight:700;padding:8px 14px;border-radius:9px;cursor:pointer;",
      "  border:1px solid var(--border-light,#2a3242);background:transparent;color:var(--text,#e7ebf2);text-decoration:none;display:inline-block;}",
      ".t58-tour-btn:hover{border-color:var(--teal,#35e0b0);}",
      ".t58-tour-btn.primary{background:var(--teal,#35e0b0);border-color:var(--teal,#35e0b0);color:#04120e;}",
      ".t58-tour-btn.link{border-color:transparent;color:var(--teal,#35e0b0);padding-left:0;padding-right:6px;}",
      ".t58-tour-btn.quiet{border-color:transparent;color:var(--text-muted,#8a93a6);font-weight:600;padding-left:4px;padding-right:4px;}",
      ".t58-tour-btn:focus-visible,.t58-tour-x:focus-visible,.t58-tour-dot:focus-visible{outline:2px solid var(--teal,#35e0b0);outline-offset:2px;}",
      "@media (max-width:760px){.t58-tour-card{left:10px!important;right:10px;top:auto!important;width:auto;",
      "  bottom:calc(10px + env(safe-area-inset-bottom,0px));padding:14px 16px 12px;max-height:62vh;overflow-y:auto;}}",
      "@media (prefers-reduced-motion:reduce){.t58-tour-spot,.t58-tour-card{transition:none;}}"
    ].join("\n");
    var style = document.createElement("style");
    style.id = "t58-tour-css";
    style.textContent = css;
    document.head.appendChild(style);
  }

  /* ---- tiny DOM builders (no innerHTML) --------------------------------- */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  // "plain **bold** plain" -> text nodes + <strong>
  function rich(parent, str) {
    var parts = String(str).split("**");
    for (var k = 0; k < parts.length; k++) {
      if (!parts[k]) continue;
      if (k % 2 === 1) parent.appendChild(el("strong", null, parts[k]));
      else parent.appendChild(document.createTextNode(parts[k]));
    }
    return parent;
  }

  /* ---- sidebar group open/close bookkeeping ---------------------------- */
  function openGroup(details) {
    if (!details) return;
    if (!details.open) {
      details.open = true;
      state.opened.push(details);
    }
  }
  function closeOpenedExcept(keep) {
    var still = [];
    state.opened.forEach(function (d) {
      if (d === keep) { still.push(d); return; }
      d.open = false;
    });
    state.opened = still;
  }

  /* ---- mobile drawer --------------------------------------------------- */
  function drawerParts() {
    return {
      btn: document.getElementById("t58-mobile-menu-btn"),
      sidebar: document.getElementById("t58-sidebar")
    };
  }
  function ensureDrawer(open) {
    var p = drawerParts();
    if (!p.btn || !p.sidebar) return;
    var isOpen = p.sidebar.classList.contains("open");
    if (open && !isOpen) { p.btn.click(); state.openedDrawer = true; }
    if (!open && isOpen && state.openedDrawer) { p.btn.click(); state.openedDrawer = false; }
  }

  /* ---- geometry / positioning ------------------------------------------ */
  function targetRect(t) {
    var r = t.getBoundingClientRect();
    if (!r.width && !r.height) return null;
    return r;
  }

  function place() {
    if (!state) return;
    var t = state.target;
    var spot = state.els.spot, card = state.els.card;
    var r = t ? targetRect(t) : null;
    var mobile = isMobile();
    var vw = window.innerWidth, vh = window.innerHeight;
    var cw = card.offsetWidth, ch = card.offsetHeight, gap = 18, m = 12;

    // Where can the card go without covering `rc`? (desktop only)
    function candidates(rc) {
      var cy = rc.top + rc.height / 2 - ch / 2;
      var cx = rc.left + rc.width / 2 - cw / 2;
      return [
        { ok: rc.right + gap + cw <= vw - m, l: rc.right + gap, t: cy },   // right of it
        { ok: rc.bottom + gap + ch <= vh - m, l: cx, t: rc.bottom + gap }, // below it
        { ok: rc.top - gap - ch >= m, l: cx, t: rc.top - gap - ch },       // above it
        { ok: rc.left - gap - cw >= m, l: rc.left - gap - cw, t: cy }      // left of it
      ];
    }
    function firstFit(list) {
      for (var k = 0; k < list.length; k++) if (list[k].ok) return list[k];
      return null;
    }

    var pick = null;
    var box = r; // the region we actually spotlight
    if (r && !mobile) {
      pick = firstFit(candidates(r));
      if (!pick) {
        // Target is too big for the card to sit beside it: spotlight just its top part,
        // leaving room for the card underneath.
        var room = vh - m - ch - gap - r.top;
        if (room >= 60) {
          box = { top: r.top, left: r.left, width: r.width, height: Math.min(r.height, room),
                  right: r.right, bottom: r.top + Math.min(r.height, room) };
          pick = firstFit(candidates(box));
        }
      }
    }

    if (!box) {
      spot.classList.add("t58-tour-nospot");
      spot.style.top = spot.style.left = spot.style.width = spot.style.height = "";
    } else {
      var pad = 5;
      spot.classList.remove("t58-tour-nospot");
      spot.style.top = (box.top - pad) + "px";
      spot.style.left = (box.left - pad) + "px";
      spot.style.width = (box.width + pad * 2) + "px";
      spot.style.height = (box.height + pad * 2) + "px";
    }

    if (mobile) { // CSS pins the card to the bottom-sheet position
      card.style.left = card.style.top = "";
      return;
    }

    var left, top;
    if (pick) { left = pick.l; top = pick.t; }
    else if (!r) { left = (vw - cw) / 2; top = (vh - ch) / 2; }
    else { left = (vw - cw) / 2; top = vh - ch - m; } // last resort: bottom-center
    left = Math.min(Math.max(m, left), Math.max(m, vw - cw - m));
    top = Math.min(Math.max(m, top), Math.max(m, vh - ch - m));
    card.style.left = Math.round(left) + "px";
    card.style.top = Math.round(top) + "px";
  }

  // On phones the card is a bottom sheet; keep the target above it.
  function keepTargetAboveSheet() {
    if (!isMobile() || !state.target) return;
    var r = targetRect(state.target);
    if (!r) return;
    var limit = window.innerHeight - state.els.card.offsetHeight - 24;
    if (r.bottom > limit) {
      var scroller = state.target.closest(".t58-sidebar") || document.scrollingElement || document.documentElement;
      scroller.scrollTop += (r.bottom - limit);
    }
  }

  /* ---- rendering a step ------------------------------------------------ */
  function render() {
    var step = STEPS[state.i];
    var last = state.i === STEPS.length - 1;
    var card = state.els.card;
    while (card.firstChild) card.removeChild(card.firstChild);

    var top = el("div", "t58-tour-top");
    top.appendChild(el("span", "t58-tour-step", "Step " + (state.i + 1) + " of " + STEPS.length));
    var x = el("button", "t58-tour-x", "\u00D7");
    x.type = "button";
    x.setAttribute("aria-label", "Close tour");
    x.addEventListener("click", function () { finish("skipped"); });
    top.appendChild(x);
    card.appendChild(top);

    var h = el("h3", "t58-tour-title", step.title);
    h.id = "t58-tour-title";
    card.appendChild(h);

    step.text.forEach(function (p) { card.appendChild(rich(el("p"), p)); });

    if (step.chips) {
      var chips = el("div", "t58-tour-chips");
      step.chips.forEach(function (c, k) {
        if (k) chips.appendChild(el("span", "t58-tour-arrow", "\u2192"));
        chips.appendChild(el("span", "t58-tour-chip", c));
      });
      card.appendChild(chips);
    }

    var dots = el("div", "t58-tour-dots");
    STEPS.forEach(function (_s, k) {
      var d = el("button", "t58-tour-dot" + (k === state.i ? " on" : ""));
      d.type = "button";
      d.setAttribute("aria-label", "Go to step " + (k + 1));
      d.addEventListener("click", function () { go(k); });
      dots.appendChild(d);
    });
    card.appendChild(dots);

    var actions = el("div", "t58-tour-actions");
    if (step.action && !step.action.primary) {
      var a = el("a", "t58-tour-btn link", step.action.label + " \u2197");
      a.href = step.action.href;
      a.addEventListener("click", function () { markSeen("opened-page"); });
      actions.appendChild(a);
    }
    actions.appendChild(el("span", "t58-tour-spacer"));

    if (!last) {
      var skip = el("button", "t58-tour-btn quiet", "Skip");
      skip.type = "button";
      skip.addEventListener("click", function () { finish("skipped"); });
      actions.appendChild(skip);
    }
    if (state.i > 0) {
      var back = el("button", "t58-tour-btn", "Back");
      back.type = "button";
      back.addEventListener("click", function () { go(state.i - 1); });
      actions.appendChild(back);
    }
    if (last) {
      var done = el("button", "t58-tour-btn", "Finish");
      done.type = "button";
      done.addEventListener("click", function () { finish("done"); });
      actions.appendChild(done);
      if (step.action && step.action.primary) {
        var go1 = el("a", "t58-tour-btn primary", step.action.label);
        go1.href = step.action.href;
        go1.addEventListener("click", function () { markSeen("done"); });
        actions.appendChild(go1);
      }
    } else {
      var next = el("button", "t58-tour-btn primary", state.i === 0 ? "Show me around" : "Next");
      next.type = "button";
      next.setAttribute("data-t58-tour-next", "1");
      next.addEventListener("click", function () { go(state.i + 1); });
      actions.appendChild(next);
    }
    card.appendChild(actions);
  }

  function go(i) {
    if (!state) return;
    if (i < 0 || i >= STEPS.length) return;
    state.i = i;
    var step = STEPS[i];

    // Make sure the sidebar item we're pointing at is actually visible.
    var grp = step.group ? step.group() : null;
    closeOpenedExcept(grp);
    openGroup(grp);
    var needsSidebar = !!(step.target && step.target() && step.target().closest(".t58-sidebar"));
    if (isMobile()) ensureDrawer(needsSidebar);

    state.target = step.target ? step.target() : null;
    if (state.target && !targetRect(state.target)) state.target = null; // hidden -> centered card

    render();

    if (state.target) {
      var block = step.scroll || "center";
      try { state.target.scrollIntoView({ block: block, inline: "nearest" }); } catch (e) {}
      if (block === "start" && !state.target.closest(".t58-sidebar")) {
        try { window.scrollBy(0, -16); } catch (e) {} // breathing room above the spotlight
      }
    }
    keepTargetAboveSheet();
    place();
    // Layout can shift a frame later (details opening, fonts): re-place once more.
    window.requestAnimationFrame(function () { if (state) { keepTargetAboveSheet(); place(); } });

    var nextBtn = state.els.card.querySelector("[data-t58-tour-next]") ||
                  state.els.card.querySelector(".t58-tour-btn.primary");
    if (nextBtn) { try { nextBtn.focus({ preventScroll: true }); } catch (e) {} }
  }

  /* ---- lifecycle -------------------------------------------------------- */
  function onKey(e) {
    if (!state) return;
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish("skipped"); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(Math.min(state.i + 1, STEPS.length - 1)); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(Math.max(state.i - 1, 0)); }
    else if (e.key === "Tab") { // keep focus inside the card
      var f = state.els.card.querySelectorAll("a[href],button");
      if (!f.length) return;
      var first = f[0], lastEl = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); lastEl.focus(); }
      else if (!e.shiftKey && document.activeElement === lastEl) { e.preventDefault(); first.focus(); }
      else if (!state.els.card.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
    }
  }
  function onReflow() { if (state) { keepTargetAboveSheet(); place(); } }

  function start() {
    if (state) return;
    injectCss();
    state = {
      i: 0, opened: [], target: null, openedDrawer: false,
      prevFocus: document.activeElement,
      els: { block: el("div", "t58-tour-block"), spot: el("div", "t58-tour-spot t58-tour-nospot"), card: el("div", "t58-tour-card") }
    };
    state.els.card.setAttribute("role", "dialog");
    state.els.card.setAttribute("aria-modal", "true");
    state.els.card.setAttribute("aria-labelledby", "t58-tour-title");
    document.body.appendChild(state.els.block);
    document.body.appendChild(state.els.spot);
    document.body.appendChild(state.els.card);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", onReflow);
    window.addEventListener("scroll", onReflow, true);
    go(0);
  }

  function finish(how) {
    if (!state) return;
    markSeen(how);
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("resize", onReflow);
    window.removeEventListener("scroll", onReflow, true);
    closeOpenedExcept(null);
    if (isMobile()) ensureDrawer(false);
    ["block", "spot", "card"].forEach(function (k) {
      var n = state.els[k];
      if (n && n.parentNode) n.parentNode.removeChild(n);
    });
    var prev = state.prevFocus;
    state = null;
    if (prev && prev.focus) { try { prev.focus({ preventScroll: true }); } catch (e) {} }
  }

  /* ---- wiring ----------------------------------------------------------- */
  document.addEventListener("click", function (e) {
    var t = e.target && e.target.closest ? e.target.closest("[data-t58-tour-start]") : null;
    if (t) { e.preventDefault(); if (state) finish("done"); start(); }
  });

  window.T58Tour = {
    start: start,
    stop: function () { finish("skipped"); },
    reset: function () { try { localStorage.removeItem(STORE_KEY); } catch (e) {} },
    _steps: STEPS // exposed for tests
  };

  function autoStart() {
    var q = "";
    try { q = new URLSearchParams(window.location.search).get("tour"); } catch (e) {}
    if (q === "0") return;
    if (q === "1" || !seen()) {
      // let the sidebar finish restoring its own open/closed groups first
      window.setTimeout(start, 650);
    }
  }
  if (document.readyState === "complete") autoStart();
  else window.addEventListener("load", autoStart);
})();
