/*
 * T58 multi-select dropdown.
 *
 * Turns any container full of checkboxes into a compact dropdown:
 *
 *   <div id="evo-families-list" data-t58-multiselect
 *        data-noun="families" data-empty-text="None selected -- every family is searched">
 *     <div class="checkbox-row"><input type="checkbox" name="families" value="x" id="a"><label for="a">X</label></div>
 *     ...
 *   </div>
 *
 * The ORIGINAL checkbox inputs are kept (just moved into the dropdown panel),
 * so form submission, name/value pairs, and any page script that reads
 * `input[name=...]:checked` keep working exactly as before -- nothing on the
 * server side changes.
 *
 * Features: summary button ("12 of 40 selected"), Select all / Clear buttons,
 * a search box (Select all then applies to the rows currently shown),
 * group headings (.dataset-group-label) that hide when their rows are
 * filtered out, close on outside click / Escape.
 *
 * Optional attributes on the container:
 *   data-noun        plural noun for the summary        (default "items")
 *   data-empty-text  summary when nothing is checked    (default "None selected")
 *   data-searchable  "false" to hide the search box     (default: shown when > 8 rows)
 */
(function () {
  "use strict";

  var STYLE_ID = "t58-ms-style";
  var CSS = [
    ".t58-ms { position: relative; margin: 4px 0 8px; }",
    ".t58-ms-toggle { display: flex; align-items: center; justify-content: space-between; gap: 10px;",
    "  width: 100%; padding: 11px 12px; border-radius: 8px; cursor: pointer; text-align: left;",
    "  border: 1px solid var(--border-light, #333); background: var(--panel-3, #161b22);",
    "  color: var(--text, #e6edf3); font-size: 14px; font-family: inherit; margin: 0; }",
    ".t58-ms-toggle:hover, .t58-ms.open .t58-ms-toggle { border-color: var(--teal, #35e0b0); }",
    ".t58-ms-toggle:focus-visible { outline: none; box-shadow: 0 0 0 3px rgba(53,224,176,0.2); }",
    ".t58-ms-summary { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
    ".t58-ms-summary.empty { color: var(--text-muted, #9aa4b2); }",
    ".t58-ms-count { flex: 0 0 auto; min-width: 22px; padding: 1px 8px; border-radius: 999px; font-size: 12px;",
    "  font-weight: 700; text-align: center; background: var(--teal, #35e0b0); color: #04120e; }",
    ".t58-ms-count[hidden] { display: none; }",
    ".t58-ms-caret { flex: 0 0 auto; color: var(--text-muted, #9aa4b2); font-size: 12px; transition: transform .15s; }",
    ".t58-ms.open .t58-ms-caret { transform: rotate(180deg); }",
    ".t58-ms-panel { position: absolute; left: 0; right: 0; top: calc(100% + 4px); z-index: 60;",
    "  background: var(--panel-2, #10151c); border: 1px solid var(--border-light, #333); border-radius: 10px;",
    "  box-shadow: 0 12px 32px rgba(0,0,0,0.55); overflow: hidden; }",
    ".t58-ms-panel[hidden] { display: none; }",
    ".t58-ms-tools { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px; border-bottom: 1px solid var(--border, #262c36);",
    "  background: var(--panel-3, #161b22); }",
    ".t58-ms-search { flex: 1 1 140px; min-width: 0; width: auto !important; margin: 0 !important;",
    "  padding: 7px 10px !important; font-size: 13px !important; border-radius: 6px !important; }",
    ".t58-ms-btn { flex: 0 0 auto; margin: 0 !important; padding: 7px 12px; border-radius: 6px; font-size: 12px;",
    "  font-weight: 600; cursor: pointer; font-family: inherit; width: auto !important;",
    "  border: 1px solid var(--border-light, #333); background: var(--panel, #0d1117); color: var(--text, #e6edf3); }",
    ".t58-ms-btn:hover { border-color: var(--teal, #35e0b0); color: var(--teal, #35e0b0); }",
    ".t58-ms-btn.primary { background: var(--teal, #35e0b0); border-color: var(--teal, #35e0b0); color: #04120e; }",
    ".t58-ms-btn.primary:hover { color: #04120e; filter: brightness(1.08); }",
    ".t58-ms-list { max-height: 300px; overflow-y: auto; padding: 4px 6px 6px; }",
    /* neutralise page-level styling of the moved list (e.g. the bordered .dataset-list box) */
    ".t58-ms-panel .t58-ms-list.dataset-list { border: 0; border-radius: 0; }",
    ".t58-ms-list .checkbox-row, .t58-ms-list .dataset-row { display: flex; align-items: flex-start; gap: 9px;",
    "  padding: 7px 8px; margin: 0 !important; border-radius: 6px; background: none !important; border: 0 !important; }",
    ".t58-ms-list .checkbox-row:hover, .t58-ms-list .dataset-row:hover { background: var(--panel-3, #161b22) !important; }",
    ".t58-ms-list input[type=checkbox] { flex: 0 0 auto; width: 16px !important; height: 16px; margin: 2px 0 0 !important;",
    "  accent-color: var(--teal, #35e0b0); }",
    ".t58-ms-list label { margin: 0 !important; color: var(--text, #e6edf3) !important; font-size: 13px !important;",
    "  line-height: 1.4; cursor: pointer; white-space: normal; }",
    ".t58-ms-list .dataset-row { cursor: pointer; }",
    ".t58-ms-list .dataset-name { color: var(--text, #e6edf3) !important; word-break: break-word; }",
    ".t58-ms-list .dataset-size { color: var(--text-muted, #9aa4b2) !important; white-space: nowrap; }",
    ".t58-ms-list .dataset-group-label { padding: 6px 8px 2px; font-size: 11px; text-transform: uppercase;",
    "  letter-spacing: .05em; color: var(--text-muted, #9aa4b2) !important; margin: 8px 0 2px !important; }",
    ".t58-ms-none { padding: 14px; text-align: center; font-size: 13px; color: var(--text-muted, #9aa4b2); }",
    ".t58-ms-hidden { display: none !important; }"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
  }

  function rowFor(input, list) {
    var row = input.closest(".checkbox-row, .dataset-row, label");
    return row && list.contains(row) ? row : input.parentNode;
  }

  function build(list) {
    if (list.getAttribute("data-t58ms-ready")) return null;
    var inputs = Array.prototype.slice.call(list.querySelectorAll("input[type=checkbox]"));
    if (!inputs.length) return null;
    list.setAttribute("data-t58ms-ready", "1");

    var noun = list.getAttribute("data-noun") || "items";
    var emptyText = list.getAttribute("data-empty-text") || "None selected";
    var searchAttr = list.getAttribute("data-searchable");
    var searchable = searchAttr ? searchAttr !== "false" : inputs.length > 8;

    var wrap = document.createElement("div");
    wrap.className = "t58-ms";
    list.parentNode.insertBefore(wrap, list);

    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "t58-ms-toggle";
    toggle.setAttribute("aria-haspopup", "listbox");
    toggle.setAttribute("aria-expanded", "false");
    var summary = document.createElement("span");
    summary.className = "t58-ms-summary";
    var count = document.createElement("span");
    count.className = "t58-ms-count";
    var caret = document.createElement("span");
    caret.className = "t58-ms-caret";
    caret.setAttribute("aria-hidden", "true");
    caret.textContent = "\u25BE";
    toggle.appendChild(summary);
    toggle.appendChild(count);
    toggle.appendChild(caret);

    var panel = document.createElement("div");
    panel.className = "t58-ms-panel";
    panel.hidden = true;

    var tools = document.createElement("div");
    tools.className = "t58-ms-tools";
    var search = null;
    if (searchable) {
      search = document.createElement("input");
      search.type = "text";
      search.className = "t58-ms-search";
      search.placeholder = "Search " + noun + "\u2026";
      search.setAttribute("aria-label", "Search " + noun);
      tools.appendChild(search);
    }
    var selectAll = document.createElement("button");
    selectAll.type = "button";
    selectAll.className = "t58-ms-btn primary";
    selectAll.textContent = "Select all";
    var clear = document.createElement("button");
    clear.type = "button";
    clear.className = "t58-ms-btn";
    clear.textContent = "Clear";
    tools.appendChild(selectAll);
    tools.appendChild(clear);

    var none = document.createElement("div");
    none.className = "t58-ms-none";
    none.textContent = "No matches";
    none.hidden = true;

    list.classList.add("t58-ms-list");
    panel.appendChild(tools);
    panel.appendChild(list);
    panel.appendChild(none);
    wrap.appendChild(toggle);
    wrap.appendChild(panel);

    var rows = inputs.map(function (input) { return { input: input, row: rowFor(input, list) }; });

    function isShown(entry) { return !entry.row.classList.contains("t58-ms-hidden"); }

    function refresh() {
      var n = 0;
      rows.forEach(function (e) { if (e.input.checked) n++; });
      var total = rows.length;
      count.hidden = n === 0;
      count.textContent = String(n);
      if (n === 0) {
        summary.textContent = emptyText;
        summary.classList.add("empty");
      } else if (n === total) {
        summary.textContent = "All " + total + " " + noun + " selected";
        summary.classList.remove("empty");
      } else {
        summary.textContent = n + " of " + total + " " + noun + " selected";
        summary.classList.remove("empty");
      }
      summary.title = summary.textContent;
    }

    function applyFilter() {
      var q = search ? search.value.trim().toLowerCase() : "";
      var anyShown = false;
      rows.forEach(function (e) {
        var match = !q || (e.row.textContent || "").toLowerCase().indexOf(q) !== -1;
        e.row.classList.toggle("t58-ms-hidden", !match);
        if (match) anyShown = true;
      });
      // hide group headings that have no visible row after them
      var groups = list.querySelectorAll(".dataset-group-label");
      Array.prototype.forEach.call(groups, function (g) {
        var el = g.nextElementSibling, visible = false;
        while (el && !el.classList.contains("dataset-group-label")) {
          if (!el.classList.contains("t58-ms-hidden")) { visible = true; break; }
          el = el.nextElementSibling;
        }
        g.classList.toggle("t58-ms-hidden", !visible);
      });
      none.hidden = anyShown;
      selectAll.textContent = q ? "Select all shown" : "Select all";
    }

    function setChecked(checked, onlyShown) {
      rows.forEach(function (e) {
        if (onlyShown && !isShown(e)) return;
        if (e.input.checked !== checked) {
          e.input.checked = checked;
          e.input.dispatchEvent(new Event("change", { bubbles: true }));
        }
      });
      refresh();
    }

    function open() {
      panel.hidden = false;
      wrap.classList.add("open");
      toggle.setAttribute("aria-expanded", "true");
      if (search) search.focus();
    }
    function close() {
      panel.hidden = true;
      wrap.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    }

    toggle.addEventListener("click", function () { panel.hidden ? open() : close(); });
    selectAll.addEventListener("click", function () { setChecked(true, true); });
    clear.addEventListener("click", function () { setChecked(false, false); });
    if (search) {
      search.addEventListener("input", applyFilter);
      search.addEventListener("keydown", function (e) { if (e.key === "Enter") e.preventDefault(); });
    }
    list.addEventListener("change", refresh);
    document.addEventListener("click", function (e) { if (!wrap.contains(e.target)) close(); });
    wrap.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { close(); toggle.focus(); }
    });
    window.addEventListener("pageshow", refresh);

    refresh();
    applyFilter();
    return { refresh: refresh, open: open, close: close };
  }

  var registry = {};

  function initAll() {
    injectStyle();
    var lists = document.querySelectorAll("[data-t58-multiselect]");
    Array.prototype.forEach.call(lists, function (list) {
      var api = build(list);
      if (api && list.id) registry[list.id] = api;
    });
  }

  /* Re-sync the summary after page script changes checkboxes programmatically. */
  window.t58MultiSelectRefresh = function (id) {
    if (registry[id]) registry[id].refresh();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
