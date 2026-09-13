/*
 * Reusable "export this table to CSV" component.
 *
 * Closes the gap flagged against the app: CSV export existed in only two
 * places in the whole codebase, hand-wired per page, instead of being a
 * universal action every leaderboard/table gets "for free" (Search Lab
 * leaderboards, Monte Carlo distribution tables, Evolution Lab
 * generations, the new Strategy Compare table, etc.).
 *
 * USAGE -- two ways, both work with the exact same script:
 *
 * 1) Zero-JS, declarative (preferred for existing/new leaderboard tables):
 *    Add a `data-csv-export="some_filename"` attribute to any <table>:
 *        <table data-csv-export="search_lab_leaderboard">...</table>
 *    On page load, this script finds every such table and injects a small
 *    "Export CSV" button directly above it. Clicking it reads the table's
 *    current <thead>/<tbody> rows (whatever is in the DOM at click time --
 *    correct even for tables built/updated by JS after page load, e.g. a
 *    leaderboard re-sorted client-side) and downloads a CSV. No server
 *    round-trip, no backend route, works offline.
 *
 * 2) Programmatic, for data that isn't (only) in an HTML table -- e.g. a
 *    Monte Carlo return-distribution array kept in a JS variable:
 *        T58ExportCSV.downloadRows('mc_return_distribution.csv',
 *            ['simulation', 'return_pct'],
 *            distribution.map((v, i) => [i + 1, v]));
 *
 * Include this ONCE, in the shared sidebar/base template
 * (app/web/templates/_sidebar.html is already included by every page --
 * see this file's integration note in INTEGRATION.md), not per-page.
 */
(function () {
    "use strict";

    function csvEscape(value) {
        const s = value === null || value === undefined ? "" : String(value);
        if (/[",\n]/.test(s)) {
            return '"' + s.replace(/"/g, '""') + '"';
        }
        return s;
    }

    function rowsToCsv(headers, rows) {
        const lines = [headers.map(csvEscape).join(",")];
        for (const row of rows) {
            lines.push(row.map(csvEscape).join(","));
        }
        return lines.join("\r\n");
    }

    function triggerDownload(filename, csvText) {
        const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename.endsWith(".csv") ? filename : filename + ".csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function downloadRows(filename, headers, rows) {
        triggerDownload(filename, rowsToCsv(headers, rows));
    }

    function tableToRows(table) {
        const headerCells = table.querySelectorAll("thead th");
        const headers = headerCells.length
            ? Array.from(headerCells).map((th) => th.textContent.trim())
            : Array.from(table.querySelectorAll("tr:first-child th, tr:first-child td")).map((c) => c.textContent.trim());

        const bodyRows = table.querySelectorAll("thead") ? table.querySelectorAll("tbody tr") : table.querySelectorAll("tr");
        const rows = [];
        bodyRows.forEach((tr) => {
            const cells = tr.querySelectorAll("td, th");
            if (cells.length) {
                rows.push(Array.from(cells).map((c) => c.textContent.trim()));
            }
        });
        return { headers, rows };
    }

    function exportTable(table, filename) {
        const { headers, rows } = tableToRows(table);
        if (!rows.length) {
            alert("Nothing to export -- this table is empty.");
            return;
        }
        downloadRows(filename || "export.csv", headers, rows);
    }

    function makeButton(filename, table) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "\u2b07 Export CSV";
        btn.className = "t58-csv-export-btn";
        btn.style.cssText = "margin:6px 0;padding:4px 10px;font-size:12px;cursor:pointer;";
        btn.addEventListener("click", () => exportTable(table, filename));
        return btn;
    }

    function attachAll() {
        document.querySelectorAll("table[data-csv-export]").forEach((table) => {
            if (table.dataset.csvExportAttached) return;  // don't double-attach on repeated calls
            table.dataset.csvExportAttached = "1";
            const filename = table.getAttribute("data-csv-export") || "export";
            table.parentNode.insertBefore(makeButton(filename, table), table);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachAll);
    } else {
        attachAll();
    }

    // Re-scan on demand: call this after JS-rendering a new table (e.g. a
    // leaderboard that loads via fetch() after the page itself has loaded).
    window.T58ExportCSV = { downloadRows, exportTable, attachAll };
})();
