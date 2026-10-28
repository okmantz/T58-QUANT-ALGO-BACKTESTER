/*
 * Floating Project Chat widget -- client-side logic.
 * Server API: app/web/project_routes.py (all under /api/projects).
 *
 * State kept here is intentionally minimal: the server (via the Flask
 * session cookie) is the source of truth for "which project is active",
 * and each project's chat history is fully persisted server-side (see
 * app.orchestration.projects) -- this file just renders whatever the
 * server returns and never assumes anything survives a page reload on
 * its own.
 */
(function () {
  "use strict";

  var API_BASE = "/api/projects";
  var ACTIVITY_POLL_MS = 4000;
  // Per LOGIN-SESSION panel state (open/collapsed, active tab, which briefing sections are expanded). The server stamps each
  // login (a fresh browser session, activation, lock-unlock or log-out) with an id in data-login on the launcher. State is
  // only honoured while that id is unchanged, so: the briefing may pop open ONCE at the start of a login session, and once the
  // person collapses it (or opens it) that choice sticks across every page until they log off and back on.
  var SESS_KEY = "t58_pc_session";
  var sess = null;

  var state = {
    project: null,       // full project dict {id, name, chat_history, ...} or null
    projectList: [],      // [{id, name, updated_at, message_count}, ...]
    activityTimer: null,
  };

  var els = {};

  function $(id) { return document.getElementById(id); }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  function apiFetch(path, options) {
    options = options || {};
    options.headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
    return fetch(API_BASE + path, options).then(function (resp) {
      return resp.json().catch(function () { return {}; });
    });
  }

  // -------------------------------------------------------------------
  // Panel open/close
  // -------------------------------------------------------------------

  function loginId() {
    var l = document.getElementById("t58-pc-launcher");
    return (l && l.getAttribute("data-login")) || "none";
  }

  function loadSess() {
    var id = loginId();
    try {
      var raw = window.localStorage.getItem(SESS_KEY);
      var s = raw ? JSON.parse(raw) : null;
      if (s && s.login === id) { s.sections = s.sections || {}; return s; }
    } catch (e) { /* storage blocked or corrupt: start fresh */ }
    // A new login session: collapsed, nothing remembered, and the briefing is allowed to pop open once.
    return { login: id, open: false, tab: "chat", first: false, sections: {} };
  }

  function saveSess() {
    if (!sess) return;
    try { window.localStorage.setItem(SESS_KEY, JSON.stringify(sess)); } catch (e) { /* storage blocked: fine */ }
  }

  function rememberOpen(isOpen) {
    if (!sess) sess = loadSess();
    sess.open = !!isOpen;
    saveSess();
  }

  function wasOpen() {
    if (!sess) sess = loadSess();
    return !!sess.open;
  }

  function setLauncherState(isOpen) {
    els.launcher.classList.toggle("open", isOpen);
    els.launcher.setAttribute("aria-expanded", isOpen ? "true" : "false");
    var label = isOpen ? "Collapse Project Chat" : "Open Project Chat";
    els.launcher.title = label;
    els.launcher.setAttribute("aria-label", label);
    if (els.launcherIcon) els.launcherIcon.innerHTML = isOpen ? "&#9662;" : "&#128172;";
  }

  function openPanel() {
    els.panel.hidden = false;
    setLauncherState(true);
    rememberOpen(true);
    refreshProjectList().then(function () {
      // Keep whichever tab was showing (Brief / Activity / Chat) -- never yank the person back to Chat.
      showTab((sess && sess.tab) || "chat");
    });
  }

  function closePanel() {
    els.panel.hidden = true;
    setLauncherState(false);
    rememberOpen(false);
    stopActivityPolling();
  }

  // Clicking the floating bubble collapses the panel when it is open and expands it when it is not.
  function togglePanel() {
    if (els.panel.hidden) { openPanel(); } else { closePanel(); }
  }

  function showTab(name) {
    if (!sess) sess = loadSess();
    sess.tab = name;
    saveSess();
    var tabs = els.panel.querySelectorAll(".t58-pc-tab");
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].classList.toggle("active", tabs[i].getAttribute("data-tab") === name);
    }
    els.tabChat.hidden = name !== "chat";
    els.tabActivity.hidden = name !== "activity";
    if (els.tabBrief) els.tabBrief.hidden = name !== "brief";
    els.empty.hidden = name === "brief" || !!state.project || (state.projectList || []).length > 0;
    if (name === "brief") { loadBrief(); startBriefPolling(); } else { stopBriefPolling(); }
    if (name === "activity" && state.project) {
      refreshActivity();
      startActivityPolling();
    } else {
      stopActivityPolling();
    }
  }

  // -------------------------------------------------------------------
  // Morning Briefing (/api/briefing) -- built only from saved results; collapsible sections, refreshed through the day
  // -------------------------------------------------------------------
  var briefTimer = null, briefEnabled = true;
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  function renderBrief(d) {
    var box = document.getElementById("t58-pc-brief"); if (!box) return;
    if (!sess) sess = loadSess();
    var open = {};
    box.querySelectorAll(".t58-pc-bs").forEach(function (s) { open[s.getAttribute("data-id")] = !s.querySelector(".t58-pc-bb").hidden; });
    Object.keys(sess.sections || {}).forEach(function (k) { if (!(k in open)) open[k] = !!sess.sections[k]; });
    var html = '<div class="t58-pc-bh"><b>T58 Morning Briefing</b><span>' + new Date(d.generated_at * 1000).toLocaleTimeString([], {hour: "numeric", minute: "2-digit"}) + '</span></div>';
    if (d.empty) html += '<div class="t58-pc-bempty">' + esc(d.empty_line) + '</div>';
    (d.sections || []).forEach(function (s, i) {
      var isOpen = (s.id in open) ? open[s.id] : i < 2;
      html += '<div class="t58-pc-bs" data-id="' + esc(s.id) + '"><button type="button" aria-expanded="' + isOpen + '"><span><small>' + esc(s.title) + '</small>' + esc(s.headline) + '</span><span aria-hidden="true">' + (isOpen ? "&#9662;" : "&#9656;") + '</span></button>' +
        '<div class="t58-pc-bb"' + (isOpen ? "" : " hidden") + '>' + (s.lines || []).map(function (l) { return "<div>" + esc(l) + "</div>"; }).join("") +
        (s.action ? '<a href="' + esc(s.action.href) + '">' + esc(s.action.label) + " &rarr;</a>" : "") +
        (s.evidence ? '<div class="ev">Based on: ' + esc(s.evidence) + "</div>" : "") + "</div></div>";
    });
    box.innerHTML = html;
    box.querySelectorAll(".t58-pc-bs > button").forEach(function (b) {
      b.addEventListener("click", function () {
        var body = b.parentNode.querySelector(".t58-pc-bb"); body.hidden = !body.hidden;
        b.setAttribute("aria-expanded", String(!body.hidden)); b.lastChild.innerHTML = body.hidden ? "&#9656;" : "&#9662;";
        sess.sections[b.parentNode.getAttribute("data-id")] = !body.hidden; saveSess();
      });
    });
  }
  function applyBriefEnabled(on) {
    briefEnabled = on;
    var tab = document.getElementById("t58-pc-brief-tab"); if (tab) tab.hidden = !on;
  }
  function loadBrief(auto) {
    return fetch("/api/briefing").then(function (r) { return r.json(); }).then(function (d) {
      applyBriefEnabled(d.enabled !== false);
      if (!briefEnabled) { if (els.tabBrief && !els.tabBrief.hidden) showTab("chat"); return d; }
      renderBrief(d);
      var dot = document.getElementById("t58-pc-brief-dot");
      if (dot) dot.hidden = !(d.due && els.panel.hidden);
      if (auto && d.due) {                       // first access of the period: pop it open once, collapsible via the header button
        openPanel(); showTab("brief");
        fetch("/api/briefing/seen", {method: "POST"}).catch(function () {});
      }
      return d;
    }).catch(function () {});
  }
  function startBriefPolling() { stopBriefPolling(); briefTimer = setInterval(function () { if (!document.hidden) loadBrief(); }, 10 * 60 * 1000); }
  function stopBriefPolling() { if (briefTimer) { clearInterval(briefTimer); briefTimer = null; } }

  // -------------------------------------------------------------------
  // Projects: list / create / rename / delete / activate
  // -------------------------------------------------------------------

  function refreshProjectList() {
    return apiFetch("").then(function (data) {
      state.projectList = data.projects || [];
      renderProjectSelect();
      var activeId = data.active_project_id;
      if (activeId) {
        return loadProject(activeId, true);
      }
      state.project = null;
      renderEmptyState();
    });
  }

  function renderProjectSelect() {
    var select = els.projectSelect;
    select.innerHTML = "";
    if (!state.projectList.length) {
      var opt = document.createElement("option");
      opt.textContent = "No projects yet";
      opt.value = "";
      select.appendChild(opt);
      return;
    }
    state.projectList.forEach(function (p) {
      var opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      if (state.project && state.project.id === p.id) opt.selected = true;
      select.appendChild(opt);
    });
  }

  function renderEmptyState() {
    var hasProjects = state.projectList.length > 0;
    els.empty.hidden = hasProjects || (els.tabBrief && !els.tabBrief.hidden);
    els.tabChat.hidden = !hasProjects;
    els.tabActivity.hidden = true;
  }

  function loadProject(projectId, keepTab) {
    return apiFetch("/" + projectId).then(function (data) {
      if (!data.project) return;
      state.project = data.project;
      renderProjectSelect();
      els.empty.hidden = true;
      renderMessages();
      // Re-loading the active project on a page load must not switch away from the Brief/Activity tab.
      showTab(keepTab && sess && sess.tab ? sess.tab : "chat");
    });
  }

  function createProject() {
    var name = window.prompt("Project name:", "");
    if (name === null) return;
    apiFetch("", { method: "POST", body: JSON.stringify({ name: name }) }).then(function (data) {
      if (!data.project) return;
      return refreshProjectList().then(function () { return loadProject(data.project.id); });
    });
  }

  function renameProject() {
    if (!state.project) return;
    var name = window.prompt("Rename project:", state.project.name);
    if (name === null || !name.trim()) return;
    apiFetch("/" + state.project.id, { method: "PATCH", body: JSON.stringify({ name: name }) })
      .then(function (data) {
        if (data.project) {
          state.project = data.project;
          renderProjectSelect();
        }
      });
  }

  function deleteProject() {
    if (!state.project) return;
    if (!window.confirm('Delete project "' + state.project.name + '"? This cannot be undone.')) return;
    apiFetch("/" + state.project.id, { method: "DELETE" }).then(function () {
      state.project = null;
      refreshProjectList();
    });
  }

  function onProjectSelectChange() {
    var id = els.projectSelect.value;
    if (!id) return;
    apiFetch("/" + id + "/activate", { method: "POST" }).then(function () {
      loadProject(id);
    });
  }

  // -------------------------------------------------------------------
  // Chat
  // -------------------------------------------------------------------

  function renderMessages() {
    var container = els.messages;
    container.innerHTML = "";
    var history = (state.project && state.project.chat_history) || [];
    history.forEach(function (turn) {
      appendMessageEl(turn.role, turn.content);
    });
    container.scrollTop = container.scrollHeight;
  }

  function appendMessageEl(role, content) {
    var div = document.createElement("div");
    div.className = "t58-pc-msg " + escapeHtml(role);
    div.innerHTML = escapeHtml(content);
    els.messages.appendChild(div);
    els.messages.scrollTop = els.messages.scrollHeight;
    return div;
  }

  function sendMessage(event) {
    event.preventDefault();
    if (!state.project) return;
    var text = els.input.value.trim();
    if (!text) return;
    els.input.value = "";
    els.input.style.height = "auto";
    appendMessageEl("user", text);
    els.sendBtn.disabled = true;
    els.typing.hidden = false;

    apiFetch("/" + state.project.id + "/chat", { method: "POST", body: JSON.stringify({ message: text }) })
      .then(function (data) {
        els.typing.hidden = true;
        els.sendBtn.disabled = false;
        if (data.error) {
          var div = appendMessageEl("error", data.error);
          div.className = "t58-pc-msg error";
          return;
        }
        appendMessageEl("assistant", data.reply || "");
        if (data.project) state.project = data.project;
      })
      .catch(function () {
        els.typing.hidden = true;
        els.sendBtn.disabled = false;
        var div = appendMessageEl("error", "Request failed -- check your connection and try again.");
        div.className = "t58-pc-msg error";
      });
  }

  // -------------------------------------------------------------------
  // Activity feed (read-only)
  // -------------------------------------------------------------------

  function refreshActivity() {
    if (!state.project) return;
    apiFetch("/" + state.project.id + "/activity").then(function (data) {
      renderActivity(data.jobs || []);
    });
  }

  function renderActivity(jobs) {
    var container = els.activityList;
    container.innerHTML = "";
    if (!jobs.length) {
      var empty = document.createElement("div");
      empty.className = "t58-pc-activity-empty";
      empty.textContent = "No background jobs started under this project yet. Runs started from " +
        "Search Lab, Full Pipeline, and other tools while this project is active will show up here.";
      container.appendChild(empty);
      return;
    }
    jobs.forEach(function (job) {
      var status = job.error ? "failed" : (job.cancelled ? "cancelled" : (job.done ? "done" : "running"));
      var div = document.createElement("div");
      div.className = "t58-pc-job";
      var title = document.createElement("div");
      title.className = "t58-pc-job-title";
      title.innerHTML = "<span>" + escapeHtml(job.tool || job.job_id) + "</span>" +
        "<span class=\"t58-pc-job-status " + status + "\">" + escapeHtml(status) + "</span>";
      div.appendChild(title);

      if (job.progress && job.progress.banner) {
        var banner = document.createElement("div");
        banner.className = "t58-pc-job-banner";
        banner.textContent = job.progress.banner;
        div.appendChild(banner);
      }

      var meta = document.createElement("div");
      meta.className = "t58-pc-job-meta";
      var metaBits = [];
      if (typeof job.elapsed_seconds === "number") metaBits.push(Math.round(job.elapsed_seconds) + "s elapsed");
      if (job.instrument) metaBits.push(escapeHtml(job.instrument));
      if (job.error) metaBits.push(escapeHtml(job.error));
      meta.textContent = metaBits.join(" \u2022 ");
      div.appendChild(meta);

      if (job.log_tail && job.log_tail.length) {
        var log = document.createElement("div");
        log.className = "t58-pc-job-log";
        log.textContent = job.log_tail.join("\n");
        div.appendChild(log);
      }
      container.appendChild(div);
    });
  }

  function startActivityPolling() {
    stopActivityPolling();
    state.activityTimer = window.setInterval(refreshActivity, ACTIVITY_POLL_MS);
  }

  function stopActivityPolling() {
    if (state.activityTimer) {
      window.clearInterval(state.activityTimer);
      state.activityTimer = null;
    }
  }

  // -------------------------------------------------------------------
  // Wiring
  // -------------------------------------------------------------------

  function init() {
    els.launcher = $("t58-pc-launcher");
    els.panel = $("t58-pc-panel");
    if (!els.launcher || !els.panel) return; // partial not included on this page

    els.closeBtn = $("t58-pc-close");
    els.projectSelect = $("t58-pc-project-select");
    els.newBtn = $("t58-pc-new-project");
    els.renameBtn = $("t58-pc-rename-project");
    els.deleteBtn = $("t58-pc-delete-project");
    els.empty = $("t58-pc-empty");
    els.tabChat = $("t58-pc-tab-chat");
    els.tabActivity = $("t58-pc-tab-activity");
    els.tabBrief = $("t58-pc-tab-brief");
    els.messages = $("t58-pc-messages");
    els.typing = $("t58-pc-typing");
    els.form = $("t58-pc-form");
    els.input = $("t58-pc-input");
    els.sendBtn = els.form.querySelector(".t58-pc-send");
    els.activityList = $("t58-pc-activity-list");

    els.launcherIcon = $("t58-pc-launcher-icon");

    els.launcher.addEventListener("click", togglePanel);
    els.launcher.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); togglePanel(); }
    });
    els.closeBtn.addEventListener("click", closePanel);
    els.newBtn.addEventListener("click", createProject);
    els.renameBtn.addEventListener("click", renameProject);
    els.deleteBtn.addEventListener("click", deleteProject);
    els.projectSelect.addEventListener("change", onProjectSelectChange);
    els.form.addEventListener("submit", sendMessage);

    var tabs = els.panel.querySelectorAll(".t58-pc-tab");
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].addEventListener("click", function (e) { showTab(e.currentTarget.getAttribute("data-tab")); });
    }

    els.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        els.form.requestSubmit ? els.form.requestSubmit() : sendMessage(e);
      }
    });
    els.input.addEventListener("input", function () {
      els.input.style.height = "auto";
      els.input.style.height = Math.min(els.input.scrollHeight, 90) + "px";
    });

    // Start collapsed (out of the way) unless it was left open earlier in THIS login session.
    sess = loadSess();
    var firstPageOfSession = !sess.first;
    sess.first = true;
    saveSess();
    if (wasOpen()) { openPanel(); } else { closePanel(); }
    // Morning Briefing: may pop open ONCE, on the first page of a login session and only when due. On every later page
    // (and after the person collapses it) it just keeps the tab's dot current and stays however it was left.
    setTimeout(function () { loadBrief(firstPageOfSession); }, 1200);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
