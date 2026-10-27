/* Library page: Books (typographic covers + your own PDFs), Papers (from research/ reading lists), Bible (ESV via your own key, KJV offline), Focus sessions. */
(function () {
  var PG = window.PG, root = document.getElementById('lib-root');
  var BOOKS = [
    ['Thinking in Bets', 'Annie Duke', 262], ['The Psychology of Money', 'Morgan Housel', 38], ['Trading in the Zone', 'Mark Douglas', 168], ['The Mental Game of Trading', 'Jared Tendler', 290],
    ['Market Wizards', 'Jack D. Schwager', 212], ['The Best Losers Win', 'Tom Hougaard', 12], ['Quantitative Trading', 'Ernest P. Chan', 196], ['The Art of Statistics', 'David Spiegelhalter', 330],
    ['Thinking, Fast and Slow', 'Daniel Kahneman', 48], ['Fooled by Randomness', 'Nassim Nicholas Taleb', 350], ['The Black Swan', 'Nassim Nicholas Taleb', 232], ['Antifragile', 'Nassim Nicholas Taleb', 150],
    ['Reminiscences of a Stock Operator', 'Edwin Lefèvre', 28], ['Advances in Financial Machine Learning', 'Marcos López de Prado', 244], ['The Signal and the Noise', 'Nate Silver', 186], ['A Random Walk Down Wall Street', 'Burton G. Malkiel', 108],
    ['Algorithmic Trading', 'Ernest P. Chan', 130], ['Trading Psychology 2.0', 'Brett N. Steenbarger', 272]
  ].map(function (b, i) { return { id: PG_slug(b[0]), title: b[0], author: b[1], hue: b[2], i: i }; });
  function PG_slug(t) { return t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 80) || 'item'; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function csrfFetch(url, opts) { return fetch(url, Object.assign({ credentials: 'same-origin' }, opts)); }

  var tabs = [['books', 'Books'], ['papers', 'Research papers'], ['bible', 'Bible'], ['focus', 'Focus']], cur = null;
  var tabBar = PG.el('div', 'pg-tabs'), body = PG.el('div', 'pg-stage'); root.appendChild(tabBar); root.appendChild(body);
  tabs.forEach(function (t) { var b = PG.el('button', 'pg-tab', t[1]); b.type = 'button'; b.dataset.id = t[0]; b.onclick = function () { show(t[0]); }; tabBar.appendChild(b); });
  function show(id) {
    if (cur && cur.destroy) cur.destroy(); [].forEach.call(tabBar.children, function (b) { b.classList.toggle('on', b.dataset.id === id); }); body.innerHTML = '';
    cur = ({ books: books, papers: papers, bible: bible, focus: focus })[id](body) || null; try { history.replaceState(null, '', '#' + id); } catch (e) {}
  }

  /* ---------------------------------------------------------------- PDF viewer */
  var V = { el: document.getElementById('lib-viewer'), frame: document.getElementById('lib-v-frame'), title: document.getElementById('lib-v-title'), open: document.getElementById('lib-v-open') };
  function openPdf(url, title) { V.frame.src = url + '#view=FitH'; V.title.textContent = title; V.open.href = url; V.el.classList.add('on'); }
  document.getElementById('lib-v-close').onclick = function () { V.el.classList.remove('on'); V.frame.src = 'about:blank'; };
  window.addEventListener('keydown', function (e) { if (e.key === 'Escape') { if (V.el.classList.contains('on')) document.getElementById('lib-v-close').click(); else document.getElementById('lib-modal').classList.remove('on'); } });
  function upload(kind, id, file, done) {
    var fd = new FormData(); fd.append('kind', kind); fd.append('id', id); fd.append('file', file);
    csrfFetch('/api/library/upload', { method: 'POST', body: fd }).then(function (r) { return r.json(); }).then(function (j) { done(j); }).catch(function () { done({ ok: false, error: 'Upload failed.' }); });
  }
  function pickPdf(cb) { var inp = document.createElement('input'); inp.type = 'file'; inp.accept = 'application/pdf,.pdf'; inp.onchange = function () { if (inp.files[0]) cb(inp.files[0]); }; inp.click(); }

  /* ---------------------------------------------------------------- Books */
  function coverHtml(b, pdf) {
    var shapes = ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="46" fill="none" stroke="#fff" stroke-width="6"/><circle cx="50" cy="50" r="26" fill="none" stroke="#fff" stroke-width="6"/></svg>', '<svg viewBox="0 0 100 100"><path d="M0 80 25 50 45 62 70 25 100 45V100H0z" fill="#fff"/></svg>', '<svg viewBox="0 0 100 100"><g fill="none" stroke="#fff" stroke-width="5"><path d="M10 90V10M10 90h80"/><path d="M10 70 35 50 55 60 90 20"/></g></svg>', '<svg viewBox="0 0 100 100"><g fill="#fff"><rect x="12" y="55" width="16" height="35"/><rect x="34" y="35" width="16" height="55"/><rect x="56" y="48" width="16" height="42"/><rect x="78" y="20" width="16" height="70"/></g></svg>', '<svg viewBox="0 0 100 100"><g fill="none" stroke="#fff" stroke-width="4"><path d="M50 8 92 50 50 92 8 50z"/><path d="M50 26 74 50 50 74 26 50z"/></g></svg>', '<svg viewBox="0 0 100 100"><g fill="none" stroke="#fff" stroke-width="4"><path d="M5 60C25 20 40 100 60 50S85 20 95 40"/><path d="M5 80C25 40 40 120 60 70S85 40 95 60"/></g></svg>'];
    return '<div class="cover" style="--h:' + b.hue + '">' + (pdf ? '<span class="pdf">PDF</span>' : '') + shapes[b.i % shapes.length] + '<div><div class="ttl">' + esc(b.title) + '</div><div class="rule"></div></div><div class="au">' + esc(b.author) + '</div></div>';
  }
  function books(host) {
    var st = PG.store.get('lib-books', {}), pdfs = {}, filter = 'all', q = '';
    var bar = PG.el('div', 'pg-bar'); host.appendChild(bar);
    var search = PG.el('input', 'lib-search'); search.placeholder = 'Search title or author'; search.type = 'search'; bar.appendChild(search);
    var seg = PG.seg(bar, [['all', 'All'], ['reading', 'Reading'], ['want', 'Want to read'], ['done', 'Finished'], ['pdf', 'Has PDF']], 'all', function (v) { filter = v; draw(); });
    var cont = PG.el('div'); host.appendChild(cont); host.appendChild(PG.el('p', 'pg-hint', 'Covers are original typographic designs, not the publishers’ artwork. Own a book as a PDF? Open it here and tap “Add your PDF”: the file is saved on this computer only and opens in the in-app reader.'));
    function draw() {
      var list = BOOKS.filter(function (b) { var s = st[b.id] || {}; return (!q || (b.title + ' ' + b.author).toLowerCase().indexOf(q) >= 0) && (filter === 'all' || (filter === 'pdf' ? pdfs[b.id] : s.status === filter)); });
      var reading = BOOKS.filter(function (b) { return (st[b.id] || {}).status === 'reading'; });
      var h = '';
      if (reading.length && filter === 'all' && !q) { h += '<h4 style="margin:14px 0 0;font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted)">Continue reading</h4><div class="lib-shelf">' + reading.map(card).join('') + '</div><h4 style="margin:22px 0 0;font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted)">Library</h4>'; }
      h += '<div class="lib-shelf">' + (list.length ? list.map(card).join('') : '<p class="pg-hint">Nothing matches.</p>') + '</div>'; cont.innerHTML = h;
      [].forEach.call(cont.querySelectorAll('.lib-book'), function (el) { el.onclick = function () { detail(el.dataset.id); }; });
    }
    function card(b) { var s = st[b.id] || {}; return '<button class="lib-book" type="button" data-id="' + b.id + '">' + coverHtml(b, pdfs[b.id]) + '<span class="meta"><b>' + esc(b.title) + '</b>' + esc(b.author) + '</span>' + (s.status === 'reading' ? '<span class="lib-prog"><i style="width:' + (s.pct || 0) + '%"></i></span>' : s.status === 'done' ? '<span class="meta" style="color:var(--teal)">Finished</span>' : '') + '</button>'; }
    function detail(id) {
      var b = BOOKS.filter(function (x) { return x.id === id; })[0], s = st[id] = st[id] || {}, M = document.getElementById('lib-modal');
      function render() {
        M.innerHTML = '<div class="lib-dlg" role="dialog" aria-label="' + esc(b.title) + '"><div>' + coverHtml(b, pdfs[id]) + '</div><div><h3>' + esc(b.title) + '</h3><div class="by">' + esc(b.author) + '</div><div id="d-seg"></div><div style="margin:14px 0 4px;font-size:12.5px;color:var(--text-muted)">Progress: <b id="d-pct">' + (s.pct || 0) + '</b>%</div><input id="d-rng" type="range" min="0" max="100" value="' + (s.pct || 0) + '" style="width:100%"><div class="pg-bar" id="d-act" style="margin-top:14px"></div><p class="pg-hint" id="d-msg"></p></div></div>';
        PG.seg(M.querySelector('#d-seg'), [['want', 'Want to read'], ['reading', 'Reading'], ['done', 'Finished']], s.status || '', function (v) { s.status = v; if (v === 'done') s.pct = 100; save(); render(); });
        var rng = M.querySelector('#d-rng'); rng.oninput = function () { s.pct = +rng.value; M.querySelector('#d-pct').textContent = s.pct; if (!s.status || s.status === 'want') s.status = 'reading'; save(false); };
        var act = M.querySelector('#d-act');
        if (pdfs[id]) { PG.button(act, 'Read PDF', function () { M.classList.remove('on'); openPdf('/library-files/books/' + id + '.pdf', b.title); if (!s.status || s.status === 'want') { s.status = 'reading'; save(); } }, true); PG.button(act, 'Replace PDF', add); PG.button(act, 'Remove', function () { csrfFetch('/api/library/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'books', id: id }) }).then(function () { delete pdfs[id]; render(); draw(); }); }); }
        else PG.button(act, 'Add your PDF', add, true);
        PG.button(act, 'Close', function () { M.classList.remove('on'); draw(); });
        function add() { pickPdf(function (f) { M.querySelector('#d-msg').textContent = 'Uploading...'; upload('books', id, f, function (j) { if (j.ok) { pdfs[id] = true; render(); draw(); } else M.querySelector('#d-msg').textContent = j.error || 'Upload failed.'; }); }); }
      }
      function save(re) { PG.store.set('lib-books', st); }
      M.onclick = function (e) { if (e.target === M) { M.classList.remove('on'); draw(); } }; render(); M.classList.add('on');
    }
    search.oninput = function () { q = search.value.trim().toLowerCase(); draw(); };
    fetch('/api/library/books').then(function (r) { return r.json(); }).then(function (j) { (j.pdfs || []).forEach(function (s) { pdfs[s] = true; }); draw(); }).catch(draw); draw();
  }

  /* ---------------------------------------------------------------- Papers */
  function papers(host) {
    var data = null, q = '', only = false; var bar = PG.el('div', 'pg-bar'); host.appendChild(bar);
    var search = PG.el('input', 'lib-search'); search.placeholder = 'Search papers'; search.type = 'search'; bar.appendChild(search); PG.seg(bar, [[false, 'All papers'], [true, 'Has PDF']], false, function (v) { only = v; draw(); });
    var cont = PG.el('div'); host.appendChild(cont); host.appendChild(PG.el('p', 'pg-hint', 'The list comes from the reading lists in your repo’s research/ folder. Publisher pages open in a new tab. Attach the PDF you download and it opens here, and the AI assistant can also draw on it (it lives in research/).'));
    cont.textContent = 'Loading...';
    function domain(u) { try { return new URL(u).hostname.replace(/^www\./, ''); } catch (e) { return u; } }
    function draw() {
      if (!data) return; var h = '';
      data.categories.forEach(function (c) {
        var ps = c.papers.filter(function (p) { return (!q || (p.title + ' ' + p.role).toLowerCase().indexOf(q) >= 0) && (!only || p.pdf); }); if (!ps.length) return;
        h += '<div class="lib-cat"><h4>' + esc(c.name) + ' <span class="pg-stat">' + ps.length + '</span></h4>' + ps.map(function (p) {
          var m = p.title.match(/^(.*?)\s-\s(.*)$/), au = m ? m[1] : '', ti = m ? m[2] : p.title;
          return '<div class="lib-paper" data-id="' + esc(p.id) + '"><div class="t">' + esc(ti.replace(/^"|"$/g, '')) + '</div>' + (au ? '<div class="a">' + esc(au) + '</div>' : '') + (p.role ? '<div class="r">' + esc(p.role) + '</div>' : '') +
            '<div class="act">' + (p.pdf ? '<button class="pg-btn pri" data-act="read" type="button">Read PDF</button>' : '<button class="pg-btn" data-act="add" type="button">Attach PDF</button>') + (p.pdf ? '<button class="pg-btn" data-act="del" type="button">Remove</button>' : '') + p.links.slice(0, 4).map(function (u) { return '<a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(domain(u)) + '</a>'; }).join('') + '</div></div>';
        }).join('') + '</div>';
      });
      cont.innerHTML = h || '<p class="pg-hint">No papers match.</p>';
      [].forEach.call(cont.querySelectorAll('.lib-paper'), function (el) {
        var id = el.dataset.id, t = el.querySelector('.t').textContent;
        [].forEach.call(el.querySelectorAll('button'), function (b) { b.onclick = function () {
          var a = b.dataset.act; if (a === 'read') openPdf('/library-files/papers/' + id + '.pdf', t);
          else if (a === 'add') pickPdf(function (f) { b.textContent = 'Uploading...'; upload('papers', id, f, function (j) { if (j.ok) load(); else { b.textContent = j.error || 'Failed'; } }); });
          else if (a === 'del') csrfFetch('/api/library/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: 'papers', id: id }) }).then(load);
        }; });
      });
    }
    function load() { fetch('/api/library/papers').then(function (r) { return r.json(); }).then(function (j) { data = j; draw(); }).catch(function () { cont.textContent = 'Could not load the research list.'; }); }
    search.oninput = function () { q = search.value.trim().toLowerCase(); draw(); }; load();
  }

  /* ---------------------------------------------------------------- Bible */
  var OT = [['Genesis', 50], ['Exodus', 40], ['Leviticus', 27], ['Numbers', 36], ['Deuteronomy', 34], ['Joshua', 24], ['Judges', 21], ['Ruth', 4], ['1 Samuel', 31], ['2 Samuel', 24], ['1 Kings', 22], ['2 Kings', 25], ['1 Chronicles', 29], ['2 Chronicles', 36], ['Ezra', 10], ['Nehemiah', 13], ['Esther', 10], ['Job', 42], ['Psalms', 150], ['Proverbs', 31], ['Ecclesiastes', 12], ['Song of Solomon', 8], ['Isaiah', 66], ['Jeremiah', 52], ['Lamentations', 5], ['Ezekiel', 48], ['Daniel', 12], ['Hosea', 14], ['Joel', 3], ['Amos', 9], ['Obadiah', 1], ['Jonah', 4], ['Micah', 7], ['Nahum', 3], ['Habakkuk', 3], ['Zephaniah', 3], ['Haggai', 2], ['Zechariah', 14], ['Malachi', 4]];
  var NT = [['Matthew', 28], ['Mark', 16], ['Luke', 24], ['John', 21], ['Acts', 28], ['Romans', 16], ['1 Corinthians', 16], ['2 Corinthians', 13], ['Galatians', 6], ['Ephesians', 6], ['Philippians', 4], ['Colossians', 4], ['1 Thessalonians', 5], ['2 Thessalonians', 3], ['1 Timothy', 6], ['2 Timothy', 4], ['Titus', 3], ['Philemon', 1], ['Hebrews', 13], ['James', 5], ['1 Peter', 5], ['2 Peter', 3], ['1 John', 5], ['2 John', 1], ['3 John', 1], ['Jude', 1], ['Revelation', 22]];
  var ALL = OT.concat(NT);
  function bible(host) {
    var pos = PG.store.get('bible-pos', { book: 'John', ch: 1, ver: 'esv' }), ver = pos.ver || 'esv', fs = PG.store.get('bible-fs', 17), esvOk = false, dead = false, kjvCache = {};
    var bar = PG.el('div', 'pg-bar'); host.appendChild(bar); var vseg = PG.seg(bar, [['esv', 'ESV'], ['kjv', 'KJV (offline)']], ver, function (v) { ver = v; pos.ver = v; PG.store.set('bible-pos', pos); load(); });
    PG.button(bar, 'A−', function () { fs = Math.max(13, fs - 1); PG.store.set('bible-fs', fs); text.style.setProperty('--bfs', fs + 'px'); }); PG.button(bar, 'A+', function () { fs = Math.min(28, fs + 1); PG.store.set('bible-fs', fs); text.style.setProperty('--bfs', fs + 'px'); });
    var wrap = PG.el('div', 'bible'); host.appendChild(wrap); var nav = PG.el('div', 'bible-nav'), main = PG.el('div'); wrap.appendChild(nav); wrap.appendChild(main);
    var chs = PG.el('div', 'bible-chs'), text = PG.el('div', 'bible-text'), foot = PG.el('p', 'pg-hint'); text.style.setProperty('--bfs', fs + 'px'); var pn = PG.el('div', 'pg-bar'); main.appendChild(chs); main.appendChild(text); main.appendChild(pn); main.appendChild(foot);
    function drawNav() { var h = '<div class="grp">Old Testament</div>'; OT.forEach(function (b) { h += '<button type="button" data-b="' + b[0] + '"' + (b[0] === pos.book ? ' class="on"' : '') + '>' + b[0] + '</button>'; }); h += '<div class="grp">New Testament</div>'; NT.forEach(function (b) { h += '<button type="button" data-b="' + b[0] + '"' + (b[0] === pos.book ? ' class="on"' : '') + '>' + b[0] + '</button>'; }); nav.innerHTML = h; [].forEach.call(nav.querySelectorAll('button'), function (b) { b.onclick = function () { pos.book = b.dataset.b; pos.ch = 1; drawNav(); load(); }; }); var on = nav.querySelector('.on'); if (on && on.scrollIntoView) { try { nav.scrollTop = on.offsetTop - 80; } catch (e) {} } }
    function nch() { return ALL.filter(function (b) { return b[0] === pos.book; })[0][1]; }
    function drawChs() { var n = nch(), h = ''; for (var i = 1; i <= n; i++) h += '<button type="button" data-c="' + i + '"' + (i === pos.ch ? ' class="on"' : '') + '>' + i + '</button>'; chs.innerHTML = h; [].forEach.call(chs.querySelectorAll('button'), function (b) { b.onclick = function () { pos.ch = +b.dataset.c; load(); }; }); pn.innerHTML = ''; PG.button(pn, '← Previous', function () { step(-1); }); PG.button(pn, 'Next →', function () { step(1); }, true); }
    function step(d) { var i = ALL.findIndex(function (b) { return b[0] === pos.book; }), c = pos.ch + d; if (c < 1) { if (i > 0) { pos.book = ALL[i - 1][0]; c = ALL[i - 1][1]; drawNav(); } else c = 1; } else if (c > nch()) { if (i < ALL.length - 1) { pos.book = ALL[i + 1][0]; c = 1; drawNav(); } else c = nch(); } pos.ch = c; load(); main.scrollIntoView && window.scrollTo({ top: 0 }); }
    function load() {
      PG.store.set('bible-pos', pos); drawChs(); text.innerHTML = '<p class="pg-hint">Loading...</p>'; foot.textContent = '';
      if (ver === 'kjv') {
        var idx = ALL.findIndex(function (b) { return b[0] === pos.book; });
        (kjvCache[idx] ? Promise.resolve(kjvCache[idx]) : fetch('/static/bible/kjv/' + idx + '.json').then(function (r) { return r.json(); })).then(function (b) {
          kjvCache[idx] = b; if (dead) return; var vs = b[pos.ch - 1] || [];
          text.innerHTML = '<h5>' + pos.book + ' ' + pos.ch + ' &middot; King James Version</h5><p>' + vs.map(function (t, i) { return '<sup>' + (i + 1) + '</sup>' + esc(t); }).join(' ') + '</p>'; foot.textContent = 'King James Version: public domain. Works fully offline.';
        }).catch(function () { text.textContent = 'Could not load this book.'; });
        return;
      }
      fetch('/api/bible/esv?q=' + encodeURIComponent(pos.book + ' ' + pos.ch)).then(function (r) { return r.json().then(function (j) { return [r.status, j]; }); }).then(function (a) {
        if (dead) return; var j = a[1];
        if (a[0] === 401 || j.error === 'no_key') { setup(); return; }
        if (!j.ok) { text.innerHTML = '<p>' + esc(j.error || 'Could not load this chapter.') + '</p><p class="pg-hint">Switch to the KJV for offline reading.</p>'; return; }
        var paras = j.text.split(/\n\s*\n/).map(function (s) { return s.trim(); }).filter(Boolean), h = '<h5>' + esc(j.ref) + ' &middot; ESV</h5>';
        paras.forEach(function (p) { if (!/\[\d+\]/.test(p) && p.length < 90) h += '<h5 style="margin-top:1.2em">' + esc(p) + '</h5>'; else h += '<p>' + esc(p.replace(/\s*\n\s*/g, ' ')).replace(/\[(\d+)\]/g, '<sup>$1</sup>') + '</p>'; });
        text.innerHTML = h; foot.textContent = j.copyright + (j.cached ? ' (saved copy)' : '');
      }).catch(function () { text.innerHTML = '<p>The ESV needs a connection the first time you open a chapter. Chapters you have read before work offline. The KJV works offline.</p>'; });
    }
    function setup() {
      text.innerHTML = '<h5>Connect the ESV</h5><p>The ESV text belongs to Crossway, so it is not bundled with the app. Get a free personal key at <a href="https://api.esv.org/" target="_blank" rel="noopener" style="color:var(--teal)">api.esv.org</a> (sign in, create an application, copy the key), paste it below once, and chapters you open will be saved for offline reading.</p><div class="pg-bar"><input class="lib-search" id="esv-key" type="password" placeholder="Paste your ESV API key" autocomplete="off"></div><div class="pg-bar"></div><p class="pg-hint" id="esv-msg">Meanwhile the KJV tab reads fully offline.</p>';
      var bt = PG.button(text.querySelectorAll('.pg-bar')[1], 'Save key', function () { csrfFetch('/api/bible/esv/key', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: text.querySelector('#esv-key').value }) }).then(function (r) { return r.json(); }).then(function (j) { if (j.ok) load(); else text.querySelector('#esv-msg').textContent = j.error; }); }, true);
      PG.button(text.querySelectorAll('.pg-bar')[1], 'Use KJV instead', function () { ver = 'kjv'; pos.ver = 'kjv'; [].forEach.call(vseg.children, function (c, i) { c.classList.toggle('on', i === 1); }); load(); });
    }
    drawNav(); load();
    return { destroy: function () { dead = true; } };
  }

  /* ---------------------------------------------------------------- Focus */
  function focus(host) {
    var active = PG.store.get('focus-active', null), minutes = 20, sound = PG.store.get('focus-sound', true), tick = null, sessions = [];
    host.appendChild(PG.el('p', 'pg-hint', 'Optional. Pick a length and read or work without interruption. No streaks, no reminders: the timer only runs when you start it.'));
    var cards = PG.el('div', 'focus-cards'); host.appendChild(cards);
    [10, 20, 30].forEach(function (m) { var b = PG.el('button', 'focus-card' + (m === minutes ? ' on' : ''), '<b>' + m + '</b>minutes'); b.type = 'button'; b.onclick = function () { if (active) return; minutes = m; [].forEach.call(cards.children, function (c) { c.classList.remove('on'); }); b.classList.add('on'); drawRing(); }; cards.appendChild(b); });
    var label = PG.el('input', 'lib-search'); label.placeholder = 'What are you focusing on? (optional)'; label.style.cssText = 'margin-top:12px;width:100%;max-width:560px'; host.appendChild(label);
    var ring = PG.el('div', 'focus-ring', '<span>20:00</span>'); host.appendChild(ring);
    var bar = PG.el('div', 'pg-bar'); host.appendChild(bar); var bGo = PG.button(bar, 'Start', function () { start(); }, true), bStop = PG.button(bar, 'End early', function () { finish(false); }); bStop.style.display = 'none';
    var snd = PG.el('label', 'pg-stat', '<input type="checkbox"' + (sound ? ' checked' : '') + '> Soft chime at the end'); snd.querySelector('input').onchange = function (e) { sound = e.target.checked; PG.store.set('focus-sound', sound); }; bar.appendChild(snd);
    var sum = PG.el('p', 'pg-msg'); host.appendChild(sum); var hist = PG.el('div', 'focus-hist'); host.appendChild(hist);
    function drawRing() { var left = active ? Math.max(0, (active.end - Date.now()) / 1000) : minutes * 60, tot = (active ? active.minutes : minutes) * 60; ring.style.setProperty('--p', active ? (1 - left / tot) * 100 : 0); ring.firstChild.textContent = PG.fmtTime(Math.ceil(left)); if (active) document.title = PG.fmtTime(Math.ceil(left)) + ' · Focus'; }
    var baseTitle = document.title;
    function start() { active = { minutes: minutes, start: Date.now(), end: Date.now() + minutes * 60000, label: label.value.trim() }; PG.store.set('focus-active', active); ui(); loop(); }
    function ui() { bGo.style.display = active ? 'none' : ''; bStop.style.display = active ? '' : 'none'; label.disabled = !!active; if (active) label.value = active.label || ''; drawRing(); }
    function loop() { clearInterval(tick); tick = setInterval(function () { if (!active) return; drawRing(); if (Date.now() >= active.end) finish(true); }, 500); }
    function chime() { if (!sound) return; try { var A = window.AudioContext || window.webkitAudioContext, c = new A(); [660, 880, 1100].forEach(function (f, i) { var o = c.createOscillator(), g = c.createGain(); o.frequency.value = f; o.type = 'sine'; g.gain.setValueAtTime(0, c.currentTime + i * .28); g.gain.linearRampToValueAtTime(.12, c.currentTime + i * .28 + .05); g.gain.exponentialRampToValueAtTime(.001, c.currentTime + i * .28 + 1.1); o.connect(g); g.connect(c.destination); o.start(c.currentTime + i * .28); o.stop(c.currentTime + i * .28 + 1.2); }); } catch (e) {} }
    function finish(done) {
      if (!active) return; var a = active, elapsed = Math.min(a.minutes, (Date.now() - a.start) / 60000); active = null; PG.store.set('focus-active', null); clearInterval(tick); document.title = baseTitle;
      if (done) chime(); if (done || elapsed >= 1) csrfFetch('/api/library/focus', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ minutes: a.minutes, done_minutes: elapsed, completed: !!done, label: a.label }) }).then(function () { loadHist(); });
      ui(); sum.textContent = done ? 'Session complete. Nicely done.' : 'Session ended early. Every minute counts.';
    }
    function loadHist() { fetch('/api/library/focus').then(function (r) { return r.json(); }).then(function (j) { sessions = j.sessions || []; drawHist(); }).catch(function () {}); }
    function drawHist() {
      var week = Date.now() - 7 * 864e5, wk = sessions.filter(function (s) { return s.at * 1000 > week; }), mins = wk.reduce(function (a, s) { return a + s.done_minutes; }, 0);
      var byDay = {}; sessions.slice().reverse().forEach(function (s) { var d = new Date(s.at * 1000).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' }); (byDay[d] = byDay[d] || []).push(s); });
      var h = '<h4 style="margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted)">Session history</h4><p class="pg-hint" style="margin:0 0 8px">Last 7 days: <b>' + Math.round(mins) + ' min</b> across ' + wk.length + ' session' + (wk.length === 1 ? '' : 's') + '.</p>';
      Object.keys(byDay).slice(0, 10).forEach(function (d) { h += '<div class="d"><span><b>' + d + '</b></span><span>' + byDay[d].map(function (s) { return Math.round(s.done_minutes) + 'm' + (s.completed ? '' : ' (early)') + (s.label ? ' · ' + esc(s.label) : ''); }).join(' &nbsp; ') + '</span></div>'; });
      if (!sessions.length) h += '<p class="pg-hint">No sessions yet.</p>'; else h += '<p style="margin-top:10px"><button class="pg-btn" id="fh-clear" type="button">Clear history</button></p>'; hist.innerHTML = h;
      var c = hist.querySelector('#fh-clear'); if (c) c.onclick = function () { csrfFetch('/api/library/focus/clear', { method: 'POST' }).then(loadHist); };
    }
    if (active) { if (Date.now() >= active.end) finish(true); else { minutes = active.minutes; loop(); } } ui(); loadHist();
    return { destroy: function () { clearInterval(tick); document.title = baseTitle; } };
  }

  show((location.hash || '').slice(1) in { books: 1, papers: 1, bible: 1, focus: 1 } ? location.hash.slice(1) : 'books');
})();
