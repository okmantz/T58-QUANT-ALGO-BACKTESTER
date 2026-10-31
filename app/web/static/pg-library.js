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
    var FREE = [
      ['An Introduction to Statistical Learning', 'James, Witten, Hastie, Tibshirani', 'https://www.statlearning.com/'],
      ['The Elements of Statistical Learning', 'Hastie, Tibshirani, Friedman', 'https://hastie.su.domains/ElemStatLearn/'],
      ['Forecasting: Principles and Practice', 'Hyndman, Athanasopoulos', 'https://otexts.com/fpp3/'],
      ['Reinforcement Learning: An Introduction', 'Sutton, Barto', 'http://incompleteideas.net/book/the-book-2nd.html'],
      ['Deep Learning', 'Goodfellow, Bengio, Courville', 'https://www.deeplearningbook.org/'],
      ['Mathematics for Machine Learning', 'Deisenroth, Faisal, Ong', 'https://mml-book.com/'],
      ['Convex Optimization', 'Boyd, Vandenberghe', 'https://web.stanford.edu/~boyd/cvxbook/'],
      ['Think Stats', 'Allen B. Downey', 'https://greenteapress.com/wp/think-stats-2e/'],
      ['Think Bayes', 'Allen B. Downey', 'https://allendowney.github.io/ThinkBayes2/'],
      ['Introduction to Probability', 'Grinstead, Snell', 'https://math.dartmouth.edu/~prob/prob/prob.pdf'],
      ['Python for Data Analysis', 'Wes McKinney', 'https://wesmckinney.com/book/'],
      ['The Statistical Consequences of Fat Tails', 'Nassim Nicholas Taleb', 'https://arxiv.org/abs/2001.10488']
    ];
    host.appendChild((function () { var d = PG.el('details', 'lib-free'); d.innerHTML = '<summary style="cursor:pointer;margin-top:18px;font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted)">Free, legal books to read online or download</summary><ul style="margin:10px 0 0;padding-left:18px;line-height:1.7;font-size:13.5px">' + FREE.map(function (f) { return '<li><a href="' + f[2] + '" target="_blank" rel="noopener" style="color:var(--teal)">' + esc(f[0]) + '</a> <span style="color:var(--text-muted)">· ' + esc(f[1]) + '</span></li>'; }).join('') + '</ul><p class="pg-hint">Published free by the authors or their publishers. Links open the official page; save the PDF and use “Add your PDF” on a shelf book if you want it in the in-app reader.</p>'; return d; })());
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
    var data = null, q = '', filter = 'all', collapsed = PG.store.get('lib-paper-collapsed', {});
    var hero = PG.el('div', 'lib-phero'); host.appendChild(hero);
    var bar = PG.el('div', 'pg-bar'); host.appendChild(bar);
    var search = PG.el('input', 'lib-search'); search.placeholder = 'Search title, author or topic'; search.type = 'search'; bar.appendChild(search);
    PG.seg(bar, [['all', 'All'], ['liked', '♥ Liked'], ['unread', 'Unread'], ['read', 'Read'], ['pdf', 'PDF in app'], ['links', 'Links only']], 'all', function (v) { filter = v; draw(); });
    var cont = PG.el('div'); host.appendChild(cont);
    host.appendChild(PG.el('p', 'pg-hint', 'Papers marked PDF are stored with the app and open in the reader here. “Links only” entries open the publisher’s page in a new tab. Attach a PDF you download to any of them and it opens here too. Likes and read marks are saved to your account.'));
    cont.textContent = 'Loading...';
    function domain(u) { try { return new URL(u).hostname.replace(/^www\./, ''); } catch (e) { return u; } }
    function setState(p, patch) {
      for (var k in patch) p[k] = patch[k]; patch.id = p.id;
      csrfFetch('/api/library/paper-state', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) }).catch(function () {});
    }
    function match(p, c) {
      if (q && (p.title + ' ' + p.authors + ' ' + p.role + ' ' + c.name).toLowerCase().indexOf(q) < 0) return false;
      if (filter === 'liked') return p.liked; if (filter === 'unread') return !p.read; if (filter === 'read') return p.read;
      if (filter === 'pdf') return p.pdf; if (filter === 'links') return !p.pdf; return true;
    }
    function card(p) {
      var lk = p.links.slice(0, 3).map(function (u) { return '<a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(domain(u)) + '</a>'; }).join('');
      return '<article class="lib-paper' + (p.read ? ' is-read' : '') + '" data-id="' + esc(p.id) + '">' +
        '<div class="top"><span class="badge ' + (p.pdf ? 'pdf' : 'lnk') + '">' + (p.pdf ? 'PDF' : 'LINK') + '</span>' +
        '<button class="ico like' + (p.liked ? ' on' : '') + '" data-act="like" type="button" aria-pressed="' + !!p.liked + '" title="Like">♥</button></div>' +
        '<div class="t">' + esc(p.title) + '</div>' + (p.authors ? '<div class="a">' + esc(p.authors) + '</div>' : '') + (p.role ? '<div class="r">' + esc(p.role) + '</div>' : '') +
        '<div class="act">' + (p.pdf ? '<button class="pg-btn pri" data-act="read" type="button">Read</button>' : '<button class="pg-btn" data-act="add" type="button">Attach PDF</button>') +
        '<button class="pg-btn' + (p.read ? ' done' : '') + '" data-act="mark" type="button">' + (p.read ? '✓ Read' : 'Mark read') + '</button>' +
        (p.pdf && !p.bundled ? '<button class="pg-btn" data-act="del" type="button">Remove</button>' : '') + lk + '</div></article>';
    }
    function draw() {
      if (!data) return;
      var all = [], h = '';
      data.categories.forEach(function (c) { c.papers.forEach(function (p) { all.push(p); }); });
      hero.innerHTML = '<div><b>' + all.length + '</b><span>papers</span></div><div><b>' + all.filter(function (p) { return p.pdf; }).length + '</b><span>readable here</span></div><div><b>' + all.filter(function (p) { return p.liked; }).length + '</b><span>liked</span></div><div><b>' + all.filter(function (p) { return p.read; }).length + '</b><span>read</span></div>';
      var groups = [['library', 'Research library'], ['reading', 'Reading lists'], ['guides', 'Strategy guides']];
      groups.forEach(function (g) {
        var cs = data.categories.filter(function (c) { return c.group === g[0]; }), gh = '';
        cs.forEach(function (c) {
          var ps = c.papers.filter(function (p) { return match(p, c); });
          var docs = (g[0] === 'guides' && !q && filter === 'all') ? c.docs : [];
          if (!ps.length && !docs.length) return;
          var shut = collapsed[c.id] && !q;
          gh += '<section class="lib-cat"><button class="cat-h" type="button" data-cat="' + esc(c.id) + '" aria-expanded="' + !shut + '"><span class="chev">' + (shut ? '▸' : '▾') + '</span>' + esc(c.name) + '<span class="pg-stat">' + ps.length + '</span></button>' +
            (shut ? '' : '<div class="lib-pgrid">' + docs.map(function (d) { return '<article class="lib-paper doc"><div class="top"><span class="badge doc">GUIDE</span></div><div class="t">' + esc(d.name.replace(/\.md$/, '').replace(/_/g, ' ')) + '</div><div class="act"><button class="pg-btn pri" data-doc="' + esc(d.path) + '" type="button">Open guide</button></div></article>'; }).join('') + ps.map(card).join('') + '</div>') + '</section>';
        });
        if (gh) h += '<h3 class="lib-group">' + g[1] + '</h3>' + gh;
      });
      cont.innerHTML = h || '<p class="pg-hint">No papers match.</p>';
      [].forEach.call(cont.querySelectorAll('.cat-h'), function (b) { b.onclick = function () { collapsed[b.dataset.cat] = !collapsed[b.dataset.cat]; PG.store.set('lib-paper-collapsed', collapsed); draw(); }; });
      [].forEach.call(cont.querySelectorAll('[data-doc]'), function (b) { b.onclick = function () { openPdf('/research-docs/' + b.dataset.doc, b.closest('.lib-paper').querySelector('.t').textContent); }; });
      [].forEach.call(cont.querySelectorAll('article.lib-paper[data-id]'), function (el) {
        var id = el.dataset.id, p = all.filter(function (x) { return x.id === id; })[0], t = p.title;
        [].forEach.call(el.querySelectorAll('button'), function (b) { b.onclick = function () {
          var a = b.dataset.act;
          if (a === 'read') { openPdf('/library-files/papers/' + id + '.pdf', t); if (!p.read) { setState(p, { read: true }); draw(); } }
          else if (a === 'like') { setState(p, { liked: !p.liked }); draw(); }
          else if (a === 'mark') { setState(p, { read: !p.read }); draw(); }
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
    var NS = 'http://www.w3.org/2000/svg', R = 112, C = 2 * Math.PI * R, baseTitle = document.title;
    var root = PG.el('div', 'fx'); host.appendChild(root);
    root.innerHTML =
      '<div class="fx-stage">' +
        '<div class="fx-dial" id="fx-dial">' +
          '<svg viewBox="0 0 260 260" aria-hidden="true"><defs><linearGradient id="fx-g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7ef0d4"/><stop offset=".55" stop-color="#4cc9f0"/><stop offset="1" stop-color="#8b7cff"/></linearGradient></defs>' +
          '<g id="fx-ticks"></g><circle class="fx-track" cx="130" cy="130" r="' + R + '"/><circle class="fx-prog" id="fx-prog" cx="130" cy="130" r="' + R + '" stroke-dasharray="' + C.toFixed(2) + '" stroke-dashoffset="' + C.toFixed(2) + '" transform="rotate(-90 130 130)"/>' +
          '<circle class="fx-dot" id="fx-dot" r="5" cx="130" cy="18"/></svg>' +
          '<div class="fx-read"><span class="fx-time" id="fx-time">20:00</span><span class="fx-state" id="fx-state">Ready when you are</span></div>' +
        '</div>' +
        '<div class="fx-side">' +
          '<h3 class="fx-title">Deep focus</h3>' +
          '<p class="fx-sub">One uninterrupted block. No streaks, no reminders &mdash; the timer only runs when you start it.</p>' +
          '<div class="fx-lbl">Length</div><div class="fx-chips" id="fx-chips" role="radiogroup" aria-label="Session length"></div>' +
          '<div class="fx-lbl">Intention</div><input class="fx-input" id="fx-label" maxlength="80" placeholder="What are you focusing on? (optional)" autocomplete="off">' +
          '<div class="fx-actions"><button class="fx-go" id="fx-go" type="button">Begin session</button><button class="fx-stop" id="fx-stop" type="button" hidden>End early</button></div>' +
          '<label class="fx-switch"><input type="checkbox" id="fx-snd"' + (sound ? ' checked' : '') + '><i></i><span>Soft chime at the end</span></label>' +
        '</div>' +
      '</div>' +
      '<div class="fx-msg" id="fx-msg" role="status" aria-live="polite"></div>' +
      '<div class="fx-stats" id="fx-stats"></div>' +
      '<div class="fx-hist" id="fx-hist"></div>';
    var $ = function (id) { return root.querySelector('#' + id); };
    var dial = $('fx-dial'), prog = $('fx-prog'), dot = $('fx-dot'), timeEl = $('fx-time'), stateEl = $('fx-state'), chips = $('fx-chips'), label = $('fx-label'),
      bGo = $('fx-go'), bStop = $('fx-stop'), msg = $('fx-msg'), stats = $('fx-stats'), hist = $('fx-hist');
    var ticks = $('fx-ticks');
    for (var t = 0; t < 60; t++) {                       // minute ticks around the dial
      var a = t / 60 * 2 * Math.PI, major = t % 5 === 0, r1 = major ? 121 : 124, r2 = 128, l = document.createElementNS(NS, 'line');
      l.setAttribute('x1', 130 + Math.sin(a) * r1); l.setAttribute('y1', 130 - Math.cos(a) * r1); l.setAttribute('x2', 130 + Math.sin(a) * r2); l.setAttribute('y2', 130 - Math.cos(a) * r2);
      l.setAttribute('class', major ? 'fx-tk major' : 'fx-tk'); ticks.appendChild(l);
    }
    [10, 20, 30, 45, 60].forEach(function (m) {
      var b = PG.el('button', 'fx-chip' + (m === minutes ? ' on' : ''), '<b>' + m + '</b><small>min</small>'); b.type = 'button'; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', m === minutes ? 'true' : 'false');
      b.onclick = function () { if (active) return; minutes = m; [].forEach.call(chips.children, function (c) { var on = c === b; c.classList.toggle('on', on); c.setAttribute('aria-checked', on ? 'true' : 'false'); }); draw(); };
      chips.appendChild(b);
    });
    $('fx-snd').onchange = function (e) { sound = e.target.checked; PG.store.set('focus-sound', sound); };

    function draw() {
      var tot = (active ? active.minutes : minutes) * 60, left = active ? Math.max(0, (active.end - Date.now()) / 1000) : tot, frac = active ? 1 - left / tot : 0;
      prog.setAttribute('stroke-dashoffset', (C * (1 - frac)).toFixed(2));
      var ang = frac * 2 * Math.PI; dot.setAttribute('cx', (130 + Math.sin(ang) * R).toFixed(2)); dot.setAttribute('cy', (130 - Math.cos(ang) * R).toFixed(2));
      timeEl.textContent = PG.fmtTime(Math.ceil(left));
      stateEl.textContent = active ? (active.label ? active.label : 'In session') : 'Ready when you are';
      dial.classList.toggle('live', !!active);
      if (active) document.title = PG.fmtTime(Math.ceil(left)) + ' · Focus';
    }
    function ui() { bGo.hidden = !!active; bStop.hidden = !active; label.disabled = !!active; chips.classList.toggle('lock', !!active); if (active) label.value = active.label || ''; draw(); }
    function start() { active = { minutes: minutes, start: Date.now(), end: Date.now() + minutes * 60000, label: label.value.trim() }; PG.store.set('focus-active', active); msg.textContent = ''; ui(); loop(); }
    function loop() { clearInterval(tick); tick = setInterval(function () { if (!active) return; draw(); if (Date.now() >= active.end) finish(true); }, 500); }
    bGo.onclick = start; bStop.onclick = function () { finish(false); };
    function chime() { if (!sound) return; try { var A = window.AudioContext || window.webkitAudioContext, c = new A(); [660, 880, 1100].forEach(function (f, i) { var o = c.createOscillator(), g = c.createGain(); o.frequency.value = f; o.type = 'sine'; g.gain.setValueAtTime(0, c.currentTime + i * .28); g.gain.linearRampToValueAtTime(.12, c.currentTime + i * .28 + .05); g.gain.exponentialRampToValueAtTime(.001, c.currentTime + i * .28 + 1.1); o.connect(g); g.connect(c.destination); o.start(c.currentTime + i * .28); o.stop(c.currentTime + i * .28 + 1.2); }); } catch (e) {} }
    function finish(done) {
      if (!active) return; var a = active, elapsed = Math.min(a.minutes, (Date.now() - a.start) / 60000); active = null; PG.store.set('focus-active', null); clearInterval(tick); document.title = baseTitle;
      if (done) chime();
      if (done || elapsed >= 1) csrfFetch('/api/library/focus', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ minutes: a.minutes, done_minutes: elapsed, completed: !!done, label: a.label || '' }) }).then(loadHist).catch(function () {});
      ui(); msg.textContent = done ? 'Session complete. Nicely done.' : 'Session ended early. Every minute counts.';
    }
    function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
    function loadHist() { fetch('/api/library/focus').then(function (r) { return r.json(); }).then(function (j) { sessions = j.sessions || []; drawHist(); }).catch(function () {}); }
    function drawHist() {
      var now = new Date(), day0 = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime(), DAY = 864e5, bars = [], total = 0, done = 0, n7 = 0, today = 0;
      for (var i = 6; i >= 0; i--) bars.push({ t: day0 - i * DAY, m: 0 });
      sessions.forEach(function (s) {
        var at = s.at * 1000, m = +s.done_minutes || 0;
        if (at >= day0) today += m;
        if (at >= day0 - 6 * DAY) { n7++; total += m; if (s.completed) done++; for (var k = 0; k < 7; k++) if (at >= bars[k].t && at < bars[k].t + DAY) bars[k].m += m; }
      });
      var peak = Math.max.apply(null, bars.map(function (b) { return b.m; }).concat([30]));
      stats.innerHTML =
        '<div class="fx-stat"><b>' + Math.round(today) + '<small>min</small></b><span>Today</span></div>' +
        '<div class="fx-stat"><b>' + Math.round(total) + '<small>min</small></b><span>Last 7 days</span></div>' +
        '<div class="fx-stat"><b>' + n7 + '</b><span>Sessions</span></div>' +
        '<div class="fx-stat"><b>' + (n7 ? Math.round(done / n7 * 100) : 0) + '<small>%</small></b><span>Completed</span></div>' +
        '<div class="fx-week" aria-label="Minutes per day, last 7 days">' + bars.map(function (b) {
          var d = new Date(b.t); return '<div class="c"><i style="height:' + Math.max(3, Math.round(b.m / peak * 100)) + '%" title="' + Math.round(b.m) + ' min"></i><em>' + d.toLocaleDateString(undefined, { weekday: 'narrow' }) + '</em></div>';
        }).join('') + '</div>';
      var byDay = {}; sessions.slice().reverse().forEach(function (s) { var d = new Date(s.at * 1000).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' }); (byDay[d] = byDay[d] || []).push(s); });
      var h = '<h4>Session history</h4>';
      Object.keys(byDay).slice(0, 10).forEach(function (d) {
        h += '<div class="r"><span class="d">' + esc(d) + '</span><span class="s">' + byDay[d].map(function (s) { return '<i class="' + (s.completed ? '' : 'early') + '">' + Math.round(s.done_minutes) + 'm' + (s.completed ? '' : ' early') + (s.label ? ' &middot; ' + esc(s.label) : '') + '</i>'; }).join('') + '</span></div>';
      });
      if (!sessions.length) h += '<p class="fx-empty">No sessions yet. Your first one will appear here.</p>'; else h += '<p><button class="fx-link" id="fh-clear" type="button">Clear history</button></p>';
      hist.innerHTML = h;
      var c = hist.querySelector('#fh-clear'); if (c) c.onclick = function () { csrfFetch('/api/library/focus/clear', { method: 'POST' }).then(loadHist); };
    }
    if (active) { if (Date.now() >= active.end) finish(true); else { minutes = active.minutes; [].forEach.call(chips.children, function (c) { c.classList.toggle('on', c.firstChild.textContent === String(minutes)); }); loop(); } } ui(); loadHist();
    return { destroy: function () { clearInterval(tick); document.title = baseTitle; } };
  }

  show((location.hash || '').slice(1) in { books: 1, papers: 1, bible: 1, focus: 1 } ? location.hash.slice(1) : 'books');
})();
