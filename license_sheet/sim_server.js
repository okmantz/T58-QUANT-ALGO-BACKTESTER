// Local stand-in for the deployed Apps Script web app, used by
// tests/licensing/test_sheet_mode.py. It loads the REAL Code.gs, keeps the sheet
// rows in memory, and serves doGet-equivalent answers -- including the 302
// redirect Google's /exec URL performs -- so the Python client is tested against
// the actual license logic, not a re-implementation of it.
//
//   node license_sheet/sim_server.js      (prints "PORT <n>" when ready)
//   GET  /exec?action=..&license_key=..   -> 302 -> /answer?same query -> JSON
//   POST /__admin  {"rows": [[...]], "mode": "ok|busy|html|http500"}   (test control)
//   GET  /__dump                           -> current rows as JSON
const fs = require('fs');
const http = require('http');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');

const sandbox = { console, Utilities: { getUuid: () => crypto.randomUUID() } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'Code.gs'), 'utf8'), sandbox);
const { evaluate_ } = vm.runInContext('({evaluate_})', sandbox);

let rows = [];
let mode = 'ok';

const server = http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  if (u.pathname === '/exec') {
    res.writeHead(302, { Location: '/answer' + u.search });
    return res.end();
  }
  if (u.pathname === '/answer') {
    if (mode === 'http500') { res.writeHead(500); return res.end('boom'); }
    if (mode === 'html') { res.writeHead(200, { 'Content-Type': 'text/html' }); return res.end('<html>Sign in to Google</html>'); }
    let out;
    if (mode === 'busy') out = { ok: false, error: 'busy' };
    else {
      const r = evaluate_(rows, Object.fromEntries(u.searchParams), new Date());
      if (r.row >= 0 && r.patch) for (const col in r.patch) rows[r.row][Number(col)] = r.patch[col];
      out = r.res;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(out));
  }
  if (u.pathname === '/__dump') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(rows));
  }
  if (u.pathname === '/__admin' && req.method === 'POST') {
    let body = '';
    req.on('data', (d) => (body += d));
    req.on('end', () => {
      const msg = JSON.parse(body || '{}');
      if (msg.rows) rows = msg.rows;
      if (msg.mode) mode = msg.mode;
      res.writeHead(200); res.end('ok');
    });
    return;
  }
  res.writeHead(404); res.end('not found');
});

server.listen(0, '127.0.0.1', () => console.log('PORT ' + server.address().port));
