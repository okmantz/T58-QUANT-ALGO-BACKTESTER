// Unit tests for the pure license logic in Code.gs (no Google services needed).
//   node license_sheet/test_code_gs.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const crypto = require('crypto');

const sandbox = {
  console,
  Utilities: { getUuid: () => crypto.randomUUID() },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'Code.gs'), 'utf8'), sandbox);
const { evaluate_, makeKey_, findRow_, COL, KEY_ALPHABET } = vm.runInContext(
  '({evaluate_, makeKey_, findRow_, COL, KEY_ALPHABET})', sandbox);

// vm-context objects have a foreign Object prototype; normalise the JSON reply for deepStrictEqual.
const evaluate = (...a) => { const o = evaluate_(...a); o.res = JSON.parse(JSON.stringify(o.res)); return o; };

const NOW = new Date('2026-10-21T18:00:00Z');
const blank = () => ['', 'Unused', '', '', '', '', '', '', '', '', ''];
function row(over) { const r = blank(); for (const k in over) r[COL[k]] = over[k]; return r; }
const KEY = 'T58-AAAAA-BBBBB-CCCCC-DDDDD';
const req = (o) => Object.assign({ action: 'activate', license_key: KEY, email: 'jane@example.com', device_id: 'dev1' }, o);
let passed = 0;
function test(name, fn) { fn(); passed++; console.log('  ok  ' + name); }

test('unknown key', () => {
  const o = evaluate([row({ KEY })], req({ license_key: 'T58-NOPE' }), NOW);
  assert.deepStrictEqual(o.res, { ok: false, error: 'not_found' });
});
test('key matching is case/space insensitive', () => {
  const o = evaluate([row({ KEY })], req({ license_key: '  ' + KEY.toLowerCase() + ' ' }), NOW);
  assert.strictEqual(o.res.ok, true);
});
test('activate binds device, fills email, stamps activation + last seen', () => {
  const o = evaluate([row({ KEY })], req(), NOW);
  assert.strictEqual(o.res.ok, true);
  assert.strictEqual(o.patch[COL.DEVICE], 'dev1');
  assert.strictEqual(o.patch[COL.STATUS], 'Active');
  assert.strictEqual(o.patch[COL.EMAIL], 'jane@example.com');
  assert.strictEqual(o.patch[COL.ACTIVATED], NOW);
  assert.strictEqual(o.patch[COL.SEEN], NOW);
});
test('email on file must match', () => {
  const o = evaluate([row({ KEY, EMAIL: 'bob@example.com' })], req(), NOW);
  assert.deepStrictEqual(o.res, { ok: false, error: 'email_mismatch' });
  assert.strictEqual(o.patch, null);
});
test('email comparison ignores case', () => {
  const o = evaluate([row({ KEY, EMAIL: 'JANE@Example.com' })], req(), NOW);
  assert.strictEqual(o.res.ok, true);
});
test('second device refused', () => {
  const o = evaluate([row({ KEY, STATUS: 'Active', DEVICE: 'dev1' })], req({ device_id: 'dev2' }), NOW);
  assert.deepStrictEqual(o.res, { ok: false, error: 'device_mismatch' });
  assert.strictEqual(o.patch, null);
});
test('validate from the bound device succeeds and refreshes last seen only', () => {
  const o = evaluate([row({ KEY, STATUS: 'Active', DEVICE: 'dev1', ACTIVATED: new Date('2026-10-01') })],
    req({ action: 'validate' }), NOW);
  assert.strictEqual(o.res.ok, true);
  assert.strictEqual(o.patch[COL.SEEN], NOW);
  assert.ok(!(COL.ACTIVATED in o.patch));
});
test('revoked / suspended are refused and never written', () => {
  for (const [status, err] of [['Revoked', 'revoked'], ['Suspended', 'suspended']]) {
    const o = evaluate([row({ KEY, STATUS: status, DEVICE: 'dev1' })], req({ action: 'validate' }), NOW);
    assert.deepStrictEqual(o.res, { ok: false, error: err });
    assert.strictEqual(o.patch, null);
  }
});
test('expiry is inclusive of the last day, refused after', () => {
  const day = new Date('2026-10-21T00:00:00Z');
  assert.strictEqual(evaluate([row({ KEY, EXPIRES: day })], req(), NOW).res.ok, true);
  const yesterday = new Date('2026-10-20T00:00:00Z');
  assert.deepStrictEqual(evaluate([row({ KEY, EXPIRES: yesterday })], req(), NOW).res, { ok: false, error: 'expired' });
  assert.strictEqual(evaluate([row({ KEY, EXPIRES: '2026-10-20' })], req(), NOW).res.error, 'expired');
});
test('expires_at is returned when set', () => {
  const o = evaluate([row({ KEY, EXPIRES: new Date('2027-01-01T00:00:00Z') })], req(), NOW);
  assert.strictEqual(o.res.expires_at, '2027-01-01T00:00:00.000Z');
});
test('released device: old PC cannot silently re-claim via validate, new PC can activate', () => {
  const released = [row({ KEY, STATUS: 'Inactive', DEVICE: '' })];
  const v = evaluate(released, req({ action: 'validate', device_id: 'oldpc' }), NOW);
  assert.deepStrictEqual(v.res, { ok: false, error: 'inactive' });
  assert.strictEqual(v.patch, null);
  const a = evaluate(released, req({ device_id: 'newpc' }), NOW);
  assert.strictEqual(a.res.ok, true);
  assert.strictEqual(a.patch[COL.DEVICE], 'newpc');
});
test('deactivate frees the device from the bound machine only', () => {
  const bound = [row({ KEY, STATUS: 'Active', DEVICE: 'dev1' })];
  const bad = evaluate(bound, { action: 'deactivate', license_key: KEY, device_id: 'dev2' }, NOW);
  assert.strictEqual(bad.res.error, 'device_mismatch');
  const ok = evaluate(bound, { action: 'deactivate', license_key: KEY, device_id: 'dev1' }, NOW);
  assert.strictEqual(ok.res.ok, true);
  assert.strictEqual(ok.patch[COL.DEVICE], '');
  assert.strictEqual(ok.patch[COL.STATUS], 'Inactive');
});
test('missing device id / unknown action are bad requests', () => {
  assert.strictEqual(evaluate([row({ KEY })], req({ device_id: '' }), NOW).res.error, 'bad_request');
  assert.strictEqual(evaluate([row({ KEY })], req({ action: 'dump' }), NOW).res.error, 'bad_request');
});
test('only the asked-about key is ever described (no row data leaks)', () => {
  const o = evaluate([row({ KEY, EMAIL: 'jane@example.com', CUSTOMER: 'Jane Secret' })], req(), NOW);
  assert.deepStrictEqual(Object.keys(o.res).sort(), ['expires_at', 'ok', 'plan', 'status']);
});
test('the PLAN column is returned on activate and validate (what the app shows under Account > Subscription)', () => {
  for (const plan of ['Pro Monthly', 'Pro Annual', 'Lifetime', 'Trial', 'Complimentary']) {
    const a = evaluate([row({ KEY, PLAN: plan })], req(), NOW);
    assert.strictEqual(a.res.plan, plan);
    const v = evaluate([row({ KEY, PLAN: plan, STATUS: 'Active', DEVICE: 'dev1' })], req({ action: 'validate' }), NOW);
    assert.strictEqual(v.res.plan, plan);
  }
});
test('plan is trimmed, and blank when the sheet has none', () => {
  assert.strictEqual(evaluate([row({ KEY, PLAN: '  Lifetime ' })], req(), NOW).res.plan, 'Lifetime');
  assert.strictEqual(evaluate([row({ KEY })], req(), NOW).res.plan, '');
});
test('a failed check never reveals the plan', () => {
  const o = evaluate([row({ KEY, PLAN: 'Lifetime', STATUS: 'Revoked' })], req(), NOW);
  assert.deepStrictEqual(o.res, { ok: false, error: 'revoked' });
});
test('findRow_ handles empty list and blank key', () => {
  assert.strictEqual(findRow_([], KEY), -1);
  assert.strictEqual(findRow_([row({ KEY })], ''), -1);
});
test('generated keys: format, alphabet, uniqueness (20k)', () => {
  assert.strictEqual(KEY_ALPHABET.length, 32);
  assert.strictEqual(new Set(KEY_ALPHABET).size, 32);
  const seen = new Set();
  for (let i = 0; i < 20000; i++) {
    const k = makeKey_();
    assert.match(k, /^T58-[2-9A-HJ-NP-Z]{5}-[2-9A-HJ-NP-Z]{5}-[2-9A-HJ-NP-Z]{5}-[2-9A-HJ-NP-Z]{5}$/);
    seen.add(k);
  }
  assert.strictEqual(seen.size, 20000);
});
test('generated keys use all 32 symbols roughly uniformly (no bias bug)', () => {
  const counts = {};
  for (let i = 0; i < 5000; i++) for (const ch of makeKey_().slice(4).replace(/-/g, '')) counts[ch] = (counts[ch] || 0) + 1;
  const vals = Object.values(counts);
  assert.strictEqual(vals.length, 32);
  const mean = 5000 * 20 / 32;
  assert.ok(vals.every(v => Math.abs(v - mean) < mean * 0.12), 'distribution skewed: ' + JSON.stringify(counts));
});

console.log(`\n${passed} tests passed`);
