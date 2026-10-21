/**
 * T58 TRADING -- LICENSE CONTROL CENTER
 * ---------------------------------------------------------------------------
 * One private Google Sheet is your whole licensing system:
 *   - every license key, its customer, email, plan, expiry and status
 *   - the device each key is bound to, when it was activated, when it was last seen
 *   - a live dashboard and a one-click admin menu
 * The T58 app asks this script (deployed as a web app) "is this key good?" and
 * the script answers yes/no for THAT key only. The sheet is never exposed.
 *
 * Setup: see SETUP.md. Short version: paste this file into Extensions > Apps
 * Script, run setup() once, then Deploy > New deployment > Web app
 * (Execute as: Me, Who has access: Anyone) and copy the /exec URL.
 *
 * After EDITING this script later: Deploy > Manage deployments > Edit > Version:
 * New version > Deploy. (Saving alone does not update the live web app.)
 */

// ============================ CONFIG =======================================
var LICENSES_SHEET = 'Licenses';
var DASHBOARD_SHEET = 'Dashboard';
var GUIDE_SHEET = 'Guide';
var INITIAL_KEY_COUNT = 500;
var KEY_ALPHABET = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'; // 32 symbols, no 0/O/1/I look-alikes
var STATUSES = ['Unused', 'Active', 'Inactive', 'Suspended', 'Revoked'];
var PLANS = ['Pro Monthly', 'Pro Annual', 'Lifetime', 'Trial', 'Complimentary'];

// Column order on the Licenses sheet (0-based; sheet column = index + 1).
var COL = { KEY: 0, STATUS: 1, CUSTOMER: 2, EMAIL: 3, PLAN: 4, ISSUED: 5, EXPIRES: 6,
            DEVICE: 7, ACTIVATED: 8, SEEN: 9, NOTES: 10 };
var HEADERS = ['LICENSE KEY', 'STATUS', 'CUSTOMER', 'EMAIL', 'PLAN', 'ISSUED', 'EXPIRES',
               'DEVICE ID', 'ACTIVATED', 'LAST SEEN', 'NOTES'];
var NUM_COLS = HEADERS.length;

// T58 brand palette (matches the desktop/web app).
var C = {
  BG: '#05070A', PANEL: '#0D1017', PANEL2: '#10141C', PANEL3: '#151A24',
  BORDER: '#1C2230', BORDER_LIGHT: '#2A3242', TEXT: '#E7EBF2', MUTED: '#8A93A6', DIM: '#5B6478',
  GREEN: '#35E0B0', CYAN: '#00D4FF', VIOLET: '#7B3DFF', RED: '#FF6F6F', AMBER: '#F5C451'
};
var FONT = 'Inter';
var MONO = 'Roboto Mono';

// ===================== LICENSE LOGIC (pure -- unit tested) ==================
function norm_(s) { return String(s == null ? '' : s).trim().toUpperCase(); }
function lower_(s) { return String(s == null ? '' : s).trim().toLowerCase(); }

function findRow_(rows, key) {
  var k = norm_(key);
  if (!k) return -1;
  for (var i = 0; i < rows.length; i++) {
    if (norm_(rows[i][COL.KEY]) === k) return i;
  }
  return -1;
}

function toTime_(v) {
  if (v === '' || v == null) return NaN;
  return v instanceof Date ? v.getTime() : Date.parse(String(v));
}

/**
 * Decide the answer to one request.
 * rows: the Licenses data rows (header excluded), each an array of NUM_COLS cells.
 * req : {action, license_key, email, device_id}
 * now : Date
 * Returns {res, row, patch}: the JSON to send back, the matching row index (-1 if
 * none) and a {columnIndex: newValue} patch the caller writes to the sheet.
 * A revoked/suspended/expired/mismatched answer NEVER writes anything.
 */
function evaluate_(rows, req, now) {
  var action = lower_(req.action);
  var i = findRow_(rows, req.license_key);
  if (i < 0) return { res: { ok: false, error: 'not_found' }, row: -1, patch: null };
  var r = rows[i];

  var status = lower_(r[COL.STATUS]);
  if (status === 'revoked') return { res: { ok: false, error: 'revoked' }, row: i, patch: null };
  if (status === 'suspended') return { res: { ok: false, error: 'suspended' }, row: i, patch: null };

  var expTime = toTime_(r[COL.EXPIRES]);
  var expiresAt = isNaN(expTime) ? null : new Date(expTime).toISOString();
  // An expiry date is valid THROUGH the end of that day.
  if (!isNaN(expTime) && expTime + 86400000 - 1 < now.getTime()) {
    return { res: { ok: false, error: 'expired' }, row: i, patch: null };
  }

  var email = lower_(req.email);
  var rowEmail = lower_(r[COL.EMAIL]);
  if (rowEmail && email && rowEmail !== email) {
    return { res: { ok: false, error: 'email_mismatch' }, row: i, patch: null };
  }

  var device = String(req.device_id == null ? '' : req.device_id).trim();
  var bound = String(r[COL.DEVICE] == null ? '' : r[COL.DEVICE]).trim();

  if (action === 'deactivate') {
    if (bound && device && bound !== device) {
      return { res: { ok: false, error: 'device_mismatch' }, row: i, patch: null };
    }
    var off = {}; off[COL.DEVICE] = ''; off[COL.STATUS] = 'Inactive';
    return { res: { ok: true, status: 'inactive' }, row: i, patch: off };
  }

  if (action !== 'activate' && action !== 'validate') {
    return { res: { ok: false, error: 'bad_request' }, row: i, patch: null };
  }
  if (!device) return { res: { ok: false, error: 'bad_request' }, row: i, patch: null };
  if (bound && bound !== device) {
    return { res: { ok: false, error: 'device_mismatch' }, row: i, patch: null };
  }
  // Only an explicit activation may BIND a device. A routine validate from a
  // computer whose binding was released must not silently re-claim the key
  // before the customer's new computer activates it.
  if (!bound && action === 'validate') {
    return { res: { ok: false, error: 'inactive' }, row: i, patch: null };
  }

  var patch = {};
  patch[COL.DEVICE] = device;
  patch[COL.STATUS] = 'Active';
  patch[COL.SEEN] = now;
  if (!r[COL.ACTIVATED]) patch[COL.ACTIVATED] = now;
  if (!rowEmail && email) patch[COL.EMAIL] = email;
  return { res: { ok: true, status: 'active', expires_at: expiresAt }, row: i, patch: patch };
}

function makeKey_() {
  // One random UUID = 122 random bits (Java SecureRandom). Drop the fixed version
  // and variant nibbles, then turn 100 of the remaining bits into 20 base-32 chars.
  var hex = Utilities.getUuid().replace(/-/g, '');
  hex = hex.slice(0, 12) + hex.slice(13, 16) + hex.slice(17); // 30 random hex chars
  var bits = '';
  for (var i = 0; i < 25; i++) bits += ('0000' + parseInt(hex.charAt(i), 16).toString(2)).slice(-4);
  var chars = '';
  for (var j = 0; j < 20; j++) chars += KEY_ALPHABET.charAt(parseInt(bits.substr(j * 5, 5), 2));
  return 'T58-' + [chars.slice(0, 5), chars.slice(5, 10), chars.slice(10, 15), chars.slice(15, 20)].join('-');
}

// ============================ WEB APP ======================================
function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function getSpreadsheet_() {
  var id = PropertiesService.getScriptProperties().getProperty('SHEET_ID');
  return id ? SpreadsheetApp.openById(id) : SpreadsheetApp.getActiveSpreadsheet();
}

/** The T58 app calls this: GET <exec-url>?action=activate|validate|deactivate&license_key=..&email=..&device_id=.. */
function doGet(e) {
  var lock = LockService.getScriptLock();
  try { lock.waitLock(20000); } catch (err) { return json_({ ok: false, error: 'busy' }); }
  try {
    var sheet = getSpreadsheet_().getSheetByName(LICENSES_SHEET);
    if (!sheet) return json_({ ok: false, error: 'not_configured' });
    var last = sheet.getLastRow();
    var rows = last < 2 ? [] : sheet.getRange(2, 1, last - 1, NUM_COLS).getValues();
    var out = evaluate_(rows, (e && e.parameter) || {}, new Date());
    if (out.row >= 0 && out.patch) {
      for (var col in out.patch) sheet.getRange(out.row + 2, Number(col) + 1).setValue(out.patch[col]);
    }
    return json_(out.res);
  } catch (err2) {
    return json_({ ok: false, error: 'busy' });
  } finally {
    lock.releaseLock();
  }
}

// ============================ ADMIN MENU ====================================
function onOpen() {
  SpreadsheetApp.getUi().createMenu('T58 License')
    .addItem('Generate more keys…', 'menuGenerateKeys')
    .addSeparator()
    .addItem('Revoke selected key', 'menuRevoke')
    .addItem('Suspend selected key', 'menuSuspend')
    .addItem('Reactivate selected key', 'menuReactivate')
    .addItem('Release device (selected key)', 'menuReleaseDevice')
    .addSeparator()
    .addItem('Rebuild dashboard & styling', 'setup')
    .addToUi();
}

function selectedLicenseRow_() {
  var ss = getSpreadsheet_();
  var sheet = ss.getActiveSheet();
  var ui = SpreadsheetApp.getUi();
  if (sheet.getName() !== LICENSES_SHEET) { ui.alert('Select a key on the "Licenses" tab first.'); return null; }
  var row = sheet.getActiveRange().getRow();
  if (row < 2) { ui.alert('Click a license row first (not the header).'); return null; }
  return { sheet: sheet, row: row, key: sheet.getRange(row, COL.KEY + 1).getValue() };
}

function setStatusMenu_(status, verb) {
  var s = selectedLicenseRow_();
  if (!s) return;
  var ui = SpreadsheetApp.getUi();
  if (ui.alert(verb + ' ' + s.key + '?', 'The app will pick this up the next time that customer opens T58.',
               ui.ButtonSet.OK_CANCEL) !== ui.Button.OK) return;
  s.sheet.getRange(s.row, COL.STATUS + 1).setValue(status);
}
function menuRevoke() { setStatusMenu_('Revoked', 'Revoke'); }
function menuSuspend() { setStatusMenu_('Suspended', 'Suspend'); }
function menuReactivate() { setStatusMenu_('Active', 'Reactivate'); }

function menuReleaseDevice() {
  var s = selectedLicenseRow_();
  if (!s) return;
  var ui = SpreadsheetApp.getUi();
  if (ui.alert('Release the device bound to ' + s.key + '?',
               'The customer can then activate on a new computer.', ui.ButtonSet.OK_CANCEL) !== ui.Button.OK) return;
  s.sheet.getRange(s.row, COL.DEVICE + 1).setValue('');
  s.sheet.getRange(s.row, COL.STATUS + 1).setValue('Inactive');
}

function menuGenerateKeys() {
  var ui = SpreadsheetApp.getUi();
  var resp = ui.prompt('Generate more keys', 'How many new keys? (1-1000)', ui.ButtonSet.OK_CANCEL);
  if (resp.getSelectedButton() !== ui.Button.OK) return;
  var n = parseInt(resp.getResponseText(), 10);
  if (!(n >= 1 && n <= 1000)) { ui.alert('Enter a number from 1 to 1000.'); return; }
  appendKeys_(getSpreadsheet_().getSheetByName(LICENSES_SHEET), n);
  ui.alert('Added ' + n + ' new keys at the bottom of the Licenses tab.');
}

function appendKeys_(sheet, n) {
  var existing = {};
  var last = sheet.getLastRow();
  if (last >= 2) {
    sheet.getRange(2, 1, last - 1, 1).getValues().forEach(function (r) { existing[norm_(r[0])] = true; });
  }
  var rows = [];
  while (rows.length < n) {
    var k = makeKey_();
    if (existing[k]) continue;
    existing[k] = true;
    var row = [];
    for (var c = 0; c < NUM_COLS; c++) row.push('');
    row[COL.KEY] = k; row[COL.STATUS] = 'Unused'; row[COL.ISSUED] = '';
    rows.push(row);
  }
  var start = Math.max(last, 1) + 1;
  if (sheet.getMaxRows() < start + n) sheet.insertRowsAfter(sheet.getMaxRows(), start + n - sheet.getMaxRows() + 20);
  sheet.getRange(start, 1, n, NUM_COLS).setValues(rows);
  styleLicenses_(sheet);
}

// ============================ SETUP / STYLING ===============================
/** Run once. Safe to re-run: existing keys and customer data are never touched. */
function setup() {
  var ss = SpreadsheetApp.getActiveSpreadsheet() || getSpreadsheet_();
  PropertiesService.getScriptProperties().setProperty('SHEET_ID', ss.getId());

  var lic = ss.getSheetByName(LICENSES_SHEET) || ss.insertSheet(LICENSES_SHEET);
  var dash = ss.getSheetByName(DASHBOARD_SHEET) || ss.insertSheet(DASHBOARD_SHEET);
  var guide = ss.getSheetByName(GUIDE_SHEET) || ss.insertSheet(GUIDE_SHEET);

  if (lic.getLastRow() < 2) {
    lic.getRange(1, 1, 1, NUM_COLS).setValues([HEADERS]);
    appendKeys_(lic, INITIAL_KEY_COUNT);
  } else {
    lic.getRange(1, 1, 1, NUM_COLS).setValues([HEADERS]);
  }
  styleLicenses_(lic);
  buildDashboard_(dash, lic);
  buildGuide_(guide);

  ss.setActiveSheet(dash);
  ss.moveActiveSheet(1);
  ss.setActiveSheet(lic);
  ss.moveActiveSheet(2);
  ss.setActiveSheet(guide);
  ss.moveActiveSheet(3);
  ss.setActiveSheet(dash);
  // Remove the empty default tab if there is one.
  var def = ss.getSheetByName('Sheet1');
  if (def && def.getLastRow() === 0 && ss.getSheets().length > 3) ss.deleteSheet(def);
  ss.toast('Licenses, dashboard and guide are ready. Next: Deploy > New deployment > Web app.', 'T58 License', 8);
}

function paint_(sheet) {
  var rows = sheet.getMaxRows(), cols = sheet.getMaxColumns();
  sheet.getRange(1, 1, rows, cols).setBackground(C.BG).setFontColor(C.TEXT).setFontFamily(FONT).setFontSize(10)
       .setVerticalAlignment('middle');
}

function trimGrid_(sheet, usedRows, usedCols, extraRows) {
  var maxC = sheet.getMaxColumns();
  if (maxC > usedCols) sheet.deleteColumns(usedCols + 1, maxC - usedCols);
  var maxR = sheet.getMaxRows(), want = usedRows + extraRows;
  if (maxR > want) sheet.deleteRows(want + 1, maxR - want);
  else if (maxR < want) sheet.insertRowsAfter(maxR, want - maxR);
}

function styleLicenses_(sheet) {
  var last = Math.max(sheet.getLastRow(), 2);
  trimGrid_(sheet, last, NUM_COLS, 60);
  paint_(sheet);
  sheet.setHiddenGridlines(true);
  sheet.setTabColor(C.VIOLET);
  sheet.setFrozenRows(1);
  sheet.setFrozenColumns(1);

  var widths = [250, 105, 160, 230, 125, 100, 100, 270, 150, 150, 240];
  widths.forEach(function (w, i) { sheet.setColumnWidth(i + 1, w); });
  sheet.setRowHeight(1, 40);
  if (last > 1) sheet.setRowHeights(2, last - 1, 30);

  // Header
  var head = sheet.getRange(1, 1, 1, NUM_COLS);
  head.setBackground(C.PANEL3).setFontColor(C.CYAN).setFontWeight('bold').setFontSize(9)
      .setHorizontalAlignment('left').setBorder(false, false, true, false, false, false, C.CYAN,
                                                SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
  // Body
  var body = sheet.getRange(2, 1, last - 1, NUM_COLS);
  body.setHorizontalAlignment('left').setBorder(false, false, true, false, false, true, C.BORDER,
                                                SpreadsheetApp.BorderStyle.SOLID);
  sheet.getRange(2, COL.KEY + 1, last - 1, 1).setFontFamily(MONO).setFontColor(C.CYAN).setFontWeight('bold');
  sheet.getRange(2, COL.DEVICE + 1, last - 1, 1).setFontFamily(MONO).setFontColor(C.MUTED).setFontSize(9);
  sheet.getRange(2, COL.STATUS + 1, last - 1, 1).setHorizontalAlignment('center').setFontWeight('bold').setFontSize(9);
  sheet.getRange(2, COL.NOTES + 1, last - 1, 1).setFontColor(C.MUTED);
  [COL.ISSUED, COL.EXPIRES].forEach(function (c) {
    sheet.getRange(2, c + 1, last - 1, 1).setNumberFormat('yyyy-mm-dd');
  });
  [COL.ACTIVATED, COL.SEEN].forEach(function (c) {
    sheet.getRange(2, c + 1, last - 1, 1).setNumberFormat('yyyy-mm-dd hh:mm').setFontColor(C.MUTED);
  });

  // Dropdowns
  sheet.getRange(2, COL.STATUS + 1, last - 1, 1).setDataValidation(
    SpreadsheetApp.newDataValidation().requireValueInList(STATUSES, true).setAllowInvalid(false).build());
  sheet.getRange(2, COL.PLAN + 1, last - 1, 1).setDataValidation(
    SpreadsheetApp.newDataValidation().requireValueInList(PLANS, true).setAllowInvalid(true).build());

  // Status chips, then banding (first matching rule wins).
  var statusRange = sheet.getRange(2, COL.STATUS + 1, last - 1, 1);
  var chip = function (text, bg, fg) {
    return SpreadsheetApp.newConditionalFormatRule().whenTextEqualTo(text).setBackground(bg).setFontColor(fg)
      .setRanges([statusRange]).build();
  };
  var allBody = sheet.getRange(2, 1, last - 1, NUM_COLS);
  var rules = [
    chip('Active', '#0E2E27', C.GREEN),
    chip('Unused', '#1A2030', C.MUTED),
    chip('Inactive', '#2B2A14', C.AMBER),
    chip('Suspended', '#33270F', '#FFB347'),
    chip('Revoked', '#3A1519', C.RED),
    // Revoked / suspended rows dim the whole line so they are obvious at a glance.
    SpreadsheetApp.newConditionalFormatRule().whenFormulaSatisfied('=OR($B2="Revoked",$B2="Suspended")')
      .setFontColor(C.DIM).setRanges([allBody]).build(),
    SpreadsheetApp.newConditionalFormatRule().whenFormulaSatisfied('=ISEVEN(ROW())')
      .setBackground(C.PANEL).setRanges([allBody]).build(),
    SpreadsheetApp.newConditionalFormatRule().whenFormulaSatisfied('=ISODD(ROW())')
      .setBackground(C.PANEL2).setRanges([allBody]).build()
  ];
  sheet.setConditionalFormatRules(rules);

  // Filter
  var existing = sheet.getFilter();
  if (existing) existing.remove();
  sheet.getRange(1, 1, last, NUM_COLS).createFilter();

  // Soft-protect the columns the system writes, so a stray edit shows a warning.
  sheet.getProtections(SpreadsheetApp.ProtectionType.RANGE).forEach(function (p) { p.remove(); });
  [[1, 1, last, 1, 'License keys'], [1, COL.DEVICE + 1, last, 1, 'Bound device (set by the app)'],
   [1, COL.ACTIVATED + 1, last, 2, 'Activity (set by the app)']].forEach(function (a) {
    sheet.getRange(a[0], a[1], a[2], a[3]).protect().setDescription(a[4]).setWarningOnly(true);
  });
}

function tile_(sheet, col, label, formula, color, caption) {
  var lab = sheet.getRange(6, col, 1, 2).merge();
  lab.setValue(label).setFontColor(C.MUTED).setFontSize(8).setFontWeight('bold').setHorizontalAlignment('left');
  var val = sheet.getRange(7, col, 1, 2).merge();
  val.setFormula(formula).setFontColor(color).setFontSize(30).setFontWeight('bold').setHorizontalAlignment('left');
  var cap = sheet.getRange(8, col, 1, 2).merge();
  cap.setValue(caption).setFontColor(C.DIM).setFontSize(8).setHorizontalAlignment('left');
  var box = sheet.getRange(6, col, 3, 2);
  box.setBackground(C.PANEL2).setBorder(true, true, true, true, false, false, C.BORDER, SpreadsheetApp.BorderStyle.SOLID);
  sheet.getRange(6, col, 3, 1).setBorder(null, true, null, null, null, null, color, SpreadsheetApp.BorderStyle.SOLID_THICK);
}

function section_(sheet, row, col, width, text) {
  var r = sheet.getRange(row, col, 1, width).merge();
  r.setValue(text).setFontColor(C.CYAN).setFontSize(9).setFontWeight('bold').setHorizontalAlignment('left')
   .setBorder(false, false, true, false, false, false, C.BORDER_LIGHT, SpreadsheetApp.BorderStyle.SOLID);
}

function buildDashboard_(sheet, lic) {
  sheet.clear();
  sheet.getCharts().forEach(function (ch) { sheet.removeChart(ch); });
  sheet.getRange(1, 1, sheet.getMaxRows(), sheet.getMaxColumns()).breakApart();
  trimGrid_(sheet, 30, 12, 0);
  paint_(sheet);
  sheet.setHiddenGridlines(true);
  sheet.setTabColor(C.CYAN);
  sheet.setColumnWidth(1, 28); sheet.setColumnWidth(12, 28);
  for (var c = 2; c <= 11; c++) sheet.setColumnWidth(c, 160);
  [16, 22, 46, 24, 18, 24, 58, 20, 18].forEach(function (h, i) { sheet.setRowHeight(i + 1, h); });

  // Banner
  var tag = sheet.getRange(2, 2, 1, 10).merge();
  tag.setValue('T58 TRADING').setFontColor(C.CYAN).setFontSize(9).setFontWeight('bold').setHorizontalAlignment('left');
  var title = sheet.getRange(3, 2, 1, 10).merge();
  title.setValue('LICENSE CONTROL CENTER').setFontColor('#FFFFFF').setFontSize(24).setFontWeight('bold')
       .setHorizontalAlignment('left');
  var sub = sheet.getRange(4, 2, 1, 10).merge();
  sub.setValue('Quant Algo Backtester  ·  Validate, don’t guess.').setFontColor(C.MUTED).setFontSize(10)
     .setHorizontalAlignment('left');
  sheet.getRange(4, 2, 1, 10).setBorder(false, false, true, false, false, false, C.VIOLET,
                                        SpreadsheetApp.BorderStyle.SOLID_MEDIUM);

  var L = "Licenses!";
  tile_(sheet, 2, 'TOTAL KEYS', '=COUNTA(' + L + 'A2:A)', C.TEXT, 'in the system');
  tile_(sheet, 4, 'ACTIVE', '=COUNTIF(' + L + 'B2:B,"Active")', C.GREEN, 'activated on a device');
  tile_(sheet, 6, 'UNUSED', '=COUNTIF(' + L + 'B2:B,"Unused")', C.CYAN, 'ready to sell');
  tile_(sheet, 8, 'SEEN IN 24H', '=COUNTIF(' + L + 'J2:J,">"&(NOW()-1))', C.VIOLET, 'opened the app today');
  tile_(sheet, 10, 'SUSPENDED / REVOKED', '=COUNTIF(' + L + 'B2:B,"Suspended")+COUNTIF(' + L + 'B2:B,"Revoked")',
        C.RED, 'blocked from launching');

  // Status breakdown table + doughnut chart
  section_(sheet, 10, 2, 3, 'STATUS BREAKDOWN');
  var rows = [['Active', C.GREEN], ['Unused', C.CYAN], ['Inactive', C.AMBER], ['Suspended', '#FFB347'], ['Revoked', C.RED]];
  rows.forEach(function (r, i) {
    var row = 11 + i;
    sheet.getRange(row, 2).setValue('●  ' + r[0]).setFontColor(r[1]).setFontWeight('bold');
    sheet.getRange(row, 3).setFormula('=COUNTIF(' + L + 'B2:B,"' + r[0] + '")').setFontColor(C.TEXT)
         .setHorizontalAlignment('right');
    sheet.getRange(row, 4).setFormula('=IFERROR(C' + row + '/SUM($C$11:$C$15),0)').setNumberFormat('0%')
         .setFontColor(C.MUTED).setHorizontalAlignment('right');
    sheet.setRowHeight(row, 30);
  });
  sheet.getRange(11, 2, 5, 3).setBackground(C.PANEL2).setBorder(true, true, true, true, false, true, C.BORDER,
                                                                SpreadsheetApp.BorderStyle.SOLID);
  var chart = sheet.newChart().setChartType(Charts.ChartType.PIE)
    .addRange(sheet.getRange(11, 2, 5, 2))
    .setPosition(10, 6, 0, 0)
    .setOption('width', 800).setOption('height', 190)
    .setOption('pieHole', 0.6).setOption('backgroundColor', C.BG)
    .setOption('colors', [C.GREEN, C.CYAN, C.AMBER, '#FFB347', C.RED])
    .setOption('legend', { position: 'right', textStyle: { color: C.MUTED, fontSize: 11 } })
    .setOption('pieSliceText', 'none').setOption('chartArea', { width: '90%', height: '90%' })
    .build();
  sheet.insertChart(chart);

  // Recent activity
  section_(sheet, 18, 2, 10, 'RECENT ACTIVITY');
  var hdr = ['CUSTOMER', 'EMAIL', 'STATUS', 'LAST SEEN'];
  hdr.forEach(function (h, i) {
    sheet.getRange(19, 2 + i).setValue(h).setFontColor(C.DIM).setFontSize(8).setFontWeight('bold');
  });
  sheet.getRange(20, 2).setFormula('=IFERROR(QUERY(' + L + 'A2:K,"select C, D, B, J where J is not null order by J desc limit 8",0),"")');
  sheet.getRange(20, 5, 8, 1).setNumberFormat('yyyy-mm-dd hh:mm');
  sheet.getRange(20, 2, 8, 4).setFontColor(C.TEXT).setBackground(C.PANEL2);
  for (var r2 = 20; r2 < 28; r2++) sheet.setRowHeight(r2, 28);
  sheet.getRange(20, 2, 8, 4).setBorder(false, false, true, false, false, true, C.BORDER, SpreadsheetApp.BorderStyle.SOLID);

  sheet.getRange(29, 2, 1, 10).merge()
    .setValue('Manage keys on the Licenses tab · use the "T58 License" menu to revoke, suspend or release a device · everything here updates live.')
    .setFontColor(C.DIM).setFontSize(8).setHorizontalAlignment('left');
}

function buildGuide_(sheet) {
  sheet.clear();
  sheet.getRange(1, 1, sheet.getMaxRows(), sheet.getMaxColumns()).breakApart();
  trimGrid_(sheet, 40, 4, 0);
  paint_(sheet);
  sheet.setHiddenGridlines(true);
  sheet.setTabColor(C.DIM);
  sheet.setColumnWidth(1, 28); sheet.setColumnWidth(2, 70); sheet.setColumnWidth(3, 720); sheet.setColumnWidth(4, 28);

  sheet.getRange(2, 2, 1, 2).merge().setValue('T58 TRADING').setFontColor(C.CYAN).setFontSize(9).setFontWeight('bold');
  sheet.getRange(3, 2, 1, 2).merge().setValue('HOW THIS WORKS').setFontColor('#FFFFFF').setFontSize(22).setFontWeight('bold');
  sheet.setRowHeight(3, 44);
  sheet.getRange(4, 2, 1, 2).setBorder(false, false, true, false, false, false, C.VIOLET, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);

  var items = [
    ['SELL', 'After a sale, pick an Unused key on the Licenses tab and copy it to the customer. Fill in CUSTOMER, EMAIL, PLAN and ISSUED. If EMAIL is filled in, only that email can activate the key; if you leave it blank, the first activation fills it in.'],
    ['ACTIVATE', 'The customer enters their email and key in T58. The key locks to that computer (DEVICE ID) and the status turns Active. A second computer is refused until you release the device.'],
    ['EXPIRE', 'Put a date in EXPIRES for trials and monthly plans; the key works through the end of that day. Leave it blank for no expiry.'],
    ['REVOKE', 'Set STATUS to Revoked (or use the T58 License menu). The app locks the next time that customer opens it — and at the latest after the 3-day offline grace period.'],
    ['SUSPEND', 'Same as revoke, but meant to be temporary (a failed payment, a dispute). Set STATUS back to Active to restore it.'],
    ['NEW PC', 'T58 License menu → Release device (selected key). The customer can then activate on their new computer.'],
    ['MORE KEYS', 'T58 License menu → Generate more keys. Keys are random, 100-bit, and cannot be guessed.'],
    ['KEEP PRIVATE', 'This spreadsheet must stay Restricted (only you). Never set sharing to "Anyone with the link" — it holds every key and customer email. The app never reads the sheet directly; it only talks to the deployed web app, which answers about one key at a time.'],
    ['AFTER EDITS', 'If you ever edit the script: Deploy → Manage deployments → Edit → Version: New version → Deploy. Saving alone does not update the live endpoint.']
  ];
  items.forEach(function (it, i) {
    var row = 6 + i * 3;
    sheet.getRange(row, 2).setValue(it[0]).setFontColor(C.CYAN).setFontSize(9).setFontWeight('bold')
         .setVerticalAlignment('top');
    sheet.getRange(row, 3).setValue(it[1]).setFontColor(C.TEXT).setFontSize(10).setWrap(true)
         .setVerticalAlignment('top');
    sheet.setRowHeight(row, it[1].length > 150 ? 64 : 44);
    sheet.setRowHeight(row + 1, 8);
    sheet.getRange(row + 1, 2, 1, 2).setBorder(false, false, true, false, false, false, C.BORDER,
                                              SpreadsheetApp.BorderStyle.SOLID);
    sheet.setRowHeight(row + 2, 8);
  });
}
