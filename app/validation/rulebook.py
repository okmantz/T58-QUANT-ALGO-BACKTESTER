"""T58 Prop Firm Rulebook v2 (2026-11-07) as code: the math, the gates G0-G7, the kill line, the attempt score and the trial register.

v2 designs a strategy BACKWARDS from the account's rules and pay structure, then spends fees as a capped sequential test. This module
is the single implementation of its numbers (the app, the docs and the tests quote these and nothing else); it replaces the v1 "Edge
Rulebook" (+0.15R certificate, binary screen, six stages) entirely.

    E_plan = E_hat - E[max N] * SE            planning edge: observed edge minus the selection bias of N looks (SE = day-block
                                              standard error; sigma_R / sqrt(n_eff) in the closed form)
    E[max N] ~ (1-g) Phi^-1(1-1/N) + g Phi^-1(1-1/(N e))                          g = Euler-Mascheroni
    G3 asks E_plan >= +0.05R, so the observed edge must clear   0.05 + E[max N] * sigma_R / sqrt(n)
    LLR_t = sum d/sigma^2 (R_i - d/2), d = 0.10 (H1 = +0.10R vs H0 = no edge), kill at LLR <= -2.25   (centre +0.05R)
    Attempt score: +1.5 per paid attempt, -0.9 per unpaid; scale at +2.2, retire at -2.2, at most 6 attempts

Gates (run in order; the first failure archives the idea with the failing number -- it is not tuned and re-run):

    G0 rules sheet   G1 design envelope   G2 mechanism + pre-registration   G3 development evidence
    G4 day behaviour G5 lifecycle         G6 seal + live sim                G7 capped paid attempts

Verdict ladder: G0-G2 = LEAD, G3-G5 = FORWARD-TEST CANDIDATE, G6 = EVALUATE (buy capped attempts), a scale decision at G7 with a recorded
first payout = FUNDED. The simulated figures come from ``app.prop.lifecycle_v2`` (i.i.d. binary priors at design time; REAL day-block trades at G5).
numpy/stdlib plus the app's own bootstrap helpers.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from app.validation.edge_stats import _norm_ppf

RULEBOOK_VERSION = "v2"
RULEBOOK_DATE = "2026-11-07"

# ---- standing constants -----------------------------------------------------------------------------------------------------
E_PLAN_FLOOR_R = 0.05              # G3: the planning (deflated) edge must be at least this
BREAKEVEN_EV_EDGE_R = 0.035        # first-payout EV per attempt turns positive near this net edge (v1 table, $172 fee)
SIGMA_R_AT_2_TO_1 = 1.46           # per-trade SD of R for a 2:1 binary payoff
MAX_LOOKS_PER_FAMILY = 3           # G2: <= 3 variants per family, pre-registered
G3_MIN_TRADES = 400                # G3 floor (raised, never lowered, by the derived floor below)
G3_MIN_TRADING_DAYS = 150
G3_MIN_QUARTERS_POSITIVE = 0.60
G3_MAX_YEAR_SHARE = 0.40
G3_NEIGHBOUR_KEEP = 0.70
G4_MAX_DAY_LOSS_P = 1.0            # percent
G4_MIN_QUALIFYING_DAY_RATE = 25.0  # percent
G4_MAX_LEG_CORR = 0.30
G5_MIN_PASS_LB = 70.0              # Wilson 5th percentile, percent
G5_MAX_EVAL_BUST = 20.0
G5_MIN_PAID_GIVEN_PASS = 75.0
G5_MIN_TWO_PLUS_GIVEN_PAID = 60.0
G5_MAX_ATTEMPTS_PER_PAYOUT = 3.0
G5_PAYOUT_HAIRCUT = 0.50
G6_MIN_LIVE_SIM_DAYS = 20
G6_MAX_SLIPPAGE_RATIO = 1.25
KILL_H1_R = 0.10                   # v2 centres the kill line at +0.05R (v1 used +0.15R, centre +0.075R)
KILL_H0_R = 0.0
KILL_LLR = -2.25
ATTEMPT_PAID_SCORE = 1.5
ATTEMPT_UNPAID_SCORE = -0.9
ATTEMPT_SCALE_AT = 2.2
ATTEMPT_RETIRE_AT = -2.2
ATTEMPT_MAX = 6
P_PAID_NO_EDGE = 15.0              # percent, v2 sizing (priors the weights come from)
P_PAID_AT_PLUS_010 = 66.0
RULES_SHEET_MAX_AGE_DAYS = 30
EULER_GAMMA = 0.5772156649015329

VERDICT_REJECT, VERDICT_LEAD, VERDICT_CANDIDATE, VERDICT_EVALUATE, VERDICT_FUNDED = (
    "REJECT", "LEAD", "FORWARD-TEST CANDIDATE", "EVALUATE", "FUNDED")


# ---- core math ----------------------------------------------------------------------------------------------------------------
def edge_per_trade(p: float, b: float, c: float = 0.0) -> float:
    """E = p(b+1) - 1 - c: net edge per trade (R) from win rate p, payoff b and round-turn cost c (R)."""
    return p * (b + 1.0) - 1.0 - c


def breakeven_win_rate(b: float, c: float = 0.0) -> float:
    return (1.0 + c) / (b + 1.0)


def z_bar(n_looks: int, alpha: float = 0.05) -> float:
    """Phi^-1(1 - (alpha/2)/N): the Bonferroni bar after N looks (kept for the noise-ceiling judgments elsewhere in the app)."""
    return float(_norm_ppf(1.0 - (alpha / 2.0) / max(int(n_looks), 1)))


def expected_max_z(n_looks: int) -> float:
    """E[max of N pure-noise z-scores] (0 for N <= 1): the Bailey-Lopez de Prado approximation."""
    n = int(n_looks)
    if n <= 1:
        return 0.0
    return float((1.0 - EULER_GAMMA) * _norm_ppf(1.0 - 1.0 / n) + EULER_GAMMA * _norm_ppf(1.0 - 1.0 / (n * math.e)))


def e_plan(e_hat: float, n_looks: int, se: float) -> float:
    """The planning edge: observed edge minus E[max N] standard errors. Every sizing, EV and fee decision uses this, never the raw mean."""
    return float(e_hat) - expected_max_z(n_looks) * float(se)


def required_observed_edge(n_trades: int, n_looks: int, sigma_r: float = SIGMA_R_AT_2_TO_1, floor: float = E_PLAN_FLOOR_R) -> float:
    """The observed edge that leaves E_plan = ``floor``: floor + E[max N] * sigma_R / sqrt(n). Reproduces the Rulebook's G3 table."""
    return float(floor) + expected_max_z(n_looks) * float(sigma_r) / math.sqrt(max(int(n_trades), 1))


def g3_table(trades=(400, 600, 1000, 1500), looks=(1, 3, 5, 10, 25), sigma_r: float = SIGMA_R_AT_2_TO_1) -> list[dict]:
    return [{"trades": n, **{f"N={k}": round(required_observed_edge(n, k, sigma_r), 3) for k in looks}} for n in trades]


def trade_floor(n_looks: int, observed_edge: float, sigma_r: float = SIGMA_R_AT_2_TO_1) -> int:
    """ONE trade floor, derived from power: the fewest trades at which ``observed_edge`` still clears E_plan >= +0.05R after ``n_looks``
    looks (never below the G3 floor of 400). None of the old 100 / 300 / 400 / 600 floors survives; this replaces all four."""
    margin = float(observed_edge) - E_PLAN_FLOOR_R
    if margin <= 0:
        return 10 ** 9
    need = (expected_max_z(n_looks) * float(sigma_r) / margin) ** 2
    return int(max(G3_MIN_TRADES, math.ceil(need)))


def deflation_example(observed: float = 0.15, n_trades: int = 600, n_looks: int = 50, sigma_r: float = SIGMA_R_AT_2_TO_1) -> float:
    """The Rulebook's worked example: +0.15R at 600 trades and 50 looks deflates to about +0.014R."""
    return e_plan(observed, n_looks, sigma_r / math.sqrt(n_trades))


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    ph = wins / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def effective_bets(n_edges: int, avg_corr: float) -> float:
    """k_eff = n / (1 + (n-1) rho): independent bets in n legs of average correlation rho."""
    n = max(int(n_edges), 0)
    return 0.0 if n == 0 else n / (1.0 + (n - 1) * float(avg_corr))


# ---- Layer 2: the recentred kill line --------------------------------------------------------------------------------------
@dataclass
class LLRKillSwitch:
    """Running log-likelihood ratio of 'the edge is +0.10R' (H1) vs 'no edge' (H0): LLR += d/sigma^2 (R - d/2). Cut the strategy at <= -2.25.

    v1 tested +0.15R against 0 (line centred at +0.075R, above the +0.035R break-even) and so killed a +0.10R edge 35% of the time. v2 tests
    +0.10R against 0, centred at +0.05R: a +0.10R edge is killed in about 8% of 1,500-trade paths and a dead edge in about 82%."""
    sigma_r: float = SIGMA_R_AT_2_TO_1
    h1_r: float = KILL_H1_R
    h0_r: float = KILL_H0_R
    kill_at: float = KILL_LLR
    llr: float = 0.0
    n: int = 0
    killed_at_trade: int | None = None

    def add(self, r: float) -> dict:
        self.n += 1
        self.llr += (self.h1_r - self.h0_r) / (self.sigma_r ** 2) * (float(r) - (self.h1_r + self.h0_r) / 2.0)
        if self.killed_at_trade is None and self.llr <= self.kill_at:
            self.killed_at_trade = self.n
        return self.status()

    def status(self) -> dict:
        return {"llr": self.llr, "n": self.n, "kill_at": self.kill_at, "killed": self.killed_at_trade is not None,
                "killed_at_trade": self.killed_at_trade, "distance_to_kill": self.llr - self.kill_at,
                "h1_r": self.h1_r, "centre_r": (self.h1_r + self.h0_r) / 2.0}


def llr_path(r_values, **kw) -> dict:
    ks = LLRKillSwitch(**kw)
    for x in r_values:
        if x is not None and math.isfinite(float(x)):
            ks.add(float(x))
    return ks.status()


# ---- Layer 3 / G7: the attempt score -----------------------------------------------------------------------------------------
@dataclass
class AttemptScore:
    """Capped sequential test on paid vs unpaid attempts. Undecided at the cap means retire."""
    paid_weight: float = ATTEMPT_PAID_SCORE
    unpaid_weight: float = ATTEMPT_UNPAID_SCORE
    scale_at: float = ATTEMPT_SCALE_AT
    retire_at: float = ATTEMPT_RETIRE_AT
    max_attempts: int = ATTEMPT_MAX
    outcomes: list = field(default_factory=list)       # True = paid, False = unpaid

    def add(self, paid: bool) -> dict:
        if self.decision() == "CONTINUE":
            self.outcomes.append(bool(paid))
        return self.status()

    @property
    def score(self) -> float:
        return sum(self.paid_weight if o else self.unpaid_weight for o in self.outcomes)

    def decision(self) -> str:
        s = self.score
        if s >= self.scale_at:
            return "SCALE"
        if s <= self.retire_at:
            return "RETIRE"
        return "RETIRE" if len(self.outcomes) >= self.max_attempts else "CONTINUE"

    def status(self) -> dict:
        d = self.decision()
        undecided = d == "RETIRE" and self.retire_at < self.score < self.scale_at
        return {"score": round(self.score, 3), "attempts": len(self.outcomes), "paid": sum(self.outcomes), "decision": d,
                "left": max(self.max_attempts - len(self.outcomes), 0) if d == "CONTINUE" else 0,
                "undecided_at_cap": undecided, "scale_at": self.scale_at, "retire_at": self.retire_at,
                "fees_cap": None}


def attempt_score_from_ledger(rows: list) -> dict:
    """``rows``: [{"paid": bool}, ...] in order. Stops counting at the first decision."""
    a = AttemptScore()
    for r in rows or []:
        a.add(bool((r or {}).get("paid")))
    return a.status()


# ---- gate result -------------------------------------------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool | None            # None = could not be evaluated (a pending check cannot pass a gate)
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class GateResult:
    gate: str
    name: str
    checks: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    skipped: bool = False
    note: str = ""

    @property
    def passed(self) -> bool:
        return (not self.skipped) and bool(self.checks) and all(c.ok is True for c in self.checks)

    @property
    def incomplete(self) -> bool:
        return (not self.skipped) and any(c.ok is None for c in self.checks) and not any(c.ok is False for c in self.checks)

    @property
    def failing(self) -> list:
        return [c for c in self.checks if c.ok is False]

    def failing_number(self) -> str:
        f = self.failing or [c for c in self.checks if c.ok is None]
        return f"{f[0].name}: {f[0].detail}" if f else ""

    def to_dict(self) -> dict:
        return {"gate": self.gate, "name": self.name, "passed": self.passed, "incomplete": self.incomplete, "skipped": self.skipped,
                "checks": [c.to_dict() for c in self.checks], "data": self.data, "note": self.note,
                "failing_number": self.failing_number() if not self.passed else ""}


# ---- G0: rules sheet --------------------------------------------------------------------------------------------------------
RULES_SHEET_FIELDS = (
    ("firm", "Firm"), ("product", "Product (Pro or Flex), fixed"), ("account_size", "Account size ($)"), ("fee", "Evaluation fee ($)"),
    ("reset_fee", "Reset fee ($)"), ("monthly_fee", "Monthly fee ($)"), ("profit_target", "Profit target ($)"),
    ("max_loss", "Trailing max loss ($)"), ("soft_daily_limit", "Soft daily loss limit ($)"), ("consistency", "Consistency rule"),
    ("payout_days", "Payout: qualifying days"), ("payout_daily_min", "Payout: daily minimum ($)"),
    ("payout_cumulative_min", "Payout: cumulative minimum ($)"), ("payout_buffer", "Payout: buffer above floor ($)"),
    ("payout_cap", "Payout: cap per payout ($)"), ("split", "Profit split (%)"), ("payout_cadence", "Payout cadence"),
)
RULES_SHEET_CONFIRMATIONS = (
    ("automation_confirmed", "Automation confirmed in writing"), ("news_confirmed", "News-window trading confirmed in writing"),
    ("copy_trading_confirmed", "Copy-trading confirmed in writing"), ("overnight_confirmed", "Holding across the close confirmed in writing"),
)


def _sheet_path() -> Path:
    from app.data.storage import get_app_base_dir
    d = Path(get_app_base_dir()) / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "rules_sheet.json"


def load_rules_sheet() -> dict:
    try:
        raw = json.loads(_sheet_path().read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_rules_sheet(sheet: dict) -> None:
    _sheet_path().write_text(json.dumps(sheet, indent=2), encoding="utf-8")


def rules_sheet_from_preset(preset_key: str, *, checked_on: str | None = None) -> dict:
    """Pre-fill the G0 sheet from a prop-firm preset: only the fields the preset genuinely carries, each with the preset's own source and
    verification date. Payout terms, fees and written confirmations stay BLANK -- one blank stops everything."""
    from app.prop.presets import get_preset
    p = get_preset(preset_key)
    on = checked_on or getattr(p, "rules_checked_on", None) or getattr(p, "as_of", "") or ""
    src = getattr(p, "source_url", "") or getattr(p, "source_note", "")[:160]
    a = float(p.account_size)

    def cell(v):
        return {"value": v, "source": src, "checked_on": str(on)}
    sheet = {"firm": cell(p.firm), "account_size": cell(a), "profit_target": cell(round(a * p.evaluation_profit_target_pct / 100.0, 2)),
             "max_loss": cell(round(a * p.max_drawdown_pct / 100.0, 2)), "soft_daily_limit": cell(round(a * p.daily_loss_limit_pct / 100.0, 2)),
             "consistency": cell(p.consistency_rule_pct if p.consistency_rule_pct is not None else "none")}
    if getattr(p, "payout_frequency_days", None):
        sheet["payout_cadence"] = cell(f"{p.payout_frequency_days} trading days")
    return sheet


def g0_rules_sheet(sheet: dict | None = None, today: date | None = None) -> GateResult:
    """Each rule sourced and dated within 30 days; product fixed; automation / news / copy-trading / overnight confirmed in writing; every
    payout term and every fee filled in. One blank stops everything."""
    sheet = load_rules_sheet() if sheet is None else sheet
    today = today or date.today()
    res = GateResult("G0", "Rules sheet")
    missing, stale, unsourced = [], [], []
    for key, label in RULES_SHEET_FIELDS:
        cell = sheet.get(key) or {}
        v = cell.get("value")
        if v is None or str(v).strip() == "":
            missing.append(label)
            continue
        if not str(cell.get("source", "")).strip():
            unsourced.append(label)
        try:
            age = (today - datetime.strptime(str(cell.get("checked_on", ""))[:10], "%Y-%m-%d").date()).days
            if age > RULES_SHEET_MAX_AGE_DAYS or age < 0:
                stale.append(label)
        except ValueError:
            stale.append(label)
    res.checks.append(Check("Every rule, payout term and fee filled in", not missing,
                            "all filled" if not missing else f"{len(missing)} blank: " + ", ".join(missing[:6]) + ("..." if len(missing) > 6 else "")))
    res.checks.append(Check("Every rule sourced", not unsourced and not missing, "all sourced" if not unsourced else "no source for " + ", ".join(unsourced[:5])))
    res.checks.append(Check(f"Every rule dated within {RULES_SHEET_MAX_AGE_DAYS} days", not stale and not missing,
                            "all fresh" if not stale else f"{len(stale)} stale or undated: " + ", ".join(stale[:5])))
    prod = str((sheet.get("product") or {}).get("value", "")).strip().lower()
    res.checks.append(Check("Product fixed (Pro or Flex)", prod in ("pro", "flex", "lucidpro", "lucidflex"), prod or "not chosen"))
    for key, label in RULES_SHEET_CONFIRMATIONS:
        res.checks.append(Check(label, bool((sheet.get(key) or {}).get("value")), "confirmed" if (sheet.get(key) or {}).get("value") else "not confirmed"))
    res.data = {"missing": missing, "stale": stale, "unsourced": unsourced}
    return res


# ---- G1: design envelope ------------------------------------------------------------------------------------------------------
def g1_design_envelope(rules) -> GateResult:
    """Computed from G0: evaluation risk, funded risk, payoff band, trades a day, daily stop and best-day cap. Output is the one-page spec card."""
    from app.prop.lifecycle_v2 import V2Rules, design_envelope
    r2 = rules if isinstance(rules, V2Rules) else V2Rules.from_prop_rules(rules)
    env = design_envelope(r2)
    res = GateResult("G1", "Design envelope", data={"envelope": env})
    res.checks.append(Check("Envelope computed from the rules sheet", env["evaluation_risk"] > 0 and env["loss_limit"] > 0,
                            f"evaluation risk ${env['evaluation_risk']:,.0f}, funded risk cap ${env['funded_risk_cap']:,.0f}, daily stop ${env['daily_stop']:,.0f}, best day <= ${env['best_day_max']:,.0f}"))
    return res


SPEC_CARD_FIELDS = ("mechanism", "sign", "instruments", "session", "entry_exit", "payoff_b", "sigma_r", "trades_per_day", "evaluation_risk",
                    "funded_risk", "daily_stop", "n_looks", "ledger_ids", "seal_date", "code_hash", "rules_sheet_version")


# ---- G2: mechanism + pre-registration (the trial register) -----------------------------------------------------------------------
def _register_path() -> Path:
    from app.data.storage import get_app_base_dir
    d = Path(get_app_base_dir()) / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "trial_register.json"


def load_register() -> dict:
    try:
        raw = json.loads(_register_path().read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw.setdefault("rows", [])
            raw.setdefault("seal_dates", {})
            return raw
    except Exception:  # noqa: BLE001
        pass
    return {"rows": [], "seal_dates": {}}


def _save_register(reg: dict) -> None:
    _register_path().write_text(json.dumps(reg, indent=2), encoding="utf-8")


def code_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def family_looks(reg: dict, family: str) -> int:
    return sum(int(r.get("variants", 1)) for r in reg.get("rows", []) if str(r.get("family", "")).strip().lower() == family.strip().lower())


def total_looks(reg: dict | None = None) -> int:
    """N = every variant, filter and parameter ever registered (archived ones still count; a revival is new looks)."""
    reg = load_register() if reg is None else reg
    return sum(int(r.get("variants", 1)) + int(r.get("extra_looks", 0)) for r in reg.get("rows", []))


def looks_status(reg: dict | None = None) -> dict:
    reg = load_register() if reg is None else reg
    n = total_looks(reg)
    fams: dict = {}
    for r in reg.get("rows", []):
        k = str(r.get("family", "")) or "(none)"
        fams[k] = fams.get(k, 0) + int(r.get("variants", 1))
    return {"n_looks": n, "e_max_now": round(expected_max_z(max(n, 1)), 3), "families": fams, "max_per_family": MAX_LOOKS_PER_FAMILY,
            "families_full": sorted(k for k, v in fams.items() if v >= MAX_LOOKS_PER_FAMILY), "seal_dates": dict(reg.get("seal_dates", {})),
            "archived": sum(1 for r in reg.get("rows", []) if r.get("archived"))}


def register_trial(*, family: str, mechanism: str, counterparty: str, sign: str, instruments: list, mechanisms: int = 1, variants: int = 1,
                   code_text: str = "", sealed: dict | None = None, extra_looks: int = 0, today: date | None = None,
                   reg: dict | None = None, persist: bool = True) -> dict:
    """G2: write the pre-registration row BEFORE the first run. Refuses a family past its 3 looks, a missing counterparty / sign, thin breadth
    (< 3 instruments from MNQ / MES / M2K / MGC / MCL, or < 2 mechanisms) and a missing seal date for any instrument."""
    reg = load_register() if reg is None else reg
    problems = []
    if not mechanism.strip() or not counterparty.strip():
        problems.append("Name the counterparty (who loses money to this) in one paragraph.")
    if sign.strip() not in ("+", "-", "long", "short", "both"):
        problems.append("State the predicted sign (long, short or both) before testing.")
    used = family_looks(reg, family)
    if used + int(variants) > MAX_LOOKS_PER_FAMILY:
        problems.append(f"Family '{family}' has {used} of {MAX_LOOKS_PER_FAMILY} looks used; {variants} more would exceed the cap. Buy trades with breadth, not parameters.")
    inst = [str(i).upper() for i in (instruments or [])]
    if len(set(inst)) < 3 and int(mechanisms) < 2:
        problems.append("Breadth plan needs at least 3 instruments (MNQ, MES, M2K, MGC, MCL) or at least 2 mechanisms.")
    seal = dict(sealed or {})
    for i in set(inst):
        if not str(seal.get(i, reg.get("seal_dates", {}).get(i, ""))).strip():
            problems.append(f"No sealed window set for {i}; book one seal date per instrument (and extend the data first if it has no post-seal window).")
    if problems:
        return {"ok": False, "problems": problems, "status": looks_status(reg)}
    d = today or date.today()
    n_after = total_looks(reg) + int(variants) + int(extra_looks)
    row = {"id": f"T{len(reg['rows']) + 1}", "family": family.strip(), "mechanism": mechanism.strip(), "counterparty": counterparty.strip(),
           "sign": sign.strip(), "instruments": inst, "mechanisms": int(mechanisms), "variants": int(variants), "extra_looks": int(extra_looks),
           "code_hash": code_hash(code_text), "n_after": n_after, "e_max": round(expected_max_z(n_after), 3), "date": d.isoformat(),
           "gate": "G2", "verdict": "open", "archived": False, "failing_number": ""}
    reg["rows"].append(row)
    for i, v in seal.items():
        reg["seal_dates"][str(i).upper()] = str(v)
    if persist:
        _save_register(reg)
    return {"ok": True, "row": row, "status": looks_status(reg)}


def archive_trial(trial_id: str, gate: str, failing_number: str) -> bool:
    """A failed idea is filed with the failing number; it is never tuned and re-run. A revival is a NEW registration (new looks in N)."""
    reg = load_register()
    for r in reg["rows"]:
        if r.get("id") == trial_id:
            r.update(archived=True, gate=gate, verdict="REJECT", failing_number=str(failing_number)[:300])
            _save_register(reg)
            return True
    return False


def update_trial(trial_id: str, *, gate: str | None = None, verdict: str | None = None) -> bool:
    reg = load_register()
    for r in reg["rows"]:
        if r.get("id") == trial_id:
            if gate:
                r["gate"] = gate
            if verdict:
                r["verdict"] = str(verdict)[:200]
            _save_register(reg)
            return True
    return False


def g2_preregistration(*, family: str, trial_id: str | None = None, reg: dict | None = None, frozen: bool = True) -> GateResult:
    """Checks the register for this idea: a row exists, its family is within 3 looks, a code hash and breadth plan and sealed windows are booked."""
    reg = load_register() if reg is None else reg
    res = GateResult("G2", "Mechanism and pre-registration")
    row = next((r for r in reg["rows"] if r.get("id") == trial_id), None) if trial_id else \
        next((r for r in reversed(reg["rows"]) if str(r.get("family", "")).lower() == family.lower()), None)
    res.checks.append(Check("Trial-register row exists before the first run", row is not None, row["id"] if row else "no pre-registration row"))
    if row is None:
        return res
    res.checks.append(Check("Counterparty and predicted sign written down", bool(row.get("counterparty")) and bool(row.get("sign")), f"sign {row.get('sign')}"))
    used = family_looks(reg, family)
    res.checks.append(Check(f"<= {MAX_LOOKS_PER_FAMILY} variants in the family", used <= MAX_LOOKS_PER_FAMILY, f"{used} looks in '{family}'"))
    res.checks.append(Check("Code hash recorded", bool(row.get("code_hash")), row.get("code_hash", "")))
    breadth = len(set(row.get("instruments", []))) >= 3 or int(row.get("mechanisms", 1)) >= 2
    res.checks.append(Check("Breadth plan (>= 3 instruments or >= 2 mechanisms)", breadth, f"{len(set(row.get('instruments', [])))} instruments, {row.get('mechanisms', 1)} mechanisms"))
    seals = reg.get("seal_dates", {})
    missing = [i for i in row.get("instruments", []) if not seals.get(i)]
    res.checks.append(Check("Sealed window set per instrument", not missing, "all booked" if not missing else "missing: " + ", ".join(missing)))
    res.checks.append(Check("Test frozen (ranges fixed, code hash locked)", bool(frozen), "frozen" if frozen else "not frozen"))
    res.data = {"row": row, "n_looks": total_looks(reg)}
    return res


# ---- shared trade series -------------------------------------------------------------------------------------------------------
def _series(trades):
    """(R multiples, timestamps, directions, extra-cost R per trade) aligned, for trades that carry a planned risk."""
    rs, ts, dirs, cost = [], [], [], []
    for t in trades or []:
        ir, size, pnl = getattr(t, "initial_risk", None), getattr(t, "size", None), getattr(t, "pnl", None)
        if pnl is None or not ir or ir <= 0 or not size:
            continue
        risk = float(ir) * float(size)
        rs.append(float(pnl) / risk)
        ts.append(getattr(t, "exit_time", None) or getattr(t, "entry_time", None))
        dirs.append(int(getattr(t, "direction", 0) or 0))
        cost.append(abs(float(getattr(t, "commission", 0.0) or 0.0)) / risk)
    return np.asarray(rs, float), ts, np.asarray(dirs, int), np.asarray(cost, float)


def _quarter_share_positive(rs, ts):
    import pandas as pd
    s = pd.Series(rs, index=pd.to_datetime(pd.Series(ts), utc=True, errors="coerce"))
    s = s[s.index.notna()]
    if s.empty:
        return None, 0
    q = s.groupby([s.index.year, s.index.quarter]).sum()
    return float((q > 0).mean()), int(len(q))


def _year_share(rs, ts):
    import pandas as pd
    s = pd.Series(rs, index=pd.to_datetime(pd.Series(ts), utc=True, errors="coerce"))
    s = s[s.index.notna()]
    y = s.groupby(s.index.year).sum()
    total = float(y.sum())
    if len(y) < 3 or total <= 0:
        return None, int(len(y)), total
    return float(y.max() / total), int(len(y)), total


# ---- G3: development evidence ----------------------------------------------------------------------------------------------------
def g3_development_evidence(trades, n_looks: int = 1, *, neighbours: list | None = None, lookahead_ok: bool | None = None,
                            falling_window_positive: bool | None = None, sigma_r: float | None = None) -> GateResult:
    """Is there an edge after selection and cost? ``n_looks`` is N: every variant, filter and parameter tried (the effective trial count).

    ``neighbours`` = E_plan of the +/-20% parameter neighbours (list of floats); ``lookahead_ok`` = lookahead and prefix-invariance tests passed."""
    from app.validation.edge_stats import day_block_bootstrap_mean
    rs, ts, dirs, cost = _series(trades)
    n = int(rs.size)
    res = GateResult("G3", "Development evidence")
    if n < 5:
        res.checks.append(Check("Enough trades with a planned risk", False, f"{n} trades"))
        return res
    bs = day_block_bootstrap_mean(list(rs), ts, conf=0.95, n_boot=2000, seed=11)
    mean = float(rs.mean())
    # day-block SE of the MEAN: the bootstrap's 5th-95th percentile spread (the "std" field is the per-trade std, not an SE)
    lb, ub = bs.get("lb"), bs.get("ub")
    se = float((ub - lb) / (2 * 1.6449)) if lb is not None and ub is not None else 0.0
    if not se:
        se = float(rs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    sd = float(sigma_r or (rs.std(ddof=1) if n > 1 else SIGMA_R_AT_2_TO_1))
    ep = e_plan(mean, n_looks, se)
    n_days = int(bs.get("n_days") or 0)
    floor_n = trade_floor(n_looks, mean, sd) if mean > E_PLAN_FLOOR_R else None
    need = required_observed_edge(n, n_looks, sd)
    res.data = {"n_trades": n, "n_days": n_days, "n_looks": int(n_looks), "mean_r": mean, "se": se, "sigma_r": sd, "e_max": expected_max_z(n_looks),
                "e_plan": ep, "required_observed_edge": need, "derived_trade_floor": floor_n}
    res.checks.append(Check(f"Sample: n >= {G3_MIN_TRADES} and >= {G3_MIN_TRADING_DAYS} trading days", n >= G3_MIN_TRADES and n_days >= G3_MIN_TRADING_DAYS,
                            f"{n} trades over {n_days} days"))
    res.checks.append(Check(f"Deflated edge E_plan >= +{E_PLAN_FLOOR_R:.2f}R (N = {n_looks})", ep >= E_PLAN_FLOOR_R,
                            f"observed {mean:+.3f}R, SE {se:.3f}, E[max N] {expected_max_z(n_looks):.2f} -> E_plan {ep:+.3f}R (observed must reach {need:+.3f}R)"))
    if floor_n is not None:
        res.checks.append(Check("Derived trade floor met", n >= floor_n, f"{n} trades vs the {floor_n} this edge and N need"))
    stressed = rs - cost
    res.checks.append(Check("Survives 2x measured cost", float(stressed.mean()) > 0, f"mean {float(stressed.mean()):+.3f}R after doubling commission ({float(cost.mean()):.3f}R each)"))
    legs = []
    for lab, sel in (("long", dirs > 0), ("short", dirs < 0)):
        if sel.any():
            legs.append((lab, float(rs[sel].mean()), int(sel.sum())))
    leg_ok = all(m >= 0 for _l, m, _c in legs) if legs else None
    if leg_ok is False and falling_window_positive:
        leg_ok = True
    res.checks.append(Check("Long and short legs each >= 0 (or positive on a falling window)", leg_ok,
                            ", ".join(f"{l} {m:+.3f}R ({c})" for l, m, c in legs) or "no direction data"))
    top3 = float(np.sort(rs)[:-3].mean()) if n > 3 else None
    res.checks.append(Check("Positive with the top 3 trades removed", None if top3 is None else top3 > 0, "n/a" if top3 is None else f"{top3:+.3f}R"))
    qshare, nq = _quarter_share_positive(rs, ts)
    res.checks.append(Check(f">= {G3_MIN_QUARTERS_POSITIVE:.0%} of quarters positive", None if qshare is None else qshare >= G3_MIN_QUARTERS_POSITIVE,
                            "no dated trades" if qshare is None else f"{qshare:.0%} of {nq} quarters"))
    ysh, ny, _tot = _year_share(rs, ts)
    res.checks.append(Check(f"No year above {G3_MAX_YEAR_SHARE:.0%} of P&L", None if ysh is None and ny >= 3 else (False if ny < 3 else ysh <= G3_MAX_YEAR_SHARE),
                            f"needs 3+ calendar years (has {ny})" if ny < 3 else ("total P&L not positive" if ysh is None else f"best year {ysh:.0%} of P&L")))
    if neighbours:
        keep = [x / ep if ep > 0 else 0.0 for x in neighbours]
        res.checks.append(Check(f"+/-20% parameter neighbours keep >= {G3_NEIGHBOUR_KEEP:.0%} of E_plan", all(k >= G3_NEIGHBOUR_KEEP for k in keep),
                                f"worst neighbour keeps {min(keep):.0%}"))
    else:
        res.checks.append(Check(f"+/-20% parameter neighbours keep >= {G3_NEIGHBOUR_KEEP:.0%} of E_plan", None, "not run (pass neighbour E_plans)"))
    res.checks.append(Check("Lookahead and prefix tests pass", lookahead_ok, "not run" if lookahead_ok is None else ("passed" if lookahead_ok else "FAILED")))
    return res


# ---- G4: day behaviour --------------------------------------------------------------------------------------------------------
def g4_day_behaviour(trades, rules, *, leg_of=None, n_boot_days: int = 20_000, seed: int = 21) -> GateResult:
    """Does it act like a payout machine day by day? Day-block bootstrap of real trades at the planned (evaluation) size with the day stop on."""
    from app.prop.lifecycle_v2 import (BEST_DAY_FRAC_OF_TARGET, DAY_STOP_FRAC_OF_SOFT, DAY_STOP_R, V2Rules, day_matrix_from_trades,
                                       eval_risk)
    r2 = rules if isinstance(rules, V2Rules) else V2Rules.from_prop_rules(rules)
    mat, info = day_matrix_from_trades(trades)
    res = GateResult("G4", "Day behaviour", data={"input": info})
    if mat.shape[0] < 20:
        res.checks.append(Check("Enough trading days", False, f"{mat.shape[0]} days"))
        return res
    rng = np.random.default_rng(seed)
    rows = np.nan_to_num(mat[rng.integers(0, mat.shape[0], int(n_boot_days))], nan=0.0)
    risk = float(eval_risk(r2.loss_limit, r2.loss_limit, r2.micro_risk))
    stop_r = min(DAY_STOP_R, DAY_STOP_FRAC_OF_SOFT * r2.soft_daily_limit / risk)
    run = np.cumsum(rows, axis=1)
    hit = run <= -stop_r
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), rows.shape[1] - 1)
    day_r = run[np.arange(rows.shape[0]), first]
    day_usd = day_r * risk
    p_loss = 100.0 * float((day_usd <= -r2.soft_daily_limit).mean())
    best = float(np.percentile(day_usd, 99))
    q_rate = 100.0 * float((day_usd >= r2.payout.qualifying_day_min).mean())
    tpd = float(info.get("median_trades_per_day", 0))
    res.data.update(p_day_loss_pct=p_loss, best_day_p99=best, qualifying_day_rate=q_rate, median_trades_per_day=tpd, risk_per_trade=risk)
    res.checks.append(Check(f"P(day loss >= soft limit) <= {G4_MAX_DAY_LOSS_P:.0f}%", p_loss <= G4_MAX_DAY_LOSS_P, f"{p_loss:.2f}% at ${risk:,.0f} per R"))
    cap = BEST_DAY_FRAC_OF_TARGET * r2.profit_target
    res.checks.append(Check("Best day <= 33% of target", best <= cap, f"99th-percentile best day ${best:,.0f} vs ${cap:,.0f}"))
    res.checks.append(Check(f"Qualifying-day rate >= {G4_MIN_QUALIFYING_DAY_RATE:.0f}%", q_rate >= G4_MIN_QUALIFYING_DAY_RATE,
                            f"{q_rate:.0f}% of days reach ${r2.payout.qualifying_day_min:,.0f}"))
    res.checks.append(Check("Median 2 to 4 trades a day", 2 <= tpd <= 4, f"median {tpd:g}"))
    corr = None
    if leg_of is not None:
        import pandas as pd
        from app.data.trading_day import trading_day
        legs: dict = {}
        for t in trades or []:
            k = leg_of(t)
            legs.setdefault(k, {})
            d = trading_day(pd.Timestamp(t.entry_time))
            legs[k][d] = legs[k].get(d, 0.0) + float(t.pnl)
        if len(legs) >= 2:
            df = pd.DataFrame(legs).fillna(0.0)
            c = df.corr().to_numpy()
            iu = np.triu_indices_from(c, 1)
            corr = float(np.nanmax(np.abs(c[iu]))) if iu[0].size else 0.0
    res.checks.append(Check(f"Daily P&L correlation between legs <= {G4_MAX_LEG_CORR}", None if corr is None else corr <= G4_MAX_LEG_CORR,
                            "single stream (no legs supplied)" if corr is None else f"max |corr| {corr:.2f}"))
    return res


# ---- G5: lifecycle --------------------------------------------------------------------------------------------------------------
def g5_lifecycle(trades, rules, *, n_paths: int = 20_000, seed: int = 42, n_looks: int = 1, fee: float | None = None,
                 split_pct: float | None = None, payout=None) -> GateResult:
    """Does it pass AND keep getting paid under exact rules? Real trades, day-block paths, v2 sizing, Wilson 5th percentile bars."""
    from app.prop.lifecycle_v2 import V2Rules, day_matrix_from_trades, simulate
    r2 = rules if isinstance(rules, V2Rules) else V2Rules.from_prop_rules(rules, fee=fee or 0.0, split_pct=split_pct, payout=payout)
    if fee is not None:
        r2.fee = float(fee)
    mat, info = day_matrix_from_trades(trades)
    res = GateResult("G5", "Lifecycle", data={"input": info})
    if mat.shape[0] < 20:
        res.checks.append(Check("Enough trading days", False, f"{mat.shape[0]} days"))
        return res
    sim = simulate(mat, r2, n_paths=n_paths, seed=seed)
    # EV at the planning edge: shift every R down by the selection bias, then cut payouts 50%
    from app.validation.edge_stats import day_block_bootstrap_mean
    rs, ts, _d, _c = _series(trades)
    se = 0.0
    if rs.size >= 5:
        _b = day_block_bootstrap_mean(list(rs), ts, n_boot=800, seed=5)
        if _b.get("lb") is not None and _b.get("ub") is not None:
            se = float((_b["ub"] - _b["lb"]) / (2 * 1.6449))     # day-block SE of the mean
    bias = expected_max_z(n_looks) * se
    plan = simulate(mat - bias, r2, n_paths=max(4_000, n_paths // 4), seed=seed + 1, payout_haircut=G5_PAYOUT_HAIRCUT)
    res.data.update(sim=sim.to_dict(), plan_ev_haircut=plan.ev_per_attempt_haircut, plan_edge_shift=bias, headline=sim.headline())
    lb = sim.p_pass_ci[0]
    res.checks.append(Check(f"P(pass) lower bound >= {G5_MIN_PASS_LB:.0f}%", lb >= G5_MIN_PASS_LB, f"{sim.p_pass:.0f}% (Wilson 5th percentile {lb:.0f}%)"))
    res.checks.append(Check(f"Evaluation bust <= {G5_MAX_EVAL_BUST:.0f}%", sim.p_bust_eval <= G5_MAX_EVAL_BUST, f"{sim.p_bust_eval:.0f}%"))
    res.checks.append(Check(f"P(paid | pass) >= {G5_MIN_PAID_GIVEN_PASS:.0f}%", sim.p_paid_given_pass >= G5_MIN_PAID_GIVEN_PASS, f"{sim.p_paid_given_pass:.0f}%"))
    res.checks.append(Check(f"P(>= 2 payouts | paid) >= {G5_MIN_TWO_PLUS_GIVEN_PAID:.0f}%", sim.p_two_plus_given_paid >= G5_MIN_TWO_PLUS_GIVEN_PAID,
                            f"{sim.p_two_plus_given_paid:.0f}%"))
    ap = sim.attempts_per_payout
    res.checks.append(Check(f"Attempts per payout <= {G5_MAX_ATTEMPTS_PER_PAYOUT:g}", ap is not None and ap <= G5_MAX_ATTEMPTS_PER_PAYOUT,
                            "no payout seen" if ap is None else f"{ap:.1f}"))
    res.checks.append(Check("EV > 0 at E_plan with payouts cut 50%", plan.ev_per_attempt_haircut > 0, f"${plan.ev_per_attempt_haircut:,.0f} per attempt (fee ${r2.fee:,.0f})"))
    if not r2.payout.sourced:
        res.note = "Payout terms are the report's assumption until G0 sources them."
    return res


# ---- G6 / G7 --------------------------------------------------------------------------------------------------------------------
def g6_seal_live(*, sealed_edge_lb: float | None = None, sealed_read_once: bool | None = None, live_sim_days: int | None = None,
                 r_values=None, slippage_ratio: float | None = None, pnl_in_band: bool | None = None, rule_breaches: int | None = None) -> GateResult:
    """Permission to buy attempts, not proof of edge. Needs the sealed window read once OR >= 20 live-sim days, plus the monitor checks."""
    res = GateResult("G6", "Seal and live sim")
    sealed_ok = bool(sealed_read_once) and sealed_edge_lb is not None and sealed_edge_lb > 0
    live_ok = live_sim_days is not None and live_sim_days >= G6_MIN_LIVE_SIM_DAYS
    res.checks.append(Check(f"Sealed window read once (lower bound > 0) or >= {G6_MIN_LIVE_SIM_DAYS} live-sim days", sealed_ok or live_ok,
                            f"sealed: {'read once, lb ' + format(sealed_edge_lb, '+.3f') + 'R' if sealed_read_once and sealed_edge_lb is not None else 'not read'}; live sim: {live_sim_days or 0} days"))
    if r_values is not None and len(list(r_values)) > 0:
        k = llr_path(list(r_values))
        res.checks.append(Check("Trade-level monitor above the kill line", not k["killed"], f"LLR {k['llr']:+.2f} vs {k['kill_at']}"))
        res.data["llr"] = k
    else:
        res.checks.append(Check("Trade-level monitor above the kill line", None, "no live trades yet"))
    res.checks.append(Check(f"Realised slippage <= {G6_MAX_SLIPPAGE_RATIO:.0%} of the model", None if slippage_ratio is None else slippage_ratio <= G6_MAX_SLIPPAGE_RATIO,
                            "not measured" if slippage_ratio is None else f"{slippage_ratio:.0%}"))
    res.checks.append(Check("Daily P&L inside the bootstrap 90% band", pnl_in_band, "not measured" if pnl_in_band is None else ("inside" if pnl_in_band else "outside")))
    res.checks.append(Check("Zero rule breaches", None if rule_breaches is None else rule_breaches == 0, "not measured" if rule_breaches is None else str(rule_breaches)))
    return res


def g7_capped_attempts(outcomes: list, fee: float = 172.0) -> GateResult:
    """Does the money agree? Running score, one attempt at a time, stop at the first rule breach."""
    a = AttemptScore()
    for o in outcomes or []:
        a.add(bool(o))
    st = a.status()
    st["fees_cap"] = round(ATTEMPT_MAX * float(fee), 2)
    res = GateResult("G7", "Capped paid attempts", data=st)
    res.checks.append(Check(f"Score reaches +{ATTEMPT_SCALE_AT} before -{abs(ATTEMPT_RETIRE_AT)} (max {ATTEMPT_MAX} attempts)", True if st["decision"] == "SCALE" else (False if st["decision"] == "RETIRE" else None),
                            f"score {st['score']:+.1f} after {st['attempts']} attempts ({st['paid']} paid) -> {st['decision']}"
                            + (" (undecided at the cap = retire)" if st["undecided_at_cap"] else "")))
    return res


# ---- the pipeline ---------------------------------------------------------------------------------------------------------------
GATES = (
    {"gate": "G0", "name": "Rules sheet", "question": "Do I know the rules?",
     "rule": "Each rule sourced and dated within 30 days; product fixed (Pro or Flex); automation, news, copy-trading and overnight holds confirmed in writing; every payout term and every fee filled in. One blank stops everything.",
     "where": "Rulebook page rules sheet; prop presets verified-date tick (unverified presets cannot reach READY)."},
    {"gate": "G1", "name": "Design envelope", "question": "What must the strategy look like?",
     "rule": "Evaluation risk 7.5% of the loss limit; funded risk <= 5% of the limit and <= 5% of the cushion; payoff 1.5:1 to 2.5:1; 2 to 4 trades a day; daily stop = lower of 3R and 60% of the soft limit; best day <= 33% of target.",
     "where": "app.prop.lifecycle_v2.design_envelope; the spec card on the Rulebook page; the sizing policy every engine uses."},
    {"gate": "G2", "name": "Mechanism and pre-registration", "question": "Who loses money to this, and is the test frozen?",
     "rule": "Counterparty and predicted sign; <= 3 variants per family; code hash and trial-register row before the first run; N counts every variant, filter and parameter; breadth plan of >= 3 instruments or >= 2 mechanisms; sealed window set per instrument.",
     "where": "Trial register (Rulebook page); the Discovery Engine's lifetime trial ledger and sealed-read ledger."},
    {"gate": "G3", "name": "Development evidence", "question": "Is there an edge after selection and cost?",
     "rule": "E_plan >= +0.05R at the honest N; n >= 400 and >= 150 trading days; survives 2x cost; both legs >= 0; positive without the top 3 trades; >= 60% of quarters positive; no year above 40% of P&L; +/-20% neighbours keep >= 70%; lookahead and prefix tests pass.",
     "where": "Test report 'Rulebook v2' block; Discovery promotion step 'rulebook_g3'; pooled edge test; parameter robustness; lookahead check."},
    {"gate": "G4", "name": "Day behaviour", "question": "Does it act like a payout machine day by day?",
     "rule": "Day-block bootstrap of real trades: P(day loss >= soft limit) <= 1%; best day <= 33% of target; qualifying-day rate >= 25%; leg correlation <= 0.3; median 2 to 4 trades a day.",
     "where": "Test report 'Rulebook v2' block (day-block bootstrap)."},
    {"gate": "G5", "name": "Lifecycle", "question": "Does it pass and keep getting paid under exact rules?",
     "rule": "Real trades, 20,000 day-block paths, v2 sizing: P(pass) Wilson lower bound >= 70% and evaluation bust <= 20%; P(paid | pass) >= 75%; P(>= 2 payouts | paid) >= 60%; attempts per payout <= 3; EV > 0 at E_plan with payouts cut 50%.",
     "where": "app.prop.lifecycle_v2 (G5 block of the Test report); the Monte Carlo and survival engines keep their own views."},
    {"gate": "G6", "name": "Seal and live sim", "question": "Do unseen data and a real platform agree?",
     "rule": "Sealed window read once, or >= 20 trading days on a free sim; trade monitor above the kill line; slippage <= 125% of the model; daily P&L inside the bootstrap band; zero rule breaches. Permission to buy attempts, not proof of edge.",
     "where": "Demo account forward test (Deploy), SPRT monitor and the +0.10R kill line."},
    {"gate": "G7", "name": "Capped paid attempts", "question": "Does the money agree?",
     "rule": "Running score +1.5 per paid attempt, -0.9 per unpaid; scale at +2.2, retire at -2.2, at most 6 attempts, one attempt at a time per strategy, stop at the first rule breach.",
     "where": "Payout forecast ledger; Deploy page attempt score."},
)

GATE_NAME_MAP = {
    "Evaluation breach (ruin)": "G5 -- evaluation bust <= 20% under v2 sizing",
    "Per-attempt pass": "G5 -- Wilson lower bound of pass >= 70%",
    "Holdout confirmation": "G6 -- sealed window read once",
    "Edge after 2x costs": "G3 -- survives 2x measured cost",
    "Unseen DSR": "G3 -- deflated planning edge E_plan >= +0.05R",
    "Null test": "G3 -- distinguishable from random entries",
    "Expected value": "G5 -- EV > 0 with payouts cut 50%",
    "Trades": "G3 -- one derived trade floor (n >= 400)",
}


def ladder_stage(results: dict) -> str:
    """Verdict ladder from a {gate: GateResult} map: G0-G2 = LEAD, G3-G5 = FORWARD-TEST CANDIDATE, G6 = EVALUATE, G7 scale = FUNDED."""
    def ok(g):
        r = results.get(g)
        return bool(r is not None and r.passed)
    if ok("G7") and (results["G7"].data or {}).get("decision") == "SCALE":
        return VERDICT_FUNDED
    if all(ok(g) for g in ("G0", "G1", "G2", "G3", "G4", "G5", "G6")):
        return VERDICT_EVALUATE
    if all(ok(g) for g in ("G0", "G1", "G2", "G3", "G4", "G5")):
        return VERDICT_CANDIDATE
    if all(ok(g) for g in ("G0", "G1", "G2")):
        return VERDICT_LEAD
    return VERDICT_REJECT


def evaluate_pipeline(trades, rules, *, n_looks: int = 1, family: str = "", trial_id: str | None = None, sheet: dict | None = None,
                      neighbours: list | None = None, lookahead_ok: bool | None = None, leg_of=None, n_paths: int = 20_000,
                      seed: int = 42, fee: float | None = None, split_pct: float | None = None, seal: dict | None = None,
                      live: dict | None = None, attempts: list | None = None, skip_gates: tuple = ()) -> dict:
    """Run G0..G7 in order on a strategy's trades. The first failing gate stops the run and files the failing number; later gates are
    marked skipped. ``skip_gates`` lets a caller that has no rules sheet / register (a quick test) report those gates as NOT RUN instead of failing."""
    results: dict = {}
    stopped = None

    def run(g, fn):
        nonlocal stopped
        if stopped:
            results[g] = GateResult(g, next(x["name"] for x in GATES if x["gate"] == g), skipped=True, note=f"not run: {stopped} failed")
            return
        if g in skip_gates:
            results[g] = GateResult(g, next(x["name"] for x in GATES if x["gate"] == g), skipped=True, note="not run (not supplied for this test)")
            return
        r = fn()
        results[g] = r
        if not r.passed:
            stopped = g

    run("G0", lambda: g0_rules_sheet(sheet))
    run("G1", lambda: g1_design_envelope(rules))
    run("G2", lambda: g2_preregistration(family=family, trial_id=trial_id))
    run("G3", lambda: g3_development_evidence(trades, n_looks, neighbours=neighbours, lookahead_ok=lookahead_ok))
    run("G4", lambda: g4_day_behaviour(trades, rules, leg_of=leg_of))
    run("G5", lambda: g5_lifecycle(trades, rules, n_paths=n_paths, seed=seed, n_looks=n_looks, fee=fee, split_pct=split_pct))
    run("G6", lambda: g6_seal_live(**(live or {}), **(seal or {})))
    run("G7", lambda: g7_capped_attempts(attempts or [], fee or 172.0))
    ran = [g for g, r in results.items() if not r.skipped]
    evaluated_all = {g: r for g, r in results.items()}
    # skipped-by-caller gates count as passed for the ladder only when the caller said so; a failure always wins
    # Only the PROCESS gates (G0 rules sheet, G2 pre-registration) may be left out of a quick test; G6 / G7 (seal, live sim, paid attempts)
    # are evidence a backtest can never supply, so skipping them never lifts a strategy to EVALUATE or FUNDED.
    lenient = {"G0", "G2"}
    ladder_view = {g: (GateResult(g, r.name, checks=[Check("not required for this view", True)]) if (r.skipped and g in skip_gates and g in lenient) else r)
                   for g, r in evaluated_all.items()}
    stage = ladder_stage(ladder_view)
    failing = results[stopped].failing_number() if stopped else ""
    return {"version": RULEBOOK_VERSION, "gates": {g: r.to_dict() for g, r in results.items()}, "stopped_at": stopped, "failing_number": failing,
            "stage": stage, "ran": ran, "n_looks": int(n_looks)}


def render(pipeline: dict) -> list[str]:
    lines = [f"Rulebook v2: {pipeline['stage']}" + (f" -- stopped at {pipeline['stopped_at']}: {pipeline['failing_number']}" if pipeline.get("stopped_at") else "")]
    for g, r in pipeline["gates"].items():
        tag = "skipped" if r["skipped"] else ("PASS" if r["passed"] else ("PENDING" if r["incomplete"] else "FAIL"))
        lines.append(f"  {g} {r['name']}: {tag}" + (f" -- {r['failing_number']}" if r.get("failing_number") else ""))
    return lines


def snapshot() -> dict:
    """Everything the Rulebook page shows, in one call."""
    reg = load_register()
    return {"version": RULEBOOK_VERSION, "date": RULEBOOK_DATE,
            "constants": {"e_plan_floor_r": E_PLAN_FLOOR_R, "breakeven_ev_edge_r": BREAKEVEN_EV_EDGE_R, "kill_llr": KILL_LLR, "kill_h1_r": KILL_H1_R,
                          "max_looks_per_family": MAX_LOOKS_PER_FAMILY, "g3_min_trades": G3_MIN_TRADES, "attempt_max": ATTEMPT_MAX,
                          "attempt_paid": ATTEMPT_PAID_SCORE, "attempt_unpaid": ATTEMPT_UNPAID_SCORE, "scale_at": ATTEMPT_SCALE_AT,
                          "retire_at": ATTEMPT_RETIRE_AT},
            "g3_table": g3_table(), "gates": [dict(g) for g in GATES], "looks": looks_status(reg), "register": reg["rows"],
            "sheet": load_rules_sheet(), "sheet_fields": RULES_SHEET_FIELDS, "sheet_confirmations": RULES_SHEET_CONFIRMATIONS,
            "g0": g0_rules_sheet().to_dict(), "deflation_example": round(deflation_example(), 3)}
