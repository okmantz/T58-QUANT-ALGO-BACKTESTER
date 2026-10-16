"""Prop-firm rules and risk configuration tabs."""
from __future__ import annotations

import os
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.backtest.risk import RiskConfig, suggest_pip_size
from app.data.importer import import_csv
from app.data.instrument_specs import get_instrument_spec, known_instrument_symbols
from app.prop.presets import list_presets as list_prop_firm_presets
from app.prop.simulator import PropRules
from tkinter import END, Frame, Label, Listbox, SINGLE, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui.widgets.dialogs import _AdaptiveRuleDialog
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class PropRiskMixin:
    """Prop-firm rules and risk configuration tabs."""


    # -----------------------------------------------------------------------
    # Tab 3 — Prop rules
    # -----------------------------------------------------------------------

    def _build_prop_tab(self):
        f = self._scrollable(self.tab_prop)

        self._page_header(
            f,
            "TEST / Prop-Firm Rules",
            "Prop-Firm Rules",
            "Define the evaluation, drawdown, consistency, payout, and position constraints.",
        )

        preset_section = self._section(
            f,
            "Quick-fill from a prop-firm preset",
            "One-click starting point for a major firm's published rules -- fills in every "
            "field below, which you can still edit by hand afterward. See each preset's "
            "'as of' date; firms change these terms often, so treat this as a fast, "
            "reviewable starting point, not a live feed -- verify against the firm's own "
            "current rules page before relying on it for a real evaluation.",
        )
        preset_labels = ["(Choose a preset...)"] + [p.label for p in list_prop_firm_presets()]
        self.p_preset_combo = LabeledCombo(preset_section, "Prop-firm preset", preset_labels, default=preset_labels[0])
        self.p_preset_note = Label(
            preset_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8),
            anchor="w", justify="left", wraplength=560,
        )
        self.p_preset_note.pack(fill="x", padx=18, pady=(0, 6))
        self.p_preset_combo.combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_prop_firm_preset())

        section = self._section(
            f,
            "Account & evaluation",
            "Core evaluation parameters. Drawdown check mode: 'intrabar' "
            "monitors floating equity in real time (a floor breach can "
            "force-close a position mid-trade); 'eod' only checks once per "
            "day using that day's final balance, so an intraday dip that "
            "recovers by the close doesn't count. Match this to what your "
            "specific firm documents -- getting it backwards makes your "
            "pass-probability estimate too pessimistic or too optimistic.",
            emphasize=True,
        )

        self.p_account_size = LabeledEntry(section, "Account size ($)", 100000)
        self.p_profit_target = LabeledEntry(
            section, "Evaluation profit target (%)", 8
        )
        self.p_daily_loss = LabeledEntry(
            section, "Daily loss limit (%)", 5
        )
        self.p_max_dd = LabeledEntry(
            section, "Maximum drawdown (%)", 10
        )
        self.p_dd_type = LabeledEntry(
            section, "Drawdown type (trailing/static)", "trailing"
        )
        self.p_dd_check_mode = LabeledEntry(
            section,
            "Drawdown check mode (intrabar/eod)",
            "intrabar",
        )
        self.p_consistency = LabeledEntry(
            section, "Consistency rule (% best day of total profit)", 30
        )
        self.p_min_days = LabeledEntry(
            section, "Minimum trading days", 5
        )

        section2 = self._section(
            f,
            "Payout & position rules",
            "Optional payout and position constraints.",
        )

        self.p_payout_threshold = LabeledEntry(
            section2, "Payout threshold (extra % profit)", 0
        )
        self.p_payout_cap = LabeledEntry(
            section2, "Payout cap (% of profit, blank=100)", 100
        )
        self.p_payout_freq = LabeledEntry(
            section2, "Payout frequency (days)", 14
        )
        self.p_buffer = LabeledEntry(
            section2, "Required buffer (%)", 0
        )
        self.p_max_pos = LabeledEntry(
            section2, "Max position size (units, blank=unlimited)", ""
        )

    def _apply_prop_firm_preset(self):
        idx = self.p_preset_combo.combo.current()
        presets = list_prop_firm_presets()
        if idx <= 0 or idx - 1 >= len(presets):
            self.p_preset_note.config(text="")
            return
        p = presets[idx - 1]
        self.p_account_size.var.set(str(p.account_size))
        self.p_profit_target.var.set(str(p.evaluation_profit_target_pct))
        self.p_daily_loss.var.set(str(p.daily_loss_limit_pct))
        self.p_max_dd.var.set(str(p.max_drawdown_pct))
        self.p_dd_type.var.set(p.drawdown_type)
        self.p_dd_check_mode.var.set(p.drawdown_check_mode)
        self.p_consistency.var.set(str(p.consistency_rule_pct) if p.consistency_rule_pct is not None else "")
        self.p_min_days.var.set(str(p.min_trading_days))
        self.p_payout_threshold.var.set(str(p.payout_threshold_pct))
        self.p_payout_cap.var.set(str(p.payout_cap_pct) if p.payout_cap_pct is not None else "100")
        self.p_payout_freq.var.set(str(p.payout_frequency_days))
        self.p_buffer.var.set(str(p.required_buffer_pct))
        self.p_preset_note.config(text=f"As of {p.as_of}: {p.source_note}")

    def _build_prop_rules(self) -> PropRules:
        cap = self.p_payout_cap.get_str().strip()
        max_pos = self.p_max_pos.get_str().strip()

        return PropRules(
            account_size=self.p_account_size.get_float(100000),
            evaluation_profit_target_pct=self.p_profit_target.get_float(8),
            daily_loss_limit_pct=self.p_daily_loss.get_float(5),
            max_drawdown_pct=self.p_max_dd.get_float(10),
            drawdown_type=self.p_dd_type.get_str().strip() or "trailing",
            drawdown_check_mode=self.p_dd_check_mode.get_str().strip() or "intrabar",
            consistency_rule_pct=self.p_consistency.get_float(30),
            min_trading_days=self.p_min_days.get_int(5),
            payout_threshold_pct=self.p_payout_threshold.get_float(0),
            payout_cap_pct=float(cap) if cap else None,
            payout_frequency_days=self.p_payout_freq.get_int(14),
            required_buffer_pct=self.p_buffer.get_float(0),
            max_position_size=float(max_pos) if max_pos else None,
        )

    # -----------------------------------------------------------------------
    # Tab 4 — Risk
    # -----------------------------------------------------------------------

    def _build_risk_tab(self):
        f = self._scrollable(self.tab_risk)

        self._page_header(
            f,
            "TEST / Risk & Execution",
            "Risk & Execution",
            "Define position risk, trading frequency, transaction costs, and execution assumptions.",
        )

        section = self._section(
            f,
            "Risk configuration",
            "These parameters are passed directly into the backtest engine.",
            emphasize=True,
        )

        # RISK-001 fix: Initial balance and Account size (Tab 3, "Prop
        # rules") must be the same dollar account -- the backend already
        # forces this (app.backtest.risk.with_prop_safety_defaults makes
        # PropRules.account_size win if they ever diverge), but sharing
        # the SAME StringVar here means the person never sees the two
        # numbers disagree in the first place, using the exact mechanism
        # this class's own docstring describes for mirroring fields
        # across tabs.
        self.r_initial_balance = LabeledEntry(
            section, "Initial balance ($) -- kept in sync with Account size (Prop rules tab)",
            variable=self.p_account_size.var,
        )
        self.r_risk_mode = LabeledEntry(
            section, "Risk mode (percent/fixed)", "percent"
        )
        self.r_risk_value = LabeledEntry(
            section, "Risk per trade (% or $)", 1.0
        )
        self.r_max_trades_day = LabeledEntry(
            section, "Max trades/day", 10
        )
        self.r_commission = LabeledEntry(
            section, "Commission per trade ($)", 0
        )
        self.r_slippage = LabeledEntry(
            section, "Slippage (pips)", 0.5
        )
        self.r_spread = LabeledEntry(
            section, "Spread (pips)", 1.0
        )
        self.r_pip_size = LabeledEntry(
            section, "Pip size (e.g. 0.0001 FX)", 0.0001
        )
        # UPGRADE (instrument-metadata-had-no-ui): app.data.instrument_specs
        # was built to remove exactly the "I had to hand-derive and
        # hardcode MES=$5, MNQ=$2, MGC=$10 myself" tedium (see that
        # module's own docstring) but was never actually wired into any
        # UI anywhere -- this dropdown is that wiring. Picking a known
        # instrument fills BOTH pip_size and the new contract $/point
        # field below from KNOWN_INSTRUMENTS; leaving it on "(none)"
        # changes nothing (both fields keep working exactly as before,
        # entered by hand).
        self.r_instrument = LabeledCombo(
            section, "Or pick a known instrument (auto-fills pip size + $/point)",
            ["(none -- set manually)"] + known_instrument_symbols(),
            default="(none -- set manually)",
        )
        self.r_instrument.combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_instrument_spec())
        self.r_contract_size = LabeledEntry(
            section, "Contract $/point (blank = not lot-rounded; auto-filled by instrument pick above)", "",
        )
        # Accuracy overhaul: how a stop that is too wide for the dollar risk
        # budget is handled, and the optional 1-minute fill-order replay.
        self.r_sizing_mode = LabeledCombo(
            section, "Sizing mode (stop too wide for the budget)",
            ["fit_stop", "skip", "micro_fallback", "fixed_contracts"], default="fit_stop",
        )
        self.r_max_stop_dollars = LabeledEntry(section, "Max stop in $ (blank = none)", "")
        self.r_fixed_contracts = LabeledEntry(section, "Fixed contracts (fixed_contracts mode only)", "")
        pip_detect_row = Frame(section, bg=PANEL)
        pip_detect_row.pack(anchor="w", padx=18, pady=(0, 8))
        self._button(
            pip_detect_row, "DETECT PIP SIZE FROM DATA", self._detect_pip_size_from_data
        ).pack(side="left")
        self.pip_detect_status = Label(
            pip_detect_row, text="", bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        )
        self.pip_detect_status.pack(side="left", padx=(10, 0))

        adaptive_section = self._section(
            f, "Adaptive risk (optional)",
            "Declarative, engine-level money-management rules -- de-risk after a losing "
            "streak, cut size once a daily loss threshold is hit, or coast once a "
            "percentage of the way to the profit target. No strategy source can see its "
            "own trade outcomes at signal-generation time; these rules apply at the "
            "engine level instead, the same way the Daily Loss Limit circuit breaker "
            "(Step 3, Prop Rules) already does. Off by default -- nothing changes about "
            "position sizing unless enabled and at least one rule is added below. Active "
            "rules stack multiplicatively (e.g. two active 0.5x rules together size at "
            "0.25x, not 0.5x).",
        )
        self.adaptive_risk_enabled = LabeledCheckbox(adaptive_section, "Enable adaptive risk for Run & Report (Step 5)", False)
        self.adaptive_profit_target_pct = LabeledEntry(
            adaptive_section, "Profit target (% of balance, for 'progress to target' rules)", 8.0,
        )

        rules_frame = Frame(adaptive_section, bg=PANEL)
        rules_frame.pack(fill="both", expand=True, padx=18, pady=(2, 8))
        self.adaptive_rule_listbox = Listbox(
            rules_frame, height=5, selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1,
            highlightbackground=BORDER, font=(MONO, 9),
        )
        self.adaptive_rule_listbox.pack(side="left", fill="both", expand=True)
        rule_scroll = ttk.Scrollbar(rules_frame, orient="vertical", command=self.adaptive_rule_listbox.yview, style="T58.Vertical.TScrollbar")
        rule_scroll.pack(side="right", fill="y")
        self.adaptive_rule_listbox.config(yscrollcommand=rule_scroll.set)
        self._bind_isolated_wheel(self.adaptive_rule_listbox)

        rule_btn_row = Frame(adaptive_section, bg=PANEL)
        rule_btn_row.pack(anchor="w", padx=18, pady=(0, 12))
        self._button(rule_btn_row, "ADD RULE", self._adaptive_add_rule, primary=True).pack(side="left")
        self._button(rule_btn_row, "REMOVE SELECTED RULE", self._adaptive_remove_rule).pack(side="left", padx=8)

        self._adaptive_rules: list[AdaptiveRiskRule] = []

    def _adaptive_add_rule(self):
        rule = _AdaptiveRuleDialog.ask(self.root)
        if rule is None:
            return
        self._adaptive_rules.append(rule)
        self.adaptive_rule_listbox.insert(END, f"{rule.trigger} >= {rule.threshold}  ->  x{rule.risk_multiplier}")

    def _adaptive_remove_rule(self):
        sel = self.adaptive_rule_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.adaptive_rule_listbox.delete(idx)
        del self._adaptive_rules[idx]

    def _build_adaptive_risk_config(self) -> AdaptiveRiskConfig | None:
        """Returns None (not just a disabled config) when adaptive risk isn't
        enabled, so callers can pass it straight through to run_backtest()'s
        `adaptive_risk=` parameter unconditionally -- None there means
        "behave exactly as if this parameter never existed," same as a
        disabled config would, but avoids constructing one at all on the
        common path where nobody has touched this section."""
        if not self.adaptive_risk_enabled.get() or not self._adaptive_rules:
            return None
        target_pct = self.adaptive_profit_target_pct.get_float(8.0)
        balance = self.r_initial_balance.get_float(100000)
        return AdaptiveRiskConfig(
            enabled=True,
            rules=list(self._adaptive_rules),
            profit_target_amount=balance * target_pct / 100.0,
        )

    def _apply_instrument_spec(self):
        """Shared by both the main Step 04 Risk & Execution tab and every
        RunContextPanel-embedded tab's own risk section (see that class's
        matching method) -- looks up the picked symbol in
        app.data.instrument_specs.KNOWN_INSTRUMENTS and fills pip_size +
        the new contract $/point field from it."""
        symbol = self.r_instrument.var.get()
        spec = get_instrument_spec(symbol)
        if spec is None:
            return
        self.r_pip_size.var.set(str(spec.pip_size))
        self.r_contract_size.var.set(str(spec.contract_size))
        # COMMISSION-DEFAULT-UPGRADE (2026-09-24): only fills in a realistic
        # commission when the field is still at its 0 default -- never
        # overwrites a rate the user already typed in. Mirrors app.data.
        # instrument_specs.apply_instrument_spec's own rule for the web/
        # backend path, so a strategy picked from this dropdown can no
        # longer end up silently backtested at $0 commission.
        commission_note = ""
        if self.r_commission.get_float(0) == 0.0:
            self.r_commission.var.set(str(spec.default_commission_round_turn))
            commission_note = f", commission=${spec.default_commission_round_turn:g}/trade"
        self.pip_detect_status.config(
            text=f"Applied {spec.symbol} ({spec.description}, {spec.exchange}): "
                 f"pip_size={spec.pip_size}, ${spec.contract_size}/point per contract{commission_note}.",
            fg=GREEN,
        )

    def _build_risk_config(self) -> RiskConfig:
        contract_size_str = self.r_contract_size.get_str().strip()
        return RiskConfig(
            initial_balance=self.r_initial_balance.get_float(100000),
            risk_mode=self.r_risk_mode.get_str().strip() or "percent",
            risk_value=self.r_risk_value.get_float(1.0),
            max_trades_per_day=self.r_max_trades_day.get_int(10),
            commission_per_trade=self.r_commission.get_float(0),
            slippage_pips=self.r_slippage.get_float(0.5),
            spread_pips=self.r_spread.get_float(1.0),
            pip_size=self.r_pip_size.get_float(0.0001),
            contract_size=float(contract_size_str) if contract_size_str else None,
            **self._accuracy_risk_kwargs(),
        )

    def _accuracy_risk_kwargs(self) -> dict:
        """Sizing-mode fields (absent on tabs built without them -> defaults)."""
        out: dict = {}
        try:
            mode = self.r_sizing_mode.get_str().strip()
            if mode:
                out["sizing_mode"] = mode
            msd = self.r_max_stop_dollars.get_str().strip()
            if msd:
                out["max_stop_dollars"] = float(msd)
            fc = self.r_fixed_contracts.get_str().strip()
            if fc and mode == "fixed_contracts":
                out["fixed_contracts"] = int(float(fc))
        except (AttributeError, ValueError):
            return {k: v for k, v in out.items() if k == "sizing_mode"}
        return out

    def _detect_pip_size_from_data(self, status_label=None):
        """Suggests a pip_size from whatever's currently selected in Step 2
        (Market Data), rather than leaving pip_size at its FX default
        (0.0001) for non-FX instruments -- the single most common cause of
        a strategy's fixed-pips stop translating into a nonsensical
        position size. Only ever suggests a starting value; the person
        still confirms it by seeing it land in the field.

        `status_label` lets a second "DETECT PIP SIZE FROM DATA" button
        elsewhere (e.g. Evolution Lab's own copy) show the same result in
        its own status line, rather than only updating 04 Risk's label
        which may not be the tab currently visible. Defaults to 04 Risk's
        label so every existing call site keeps working unchanged."""
        label = status_label if status_label is not None else self.pip_detect_status
        if not self.csv_paths:
            label.config(
                text="Select a market data CSV in Step 2 first.", fg=AMBER,
            )
            return
        try:
            result = import_csv(self.csv_paths[0])
            if not result.is_valid:
                label.config(text="Couldn't read that CSV.", fg=RED)
                return
            suggested = suggest_pip_size(result.dataframe)
            self.r_pip_size.var.set(str(suggested))
            label.config(
                text=f"Suggested {suggested} from {os.path.basename(self.csv_paths[0])} "
                     f"-- confirm this matches the instrument before running a backtest.",
                fg=GREEN,
            )
        except Exception as exc:
            label.config(text=f"Couldn't detect: {exc}", fg=RED)
