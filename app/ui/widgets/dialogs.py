"""Small modal dialogs."""
from __future__ import annotations

from app.backtest.adaptive_risk import AdaptiveRiskError, AdaptiveRiskRule
from tkinter import Button, END, Entry, Frame, Label, Listbox, SINGLE, StringVar, Toplevel, messagebox, ttk
from app.ui.common import _ADAPTIVE_RISK_TRIGGERS, _ADAPTIVE_RISK_TRIGGER_HELP, _safe_font
from app.ui.theme import MONO
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
METAL_BRIGHT = ""
AMBER = ""
theme.bind(globals())


class _AdaptiveRuleDialog(Toplevel):
    """Small modal for adding one AdaptiveRiskRule -- a trigger type, a
    threshold, and a size multiplier. Used from the Risk & Execution tab's
    Adaptive Risk section; see app.backtest.adaptive_risk for the trigger
    semantics."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Add adaptive risk rule")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(parent)
        self.result: AdaptiveRiskRule | None = None

        self.trigger_var = StringVar(value=_ADAPTIVE_RISK_TRIGGERS[0])
        self.threshold_var = StringVar(value="2")
        self.multiplier_var = StringVar(value="0.5")

        pad = dict(padx=14, pady=(10, 2))
        Label(self, text="Trigger", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x", **pad)
        trigger_combo = ttk.Combobox(
            self, textvariable=self.trigger_var, values=_ADAPTIVE_RISK_TRIGGERS,
            state="readonly", width=28, font=_safe_font(9), style="T58.TCombobox",
        )
        trigger_combo.pack(fill="x", padx=14)
        trigger_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_help())

        self.help_label = Label(
            self, text=_ADAPTIVE_RISK_TRIGGER_HELP[self.trigger_var.get()],
            bg=PANEL, fg=AMBER, font=_safe_font(8), wraplength=280, justify="left", anchor="w",
        )
        self.help_label.pack(fill="x", padx=14, pady=(4, 6))

        Label(self, text="Threshold", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x", **pad)
        Entry(
            self, textvariable=self.threshold_var, bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, font=_safe_font(10),
        ).pack(fill="x", padx=14, ipady=4)

        Label(self, text="Risk multiplier (e.g. 0.5 = half size)", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x", **pad)
        Entry(
            self, textvariable=self.multiplier_var, bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, font=_safe_font(10),
        ).pack(fill="x", padx=14, ipady=4)

        btn_row = Frame(self, bg=PANEL)
        btn_row.pack(fill="x", padx=14, pady=14)
        Button(btn_row, text="Cancel", command=self._cancel).pack(side="right", padx=(8, 0))
        Button(btn_row, text="Add", command=self._confirm).pack(side="right")

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _update_help(self):
        self.help_label.config(text=_ADAPTIVE_RISK_TRIGGER_HELP[self.trigger_var.get()])

    def _confirm(self):
        try:
            threshold = float(self.threshold_var.get())
            multiplier = float(self.multiplier_var.get())
            self.result = AdaptiveRiskRule(
                trigger=self.trigger_var.get(), threshold=threshold, risk_multiplier=multiplier,
            )
        except (ValueError, AdaptiveRiskError) as exc:
            messagebox.showerror("Invalid rule", str(exc), parent=self)
            return
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    @classmethod
    def ask(cls, parent) -> AdaptiveRiskRule | None:
        dialog = cls(parent)
        parent.wait_window(dialog)
        return dialog.result


class _ChoiceListDialog(Toplevel):
    """Small modal single-select picker: a title, a prompt, a scrollable
    list of string choices, and OK/Cancel. Used wherever a quick "pick one
    of these saved items" step is needed without spinning up a whole tab
    (e.g. Multi-Asset Portfolio / Multi-Strategy Ensemble's ADD ... MANUAL
    buttons, which need to pick one saved Manual Strategy Builder config
    by name)."""

    def __init__(self, parent, title: str, prompt: str, choices: list[str]):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(parent)
        self.result: str | None = None

        Label(
            self, text=prompt, bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
            wraplength=340, justify="left", anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 6))

        list_frame = Frame(self, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=14)
        self.listbox = Listbox(
            list_frame, height=min(10, max(4, len(choices))), selectmode=SINGLE, exportselection=False,
            bg=PANEL_3, fg=TEXT, selectbackground=BORDER_LIGHT, selectforeground=METAL_BRIGHT,
            activestyle="none", relief="flat", bd=0, highlightthickness=1, highlightbackground=BORDER,
            font=(MONO, 9), width=44,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview, style="T58.Vertical.TScrollbar")
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)
        for c in choices:
            self.listbox.insert(END, c)
        if choices:
            self.listbox.selection_set(0)
        self.listbox.bind("<Double-Button-1>", lambda _e: self._confirm())

        btn_row = Frame(self, bg=PANEL)
        btn_row.pack(fill="x", padx=14, pady=14)
        Button(btn_row, text="Cancel", command=self._cancel).pack(side="right", padx=(8, 0))
        Button(btn_row, text="OK", command=self._confirm).pack(side="right")

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _confirm(self):
        sel = self.listbox.curselection()
        if sel:
            self.result = self.listbox.get(sel[0])
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    @classmethod
    def ask(cls, parent, title: str, prompt: str, choices: list[str]) -> str | None:
        dialog = cls(parent, title, prompt, choices)
        parent.wait_window(dialog)
        return dialog.result
