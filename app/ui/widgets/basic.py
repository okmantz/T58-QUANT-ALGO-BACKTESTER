"""Basic labeled form widgets."""
from __future__ import annotations

from tkinter import BooleanVar, Checkbutton, Entry, Frame, Label, StringVar, ttk
from app.ui.common import _safe_font
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
theme.bind(globals())


class LabeledEntry(Frame):
    def __init__(self, parent, label, default="", secret=False, width=20, variable=None):
        """`variable=` optionally takes an existing StringVar (e.g. from
        another tab's LabeledEntry, via `other_entry.var`) instead of
        creating a fresh one. Tkinter keeps every Entry pointed at the same
        StringVar in sync automatically, so two LabeledEntry widgets built
        this way -- one on each tab -- are the SAME setting shown/edited in
        two places, not two copies that can quietly drift apart. Used to
        mirror 03 Prop Rules / 04 Risk fields onto the Evolution Lab tab so
        a run always uses exactly what's shown there, in either place.
        `default` is ignored when `variable` is given, since the variable
        already carries whatever value its original widget set it to."""
        super().__init__(parent, bg=PANEL)
        self.configure(height=36)

        Label(
            self,
            text=label,
            width=31,
            anchor="w",
            bg=PANEL,
            fg=TEXT_MUTED,
            font=_safe_font(9),
        ).pack(side="left")

        self.var = variable if variable is not None else StringVar(value=str(default))

        entry_kwargs = dict(
            textvariable=self.var,
            width=width,
            bg=PANEL_3,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER_LIGHT,
            font=_safe_font(10),
        )
        # secret=True masks the field like a password input -- used for API
        # keys/secrets so they aren't visible over someone's shoulder or in
        # a screen share.
        if secret:
            entry_kwargs["show"] = "\u2022"

        self.entry = Entry(self, **entry_kwargs)
        self.entry.pack(side="left", ipady=5, padx=(4, 0))

        self.pack(fill="x", pady=3, padx=18)

    def get_float(self, fallback=0.0):
        try:
            return float(self.var.get())
        except ValueError:
            return fallback

    def get_int(self, fallback=0):
        try:
            return int(float(self.var.get()))
        except ValueError:
            return fallback

    def get_str(self):
        return self.var.get()


class LabeledCombo(Frame):
    def __init__(self, parent, label, values, default="", width=None):
        super().__init__(parent, bg=PANEL)

        Label(
            self, text=label, width=31, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left")

        self.var = StringVar(value=str(default))
        # width=18 was a fine default back when every combo's options were
        # short labels ("bootstrap", "shuffle", ...); Search Lab's Mode
        # dropdown has long descriptive option text (e.g. "Bulk backtest
        # -- upload multiple Python / PineScript / MQL5 files and run
        # each one") that a fixed width=18 clips hard in both the closed
        # box and ttk's popdown listbox, which sizes off the widget's own
        # width. Auto-sizing to the longest value (capped so it can't blow
        # out the layout) fixes that everywhere a combo is used, not just
        # this one spot -- pass an explicit width to opt back into the
        # old fixed-width behavior for a combo that wants it.
        if width is None:
            longest = max((len(str(v)) for v in values), default=18)
            width = min(60, max(18, longest + 2))
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=values, state="readonly",
            width=width, font=_safe_font(9), style="T58.TCombobox",
        )
        self.combo.pack(side="left", padx=(4, 0))
        self.pack(fill="x", pady=3, padx=18)

    def get_str(self):
        return self.var.get()

    def set_str(self, value):
        self.var.set(value)

    def set_options(self, values):
        """Replaces the dropdown's option list in place (e.g. after a
        background scan finds new files) -- re-sizes the popdown width to
        fit the new longest value the same way __init__ does, and clears
        the current selection if it no longer appears in the new list."""
        longest = max((len(str(v)) for v in values), default=18)
        self.combo.configure(width=min(60, max(18, longest + 2)), values=list(values))
        if self.var.get() not in values:
            self.var.set(values[0] if values else "")


class LabeledCheckbox(Frame):
    def __init__(self, parent, label, default=False):
        super().__init__(parent, bg=PANEL)

        self.var = BooleanVar(value=default)
        cb = Checkbutton(
            self, variable=self.var, bg=PANEL, activebackground=PANEL,
            highlightthickness=0, bd=0, selectcolor=PANEL_3,
        )
        cb.pack(side="left")

        Label(
            self, text=label, anchor="w",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9),
        ).pack(side="left", padx=(2, 0))
        self.pack(fill="x", pady=3, padx=18)

    def get(self) -> bool:
        return bool(self.var.get())


def _blend_hex(color_hex: str, toward_hex: str, t: float) -> str:
    """Blends color_hex toward toward_hex by fraction t (0=color, 1=toward).
    Standalone version of MainWindow._blend, usable by widgets that aren't
    the main window itself."""
    c = color_hex.lstrip("#")
    b = toward_hex.lstrip("#")
    cr, cg, cb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    br, bg_, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    r = round(cr + (br - cr) * t)
    g = round(cg + (bg_ - cg) * t)
    bl = round(cb + (bb - cb) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"
