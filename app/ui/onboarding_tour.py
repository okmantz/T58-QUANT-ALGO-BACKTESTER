"""
T58 first-run guided tour (desktop).

An 8-step, skippable walkthrough shown once, the first time the main window
appears after license activation. Each step jumps to the REAL tab it is
talking about (the sidebar highlights it and opens its group), so a new user
sees exactly where to click: Market Data -> Strategy -> Prop rules -> Risk ->
Run & Report -> Full Pipeline -> Dashboard/next steps -> User Manual.

Design notes
- Self-contained on purpose: its own small palette (kept in sync by eye with
  app.ui.main_window.THEMES["dark"], same convention as app.licensing.gate),
  so importing this never imports the 20,000-line main window.
- The card is a child Frame `place()`d over the main window, NOT a Toplevel.
  The app can draw its own custom title bar (overrideredirect), where extra
  top-level windows misbehave and steal focus.
- "Seen" is a tiny JSON flag next to the license state in data/config/, so it
  shows once per install. A read/write failure never blocks the app: if the
  flag can't be read the tour simply doesn't auto-start.
- Replay any time: OnboardingTour(window).start(), or the "Take the tour"
  button on the Dashboard.
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

BG = "#05070A"
PANEL = "#0D1017"
PANEL_2 = "#10141C"
PANEL_3 = "#151A24"
BORDER = "#1C2230"
TEXT = "#E7EBF2"
TEXT_MUTED = "#8A93A6"
GREEN = "#35E0B0"
ACCENT = "#35E0B0"

FLAG_NAME = "onboarding_tour.json"
FLAG_VERSION = 1


def _font(size=10, weight="normal"):
    return ("Segoe UI", size, weight) if weight != "normal" else ("Segoe UI", size)


# (page key used by MainWindow._show_page, title, body)
STEPS = [
    (
        "dashboard",
        "Welcome to T58 \U0001F44B",
        "T58 answers one question:\n\n"
        "If you trade this strategy under your prop firm's rules, what are your odds of "
        "passing the evaluation and reaching a payout?\n\n"
        "It takes five simple moves: Data \u2192 Strategy \u2192 Rules \u2192 Run \u2192 Verdict. "
        "This tour jumps to each screen so you know where everything lives.",
    ),
    (
        "data",
        "1 \u00B7 Add your market data",
        "Every test needs price history. Load a file here (CSV, TSV or Parquet) with the columns "
        "timestamp, open, high, low, close, volume.\n\n"
        "It is saved automatically, so you only upload once. Loading several files lets you test "
        "multiple timeframes together.",
    ),
    (
        "strategyconfig",
        "2 \u00B7 Create or bring a strategy",
        "Build one with the no-code Manual builder, or paste/upload Python, PineScript or MQL5.\n\n"
        "No idea yet? Skip ahead: Speed Run and Forge Strategy (in the CREATE section) hunt for one "
        "for you, and Generate Strategies (AI) drafts one from a plain-English idea. Everything you "
        "save lands in your Strategy Library.",
    ),
    (
        "prop",
        "3 \u00B7 Enter your prop-firm rules",
        "Account size, profit target, daily loss limit, max drawdown and payout rules.\n\n"
        "Use the quick-fill presets: FTMO, Apex, TopStep, The5%ers, FundedNext and Lucid are built "
        "in. Firms change terms often, so double-check the numbers against your own account.",
    ),
    (
        "risk",
        "4 \u00B7 Set risk & costs",
        "How much you risk per trade (fixed $ or % of equity), plus commission, slippage and spread.\n\n"
        "Tip: use \"detect from data\" for pip size, and set a realistic commission. Costs are what "
        "separate a backtest that looks great from one that survives.",
    ),
    (
        "run",
        "5 \u00B7 Run your first test",
        "Hit RUN. You get a report with your probability of passing the evaluation and reaching a "
        "payout.\n\n"
        "The backtest trades from the first bar to the last bar of your data. If the account "
        "breaches or hits its profit target, that is logged and a fresh account starts, so you "
        "always see the whole history.",
    ),
    (
        "fullpipeline",
        "6 \u00B7 Full Pipeline: the deep test",
        "One button does everything: baseline backtest, lookahead-bias check, a search for a "
        "version that holds up out-of-sample, Monte Carlo, then a plain verdict: READY, MARGINAL "
        "or NOT READY, with reasons.\n\n"
        "It is the slowest tool in the app, so try Run & Report first. Winners save to your "
        "Strategy Library automatically.",
    ),
    (
        "manual",
        "You're set \u2014 what's next?",
        "The Dashboard is your home base: after every run it tells you what you're working on, "
        "whether it's working, why, and the exact next step.\n\n"
        "Every section also has its own Start Here page, and the User Manual is the full "
        "walkthrough. Want this tour again? Use \"Take the tour\" on the Dashboard.",
    ),
]


# ---------------------------------------------------------------- seen flag
def _flag_path() -> Path | None:
    try:
        from app.data.storage import get_app_base_dir
        d = Path(get_app_base_dir()) / "data" / "config"
        d.mkdir(parents=True, exist_ok=True)
        return d / FLAG_NAME
    except Exception:
        return None


def tour_seen() -> bool:
    """True if the tour was already shown. Fails CLOSED (True) on any
    problem reading the flag so a broken disk never nags on every launch."""
    p = _flag_path()
    if p is None:
        return True
    try:
        if not p.exists():
            return False
        return bool(json.loads(p.read_text(encoding="utf-8")).get("seen"))
    except Exception:
        return True


def mark_seen(how: str = "done") -> None:
    p = _flag_path()
    if p is None:
        return
    try:
        p.write_text(json.dumps({"seen": True, "how": how, "version": FLAG_VERSION}), encoding="utf-8")
    except Exception:
        pass


def maybe_start_tour(window) -> "OnboardingTour | None":
    """Called once from launch(): starts the tour only on a first run."""
    try:
        if tour_seen():
            return None
        tour = OnboardingTour(window)
        tour.start()
        return tour
    except Exception:
        return None  # a tour problem must never stop the app from opening


# --------------------------------------------------------------------- tour
class OnboardingTour:
    def __init__(self, window, steps=None):
        self.window = window
        self.root = window.root
        self.steps = list(steps or STEPS)
        self.i = 0
        self.card: tk.Frame | None = None
        self._bound: list[tuple[str, str]] = []

    # ---- lifecycle
    def start(self) -> None:
        if self.card is not None:
            return
        self.i = 0
        self.card = tk.Frame(self.root, bg=PANEL_2, highlightthickness=1, highlightbackground=GREEN)
        self.card.place(relx=1.0, rely=1.0, anchor="se", x=-28, y=-96)  # clear of the floating chat button
        for seq, fn in (("<Escape>", lambda e: self.finish("skipped")),
                        ("<Right>", lambda e: self.go(self.i + 1)),
                        ("<Left>", lambda e: self.go(self.i - 1))):
            bid = self.root.bind(seq, fn, add="+")
            self._bound.append((seq, bid))
        self._render()

    def finish(self, how: str = "done") -> None:
        mark_seen(how)
        for seq, bid in self._bound:
            try:
                self.root.unbind(seq, bid)
            except Exception:
                pass
        self._bound = []
        if self.card is not None:
            try:
                self.card.destroy()
            except Exception:
                pass
            self.card = None
        try:
            self.window._show_page("dashboard")
        except Exception:
            pass

    # ---- navigation
    def go(self, i: int) -> None:
        if self.card is None or i < 0 or i >= len(self.steps):
            return
        self.i = i
        self._render()

    def _render(self) -> None:
        key, title, body = self.steps[self.i]
        try:
            self.window._show_page(key)  # jump to the real tab; sidebar highlights it
        except Exception:
            pass
        card = self.card
        for w in card.winfo_children():
            w.destroy()
        last = self.i == len(self.steps) - 1

        top = tk.Frame(card, bg=PANEL_2)
        top.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(top, text=f"STEP {self.i + 1} OF {len(self.steps)}", bg=PANEL_2, fg=GREEN,
                 font=_font(8, "bold")).pack(side="left")
        x = tk.Label(top, text="\u00D7", bg=PANEL_2, fg=TEXT_MUTED, font=_font(14), cursor="hand2")
        x.pack(side="right")
        x.bind("<Button-1>", lambda e: self.finish("skipped"))

        tk.Label(card, text=title, bg=PANEL_2, fg=TEXT, font=_font(13, "bold"), anchor="w",
                 justify="left", wraplength=360).pack(fill="x", padx=16, pady=(6, 4))
        tk.Label(card, text=body, bg=PANEL_2, fg=TEXT, font=_font(9), anchor="w",
                 justify="left", wraplength=360).pack(fill="x", padx=16, pady=(0, 10))

        dots = tk.Frame(card, bg=PANEL_2)
        dots.pack(anchor="w", padx=16, pady=(0, 10))
        for k in range(len(self.steps)):
            d = tk.Label(dots, text="\u25CF", bg=PANEL_2, fg=(GREEN if k == self.i else BORDER),
                         font=_font(8), cursor="hand2")
            d.pack(side="left", padx=2)
            d.bind("<Button-1>", lambda e, k=k: self.go(k))

        row = tk.Frame(card, bg=PANEL_2)
        row.pack(fill="x", padx=16, pady=(0, 14))
        if not last:
            self._btn(row, "Skip", lambda: self.finish("skipped"), muted=True).pack(side="left")
        primary = self._btn(row, "Finish" if last else ("Show me around" if self.i == 0 else "Next"),
                            (lambda: self.finish("done")) if last else (lambda: self.go(self.i + 1)),
                            primary=True)
        primary.pack(side="right")
        if self.i > 0:
            self._btn(row, "Back", lambda: self.go(self.i - 1)).pack(side="right", padx=(0, 8))
        self.primary_button = primary
        self.card.lift()

    @staticmethod
    def _btn(parent, text, command, primary=False, muted=False):
        if primary:
            bg, fg = GREEN, "#04120E"
        elif muted:
            bg, fg = PANEL_2, TEXT_MUTED
        else:
            bg, fg = PANEL_3, TEXT
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=bg,
                         activeforeground=fg, relief="flat", bd=0, cursor="hand2",
                         font=_font(9, "bold"), padx=14, pady=6)
