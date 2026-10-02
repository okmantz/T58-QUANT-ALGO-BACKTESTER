"""
Startup splash (Oct 2026). A small borderless window shown the instant the app
starts and kept up -- with a live status line and a moving progress bar -- until
the main window is fully built, so there is never a stretch where the app seems
to have done nothing (after activating, or on a normal launch).

It is a Toplevel of the app's single (hidden) Tk root, not a second Tk(): two
roots in one process make "default root" widgets bind to the wrong one.
Self-contained on purpose (own palette, tkinter only) so it can appear before
the 21k-line main window module has even been imported.
"""
from __future__ import annotations

import math
import tkinter as tk

BG, PANEL, BORDER = "#05070A", "#10141C", "#2A3242"
TEXT, MUTED, GREEN, VIOLET, CYAN = "#E7EBF2", "#8A93A6", "#35E0B0", "#7B3DFF", "#00D4FF"


def _blend(c1: str, c2: str, t: float) -> str:
    a, b = c1.lstrip("#"), c2.lstrip("#")
    return "#%02x%02x%02x" % tuple(
        round(int(a[i:i + 2], 16) + (int(b[i:i + 2], 16) - int(a[i:i + 2], 16)) * t) for i in (0, 2, 4)
    )


def _font(size=10, weight="normal"):
    return ("Segoe UI", size, weight) if weight != "normal" else ("Segoe UI", size)


class StartupSplash:
    W, H = 400, 190

    def __init__(self, root: tk.Tk):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg=BORDER)
        x = (self.win.winfo_screenwidth() - self.W) // 2
        y = (self.win.winfo_screenheight() - self.H) // 2 - 30
        self.win.geometry(f"{self.W}x{self.H}+{x}+{y}")
        try:
            self.win.attributes("-topmost", True)
        except Exception:
            pass
        body = tk.Frame(self.win, bg=PANEL)
        body.pack(fill="both", expand=True, padx=1, pady=1)

        mark = tk.Canvas(body, width=60, height=48, bg=PANEL, highlightthickness=0)
        mark.pack(pady=(24, 8))
        for i in range(58):
            edge = min(i, 57 - i)
            inset = 0.0 if edge >= 10 else 10 - math.sqrt(max(100 - (10 - edge - 0.5) ** 2, 0.0))
            mark.create_line(1 + i, 1 + inset, 1 + i, 47 - inset, fill=_blend(CYAN, VIOLET, i / 57))
        mark.create_text(30, 24, text="T58", fill="#04120E", font=_font(11, "bold"))

        tk.Label(body, text="T58 Quant Algo Backtester", bg=PANEL, fg=TEXT, font=_font(12, "bold")).pack()
        self._status = tk.Label(body, text="Starting\u2026", bg=PANEL, fg=MUTED, font=_font(9))
        self._status.pack(pady=(4, 12))
        self._bar = tk.Canvas(body, width=300, height=4, bg=BORDER, highlightthickness=0)
        self._bar.pack()
        self._pos = 0.0
        self._fill = self._bar.create_rectangle(0, 0, 0, 4, fill=GREEN, outline="")
        self.win.update()

    def set_status(self, text: str) -> None:
        """Update the status line and nudge the progress bar forward."""
        try:
            self._status.config(text=text)
            self._pos = min(self._pos + (1.0 - self._pos) * 0.08 + 0.01, 0.97)  # eases toward (never reaches) full
            self._bar.coords(self._fill, 0, 0, 300 * self._pos, 4)
        except tk.TclError:
            pass

    def close(self) -> None:
        try:
            self._bar.coords(self._fill, 0, 0, 300, 4)
            self.win.update_idletasks()
            self.win.destroy()
        except tk.TclError:
            pass
