"""
T58 Licensing Gate -- one function, ensure_licensed(), called exactly
once from app/main.py before app.ui.main_window.launch(). This is the
entire integration surface between licensing and the rest of the app:
delete this package and the one call site in app/main.py, and the
backtesting engine, strategy adapters, and every tab are completely
unaffected -- see app/licensing/client.py's own module docstring for the
full reasoning behind keeping this a boundary rather than something
threaded through individual features.

Deliberately self-contained (its own small color palette, not imported
from app.ui.main_window) so this module never needs to import the
20,000-line main UI file just to put up a login screen -- keeping the
two independent is itself part of the isolation this is built for.

tkinter is imported lazily, inside show_activation_window() (including
the ActivationWindow class definition itself, not just its
instantiation) rather than at module level -- so ensure_licensed(
interactive=False), used by every headless/scripted CLI flag in
app/main.py, never touches tkinter at all, exactly matching how
app/main.py itself already lazy-imports app.ui.main_window only for the
GUI path.
"""
from __future__ import annotations

from app.licensing import client

# A small, self-contained slice of the app's dark theme (see
# app.ui.main_window.THEMES["dark"] -- kept in sync by eye, not by
# import, on purpose; see this module's own docstring). Matches the web
# activation page (app/web/templates/activate.html).
BG = "#05070A"
PANEL = "#0D1017"
PANEL_2 = "#10141C"
PANEL_3 = "#151A24"
BORDER = "#1C2230"
BORDER_LIGHT = "#2A3242"
TEXT = "#E7EBF2"
TEXT_MUTED = "#8A93A6"
TEXT_DIM = "#5B6478"
GREEN = "#35E0B0"
GREEN_HOVER = "#5BEBC4"
RED = "#FF6F6F"
VIOLET = "#7B3DFF"
CYAN = "#00D4FF"
ACCENT = GREEN  # primary accent (kept as a name other code may import)

KEY_PLACEHOLDER = "T58-XXXX-XXXX-XXXX-XXXX"


def _blend(c1: str, c2: str, t: float) -> str:
    a, b = c1.lstrip("#"), c2.lstrip("#")
    r = [round(int(a[i:i + 2], 16) + (int(b[i:i + 2], 16) - int(a[i:i + 2], 16)) * t) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(r)


def show_activation_window(initial_message: str = "", initial_email: str = "") -> bool:
    """Builds and runs the activation window, returns True iff the person
    successfully activated a license before closing it. tkinter (and the
    ActivationWindow class itself) is defined inside this function so
    nothing in this module needs tkinter to be importable except when a
    GUI launch actually needs to show this window -- see this module's
    own docstring."""
    import math
    import queue
    import threading
    import tkinter as tk
    from pathlib import Path
    from tkinter import ttk

    def _font(size=10, weight="normal"):
        return ("Segoe UI", size, weight) if weight != "normal" else ("Segoe UI", size)

    class ActivationWindow(tk.Tk):
        """The login/activation screen -- shown when there is no valid,
        currently-active license cached locally. Two outcomes: the person
        activates (self.activated becomes True, the window closes and the
        app starts), or they close the window without activating (the app
        exits without ever reaching main_window.launch()).

        The network call runs on a worker thread, so the window never
        freezes or goes "Not Responding" while the license server answers."""

        W, H = 480, 640

        def __init__(self):
            super().__init__()
            self.activated = False
            self._busy = False
            self._results: "queue.Queue[tuple[bool, str]]" = queue.Queue()
            self.title("Activate \u2014 T58 Quant Algo Backtester")
            self.configure(bg=BG)
            x = (self.winfo_screenwidth() - self.W) // 2
            y = max((self.winfo_screenheight() - self.H) // 2 - 20, 0)
            self.geometry(f"{self.W}x{self.H}+{x}+{y}")
            self.resizable(False, False)
            self.protocol("WM_DELETE_WINDOW", self._on_close)

            try:
                icon_path = Path(__file__).resolve().parents[2] / "app" / "ui" / "assets" / "t58_mark_medium.png"
                if icon_path.exists():
                    self._icon_image = tk.PhotoImage(file=str(icon_path))
                    self.iconphoto(True, self._icon_image)
            except Exception:
                pass

            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("T58.Horizontal.TProgressbar", background=GREEN, troughcolor=PANEL_3,
                            bordercolor=PANEL_3, lightcolor=GREEN, darkcolor=GREEN)

            # --- the card: a rounded panel centred in the window (web: .activate-card)
            cw, ch = 400, 580
            self._canvas = tk.Canvas(self, width=self.W, height=self.H, bg=BG, highlightthickness=0, bd=0)
            self._canvas.pack(fill="both", expand=True)
            x0, y0 = (self.W - cw) // 2, (self.H - ch) // 2
            self._round_rect(x0, y0, x0 + cw, y0 + ch, 16, fill=PANEL_2, outline=BORDER)
            card = tk.Frame(self._canvas, bg=PANEL_2)
            self._canvas.create_window(x0 + cw // 2, y0 + ch // 2, window=card, width=cw - 4, height=ch - 4)
            pad = tk.Frame(card, bg=PANEL_2)
            pad.pack(fill="both", expand=True, padx=30, pady=(26, 18))

            # brand tile (cyan -> violet gradient, rounded) + title
            mark = tk.Canvas(pad, width=60, height=48, bg=PANEL_2, highlightthickness=0)
            mark.pack(pady=(0, 10))
            self._draw_mark(mark, 60, 48)
            tk.Label(pad, text="Activate T58", bg=PANEL_2, fg=TEXT, font=_font(18, "bold")).pack()
            tk.Label(
                pad, text="Enter the email and license key from your purchase confirmation.",
                bg=PANEL_2, fg=TEXT_MUTED, font=_font(9), wraplength=330, justify="center",
            ).pack(pady=(4, 20))

            self.email_var = tk.StringVar(self, value=initial_email)
            self.key_var = tk.StringVar(self)
            self.email_entry = self._field(pad, "Email", self.email_var)
            self.key_entry = self._field(pad, "License key", self.key_var, placeholder=KEY_PLACEHOLDER)

            self.remember_var = tk.BooleanVar(self, value=True)
            tk.Checkbutton(
                pad, text="Remember this license on this device", variable=self.remember_var,
                bg=PANEL_2, fg=TEXT_MUTED, selectcolor=PANEL_3, activebackground=PANEL_2, activeforeground=TEXT,
                font=_font(9), anchor="w", highlightthickness=0, bd=0, cursor="hand2",
            ).pack(anchor="w", pady=(2, 2))
            tk.Label(
                pad,
                text="Unchecked: this license works for today's session only \u2014 you'll be asked to "
                     "activate again next time you open the app.",
                bg=PANEL_2, fg=TEXT_DIM, font=_font(8), wraplength=330, justify="left",
            ).pack(anchor="w", pady=(0, 14))

            self.activate_btn = tk.Button(
                pad, text="Activate", command=self._on_activate, bg=GREEN, fg="#04120E", relief="flat", bd=0,
                font=_font(11, "bold"), padx=10, pady=11, cursor="hand2", activebackground=GREEN_HOVER,
                activeforeground="#04120E", disabledforeground="#0A3A2E",
            )
            self.activate_btn.pack(fill="x")
            self.activate_btn.bind("<Enter>", lambda _e: self._btn_hover(True))
            self.activate_btn.bind("<Leave>", lambda _e: self._btn_hover(False))

            self.progress = ttk.Progressbar(pad, mode="indeterminate", length=330, style="T58.Horizontal.TProgressbar")
            self.status_var = tk.StringVar(self, value=initial_message)
            self.status_label = tk.Label(
                pad, textvariable=self.status_var, bg=PANEL_2, fg=RED, font=_font(9),
                wraplength=330, justify="center",
            )
            self.status_label.pack(pady=(12, 0))
            tk.Label(
                pad, text="No license yet? Purchases and support are handled through Whop.",
                bg=PANEL_2, fg=TEXT_DIM, font=_font(8), wraplength=330, justify="center",
            ).pack(side="bottom", pady=(14, 0))

            self.bind("<Return>", lambda _e: self._on_activate())
            self.after(50, self._poll)
            (self.email_entry if not initial_email else self.key_entry).focus_set()

        # ---------------------------------------------------------- drawing helpers
        def _round_rect(self, x0, y0, x1, y1, r, **kw):
            pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
                   x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
            return self._canvas.create_polygon(pts, smooth=True, splinesteps=16, **kw)

        @staticmethod
        def _draw_mark(cv, w, h, r=10):
            for i in range(w - 2):
                t = i / max(w - 3, 1)
                edge = min(i, w - 3 - i)
                inset = 0.0 if edge >= r else r - math.sqrt(max(r * r - (r - edge - 0.5) ** 2, 0.0))
                cv.create_line(1 + i, 1 + inset, 1 + i, h - 1 - inset, fill=_blend(CYAN, VIOLET, t))
            cv.create_text(w / 2, h / 2, text="T58", fill="#04120E", font=_font(11, "bold"))

        def _field(self, parent, label, var, placeholder=None):
            tk.Label(parent, text=label, bg=PANEL_2, fg=TEXT_MUTED, font=_font(9), anchor="w").pack(fill="x")
            entry = tk.Entry(
                parent, textvariable=var, bg=PANEL_3, fg=TEXT, insertbackground=TEXT, relief="flat",
                font=_font(11), highlightthickness=1, highlightbackground=BORDER_LIGHT, highlightcolor=GREEN, bd=0,
            )
            entry.pack(fill="x", pady=(3, 14), ipady=8)
            if placeholder:
                entry._placeholder = placeholder
                var.set(placeholder)
                entry.configure(fg=TEXT_DIM)

                def _in(_e):
                    if var.get() == placeholder:
                        var.set("")
                        entry.configure(fg=TEXT)

                def _out(_e):
                    if not var.get().strip():
                        var.set(placeholder)
                        entry.configure(fg=TEXT_DIM)

                entry.bind("<FocusIn>", _in)
                entry.bind("<FocusOut>", _out)
            return entry

        def _btn_hover(self, on):
            if not self._busy:
                self.activate_btn.configure(bg=GREEN_HOVER if on else GREEN)

        # ---------------------------------------------------------- activation
        def _key_value(self) -> str:
            v = self.key_var.get().strip()
            return "" if v == KEY_PLACEHOLDER else v

        def _set_busy(self, busy: bool, label: str):
            self._busy = busy
            self.activate_btn.configure(
                state="disabled" if busy else "normal", text=label,
                bg=_blend(GREEN, PANEL_2, 0.45) if busy else GREEN,
            )
            if busy:
                self.progress.pack(fill="x", pady=(12, 0), before=self.status_label)
                self.progress.start(12)
            else:
                self.progress.stop()
                self.progress.pack_forget()

        def _on_activate(self):
            if self._busy:
                return
            email, key, remember = self.email_var.get(), self._key_value(), self.remember_var.get()
            if not email.strip() or not key:
                self.status_label.config(fg=RED)
                self.status_var.set("Enter both your email and license key.")
                return
            self.status_var.set("")
            self._set_busy(True, "Activating\u2026")

            def work():
                try:
                    self._results.put(client.activate(email, key, remember=remember))
                except Exception as exc:  # noqa: BLE001 -- never leave the window stuck on "Activating"
                    self._results.put((False, f"Activation failed: {exc}"))

            threading.Thread(target=work, daemon=True).start()

        def _poll(self):
            try:
                ok, message = self._results.get_nowait()
            except queue.Empty:
                self.after(50, self._poll)
                return
            if ok:
                self.activated = True
                self.progress.stop()
                self.activate_btn.configure(text="\u2713  Activated \u2014 starting T58\u2026", state="disabled",
                                            bg=GREEN, disabledforeground="#04120E")
                self.status_label.config(fg=GREEN)
                self.status_var.set("")
                self.after(250, self.destroy)  # brief confirmation, then straight into the app
                return
            self._set_busy(False, "Activate")
            self.status_label.config(fg=RED)
            self.status_var.set(message)
            self.after(50, self._poll)

        def _on_close(self):
            self.activated = False
            self.destroy()

    window = ActivationWindow()
    window.mainloop()
    return window.activated


def ensure_licensed(interactive: bool = True) -> bool:
    """Call once, before app.ui.main_window.launch(). Returns True if the
    app should proceed to launch, False if it should exit immediately.

    Fast path: an already-activated, currently-valid (or within its
    offline grace period) license validates silently -- no window ever
    appears, so this adds no friction to every normal launch after the
    first.

    `interactive=False` (used by every headless/scripted CLI flag in
    app/main.py) never shows the Tkinter window, and never imports
    tkinter at all -- a CLI/headless run with no valid cached license
    just fails with a clear message instead of popping up a GUI, since a
    login window has no sensible behavior in a script/cron context.
    """
    ok, message = client.validate_fast_start()
    if ok:
        return True

    if not interactive:
        print(f"T58 license check failed: {message}")
        print("Run the app normally (without any CLI flags) once to activate, or check your license status.")
        return False

    state = client.load_state()
    # "Not activated." is the normal state of a brand-new install, not an error --
    # don't greet a first-time user with red error text.
    first_run = message.strip() == "Not activated."
    activated = show_activation_window(initial_message="" if first_run else message, initial_email=state.email)
    if not activated:
        return False
    # activate() just talked to the license server and saved the result, so
    # there is nothing left to verify -- the old code made a SECOND blocking
    # network round trip here (client.validate()), which is a big part of why
    # starting the app right after activating felt slow.
    return True
