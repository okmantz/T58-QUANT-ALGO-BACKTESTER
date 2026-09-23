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
# import, on purpose; see this module's own docstring).
BG = "#05070A"
PANEL = "#0D1017"
PANEL_3 = "#151A24"
BORDER = "#1C2230"
TEXT = "#E7EBF2"
TEXT_MUTED = "#8A93A6"
GREEN = "#35E0B0"
RED = "#FF6F6F"
ACCENT = "#7B3DFF"


def show_activation_window(initial_message: str = "", initial_email: str = "") -> bool:
    """Builds and runs the activation window, returns True iff the person
    successfully activated a license before closing it. tkinter (and the
    ActivationWindow class itself) is defined inside this function so
    nothing in this module needs tkinter to be importable except when a
    GUI launch actually needs to show this window -- see this module's
    own docstring."""
    import tkinter as tk
    from tkinter import ttk

    def _safe_font(size=10, weight="normal"):
        return ("Segoe UI", size, weight) if weight != "normal" else ("Segoe UI", size)

    class ActivationWindow(tk.Tk):
        """The login/activation screen. Shown when there is no valid,
        currently-active license cached locally. Two outcomes: the
        person activates successfully (self.activated becomes True,
        window closes, ensure_licensed() proceeds to launch the app), or
        they close the window without activating (self.activated stays
        False, the app exits without ever reaching main_window.launch())."""

        def __init__(self):
            super().__init__()
            self.activated = False
            self.title("T58 Quant Algo Backtester — Activation")
            self.configure(bg=BG)
            self.geometry("440x420")
            self.resizable(False, False)
            self.protocol("WM_DELETE_WINDOW", self._on_close)

            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("T58.TEntry", fieldbackground=PANEL_3, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER)

            container = tk.Frame(self, bg=BG, padx=32, pady=28)
            container.pack(fill="both", expand=True)

            tk.Label(
                container, text="T58 QUANT ALGO BACKTESTER", bg=BG, fg=ACCENT,
                font=_safe_font(11, "bold"),
            ).pack(anchor="w")
            tk.Label(
                container, text="Activate your license", bg=BG, fg=TEXT,
                font=_safe_font(18, "bold"),
            ).pack(anchor="w", pady=(4, 2))
            tk.Label(
                container, text="Enter the email and license key from your purchase confirmation.",
                bg=BG, fg=TEXT_MUTED, font=_safe_font(9), wraplength=380, justify="left",
            ).pack(anchor="w", pady=(0, 16))

            self.status_var = tk.StringVar(value=initial_message)
            self.status_label = tk.Label(
                container, textvariable=self.status_var, bg=BG, fg=RED, font=_safe_font(9),
                wraplength=380, justify="left",
            )
            self.status_label.pack(anchor="w", pady=(0, 10))

            tk.Label(container, text="Email", bg=BG, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x")
            self.email_var = tk.StringVar(value=initial_email)
            email_entry = ttk.Entry(container, textvariable=self.email_var, style="T58.TEntry", font=_safe_font(11))
            email_entry.pack(fill="x", pady=(2, 12), ipady=4)

            tk.Label(container, text="License key", bg=BG, fg=TEXT_MUTED, font=_safe_font(9), anchor="w").pack(fill="x")
            self.key_var = tk.StringVar()
            key_entry = ttk.Entry(container, textvariable=self.key_var, style="T58.TEntry", font=_safe_font(11))
            key_entry.pack(fill="x", pady=(2, 4), ipady=4)
            tk.Label(
                container, text="Format: T58-XXXX-XXXX-XXXX-XXXX", bg=BG, fg=TEXT_MUTED, font=_safe_font(8),
            ).pack(anchor="w", pady=(0, 16))

            self.remember_var = tk.BooleanVar(value=True)
            tk.Checkbutton(
                container, text="Remember this license on this device", variable=self.remember_var,
                bg=BG, fg=TEXT_MUTED, selectcolor=PANEL_3, activebackground=BG, activeforeground=TEXT,
                font=_safe_font(9), anchor="w",
            ).pack(anchor="w", pady=(0, 16))
            tk.Label(
                container,
                text="Unchecked: this license works for today's session only -- "
                     "you'll be asked to activate again next time you open the app.",
                bg=BG, fg=TEXT_MUTED, font=_safe_font(8), wraplength=380, justify="left",
            ).pack(anchor="w", pady=(0, 4))

            self.activate_btn = tk.Button(
                container, text="Activate", command=self._on_activate, bg=ACCENT, fg="#FFFFFF",
                font=_safe_font(11, "bold"), relief="flat", padx=10, pady=10, cursor="hand2",
            )
            self.activate_btn.pack(fill="x")

            tk.Label(
                container, text="No license yet? Purchases and support are handled through Whop.",
                bg=BG, fg=TEXT_MUTED, font=_safe_font(8), pady=14,
            ).pack(anchor="w")

            self.bind("<Return>", lambda _e: self._on_activate())
            email_entry.focus_set()

        def _on_activate(self):
            self.activate_btn.config(state="disabled", text="Activating...")
            self.update_idletasks()
            ok, message = client.activate(self.email_var.get(), self.key_var.get(), remember=self.remember_var.get())
            if ok:
                self.activated = True
                self.destroy()
                return
            self.status_label.config(fg=RED)
            self.status_var.set(message)
            self.activate_btn.config(state="normal", text="Activate")

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
    ok, message = client.validate()
    if ok:
        return True

    if not interactive:
        print(f"T58 license check failed: {message}")
        print("Run the app normally (without any CLI flags) once to activate, or check your license status.")
        return False

    state = client.load_state()
    activated = show_activation_window(initial_message=message, initial_email=state.email)
    if not activated:
        return False
    # Re-validate immediately after a successful activate() so the rest
    # of this function (and the caller) only ever has one source of
    # truth (client.validate()'s own return value) for "is this okay to
    # run right now" rather than trusting activate()'s own success flag
    # a second time.
    ok, _message = client.validate()
    return ok
