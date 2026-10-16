"""Windows-specific window helpers (DPI, DWM, taskbar icon, dark title bar, messagebox focus patch)."""
from __future__ import annotations

import sys
from tkinter import Tk, messagebox
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
CURRENT_THEME = "dark"
theme.bind(globals())


# ---------------------------------------------------------------------------
# BUGFIX (Sep 2026): "clicked RUN and it said RUNNING... but nothing
# happened" / "can't type in the AI Assistant chatbox" turned out to be ONE
# root cause, not two.
#
# Every async action in this app (AI Assistant chat/daily-brief, Options
# Outlook, every Quant Lab tool, etc.) runs its work on a background thread
# and reports success/failure back via `self.root.after(0, callback)` (see
# _ai_run_async / _quant_lab_run_async). That callback runs on the Tk main
# thread, so calling messagebox.showerror() from it is normally safe -- BUT
# on Windows, a new top-level window can only be promoted to the foreground
# by a process that is already the foreground process AND is responding to
# real user input (a click, a keypress). A `root.after(...)`-driven callback
# is a TIMER callback, not a live input event, so Windows silently refuses
# the foreground-window request Tk makes when the error dialog is created.
# Combined with this app's custom borderless main window
# (root.overrideredirect(True), see MainWindow.__init__/_build_custom_titlebar),
# the dialog opens fully hidden BEHIND the main window instead of merely
# losing focus -- there's no native title bar/taskbar affordance left to
# reveal it. Since messagebox.showerror() is modal (grab_set()), that
# invisible dialog also silently swallows ALL further keyboard/mouse input
# to the whole app until it's dismissed -- which is exactly why a
# perfectly-typed AI Assistant question can appear to do nothing: an
# earlier failed/errored async action (e.g. a chat send or Options Outlook
# run before Ollama was configured) already left an invisible modal dialog
# eating every keystroke.
#
# Fix: monkeypatch tkinter.messagebox's show*/ask* functions so every call
# in this file (133+ call sites -- far too many to touch individually)
# briefly forces the root window `-topmost` immediately before showing the
# dialog. `-topmost` uses SetWindowPos(HWND_TOPMOST, ...) under the hood,
# which is a z-order change, not a foreground-activation request, so it is
# NOT subject to the same Windows restriction -- it reliably pulls the
# root (and its transient dialog, which Tk stacks above its owner
# regardless of activation state) above every other window. Reset right
# after so the app doesn't stay pinned above everything else afterward.
# ---------------------------------------------------------------------------
def _patch_messagebox_for_overrideredirect_focus() -> None:
    import tkinter as _tk

    def _wrap(fn):
        def _wrapped(*args, **kwargs):
            root = kwargs.get("parent") or _tk._default_root
            if root is not None:
                try:
                    root.deiconify()
                except Exception:
                    pass
                try:
                    root.attributes("-topmost", True)
                    root.update_idletasks()
                except Exception:
                    pass
            try:
                return fn(*args, **kwargs)
            finally:
                if root is not None:
                    try:
                        root.attributes("-topmost", False)
                    except Exception:
                        pass
        return _wrapped

    for _name in (
        "showerror", "showwarning", "showinfo",
        "askyesno", "askokcancel", "askretrycancel", "askquestion", "askyesnocancel",
    ):
        original = getattr(messagebox, _name, None)
        if original is not None and not getattr(original, "_t58_focus_patched", False):
            wrapped = _wrap(original)
            wrapped._t58_focus_patched = True
            setattr(messagebox, _name, wrapped)


def _make_dpi_aware() -> None:
    """Tell Windows this process handles its own DPI scaling.

    Without this, a Tkinter app is "DPI-unaware" as far as Windows is
    concerned: it always lays itself out assuming a 96-DPI screen, and
    Windows then bitmap-stretches the *entire already-rendered window* up
    to match the display's actual scaling factor (125%/150%/175%/200%,
    all extremely common on laptops and 4K monitors). That stretch is
    what shows up as "everything is extremely zoomed in" -- and blurry,
    since it's a raw pixel upscale, not a re-render. This must run
    *before* the first Tk() window is created; it's a no-op (silently
    caught) on macOS/Linux and on any Windows build where it's already
    been declared some other way.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # PROCESS_SYSTEM_DPI_AWARE = 1. Preferred on Windows 8.1+: renders
        # at the system's actual DPI instead of being upscaled by the OS.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes
            # Fallback for older Windows without shcore.dll.
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass  # never let this block the app from launching


def _apply_tk_scaling(root: Tk) -> None:
    """Pin Tk's font/widget scaling to the app's original 96-DPI design,
    regardless of the monitor's real DPI.

    This app was built almost entirely with literal-pixel geometry --
    fixed Frame/Canvas widths, wraplength=, hand-drawn glow charts at
    fixed pixel coordinates, fixed-size PhotoImage icons -- alongside
    point-sized fonts. Those two things only line up correctly at the
    96-DPI scaling Tk defaults to (1.333 = 96/72).

    An earlier version of this function matched Tk's scaling to the
    *real* screen DPI instead (the standard high-DPI fix for apps built
    with DPI-relative layouts). Once _make_dpi_aware() below stops
    Windows from lying about the screen's DPI, that made Tk report a
    correct but *higher* DPI on any scaled display (125%/150%/etc.) --
    so every point-sized font rendered bigger while the literal-pixel
    containers around it stayed the same fixed pixel count, and text
    started overflowing/getting clipped throughout the app. Forcing the
    fixed 1.333 baseline here instead keeps fonts and hardcoded pixel
    geometry in the same proportion the app was actually designed and
    tested at, on any monitor. Combined with DPI-awareness (which stops
    the OS from bitmap-stretching the whole window), this renders the
    app at its original, decent-sized layout, crisply, everywhere.
    """
    try:
        root.tk.call("tk", "scaling", 96.0 / 72.0)
    except Exception:
        pass


def _force_dwm_composition(root: Tk) -> None:
    """Root-causing the reported "white/black streaks while scrolling or
    switching pages" on Windows.

    This app's tab-switch is 18 full-page Frames all `.place()`-d over
    each other at the identical (0,0,1,1) rectangle within one parent,
    switched via `.lift()` (see `_scrollable`'s own docstring on that
    pattern), and its scrollable tabs embed a live Frame full of real
    child widgets inside a Canvas (`canvas.create_window(...)` in
    `_scrollable`), scrolled via `yview_scroll`. Both operations move or
    restack large areas of REAL native child windows, not just redrawn
    pixels on a single canvas surface. A plain (non-composited) Tk
    top-level on Windows is drawn directly via GDI with no back buffer:
    when the screen refreshes mid-restack or mid-scroll -- easy to hit
    on a page with several widgets/canvases -- the previous frame's
    leftover pixels are still on screen until the next full repaint
    catches up, which is exactly the transient white/black streak being
    reported. This is a well-known Tkinter-on-Windows symptom of
    uncomposited rendering, not a bug in any single drawing call, so no
    fix to one canvas or one tab would address it.

    Giving the window ANY non-1.0 `-alpha` value flips Windows' Desktop
    Window Manager into compositing that window instead of letting the
    app draw straight to screen -- every repaint then goes through DWM's
    own off-screen back buffer and is presented as one complete frame,
    which is what actually stops the tearing. 0.9999 alpha is visually
    indistinguishable from fully opaque (no perceptible transparency)
    but is enough to cross the threshold that turns compositing on. This
    is a standard, widely-used mitigation for exactly this class of
    Tkinter/Windows flicker -- not a real transparency feature being
    used here for anything visual -- and is entirely a no-op wrapped in
    try/except everywhere else (macOS/Linux Tk either already composites
    natively or ignores a change this small, so nothing changes there).

    This is a mitigation, not a guaranteed fix for every possible cause
    of on-screen tearing (a sufficiently overloaded redraw -- e.g. an
    animation still ticking on a page you've just scrolled away from --
    can still visibly lag even once DWM is compositing). If streaks
    persist after this, the next thing worth checking is whether an
    active NeuralProgress animation (`.start()`/`.tick()`) is still
    running on a tab you're no longer looking at.
    """
    if sys.platform != "win32":
        return
    try:
        root.attributes("-alpha", 0.9999)
    except Exception:
        pass  # cosmetic mitigation only -- never let this block launch


def _hex_to_colorref(hex_color: str) -> int:
    """Windows COLORREF is 0x00BBGGRR -- the reverse byte order from a
    normal #RRGGBB hex string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def _keep_taskbar_icon(root: Tk) -> None:
    """overrideredirect(True) (see MainWindow.__init__) removes the
    window's native caption AND, as a side effect on Windows, its taskbar
    button and Alt-Tab entry -- both are driven by the same WS_CAPTION /
    WS_EX_APPWINDOW window styles Windows uses to draw the native title
    bar this fix is deliberately removing. Forcing WS_EX_APPWINDOW back on
    (and WS_EX_TOOLWINDOW off) via ctypes restores both, so the window is
    borderless but still shows up in the taskbar exactly like any other
    app -- no native caption required for either. The brief withdraw/
    after(deiconify) at the end forces Windows to actually re-evaluate the
    window's taskbar presence against the new style; SetWindowLongW alone
    changes the style but doesn't reliably repaint/re-register it on an
    already-created window.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        root.withdraw()
        root.after(10, root.deiconify)
    except Exception:
        pass  # cosmetic/UX-only -- never let this block launch


def _get_work_area() -> tuple[int, int, int, int] | None:
    """Returns (x, y, width, height) of the Windows work area -- the
    screen minus the taskbar -- so MAXIMIZE on a borderless window (see
    MainWindow._titlebar_toggle_maximize) doesn't cover the taskbar the
    way a naive "fill winfo_screenwidth/height" would. Returns None on
    non-Windows or on any failure; callers fall back to full screen size.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long),
            ]

        rect = _RECT()
        SPI_GETWORKAREA = 0x0030
        ok = ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
        if not ok:
            return None
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        return None


def _apply_dark_titlebar(root: Tk) -> None:
    """Recolors the native Windows title bar to match this app's own
    dark theme instead of Windows' default -- reported as "that ugly
    blue banner at the top" (Windows renders an unthemed window's
    title bar using the user's OS accent color, which is bright blue
    on most default installs; it has nothing to do with anything this
    app draws, since Tkinter has no built-in way to touch native
    window chrome at all). DWM (Desktop Window Manager) is the only
    thing that draws that bar, so this reaches it directly via
    ctypes -- there is no Tkinter API for this.

    Two DWM window attributes, applied independently so a Windows
    version missing one still benefits from the other:
      DWMWA_USE_IMMERSIVE_DARK_MODE (20) -- switches the caption to
        Windows' own dark-mode caption style (dark background, light
        text). Supported since Windows 10 build 18985 and all of
        Windows 11.
      DWMWA_CAPTION_COLOR (35) -- Windows 11 (build 22000+) only --
        sets the EXACT caption color, so the title bar matches this
        app's current theme color (BG) instead of Windows' generic
        dark gray. On a Windows 10 build that lacks this attribute,
        DwmSetWindowAttribute simply returns an error for this call
        (caught below) and the immersive-dark-mode call above still
        replaces the blue bar with a dark one.

    Re-run this after a theme toggle (see MainWindow._toggle_theme) so
    switching to Light mode also lightens the title bar to match.
    No-op (silently caught) on macOS/Linux, and never blocks launch or
    a theme toggle if any of this fails -- purely cosmetic."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi

        user32.GetAncestor.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
        GA_ROOT = 2
        hwnd = user32.GetAncestor(wintypes.HWND(root.winfo_id()), GA_ROOT)
        if not hwnd:
            return

        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        DWMWA_CAPTION_COLOR = 35

        prefer_dark = ctypes.c_int(1 if CURRENT_THEME == "dark" else 0)
        dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(prefer_dark), ctypes.sizeof(prefer_dark),
        )
        try:
            color = ctypes.c_int(_hex_to_colorref(BG))
            dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(color), ctypes.sizeof(color),
            )
        except Exception:
            pass  # DWMWA_CAPTION_COLOR unsupported on this Windows build -- dark mode above still applied

        # UPGRADE (Sep 2026 UI pass, round 3): the two DwmSetWindowAttribute
        # calls above take effect immediately -- but only in the sense that
        # DWM now *knows* the new values. It does not automatically repaint
        # the already-drawn caption with them; on a window that's already
        # on screen, the title bar visibly stays exactly as it was until
        # something else (a resize, a move, minimize/restore) forces
        # Windows to redraw the non-client frame. That's the actual bug
        # behind "the code runs with no error, but the title bar is still
        # blue" -- SetWindowPos with SWP_FRAMECHANGED is the standard way
        # to force that redraw immediately without actually moving or
        # resizing the window (all four SWP_NO* flags below keep its
        # position/size/z-order/activation exactly as they were).
        try:
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020
            user32.SetWindowPos(
                hwnd, None, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )
        except Exception:
            pass
    except Exception:
        pass  # cosmetic only -- never let this block launch or a theme toggle


_patch_messagebox_for_overrideredirect_focus()
