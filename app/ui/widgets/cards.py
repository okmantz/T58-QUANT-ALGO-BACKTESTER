"""Card / progress widgets."""
from __future__ import annotations

import math
import time
from tkinter import Canvas, Frame
from app.ui.common import _safe_font
from app.ui.widgets.basic import _blend_hex
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_2 = ""
BORDER = ""
TEXT = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
ACCENT = ""
NEON_CYAN = ""
theme.bind(globals())


class NeuralProgress(Canvas):
    """A glowing neon loading bar -- a rounded, softly-lit track with an
    animated cyan fill, a "LOADING…." label, and a live percentage
    readout, in the same visual family as this app's reference neon-dark
    mockups. Most call sites in this app are indeterminate (a Search Lab
    generation, a Monte Carlo run, a full pipeline stage -- none report a
    real 0-100% completion), so while running this shows a smoothly
    rising ESTIMATED percentage that eases toward (but never reaches)
    ~96% -- honest about being an estimate, while still giving a much
    more readable sense of "this is progressing" than a bar that just
    bounces back and forth.

    Drop-in replacement for `ttk.Progressbar(mode="indeterminate")`:
    supports the same `.start(interval)` / `.stop()` calls (interval is
    accepted for signature compatibility but this always animates at its
    own fixed frame rate) and the same `.pack(...)` usage -- every
    existing call site swaps in unchanged. Call `.set_progress(pct)`
    instead of/in addition to `.start()` for the rare call site that DOES
    know a real percentage (e.g. "fold 3 of 8") -- it freezes the
    estimate and shows the real number until `.start()` or `.stop()` is
    called again.
    """

    FRAME_MS = 33                     # ~30fps -- smooth without burning CPU on a purely decorative animation
    ESTIMATE_CEILING_PCT = 96.0        # never implies completion on its own -- only stop() does that
    ESTIMATE_TAU_SECONDS = 14.0        # how quickly the estimate eases toward the ceiling
    DISPLAY_EASE_TAU_SECONDS = 0.35    # how quickly the DRAWN bar glides toward whatever it's targeting
    SWEEP_SECONDS = 1.8                # time for one highlight streak to cross the filled portion
    CORNER_RADIUS = 7

    def __init__(self, parent, height: int = 30, **kwargs):
        super().__init__(parent, height=height, bg=PANEL_2, highlightthickness=0, **kwargs)
        self._height = height
        self._running = False
        self._after_id = None
        self._t0 = 0.0
        self._sweep_phase = 0.0
        self._manual_pct: float | None = None   # set via set_progress(); None means "use the eased estimate"
        # UPGRADE (2026-09-03, smoothness): _current_pct() used to return
        # whichever number it was targeting (the auto-estimate curve, or
        # set_progress()'s manual value) DIRECTLY -- fine for the smooth
        # auto-estimate curve, but every set_progress() call (Search Lab
        # "Stage 1: 40/84...", a fold completing, a candidate finishing)
        # made the bar SNAP straight to the new percentage with no
        # transition at all. That instant jump between milestones was the
        # actual "choppy" bit, not the fill/glow rendering. _displayed_pct
        # is the value actually drawn; _current_pct() now eases it toward
        # whatever the real target is every frame, so a jump from 40% to
        # 55% glides there over a couple hundred ms instead of teleporting.
        self._displayed_pct = 0.0
        self._last_frame_t = time.monotonic()
        self._label = "LOADING...."
        self.bind("<Configure>", lambda _e: self._draw())
        self._draw()  # resting state before the first start()

    def start(self, interval=None):
        if self._running:
            return
        self._running = True
        self._manual_pct = None
        self._t0 = time.monotonic()
        self._last_frame_t = self._t0
        self._displayed_pct = 0.0
        self._sweep_phase = 0.0
        self._tick()

    def set_progress(self, pct: float, label: str | None = None):
        """For the rare call site that knows a real percentage (e.g. fold
        N of M). Freezes the animated estimate at this exact value until
        the next start()/stop(). Starts the sweep animation if not
        already running. The DRAWN bar still glides to this value over a
        fraction of a second rather than jumping -- see _current_pct()."""
        self._manual_pct = max(0.0, min(100.0, pct))
        if label is not None:
            self._label = label
        if not self._running:
            self._running = True
            self._last_frame_t = time.monotonic()
            self._sweep_phase = 0.0
            self._tick()
        else:
            self._draw()

    def stop(self):
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._manual_pct = None
        self._displayed_pct = 0.0
        self._label = "LOADING...."
        self._draw()

    def _current_pct(self) -> float:
        now = time.monotonic()
        dt = max(0.0, now - self._last_frame_t)
        self._last_frame_t = now
        if self._manual_pct is not None:
            target = self._manual_pct
        else:
            elapsed = max(0.0, now - self._t0)
            target = self.ESTIMATE_CEILING_PCT * (1.0 - math.exp(-elapsed / self.ESTIMATE_TAU_SECONDS))
        # First-order lag toward `target` -- the actual smoothing step.
        # Framerate-independent (uses measured dt, not an assumed frame
        # duration), so it looks identical whether _draw() was triggered
        # by the normal 30fps tick or by an extra call (e.g. a resize, or
        # set_progress() firing mid-frame).
        alpha = 1.0 - math.exp(-dt / self.DISPLAY_EASE_TAU_SECONDS)
        self._displayed_pct += (target - self._displayed_pct) * alpha
        return self._displayed_pct

    def _tick(self):
        if not self._running:
            return
        self._sweep_phase = (self._sweep_phase + self.FRAME_MS / 1000.0 / self.SWEEP_SECONDS) % 1.0
        self._draw()
        self._after_id = self.after(self.FRAME_MS, self._tick)

    def _rounded_rect_points(self, x0, y0, x1, y1, r):
        r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        return [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]

    def _draw(self):
        self.delete("all")
        width = max(self.winfo_width(), 80)
        height = self._height
        pad = 3
        x0, y0, x1, y1 = pad, pad, width - pad, height - pad
        r = self.CORNER_RADIUS
        active = self._running

        # Outer glow halo behind the track -- same nested-shrinking-alpha
        # trick GlowCard/RingProgress use elsewhere, muted to almost
        # nothing when idle so a bank of a dozen of these on one tab
        # doesn't read as a wall of neon.
        if active:
            for i, alpha in ((3, 0.05), (2, 0.10), (1, 0.18)):
                self.create_polygon(
                    self._rounded_rect_points(x0 - i, y0 - i, x1 + i, y1 + i, r + i),
                    smooth=True, fill="", outline=_blend_hex(PANEL_2, NEON_CYAN, alpha), width=1.4,
                )

        track_color = _blend_hex(PANEL_2, NEON_CYAN, 0.06)
        border_color = _blend_hex(PANEL_2, NEON_CYAN, 0.85 if active else 0.35)
        self.create_polygon(self._rounded_rect_points(x0, y0, x1, y1, r), smooth=True,
                             fill=track_color, outline=border_color, width=1.4)

        pct = self._current_pct() if active else 0.0
        inner_x0, inner_y0, inner_x1, inner_y1 = x0 + 2, y0 + 2, x1 - 2, y1 - 2
        fill_w = max(0.0, (inner_x1 - inner_x0) * (pct / 100.0))
        if fill_w > 1:
            fx1 = inner_x0 + fill_w
            fill_r = min(r - 1, fill_w / 2, (inner_y1 - inner_y0) / 2)
            # Soft glow bleeding just past the fill's own edge, then the
            # solid glowing fill itself.
            for i, alpha in ((4, 0.10), (2, 0.22)):
                self.create_polygon(
                    self._rounded_rect_points(inner_x0 - i, inner_y0 - i, fx1 + i, inner_y1 + i, fill_r + i),
                    smooth=True, fill=_blend_hex(track_color, NEON_CYAN, alpha), outline="",
                )
            self.create_polygon(
                self._rounded_rect_points(inner_x0, inner_y0, fx1, inner_y1, fill_r),
                smooth=True, fill=_blend_hex(PANEL_2, NEON_CYAN, 0.75), outline="",
            )
            # A brighter highlight streak sliding smoothly through the
            # filled region -- the "make it look smooth" motion cue,
            # independent of the estimate's own (slow, easing) growth.
            streak_w = max(18.0, fill_w * 0.22)
            streak_center = inner_x0 + self._sweep_phase * (fill_w + streak_w) - streak_w / 2
            sx0 = max(inner_x0, streak_center - streak_w / 2)
            sx1 = min(fx1, streak_center + streak_w / 2)
            if sx1 > sx0:
                self.create_polygon(
                    self._rounded_rect_points(sx0, inner_y0, sx1, inner_y1, fill_r),
                    smooth=True, fill=_blend_hex(PANEL_2, METAL_BRIGHT, 0.55), outline="",
                )

        label_color = TEXT if active else TEXT_DIM
        self.create_text(x0 + 10, height / 2, text=self._label, fill=label_color,
                          font=_safe_font(8, "bold"), anchor="w")
        pct_text = f"{pct:.0f}%" if active else ""
        if pct_text:
            self.create_text(x1 - 10, height / 2, text=pct_text, fill=NEON_CYAN,
                              font=_safe_font(8, "bold"), anchor="e")


class GlowCard(Canvas):
    """A rounded panel with a soft neon glow border -- the "gradient
    border card" look from the neon dashboard reference mockups. Tkinter
    has no native box-shadow or rounded Frame, so this draws the panel
    itself: a smoothed rounded-rect polygon for the body, a crisp accent-
    colored edge, and an outward-fading halo built from the same nested/
    dimming-shapes trick NeuralProgress uses for its pulse glow.

    Usage: build widgets into `.body` (a plain Frame) exactly like any
    other container -- `GlowCard(parent, accent=NEON_CYAN).body` is a drop
    -in replacement for `Frame(parent, bg=PANEL_2)` wherever a panel
    currently gets a flat `highlightbackground=BORDER` outline.
    """

    CORNER_RADIUS = 14

    def __init__(self, parent, accent=None, glow=True, **kwargs):
        # accent's default is resolved HERE (call time), not bound into
        # the function signature at module-load time, so it always
        # reflects whichever theme is currently active -- a literal
        # `accent=ACCENT` default would freeze at the color ACCENT held
        # when this module was first imported and never follow a later
        # theme toggle.
        super().__init__(parent, bg=BG, highlightthickness=0, **kwargs)
        self.accent = accent if accent is not None else ACCENT
        self.glow = glow
        self.body = Frame(self, bg=PANEL_2)
        self._window_id = None
        self.bind("<Configure>", lambda e: self._redraw(e.width, e.height))

    def set_accent(self, accent: str):
        self.accent = accent
        self._redraw(self.winfo_width(), self.winfo_height())

    def _rounded_rect(self, x0, y0, x1, y1, r, **kwargs):
        r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
        points = [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
            x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _redraw(self, width, height):
        if width < 8 or height < 8:
            return
        self.delete("all")
        r = self.CORNER_RADIUS
        pad = 4  # room for the glow halo to breathe past the card's own edge

        if self.glow:
            for i, alpha in ((3, 0.05), (2, 0.10), (1, 0.20)):
                self._rounded_rect(
                    pad - i * 2, pad - i * 2, width - pad + i * 2, height - pad + i * 2,
                    r + i * 2, outline=_blend_hex(BG, self.accent, alpha), fill="", width=2,
                )

        self._rounded_rect(pad, pad, width - pad, height - pad, r, fill=PANEL_2, outline=self.accent, width=1)

        inner_w, inner_h = max(width - 2 * pad - 2, 1), max(height - 2 * pad - 2, 1)
        if self._window_id is None:
            self._window_id = self.create_window(
                pad + 1, pad + 1, anchor="nw", window=self.body, width=inner_w, height=inner_h,
            )
        else:
            self.coords(self._window_id, pad + 1, pad + 1)
            self.itemconfigure(self._window_id, width=inner_w, height=inner_h)


class RoundedCard(Frame):
    """Auto-sizing card with a rounded 1px border -- the desktop twin of the
    web app's .t58-card (theme.css: panel-2 fill, 1px --border, rounded).

    Unlike GlowCard (fixed height, glow halo), this grows with whatever is
    packed into `.body`, so it can back ordinary form/section cards. The
    border is painted on a Canvas that sits *behind* `.body`; `.body` is
    inset just far enough that its square corners stay inside the rounded
    outline. Redraws are coalesced with after_idle so resizing a window
    with dozens of cards doesn't thrash.
    """

    def __init__(self, parent, fill=None, border=None, radius=12, inset=4):
        try:
            outer_bg = str(parent.cget("bg"))
        except Exception:
            outer_bg = BG
        super().__init__(parent, bg=outer_bg, highlightthickness=0, bd=0)
        self._fill = fill if fill is not None else PANEL
        self._border = border if border is not None else BORDER
        self._radius = radius
        self._last_size = (0, 0)
        self._redraw_job = None
        self._canvas = Canvas(self, bg=outer_bg, highlightthickness=0, bd=0)
        self._canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.body = Frame(self, bg=self._fill, highlightthickness=0, bd=0)
        self.body.pack(fill="both", expand=True, padx=inset, pady=inset)
        self.bind("<Configure>", self._on_configure)

    def set_border(self, color: str):
        self._border = color
        self._schedule_redraw()

    def _on_configure(self, event):
        size = (event.width, event.height)
        if size == self._last_size:
            return
        self._last_size = size
        self._schedule_redraw()

    def _schedule_redraw(self):
        if self._redraw_job is not None:
            return
        try:
            self._redraw_job = self.after_idle(self._redraw)
        except Exception:
            self._redraw_job = None

    def _redraw(self):
        self._redraw_job = None
        try:
            w, h = self._last_size
            if w < 14 or h < 14:
                return
            c = self._canvas
            c.delete("all")
            r = max(0.0, min(float(self._radius), w / 2 - 1, h / 2 - 1))
            x0, y0, x1, y1 = 0.5, 0.5, w - 1.5, h - 1.5
            pts = [
                x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
                x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
                x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
            ]
            c.create_polygon(pts, smooth=True, splinesteps=16, fill=self._fill, outline=self._border, width=1)
        except Exception:
            pass  # widget destroyed mid-redraw (e.g. theme toggle rebuild) -- nothing to paint


def _card(parent, border=None, fill=None, radius=12):
    """Drop-in for `Frame(parent, bg=PANEL, highlightthickness=1,
    highlightbackground=...)` that renders as a rounded web-style card.
    Returns the inner body Frame; its pack/grid/place calls are forwarded to
    the outer card so callers can position it exactly like the old Frame."""
    card = RoundedCard(parent, fill=fill, border=border, radius=radius)
    body = card.body
    for name in ("pack", "grid", "place", "pack_forget", "grid_forget", "grid_remove", "place_forget"):
        setattr(body, name, getattr(card, name))
    body._t58_card = card
    return body


class RingProgress(Canvas):
    """A glowing circular percentage ring -- the donut readout used for
    "progress toward a target" across the neon dashboard mockups, in place
    of a flat horizontal progress bar. Call `.set(pct)` to update.
    """

    def __init__(self, parent, size=88, thickness=9, accent=None, track=None, **kwargs):
        # Same call-time-default reasoning as GlowCard above -- resolved
        # here rather than bound into the signature, so this keeps
        # following the active theme rather than freezing at import time.
        super().__init__(parent, width=size, height=size, bg=PANEL_2, highlightthickness=0, **kwargs)
        self.size = size
        self.thickness = thickness
        self.accent = accent if accent is not None else ACCENT
        self.track = track if track is not None else BORDER
        self._pct = 0.0
        self._draw()

    def set(self, pct: float, accent: str | None = None):
        self._pct = max(0.0, min(100.0, pct))
        if accent is not None:
            self.accent = accent
        self._draw()

    def _draw(self):
        self.delete("all")
        s, t = self.size, self.thickness
        pad = t / 2 + 5
        for radius_pad, alpha in ((6, 0.05), (3, 0.14)):
            self.create_oval(
                pad - radius_pad, pad - radius_pad, s - pad + radius_pad, s - pad + radius_pad,
                outline=_blend_hex(PANEL_2, self.accent, alpha), width=t + radius_pad,
            )
        self.create_oval(pad, pad, s - pad, s - pad, outline=self.track, width=t)
        extent = -359.9 * (self._pct / 100.0)  # 359.9, not 360 -- a full-circle arc draws nothing in Tk
        if abs(extent) > 0.5:
            self.create_arc(pad, pad, s - pad, s - pad, start=90, extent=extent, style="arc",
                             outline=self.accent, width=t)
        self.create_text(s / 2, s / 2, text=f"{self._pct:.0f}%", fill=self.accent,
                          font=_safe_font(max(int(s * 0.16), 9), "bold"))
