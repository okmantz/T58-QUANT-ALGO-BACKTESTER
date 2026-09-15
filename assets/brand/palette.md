# T58 brand palette

Single source of truth for the T58 brand colors used across the logo/icon
assets in this folder, the web app (`app/web/static/theme.css`), and the
desktop app (`app/ui/main_window.py`'s `THEMES` dict). If you regenerate any
brand asset, match these exactly.

| Color      | Hex       | Role                                                              |
|------------|-----------|--------------------------------------------------------------------|
| Cyan       | `#00D4FF` | Second brand color (logo gradient, decorative NEON_CYAN dashboard accent, sidebar brand mark) |
| Purple     | `#7B3DFF` | Primary brand accent (`--violet` / `ACCENT` — buttons, focus states, primary highlights everywhere) |
| Deep Navy  | `#0B0F1A` | Background reference tone (app already uses a very close near-black navy ramp — `--bg` / `BG`) |
| Slate      | `#1A1F2E` | Panel/border reference tone (app already uses a very close ramp — `--panel-3` / `PANEL_3`) |
| White      | `#FFFFFF` | Light-theme background / on-accent text |

Fonts used in the marketing/logo artwork: **Orbitron** (headings), **Inter**
(body/text) — the app itself uses its existing system font stack; these are
for any future marketing collateral (landing pages, social banners) only.

## What actually changed vs. what stayed put

- `--violet` / `ACCENT` (the primary brand accent, already labeled as such
  in both codebases) was retuned to the exact purple above, with its
  hover/dim shades recomputed proportionally, in both the dark and light
  themes.
- `--neon-cyan` / `NEON_CYAN` (used for the sidebar brand mark and as one of
  the decorative per-tile dashboard accent colors) was retuned to the exact
  cyan above.
- The semantic pass/fail/warning colors (`--teal`/`GREEN`, `--coral`/`RED`,
  `--amber`/`AMBER`) were deliberately left alone — they carry meaning
  (win/loss/caution) independent of brand identity, and the brand kit does
  not specify replacements for them.
- The background/panel/border ramp was left alone — the app's existing
  near-black navy/slate values are already close enough to Deep Navy/Slate
  that a byte-for-byte swap would be a purely cosmetic, non-zero-risk
  change (dozens of call sites blend against these exact values for glow
  effects) for no visible difference.

## Assets in this repo

- `assets/logo/` — primary logo marks (square icon, horizontal logo,
  mark-only) at their working sizes.
- `assets/icons/` — generated app icons/favicons at each required size,
  downscaled from the master square icon.
- `assets/images/` — the GitHub README banner and report-header image.
- `assets/brand/` — supporting marketing assets (feature icons, status
  badge reference, social icons, motivational poster) used in the README.
