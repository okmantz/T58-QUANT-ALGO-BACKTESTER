# Research Library

Drop trading/quant research papers in here (`.pdf`, `.txt`, or `.md`) and
the optional local-Ollama AI assist will automatically draw on them when
it suggests parameter values during a GA search (Iterative Refinement,
Search Lab, Walk-Forward GA, and the Full Pipeline's built-in search all
share this).

## How to use it

1. Drop files directly in this folder (subfolders are fine too --
   `research/momentum/` and `research/mean_reversion/` both get scanned).
2. That's it. Nothing to run, configure, or re-index by hand -- the next
   time a GA search calls Ollama for a suggestion, it automatically
   re-scans this folder for anything new or changed since the last call.
3. Turn on AI Assist (Ollama) on the Full Pipeline tab (or wherever you're
   running a search) like normal. No separate "research mode" toggle --
   grounding from this folder is just part of every suggestion request
   once there's something in here to draw from.

## How it actually works (and its real limits)

This does **not** use embeddings, a vector database, or send anything to
the internet. When the AI assist is about to ask Ollama for suggestions,
it:

1. Extracts and splits every paper in this folder into paragraph-sized
   chunks (re-parsing only files that are new or have changed since last
   time -- see `app/ai/research_library.py`).
2. Scores every chunk against the current strategy's name and its tunable
   parameters' own labels (e.g. "EMA period", "session filter") using
   plain keyword overlap -- no AI, no network call, completely
   deterministic and free to run on every single generation.
3. Hands the two or three best-matching excerpts to Ollama as extra
   context alongside the strategy's current performance and (if this is
   a Full Pipeline run) what's already been learned about which parameter
   ranges are working, asking it to use them where they genuinely apply.

**What this means in practice:**
- It's only as good as plain keyword matching -- a paper about "trend
  persistence" won't automatically get pulled in for a strategy whose
  parameters are all named things like "fast_len"/"slow_len" with no
  shared vocabulary. Naming your strategy and its parameters with
  recognizable trading terms (as most strategies already are) is what
  makes this work well.
- It never edits, validates, or fact-checks a paper's claims, and it
  never asks Ollama to blindly follow one -- every suggested parameter
  set still goes through the exact same backtest -> prop-simulation ->
  Monte Carlo -> (for Full Pipeline) out-of-sample validation pipeline as
  any other candidate. A paper can inform a *suggestion*; it can never
  skip validation.
- A PDF that's scanned images with no real text layer won't extract
  anything useful (this app doesn't do OCR) -- if a paper doesn't seem to
  be influencing suggestions, that's the first thing to check.

## What's actually in this folder

Markdown documents (the app reads `.pdf`, `.txt` and `.md`; drop more in any time):

- `00_Start_Here/` -- the two guidance reports: A (the prop-eval math and
  what quants do) and B (evidence, math and a simple strategy spec).
- `01_Intraday_Momentum_and_Core_Signals/` through
  `04_Regime_Events_and_Data/` -- reading lists behind Report B's
  recommended strategy, with links to the primary papers.
- `05_Repo_Research_Library_Index/` -- index of the 44 PDFs this folder
  used to hold, kept so the reading lists can still refer to them.

The reading lists link out to the papers rather than bundling them; open
each link and confirm it before relying on it.
