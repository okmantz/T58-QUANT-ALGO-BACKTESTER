"""Strategy library list, metadata, rename/delete and detail viewers."""
from __future__ import annotations

import json
import os
import threading
from app.strategy.dna import extract_dna, find_common_patterns
from app.strategy.library import (
    STRATEGY_TYPES,
    StrategyAlreadyExists,
    delete_many,
    get_strategy_library_dir,
    list_all_markets,
    list_all_tags,
    list_misplaced_files,
    list_saved_strategies,
    load_strategy_text,
    rename_saved_strategy,
    save_strategy_bytes,
    save_strategy_metadata,
    save_strategy_path,
    set_strategy_status,
    set_strategy_tags,
)
from pathlib import Path
from tkinter import END, Frame, Label, Text, Toplevel, filedialog, messagebox, simpledialog, ttk
from app.ui.common import STATUS_KEY_TO_LABEL, STATUS_LABEL_TO_KEY, _safe_font
from app.ui.platform_win import _apply_dark_titlebar
from app.ui.theme import MONO
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL_3 = ""
TEXT = ""
TEXT_DIM = ""
GREEN = ""
RED = ""
BLUE = ""
AMBER = ""
theme.bind(globals())


class StrategyLibraryMixin:
    """Strategy library list, metadata, rename/delete and detail viewers."""


    def _current_session(self) -> tuple[str, str]:
        start = self.s_session_start.get_str().strip() or "08:30"
        end = self.s_session_end.get_str().strip() or "15:00"
        return start, end

    def _set_strategy_mode(self, mode: str):
        self.strategy_mode.set(mode)

        display = {
            "manual": "MANUAL STRATEGY BUILDER",
            "python": "PYTHON STRATEGY",
            "pinescript": "PINESCRIPT STRATEGY",
            "mql5": "MQL5 STRATEGY",
        }.get(mode, mode.upper())

        self.strategy_mode_label.config(
            text=f"SELECTED  •  {display}"
        )
        if hasattr(self, "strategy_library_listbox"):
            self._refresh_strategy_library()

    def _browse_strategy_file(self):
        mode = self.strategy_mode.get()
        ext = {
            "python": "*.py",
            "pinescript": "*.pine",
            "mql5": "*.mq5",
        }.get(mode, "*.*")

        path = filedialog.askopenfilename(
            filetypes=[("Strategy file", ext)]
        )

        if not path:
            return

        if mode in STRATEGY_TYPES:
            stored_path = self._save_to_library_with_overwrite_prompt(
                lambda overwrite: save_strategy_path(path, mode, overwrite=overwrite),
                fallback_path=Path(path),
            )
            if stored_path is None:
                return  # user cancelled the overwrite/rename prompt
            self.strategy_py_path = str(stored_path)
            self._active_library_strategy = (mode, stored_path.name)
            self.strategy_file_status.config(
                text=f"Selected: {stored_path.name}  (saved to library)",
                fg=GREEN,
            )
            self._refresh_strategy_library()
        else:
            self.strategy_py_path = path
            self._active_library_strategy = None
            self.strategy_file_status.config(
                text=f"Selected: {os.path.basename(path)}",
                fg=GREEN,
            )

    def _save_to_library_with_overwrite_prompt(self, save_fn, fallback_path: Path):
        """Try `save_fn(overwrite=False)`. On a name collision, ask the user
        whether to overwrite the existing saved strategy or save this one as
        a new, separately-named copy; on OSError, fall back to using the
        file in place (still usable for this run, just not library-backed).
        Returns the stored Path, or None if the user cancelled."""
        try:
            return save_fn(overwrite=False)
        except StrategyAlreadyExists as exc:
            choice = messagebox.askyesnocancel(
                "Strategy already saved",
                f"'{exc.filename}' is already in the strategy library.\n\n"
                "Yes = overwrite the saved copy with this one\n"
                "No = save this as a new, separately-named copy\n"
                "Cancel = don't save to the library",
            )
            if choice is None:
                return None
            if choice:
                return save_fn(overwrite=True)
            new_name = simpledialog.askstring(
                "Save as new copy",
                "New filename for this copy:",
                initialvalue=exc.filename,
            )
            if not new_name:
                return None
            try:
                return self._save_strategy_copy_as(fallback_path, exc.strategy_type, new_name)
            except StrategyAlreadyExists:
                messagebox.showerror(
                    "Name taken", f"'{new_name}' is also already saved. Try again with a different name."
                )
                return None
        except OSError as exc:
            messagebox.showwarning(
                "Saved locally only",
                f"Selected the file, but couldn't copy it into the strategy "
                f"library ({exc}). It will still work for this run.",
            )
            return fallback_path

    def _save_strategy_copy_as(self, source_path: Path, strategy_type: str, new_name: str) -> Path:
        """Copy source_path's bytes into the library under a caller-chosen
        filename (save_strategy_path always keeps the source's own
        filename, so a rename-on-save needs this instead)."""
        content = Path(source_path).read_bytes()
        return save_strategy_bytes(content, new_name, strategy_type, overwrite=False)

    def _refresh_strategy_library(self):
        # Same cross-thread hazard as _refresh_dashboard (see its
        # docstring) -- this is called both from UI event handlers (safe)
        # and directly from several pipelines' background worker threads
        # right after they finish (not safe). Self-guard the same way.
        if threading.current_thread() is not threading.main_thread():
            try:
                self.root.after(0, self._refresh_strategy_library)
            except Exception:
                pass
            return
        mode = self.strategy_mode.get()
        self.strategy_library_listbox.delete(0, END)
        self._strategy_library_items = []

        if mode not in STRATEGY_TYPES:
            self.strategy_library_status.config(
                text="Manual strategies aren't files — nothing to show here.",
                fg=TEXT_DIM,
            )
            return

        # Keep the market/tag filter dropdowns' options current -- new
        # values show up here as soon as they're saved on any strategy.
        markets = list_all_markets(mode)
        self.strategy_filter_market.combo["values"] = ["All markets", *markets]
        if self.strategy_filter_market.get_str() not in ("All markets", *markets):
            self.strategy_filter_market.var.set("All markets")
        tags = list_all_tags(mode)
        self.strategy_filter_tag.combo["values"] = ["All tags", *tags]
        if self.strategy_filter_tag.get_str() not in ("All tags", *tags):
            self.strategy_filter_tag.var.set("All tags")

        query = self.strategy_search.get_str().strip() if hasattr(self, "strategy_search") else ""
        market_filter = self.strategy_filter_market.get_str()
        tag_filter = self.strategy_filter_tag.get_str()
        status_filter = self.strategy_filter_status.get_str()

        self._strategy_library_items = list_saved_strategies(
            mode,
            query=query,
            market=None if market_filter in ("", "All markets") else market_filter,
            tag=None if tag_filter in ("", "All tags") else tag_filter,
            status=None if status_filter in ("", "All statuses") else STATUS_LABEL_TO_KEY.get(status_filter, status_filter),
        )
        for item in self._strategy_library_items:
            kb = item.size_bytes / 1024
            desc = item.metadata.get("description", "")
            suffix = f"  —  {desc}" if desc else ""
            lookahead = item.metadata.get("lookahead")
            if lookahead is None:
                badge = ""
            elif lookahead.get("clean"):
                badge = "  ✓clean"
            else:
                badge = "  ⚠LOOKAHEAD"
            # Pipeline-progress stage badge (Create/Test/Optimize/Validate/
            # Champion Check/Ready -- see app.strategy.library.
            # compute_pipeline_progress, mirrored on the web app's Strategy
            # Library page as a 6-pill bar). A plain Listbox can't render
            # pills, so this shows the furthest stage actually reached
            # plus the verdict when a Champion Check came back anything
            # other than READY, e.g. "Stage: Optimize" or
            # "Stage: Champion Check (NOT READY)".
            pp = item.pipeline_progress
            current_title = next((s["title"] for s in pp["stages"] if s["key"] == pp["current_stage"]), "Create")
            stage_note = f"  ·  Stage: {current_title}"
            if pp["verdict"] and pp["current_stage"] in ("champion_check", "ready") and pp["verdict"] != "READY":
                stage_note += f" ({pp['verdict']})"
            idx = self.strategy_library_listbox.size()
            self.strategy_library_listbox.insert(
                END, f"  [{item.status_display}]  {item.name}  ({kb:.1f} KB){badge}{stage_note}{suffix}"
            )
            self.strategy_library_listbox.itemconfig(idx, fg=self._status_color(item.status))

        d = get_strategy_library_dir(mode)
        total = len(list_saved_strategies(mode))
        filtered = query or market_filter not in ("", "All markets") or \
            tag_filter not in ("", "All tags") or status_filter not in ("", "All statuses")
        misplaced = list_misplaced_files(mode)
        misplaced_note = ""
        if misplaced:
            shown_names = ", ".join(misplaced[:3]) + (f", +{len(misplaced) - 3} more" if len(misplaced) > 3 else "")
            misplaced_note = (
                f"\n\u26a0 {len(misplaced)} file(s) in this folder don't match the "
                f"{mode} extension and won't show up here: {shown_names}. If one of "
                f"these is a strategy you dropped in, it's probably in the wrong "
                f"language's subfolder -- move it into strategies/"
                f"{{python|pinescript|mql5}}/ as appropriate, then REFRESH LIBRARY."
            )
        if self._strategy_library_items:
            shown = (
                f"{len(self._strategy_library_items)} of {total} saved {mode} strategy(ies)"
                if filtered else f"{total} saved {mode} strategy(ies)"
            )
            self.strategy_library_status.config(text=f"{shown}  •  {d}{misplaced_note}", fg=TEXT_DIM)
        elif total and filtered:
            self.strategy_library_status.config(
                text=f"No saved {mode} strategies match the current search/filters.{misplaced_note}", fg=TEXT_DIM,
            )
        else:
            self.strategy_library_status.config(
                text=f"No saved {mode} strategies yet. Import one above, or drop a file "
                f"directly in {d} and press REFRESH LIBRARY.{misplaced_note}",
                fg=TEXT_DIM,
            )
        self._clear_strategy_metadata_panel()
        if hasattr(self, "_batch_checklist_inner"):
            self._refresh_batch_checklist()

    def _status_color(self, status: str) -> str:
        """Rough color cue for a strategy's lifecycle status, so it reads
        at a glance in the library listbox instead of only via the text
        label -- resolved at call time (not module load) since GREEN/RED/
        AMBER/etc. are theme globals that apply_theme() can overwrite."""
        return {
            "draft": TEXT_DIM,
            "tested_failed": RED,
            "tested_passed": AMBER,
            "validated": BLUE,
            "ready_for_demo": GREEN,
            "ready_for_live": GREEN,
        }.get(status, TEXT_DIM)

    def _selected_library_item(self):
        """First selected item, for actions that only make sense on one at
        a time (load, rename, edit info)."""
        items = self._selected_library_items()
        return items[0] if items else None

    def _selected_library_items(self):
        """All currently highlighted items, for bulk actions (delete,
        export). The listbox allows multi-select (Ctrl/Cmd/Shift-click)."""
        mode = self.strategy_mode.get()
        if mode not in STRATEGY_TYPES:
            return []
        sel = self.strategy_library_listbox.curselection()
        return [self._strategy_library_items[i] for i in sel]

    def _clear_strategy_metadata_panel(self):
        if hasattr(self, "strategy_meta_description"):
            self.strategy_meta_description.var.set("")
            self.strategy_meta_market.var.set("")
            self.strategy_meta_tags.var.set("")
            self.strategy_meta_status.var.set(STATUS_KEY_TO_LABEL["draft"])
            self.strategy_meta_last_run.config(text="")

    def _on_library_selection_changed(self):
        items = self._selected_library_items()
        if len(items) != 1:
            self._clear_strategy_metadata_panel()
            if len(items) > 1:
                self.strategy_meta_last_run.config(
                    text=f"{len(items)} strategies selected — bulk delete/export only "
                    "(load/rename/info apply to a single selection)."
                )
            return
        item = items[0]
        self.strategy_meta_description.var.set(item.metadata.get("description", ""))
        self.strategy_meta_market.var.set(item.metadata.get("market", ""))
        self.strategy_meta_tags.var.set(", ".join(item.tags))
        self.strategy_meta_status.var.set(item.status_display)

        lines = []
        last_run = item.metadata.get("last_run")
        if last_run:
            lines.append("Last run — " + "  •  ".join(f"{k}: {v}" for k, v in last_run.items()))
        lookahead = item.metadata.get("lookahead")
        if lookahead:
            lines.append(f"Lookahead — {'clean' if lookahead.get('clean') else 'FAILED'}: "
                         f"{lookahead.get('summary', '')}")
        last_search = item.metadata.get("last_search")
        if last_search:
            lines.append("Last search — " + "  •  ".join(f"{k}: {v}" for k, v in last_search.items()))
        self.strategy_meta_last_run.config(
            text="\n".join(lines) if lines else "No backtest, lookahead check, or search recorded yet."
        )

    def _save_selected_library_metadata(self):
        mode = self.strategy_mode.get()
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a single saved strategy from the list first.")
            return
        save_strategy_metadata(mode, item.name, {
            "description": self.strategy_meta_description.get_str().strip(),
            "market": self.strategy_meta_market.get_str().strip(),
        })
        tags_raw = self.strategy_meta_tags.get_str().strip()
        set_strategy_tags(mode, item.name, [t.strip() for t in tags_raw.split(",")] if tags_raw else [])
        status_label_selected = self.strategy_meta_status.get_str()
        set_strategy_status(mode, item.name, STATUS_LABEL_TO_KEY.get(status_label_selected, status_label_selected))
        self._refresh_strategy_library()

    def _load_library_item_into_active_slot(self, item):
        """Loads a StoredStrategy (from the main library list) into the
        single 'active strategy' slot at the top of Step 01 Strategy
        Configuration -- the slot Full Pipeline, Refinement,
        Walk-Forward Opt, Sensitivity, and Multi-Objective
        all read via _build_strategy(). Switches the STRATEGY SOURCE mode
        too, so loading a PineScript item while Python is the active tab
        still works."""
        if self.strategy_mode.get() != item.strategy_type:
            self._set_strategy_mode(item.strategy_type)
        self.strategy_py_path = str(item.path)
        self._active_library_strategy = (item.strategy_type, item.name)
        self.strategy_file_status.config(
            text=f"Loaded from library: {item.name}  [{item.status_display}]",
            fg=GREEN,
        )

    def _load_selected_library_strategy(self):
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a saved strategy from the list first.")
            return
        self._load_library_item_into_active_slot(item)

    def _rename_selected_library_strategy(self):
        mode = self.strategy_mode.get()
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a single saved strategy from the list first.")
            return
        new_name = simpledialog.askstring(
            "Rename saved strategy", "New filename:", initialvalue=item.name
        )
        if not new_name or new_name == item.name:
            return
        try:
            new_path = rename_saved_strategy(mode, item.name, new_name, overwrite=False)
        except StrategyAlreadyExists:
            if not messagebox.askyesno(
                "Name already taken",
                f"'{new_name}' is already a saved {mode} strategy. Overwrite it?",
            ):
                return
            new_path = rename_saved_strategy(mode, item.name, new_name, overwrite=True)
        if self.strategy_py_path == str(item.path):
            self.strategy_py_path = str(new_path)
            self._active_library_strategy = (mode, new_path.name)
        self._refresh_strategy_library()

    def _delete_selected_library_strategy(self):
        mode = self.strategy_mode.get()
        items = self._selected_library_items()
        if not items:
            messagebox.showinfo("No selection", "Select one or more saved strategies from the list first.")
            return
        names = ", ".join(i.name for i in items)
        prompt = (
            f"Permanently delete '{items[0].name}' from the strategy library? This cannot be undone."
            if len(items) == 1 else
            f"Permanently delete {len(items)} strategies from the library? This cannot be undone.\n\n{names}"
        )
        if not messagebox.askyesno("Delete saved strategy(ies)", prompt):
            return
        deleted, failed = delete_many((mode, i.name) for i in items)
        if self._active_library_strategy and self._active_library_strategy[0] == mode and \
                any(self._active_library_strategy[1] == i.name for i in items):
            self.strategy_py_path = None
            self._active_library_strategy = None
            self.strategy_file_status.config(
                text="Only needed for Python / PineScript / MQL5 modes.",
                fg=TEXT_DIM,
            )
        if failed:
            messagebox.showwarning("Some deletions failed", "\n".join(failed))
        self._refresh_strategy_library()

    def _show_text_viewer(self, title: str, text: str):
        """Read-only popup showing raw text (strategy source, or a
        manual strategy's built config as JSON) with copy-to-clipboard --
        used by VIEW CODE / CONFIG."""
        win = Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.geometry("860x640")
        try:
            _apply_dark_titlebar(win)
        except Exception:
            pass  # cosmetic only -- see _apply_dark_titlebar's own docstring

        Label(
            win, text=title, bg=BG, fg=TEXT, font=_safe_font(11, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        text_frame = Frame(win, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        txt = Text(
            text_frame, wrap="none", bg=PANEL_3, fg=TEXT, insertbackground=TEXT,
            font=(MONO, 10), relief="flat", bd=0,
        )
        vs = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview, style="T58.Vertical.TScrollbar")
        hs = ttk.Scrollbar(text_frame, orient="horizontal", command=txt.xview)
        txt.config(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self._bind_isolated_wheel(txt)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        txt.insert("1.0", text)
        txt.config(state="disabled")

        btn_row = Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=14, pady=(0, 12))

        def _copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

        self._button(btn_row, "COPY TO CLIPBOARD", _copy, primary=True).pack(side="left")
        self._button(btn_row, "CLOSE", win.destroy).pack(side="left", padx=8)

    def _view_selected_strategy_detail(self) -> None:
        """The Strategy Library equivalent of the Evolution Lab
        leaderboard's VIEW DETAILS / double-click popup: everything this
        saved strategy's own metadata sidecar already knows (last
        backtest, lookahead check, last search result) plus its full
        code/config in one read-only window, for ANY saved strategy
        (Python, PineScript, MQL5, or Manual) -- not just Evolution Lab
        leaders. Metadata comes from the same .meta.json sidecar
        _on_library_selection_changed() already reads for the inline
        panel; this just also pulls in the source/config text so there's
        one place to see both without leaving the library list.
        """
        mode = self.strategy_mode.get()
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a saved strategy from the list first.")
            return

        def fmt_kv(d: dict) -> str:
            return "  •  ".join(f"{k}: {v}" for k, v in d.items()) if d else "n/a"

        lines = [
            f"Strategy: {item.name}",
            f"Type: {mode}",
            f"Status: {item.status_display}",
            f"Description: {item.metadata.get('description') or 'n/a'}",
            f"Market: {item.metadata.get('market') or 'n/a'}",
            f"Tags: {', '.join(item.tags) if item.tags else 'n/a'}",
            "",
            "-- Last backtest / batch / Full Pipeline result --",
        ]
        last_run = item.metadata.get("last_run")
        lines.append(f"  {fmt_kv(last_run)}" if last_run else "  Not tested yet.")
        lines.append("")
        lines.append("-- Lookahead check --")
        lookahead = item.metadata.get("lookahead")
        if lookahead:
            lines.append(f"  {'CLEAN' if lookahead.get('clean') else 'FAILED'}: {lookahead.get('summary', '')}")
        else:
            lines.append("  Not run yet.")
        lines.append("")
        lines.append("-- Last Search Lab / Optimize result --")
        last_search = item.metadata.get("last_search")
        lines.append(f"  {fmt_kv(last_search)}" if last_search else "  Not run yet.")
        lines.append("")
        lines.append("-- Code / config --")
        try:
            source_text = load_strategy_text(mode, item.name)
        except Exception as exc:
            source_text = f"(could not load source: {exc})"
        lines.append(source_text)

        self._show_text_viewer(f"Strategy detail -- {item.name}  ({mode})", "\n".join(lines))

    def _view_selected_strategy_code(self):
        mode = self.strategy_mode.get()
        if mode == "manual":
            # A saved manual strategy selected in the library (e.g. one
            # promoted from the Evolution Lab leaderboard) takes priority
            # over the live in-progress builder form -- otherwise this
            # always showed whatever happened to be typed into Step 2's
            # form right now, never the actually-selected library entry,
            # which meant a promoted manual strategy's config could never
            # be viewed here at all.
            item = self._selected_library_item()
            if item is not None:
                try:
                    text = load_strategy_text(mode, item.name)
                except Exception as exc:
                    messagebox.showerror("Could not load", str(exc))
                    return
                self._show_text_viewer(f"{item.name}  (manual)", text)
                return
            try:
                strategy = self._build_strategy()
            except Exception as exc:
                messagebox.showerror("Could not build strategy", str(exc))
                return
            text = json.dumps(strategy.config, indent=2)
            self._show_text_viewer(f"Manual strategy config -- {strategy.config.get('name', 'Manual Strategy')}", text)
            return
        if mode not in STRATEGY_TYPES:
            messagebox.showinfo("No strategy type selected", "Choose Python, PineScript, MQL5, or Manual above first.")
            return
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a saved strategy from the list first.")
            return
        try:
            text = load_strategy_text(mode, item.name)
        except Exception as exc:
            messagebox.showerror("Could not load", str(exc))
            return
        self._show_text_viewer(f"{item.name}  ({mode})", text)

    def _view_selected_strategy_dna(self):
        mode = self.strategy_mode.get()
        if mode not in STRATEGY_TYPES:
            messagebox.showinfo("No strategy type selected", "Choose Python, PineScript, MQL5, or Manual above first.")
            return
        item = self._selected_library_item()
        if item is None:
            messagebox.showinfo("No selection", "Select a saved strategy from the list first.")
            return
        try:
            text = load_strategy_text(mode, item.name)
        except Exception as exc:
            messagebox.showerror("Could not load", str(exc))
            return
        dna = extract_dna(mode, text)
        tree = dna.render_tree(item.name)
        matched_lines = "\n".join(
            f"  {tag}: {', '.join(terms)}" for tag, terms in sorted(dna.matched_terms.items())
        )
        body = tree + ("\n\nMatched on:\n" + matched_lines if matched_lines else "\n\nNo genes matched -- this strategy doesn't contain any of the keywords app.strategy.dna looks for.")
        self._show_text_viewer(f"Strategy DNA -- {item.name}", body)

    def _find_common_dna_patterns_library(self):
        """Walks every saved Python/PineScript/MQL5/Manual strategy,
        extracts its DNA, and defines 'top performer' as any strategy
        whose recorded last_run.passed_evaluation is True (the same
        signal the Dashboard and Strategy Library badges already use) --
        then runs app.strategy.dna.find_common_patterns to report which
        gene combinations are disproportionately common among the
        strategies that actually passed versus the ones that didn't."""
        entries = []
        for stype in STRATEGY_TYPES:
            for item in list_saved_strategies(stype):
                try:
                    text = load_strategy_text(stype, item.name)
                except Exception:
                    continue
                dna = extract_dna(stype, text)
                last_run = item.metadata.get("last_run") or {}
                is_top = bool(last_run.get("passed_evaluation"))
                entries.append((f"{item.name} ({stype})", dna, is_top))

        n_top = sum(1 for _, _, is_top in entries if is_top)
        if len(entries) < 3:
            messagebox.showinfo(
                "Not enough strategies",
                "Save and run at least a few strategies first (05 Run & Report, Full Pipeline, "
                "Quick Optimize, or a Batch Test all record a last_run result) so there's "
                "something to compare.",
            )
            return
        if n_top < 2:
            messagebox.showinfo(
                "Not enough passing strategies yet",
                f"{len(entries)} strategies are in the library, but only {n_top} have a recorded "
                "evaluation PASS -- run more of them (or run the ones you have) through 05 Run & "
                "Report, Full Pipeline, or a Batch Test first so there are at least 2 winners to "
                "compare against the rest.",
            )
            return

        patterns = find_common_patterns(entries, min_combo_size=2, max_combo_size=4, min_support_top=2, min_lift=1.2)
        n_rest = len(entries) - n_top
        header = (
            f"{len(entries)} strategies compared -- {n_top} passed their evaluation (top performers), "
            f"{n_rest} did not or haven't been run.\n\n"
        )
        if not patterns:
            body = header + "No gene combination was both common enough among the winners and rare enough among the rest to call a pattern -- try again once more strategies have been tested."
        else:
            body = header + "Patterns disproportionately common among the winners (highest lift first):\n\n" + \
                "\n".join(f"{i+1}. {p.render_line()}" for i, p in enumerate(patterns))
        self._show_text_viewer("Strategy DNA -- Common Patterns Across the Library", body)
