"""Education and Resources tabs."""
from __future__ import annotations

import webbrowser
from tkinter import END, Frame, Label, Text, ttk
from app.ui.common import _safe_font
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_3 = ""
BORDER = ""
BORDER_LIGHT = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
METAL_BRIGHT = ""
GREEN = ""
AMBER = ""
ACCENT_HOVER = ""
NEON_CYAN = ""
NEON_LIME = ""
NEON_VIOLET = ""
NEON_AMBER = ""
theme.bind(globals())


class LearnMixin:
    """Education and Resources tabs."""


    def _build_education_tab(self):
        """Quant/algo-trading CONCEPT lessons -- what a number or a tab
        actually means, as opposed to _build_manual_tab (which button to
        click) and _build_resources_tab (external trading-fundamentals
        links). Content lives in app.education.content as one shared
        source both this tab and the web app's /education route render
        from, so the two can never drift apart -- this method is a pure
        renderer, same Text-widget-with-tags approach as _build_manual_tab,
        with zero lesson copy of its own."""
        from app.education.content import list_sections

        f = self.tab_education
        self._page_header(
            f, "GUIDE", "\U0001F393 Education",
            "What the numbers and tabs in this app actually mean, and how to think quantitatively about "
            "strategy research -- not a generic trading course. If you only read one section, make it "
            "Validation Lab.",
        )

        wrap = Frame(f, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        wrap.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        text = Text(
            wrap, wrap="word", bg=PANEL, fg=TEXT, relief="flat", bd=0,
            highlightthickness=0, font=_safe_font(9), padx=22, pady=18, cursor="arrow",
            spacing1=1, spacing3=1,
        )
        scrollbar = ttk.Scrollbar(wrap, orient="vertical", command=text.yview, style="T58.Vertical.TScrollbar")
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            text.bind(seq, lambda e: self._generic_text_wheel(text, e))

        text.tag_configure("h1", font=_safe_font(17, "bold"), foreground=ACCENT_HOVER, spacing1=4, spacing3=10)
        text.tag_configure("h2", font=_safe_font(12, "bold"), foreground=NEON_CYAN, spacing1=20, spacing3=8)
        text.tag_configure("h3", font=_safe_font(10, "bold"), foreground=METAL_BRIGHT, spacing1=12, spacing3=4)
        text.tag_configure("subtitle", font=_safe_font(9, "italic"), foreground=TEXT_DIM, spacing3=10)
        text.tag_configure("body", font=_safe_font(9), foreground=TEXT_MUTED, spacing3=5, lmargin1=2, lmargin2=2)
        text.tag_configure("bullet", font=_safe_font(9), foreground=TEXT_MUTED, lmargin1=22, lmargin2=38, spacing3=4)
        text.tag_configure("numstep", font=_safe_font(9, "bold"), foreground=TEXT, lmargin1=4, lmargin2=26, spacing1=6, spacing3=2)
        text.tag_configure("example", font=_safe_font(9, "italic"), foreground=NEON_LIME, lmargin1=22, lmargin2=22, spacing1=6, spacing3=8)
        text.tag_configure("warn", font=_safe_font(9, "bold"), foreground=AMBER, lmargin1=4, lmargin2=20, spacing1=8, spacing3=8)
        text.tag_configure("tip", font=_safe_font(9), foreground=GREEN, lmargin1=4, lmargin2=20, spacing1=6, spacing3=8)
        text.tag_configure("see_also", font=_safe_font(8), foreground=TEXT_DIM, lmargin1=4, spacing1=2, spacing3=14)
        text.tag_configure("divider", font=_safe_font(4), foreground=BORDER, spacing3=14)

        def h1(t):
            text.insert(END, t + "\n", "h1")

        def h2(t):
            text.insert(END, t + "\n", "h2")

        def h3(t):
            text.insert(END, t + "\n", "h3")

        def subtitle(t):
            text.insert(END, t + "\n", "subtitle")

        def body(t):
            text.insert(END, t + "\n", "body")

        def bullet(t):
            text.insert(END, "•  " + t + "\n", "bullet")

        def numstep(n, t):
            text.insert(END, f"{n}.  " + t + "\n", "numstep")

        def example_line(t):
            text.insert(END, "\u201c" + t + "\u201d\n", "example")

        def warn(t):
            text.insert(END, "⚠  " + t + "\n", "warn")

        def tip(t):
            text.insert(END, "✓  " + t + "\n", "tip")

        def see_also(items):
            text.insert(END, "See also: " + " · ".join(items) + "\n", "see_also")

        def rule():
            text.insert(END, "―" * 90 + "\n", "divider")

        h1("\U0001F393 EDUCATION")
        body(
            "What the numbers and tabs in this app actually mean, and how to think quantitatively about "
            "strategy research -- not a generic trading course. If you only read one section, make it "
            "Validation Lab."
        )
        rule()

        for section in list_sections():
            h1(f"{section.icon}  {section.title.upper()}")
            subtitle(section.subtitle)
            for lesson in section.lessons:
                h2(lesson.title)
                for blk in lesson.blocks:
                    if blk.kind == "p":
                        body(blk.text)
                    elif blk.kind == "bullets":
                        for item in blk.items:
                            bullet(item)
                    elif blk.kind == "steps":
                        for i, item in enumerate(blk.items, start=1):
                            numstep(i, item)
                    elif blk.kind == "tip":
                        tip(blk.text)
                    elif blk.kind == "warn":
                        warn(blk.text)
                    elif blk.kind == "example":
                        example_line(blk.text)
                if lesson.see_also:
                    see_also(lesson.see_also)
            rule()

        text.config(state="disabled")

    def _build_resources_tab(self):
        """Desktop port of the web app's /resources page (app/web/templates/
        resources.html) -- purely static educational content (no engine, no
        upload), so this just mirrors that page's copy and links using the
        same section/button widgets every other tab uses."""
        f = self._scrollable(self.tab_resources)
        self._page_header(
            f, "GUIDE", "Resources",
            "A running start on trading fundamentals before you backtest your first strategy -- "
            "no engine, no upload, just reading.",
        )

        guide_section = self._section(
            f, "\u2605 T58 guide -- 30-Day Trading Quickstart Guide",
            "From zero to sim trading in 30 days. Built around T58's own MS-LSD framework -- the "
            "same market-reading model behind this app's Manual Strategy Builder's advanced "
            "conditions (Break of Structure, Liquidity Sweep, Fair Value Gap, Order Block, and "
            "the rest).",
            emphasize=True,
        )

        mslsd_row = Frame(guide_section, bg=PANEL)
        mslsd_row.pack(anchor="w", padx=18, pady=(0, 10))
        for letter, label, color in (
            ("M", "Market Structure", NEON_CYAN),
            ("L", "Liquidity", NEON_LIME),
            ("S", "Supply & Demand", NEON_AMBER),
            ("D", "Entry Models", NEON_VIOLET),
        ):
            chip = Frame(mslsd_row, bg=PANEL_3, highlightthickness=1, highlightbackground=BORDER_LIGHT)
            chip.pack(side="left", padx=(0, 8), pady=(0, 4))
            Label(chip, text=f" {letter} ", bg=color, fg="#04120e", font=_safe_font(8, "bold")).pack(
                side="left", padx=(6, 6), pady=5,
            )
            Label(chip, text=label, bg=PANEL_3, fg=TEXT, font=_safe_font(8)).pack(side="left", padx=(0, 10))

        week_row = Frame(guide_section, bg=PANEL)
        week_row.pack(fill="x", padx=18, pady=(0, 10))
        for num, label in (
            ("Week 1", "Setup & Market Structure"),
            ("Week 2", "Liquidity"),
            ("Week 3", "Supply & Demand"),
            ("Week 4", "Entry Models & Sim Trading"),
        ):
            cell = Frame(week_row, bg=PANEL_3, highlightthickness=1, highlightbackground=BORDER)
            cell.pack(side="left", fill="both", expand=True, padx=(0, 8))
            Label(cell, text=num.upper(), bg=PANEL_3, fg=TEXT_DIM, font=_safe_font(7, "bold")).pack(
                anchor="w", padx=10, pady=(8, 2),
            )
            Label(
                cell, text=label, bg=PANEL_3, fg=TEXT, font=_safe_font(9, "bold"), wraplength=150, justify="left",
            ).pack(anchor="w", padx=10, pady=(0, 8))

        guide_url = "https://docs.google.com/document/d/17tdRY_tzpHOdw8_a1EpAgOmradTbu02Dlo6BOAnK0Wg/edit?usp=drivesdk"
        guide_btn_row = Frame(guide_section, bg=PANEL)
        guide_btn_row.pack(anchor="w", padx=18, pady=(0, 4))
        self._button(guide_btn_row, "OPEN THE GUIDE \u2192", lambda: webbrowser.open(guide_url), primary=True).pack(
            side="left",
        )
        Label(
            guide_section, text="Opens in Google Docs, in your default browser.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        ).pack(anchor="w", padx=18, pady=(0, 14))

        mslsd_section = self._section(
            f, "\u2605 T58 strategy -- MS-LSD Strategy (Market Structure + Liquidity + Supply & Demand)",
            "\"Never predict the markets. Instead, participate in probability.\" Price moves because "
            "of liquidity and order flow through banks, institutions, and news -- MS-LSD is the "
            "framework for reading that flow, and it's the same logic wired into this app's Manual "
            "Strategy Builder's advanced conditions.",
            emphasize=True,
        )

        steps_row = Frame(mslsd_section, bg=PANEL)
        steps_row.pack(fill="x", padx=18, pady=(0, 10))
        for num, label in (
            ("STEP 1", "HTF directional bias"),
            ("STEP 2", "Key liquidity"),
            ("STEP 3", "Area of interest (S&D zone below liquidity)"),
            ("STEP 4", "Standard confirmation entry (LTF ChoCh)"),
        ):
            cell = Frame(steps_row, bg=PANEL_3, highlightthickness=1, highlightbackground=BORDER)
            cell.pack(side="left", fill="both", expand=True, padx=(0, 8))
            Label(cell, text=num, bg=PANEL_3, fg=NEON_CYAN, font=_safe_font(7, "bold")).pack(
                anchor="w", padx=10, pady=(8, 2),
            )
            Label(
                cell, text=label, bg=PANEL_3, fg=TEXT, font=_safe_font(9, "bold"), wraplength=150, justify="left",
            ).pack(anchor="w", padx=10, pady=(0, 8))

        Label(
            mslsd_section, text="Timeframe alignment -- Bias \u2192 Structure \u2192 Entry",
            bg=PANEL, fg=TEXT, font=_safe_font(10, "bold"),
        ).pack(anchor="w", padx=18, pady=(6, 2))
        Label(
            mslsd_section,
            text="Daily \u2192 1hr \u2192 5m, or 4hr \u2192 15m \u2192 1m. Always start on the higher "
                 "timeframe and step down.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=880, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 4))

        Label(
            mslsd_section,
            text=(
                "The full write-up -- how to read market structure, the three types of liquidity, "
                "what makes a valid order block, and both entry models in detail -- lives in T58's "
                "strategy doc rather than pasted here, so it always reflects the latest version."
            ),
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=880, justify="left",
        ).pack(anchor="w", padx=18, pady=(2, 10))

        mslsd_doc_url = "https://docs.google.com/document/d/14jubETVbumncdLJrTrw2ke-NxFPXkT34qzH4_Q1BguA/edit?usp=drivesdk"
        self._button(
            mslsd_section, "OPEN THE MS-LSD STRATEGY DOC \u2192", lambda: webbrowser.open(mslsd_doc_url), primary=True,
        ).pack(anchor="w", padx=18, pady=(0, 4))

        Label(
            mslsd_section, text="This is T58's own framework -- the same one this app's engine tests "
            "strategies against. Opens in Google Docs, in your default browser.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(8),
        ).pack(anchor="w", padx=18, pady=(0, 14))

        discord_section = self._section(f, "T58 Trading Discord")
        discord_row = Frame(discord_section, bg=PANEL)
        discord_row.pack(fill="x", padx=18, pady=(0, 14))
        Label(
            discord_row,
            text="Join the T58 Trading community to discuss MS-LSD setups, prop-firm evals, and "
            "get support directly from T58.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=620, justify="left",
        ).pack(side="left", padx=(0, 14))
        discord_url = "https://whop.com/t58-trading/t58-trading-discord/"
        self._button(
            discord_row, "JOIN THE DISCORD \u2192", lambda: webbrowser.open(discord_url), primary=True,
        ).pack(side="left")

        more_section = self._section(f, "More free places to build your foundation")
        for label, url, badge, desc in (
            (
                "BabyPips \u2014 School of Pipsology", "https://www.babypips.com/learn/forex", "COURSE",
                "The standard free, self-paced forex course -- starts from absolute basics and builds up.",
            ),
            (
                "Investopedia \u2014 Trading", "https://www.investopedia.com/trading-4689660", "REFERENCE",
                "A reliable reference for looking up any term, order type, or concept as it comes up.",
            ),
            (
                "CME Group Education", "https://www.cmegroup.com/education.html", "EDUCATION",
                "Free courses on futures, options, and risk management from a major exchange operator.",
            ),
        ):
            card = Frame(more_section, bg=PANEL_3, highlightthickness=1, highlightbackground=BORDER, cursor="hand2")
            card.pack(fill="x", padx=18, pady=(0, 8))
            badge_lbl = Label(card, text=badge, bg=PANEL_3, fg=TEXT_DIM, font=_safe_font(7, "bold"), cursor="hand2")
            badge_lbl.pack(anchor="w", padx=14, pady=(10, 2))
            title_lbl = Label(card, text=label, bg=PANEL_3, fg=METAL_BRIGHT, font=_safe_font(10, "bold"), cursor="hand2")
            title_lbl.pack(anchor="w", padx=14)
            desc_lbl = Label(
                card, text=desc, bg=PANEL_3, fg=TEXT_MUTED, font=_safe_font(8), wraplength=820,
                justify="left", cursor="hand2",
            )
            desc_lbl.pack(anchor="w", padx=14, pady=(2, 10))
            for widget in (card, badge_lbl, title_lbl, desc_lbl):
                widget.bind("<Button-1>", lambda _e, u=url: webbrowser.open(u))

        Label(
            f,
            text="Educational content only, not financial advice -- the same disclaimer that applies to "
                 "every simulated result elsewhere in this app applies here too. External links open in "
                 "your default browser to third-party sites this app doesn't control.",
            bg=BG, fg=TEXT_DIM, font=_safe_font(8), wraplength=900, justify="left",
        ).pack(anchor="w", padx=24, pady=(4, 24))
