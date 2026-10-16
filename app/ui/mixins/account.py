"""Account, license, updates, profile and notification settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ui.widgets.run_context import RunContextPanel

import threading
from app.validation.integrity_check import run_integrity_check
from pathlib import Path
from tkinter import Frame, Label, Text, Toplevel, filedialog, messagebox
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui.widgets.basic import LabeledCheckbox, LabeledCombo, LabeledEntry
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
BG = ""
PANEL = ""
PANEL_3 = ""
TEXT = ""
TEXT_MUTED = ""
TEXT_DIM = ""
GREEN = ""
RED = ""
AMBER = ""
theme.bind(globals())


class AccountMixin:
    """Account, license, updates, profile and notification settings."""


    # -----------------------------------------------------------------------
    # Account -- desktop parity for the web app's Account section
    # (currently just notification settings; see app.web.notifications).
    # Self-contained settings page, not tied to any run.
    # -----------------------------------------------------------------------

    def _build_account_tab(self):
        f = self._scrollable(self.tab_account)

        self._page_header(
            f,
            "ACCOUNT / Account settings",
            "\u2699 Account",
            "Profile, Subscription, Security, and App info -- desktop parity with the web app's "
            "own /settings/account page (both read/write the exact same local JSON files under "
            "this computer's app data folder, so either build always shows the same values). "
            "Below that: Notification Settings, so a long-running job (Evolution Lab, Search Lab, "
            "Full Pipeline) can email you the moment it finishes instead of you keeping a progress "
            "window open and watching it. Everything on this tab is stored locally on this "
            "computer only -- this app has no cloud account or login of its own.",
        )

        from app.accounts.app_info import APP_VERSION
        from app.accounts.settings import load_account_settings
        from app.accounts.subscription import LICENSE_STATUS_CHOICES, load_subscription
        from app.web.notifications import load_notification_settings

        acct_saved = load_account_settings()
        sub_saved = load_subscription()
        saved = load_notification_settings()

        # ================= PROFILE =================
        profile_section = self._section(
            f, "Profile",
            "Your own name/username/email -- used to personalize exported reports; not a login, "
            "just local profile info. Password status is shown here; set or change it in the "
            "Security section below.",
            emphasize=True,
        )
        self.acct_display_name = LabeledEntry(profile_section, "Name", acct_saved.display_name, width=32)
        self.acct_username = LabeledEntry(profile_section, "Username", acct_saved.username, width=32)
        self.acct_email = LabeledEntry(profile_section, "Email", acct_saved.email, width=32)
        self.acct_company = LabeledEntry(profile_section, "Company (optional)", acct_saved.company, width=32)

        # UPGRADE (profile picture): stored exactly the way the web app's
        # own /settings/account/profile-picture route stores it -- the raw
        # file under data/config/profile_picture.<ext>, with its filename
        # on AccountSettings.profile_picture_filename -- so whichever
        # build (web or desktop) sets one, the other build's Account page
        # shows the same picture; there's no separate desktop-only copy
        # of this setting. Preview uses Pillow (already a dependency via
        # qrcode[pil]) so PNG/JPEG/GIF/WEBP all preview, not just the
        # PNG/GIF Tkinter's own PhotoImage supports natively -- degrades
        # to a plain filename label (still fully functional otherwise) if
        # Pillow can't load a given file rather than raising.
        pic_row = Frame(profile_section, bg=PANEL)
        pic_row.pack(fill="x", padx=18, pady=(10, 2))
        self.acct_pic_preview_label = Label(pic_row, bg=PANEL_3, width=8, height=4)
        self.acct_pic_preview_label.pack(side="left")
        pic_btn_col = Frame(pic_row, bg=PANEL)
        pic_btn_col.pack(side="left", padx=(12, 0), fill="y")
        Label(pic_btn_col, text="Profile picture", bg=PANEL, fg="#ccc", font=_safe_font(9)).pack(anchor="w")
        pic_btn_row = Frame(pic_btn_col, bg=PANEL)
        pic_btn_row.pack(anchor="w", pady=(4, 0))
        self._button(pic_btn_row, "UPLOAD", self._account_upload_profile_picture).pack(side="left")
        self._button(pic_btn_row, "REMOVE", self._account_remove_profile_picture).pack(side="left", padx=8)
        self._refresh_profile_picture_preview(acct_saved.profile_picture_filename)

        pw_row = Frame(profile_section, bg=PANEL)
        pw_row.pack(fill="x", padx=18, pady=(10, 2))
        Label(pw_row, text="Password", bg=PANEL, fg="#ccc", font=_safe_font(9)).pack(side="left")
        self.acct_password_status_label = Label(
            pw_row,
            text="\u25cf  Set" if acct_saved.has_password else "\u25cb  Not set",
            bg=PANEL, fg=(GREEN if acct_saved.has_password else TEXT_DIM), font=_safe_font(9, "bold"),
        )
        self.acct_password_status_label.pack(side="left", padx=(8, 0))
        Label(
            profile_section, text="Set or change your password in the Security section below.",
            bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8),
        ).pack(anchor="w", padx=18, pady=(0, 10))
        acct_btn_row = Frame(profile_section, bg=PANEL)
        acct_btn_row.pack(anchor="w", padx=18, pady=(4, 12))
        self._button(acct_btn_row, "SAVE PROFILE", self._save_account_profile, primary=True).pack(side="left")
        self.acct_profile_status = Label(profile_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8))
        self.acct_profile_status.pack(anchor="w", padx=18, pady=(0, 8))

        # ================= SUBSCRIPTION (legacy notes) =================
        subscription_section = self._section(
            f, "Subscription notes",
            "Free-form fields for your own reference -- not verified against anything. See the "
            "License section below for this device's REAL, server-verified activation status.",
            emphasize=True,
        )
        self.acct_sub_plan = LabeledEntry(subscription_section, "Plan", sub_saved.plan, width=32)
        self.acct_sub_license_key = LabeledEntry(subscription_section, "License key", sub_saved.license_key, width=32)
        self.acct_sub_status = LabeledCombo(
            subscription_section, "License status",
            [c.capitalize() for c in LICENSE_STATUS_CHOICES],
            sub_saved.license_status.capitalize() if sub_saved.license_status in LICENSE_STATUS_CHOICES else "Unset",
        )
        self.acct_sub_renewal = LabeledEntry(subscription_section, "Renewal / expiration date", sub_saved.renewal_date, width=32)
        sub_btn_row = Frame(subscription_section, bg=PANEL)
        sub_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(sub_btn_row, "SAVE SUBSCRIPTION INFO", self._save_subscription_info, primary=True).pack(side="left")
        self._button(sub_btn_row, "CHECK KEY FORMAT", self._check_license_key_format).pack(side="left", padx=8)
        self.acct_sub_status_label = Label(subscription_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=760, justify="left")
        self.acct_sub_status_label.pack(anchor="w", padx=18, pady=(6, 8))

        # ================= LICENSE (real, server-verified) =================
        # UPGRADE (licensing): this is the actual activation this
        # install is running under -- see app/licensing/client.py, the
        # one module this section reads from. Read-only except for
        # "Deactivate this device", which frees this license's device
        # slot on the license server so it can be activated on a
        # different machine (the "logout" the licensing spec asked for --
        # there is no separate desktop login session to log out of, per
        # the Security section below's own note, so this IS the
        # meaningful logout action for this app).
        license_section = self._section(
            f, "License",
            "This device's real, server-verified activation -- refreshed every time the app "
            "checks in with the license server (see app/licensing for how).",
            emphasize=True,
        )
        from app.licensing import client as _license_client
        lic_state = _license_client.load_state()
        lic_rows = Frame(license_section, bg=PANEL)
        lic_rows.pack(anchor="w", padx=18, pady=(4, 8), fill="x")
        for label, value in (
            ("Email", lic_state.email or "-"),
            ("Status", (lic_state.status or "not activated").upper()),
            ("Expires", lic_state.expires_at or "no fixed expiry"),
            ("Last verified", lic_state.last_validated_at or "never"),
        ):
            row = Frame(lic_rows, bg=PANEL)
            row.pack(anchor="w", fill="x", pady=1)
            Label(row, text=label, bg=PANEL, fg=TEXT_MUTED, font=_safe_font(9), width=14, anchor="w").pack(side="left")
            Label(row, text=value, bg=PANEL, fg=(GREEN if label == "Status" and lic_state.status == "active" else TEXT), font=_safe_font(9, "bold"), anchor="w").pack(side="left")
        lic_btn_row = Frame(license_section, bg=PANEL)
        lic_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(lic_btn_row, "DEACTIVATE THIS DEVICE", self._account_deactivate_license).pack(side="left")
        self.acct_license_status_label = Label(license_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=760, justify="left")
        self.acct_license_status_label.pack(anchor="w", padx=18, pady=(6, 8))


        # ================= SECURITY =================
        security_section = self._section(
            f, "Security",
            "An OPTIONAL local app lock -- useful mainly if you use Mobile Access on a shared "
            "Wi-Fi, since it's the same lock the web app's own Account Settings page sets. The "
            "desktop app itself has no login session of its own to lock or log out of.",
            emphasize=True,
        )
        self.acct_new_password = LabeledEntry(security_section, "New password (blank = remove lock)", "", secret=True, width=32)
        self.acct_confirm_password = LabeledEntry(security_section, "Confirm password", "", secret=True, width=32)
        sec_btn_row = Frame(security_section, bg=PANEL)
        sec_btn_row.pack(anchor="w", padx=18, pady=(4, 4))
        self._button(sec_btn_row, "CHANGE PASSWORD", self._account_change_password, primary=True).pack(side="left")
        self._button(sec_btn_row, "LOG OUT", self._account_log_out).pack(side="left", padx=8)
        self._button(sec_btn_row, "DELETE ACCOUNT", self._account_delete).pack(side="left", padx=8)
        self.acct_security_status = Label(security_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=760, justify="left")
        self.acct_security_status.pack(anchor="w", padx=18, pady=(6, 8))

        # ================= APP =================
        app_section = self._section(f, "App", emphasize=True)
        Label(app_section, text=f"App version: {APP_VERSION}", bg=PANEL, fg=TEXT, font=_safe_font(10, "bold")).pack(
            anchor="w", padx=18, pady=(2, 8),
        )
        self._button(app_section, "CHECK FOR UPDATES", self._account_check_for_updates).pack(anchor="w", padx=18)
        # v7: actually download a found update (checksum-verified) instead of
        # just telling the user to visit GitHub. Enabled by
        # _account_check_for_updates only when an update is available.
        self.acct_download_update_btn = self._button(app_section, "DOWNLOAD UPDATE", self._account_download_update)
        self.acct_download_update_btn.pack(anchor="w", padx=18, pady=(6, 0))
        self.acct_download_update_btn.configure(state="disabled")
        self._acct_pending_update = None
        self.acct_update_status = Label(app_section, text="", bg=PANEL, fg=TEXT_MUTED, font=_safe_font(8), wraplength=760, justify="left")
        self.acct_update_status.pack(anchor="w", padx=18, pady=(8, 8))

        dest_section = self._section(f, "Notification Settings -- where to send it", emphasize=True)
        self.acct_notify_email = LabeledEntry(dest_section, "Notification email", saved.notify_email, width=32)
        self.acct_email_enabled = LabeledCheckbox(dest_section, "Email notifications enabled", saved.email_enabled)

        smtp_section = self._section(
            f, "Outgoing mail server (SMTP)",
            "Any SMTP-capable mail account works, including a free Gmail/Outlook account -- "
            "for Gmail specifically, use an app password, not your normal login password. "
            "Nothing here is sent anywhere except directly to the SMTP host you enter.",
        )
        self.acct_smtp_host = LabeledEntry(smtp_section, "SMTP host", saved.smtp_host, width=32)
        self.acct_smtp_port = LabeledEntry(smtp_section, "SMTP port", saved.smtp_port)
        self.acct_smtp_username = LabeledEntry(smtp_section, "SMTP username", saved.smtp_username, width=32)
        self.acct_smtp_password = LabeledEntry(
            smtp_section, "SMTP password (blank = keep current)", "", secret=True, width=32,
        )
        self.acct_smtp_from = LabeledEntry(smtp_section, "\"From\" address", saved.smtp_from, width=32)

        # UPGRADE (Discord + Telegram): the backend (app.web.notifications)
        # already fully supports both -- real senders, is_usable checks --
        # this was just never surfaced on the desktop Account tab (web's
        # /settings/notifications page already has it). Same file, same
        # settings object as email/SMTP above; SAVE below writes all of it
        # together in one NotificationSettings.
        discord_section = self._section(
            f, "Discord",
            "Paste an Incoming Webhook URL from a Discord channel's Integrations settings -- "
            "a job-finished notification posts there. Leave blank to disable.",
        )
        self.acct_discord_webhook = LabeledEntry(
            discord_section, "Discord webhook URL", saved.discord_webhook_url, secret=True, width=48,
        )

        telegram_section = self._section(
            f, "Telegram",
            "Create a bot via @BotFather to get a bot token, then message the bot once and use "
            "https://api.telegram.org/bot<token>/getUpdates to find your chat ID. Both fields are "
            "required for Telegram notifications to send.",
        )
        self.acct_telegram_token = LabeledEntry(
            telegram_section, "Telegram bot token", saved.telegram_bot_token, secret=True, width=48,
        )
        self.acct_telegram_chat_id = LabeledEntry(telegram_section, "Telegram chat ID", saved.telegram_chat_id, width=32)

        notif_test_row = Frame(f, bg=BG)
        notif_test_row.pack(fill="x", padx=24, pady=(0, 4))
        self._button(notif_test_row, "TEST EMAIL", lambda: self._test_notification_channel("email")).pack(side="left")
        self._button(notif_test_row, "TEST DISCORD", lambda: self._test_notification_channel("discord")).pack(side="left", padx=8)
        self._button(notif_test_row, "TEST TELEGRAM", lambda: self._test_notification_channel("telegram")).pack(side="left")
        self.acct_notif_test_status = Label(f, text="", bg=BG, fg=TEXT_MUTED, font=_safe_font(8), wraplength=760, justify="left")
        self.acct_notif_test_status.pack(anchor="w", padx=26, pady=(4, 2))

        btn_row = Frame(f, bg=BG)
        btn_row.pack(fill="x", padx=24, pady=10)
        self._button(btn_row, "SAVE", self._save_account_settings, primary=True).pack(side="left")

        self.acct_status = Label(f, text="", bg=BG, fg=TEXT_MUTED, font=_safe_font(9))
        self.acct_status.pack(anchor="w", padx=26, pady=(4, 2))

    def _save_account_profile(self):
        from app.accounts.settings import load_account_settings, save_account_settings

        # Keep password_hash untouched -- Security's CHANGE PASSWORD button
        # below is the only way to alter it, same convention as the web
        # app's own /settings/account/save route.
        settings = load_account_settings()
        settings.display_name = self.acct_display_name.get_str().strip()
        settings.username = self.acct_username.get_str().strip()
        settings.email = self.acct_email.get_str().strip()
        settings.company = self.acct_company.get_str().strip()
        save_account_settings(settings)
        self.acct_profile_status.config(text="\u25cf  Saved.", fg=GREEN)

    def _save_subscription_info(self):
        from app.accounts.subscription import LICENSE_STATUS_CHOICES, SubscriptionInfo, save_subscription

        status = self.acct_sub_status.get_str().strip().lower()
        if status not in LICENSE_STATUS_CHOICES:
            status = "unset"
        save_subscription(SubscriptionInfo(
            plan=self.acct_sub_plan.get_str().strip(),
            license_key=self.acct_sub_license_key.get_str().strip(),
            license_status=status,
            renewal_date=self.acct_sub_renewal.get_str().strip(),
        ))
        self.acct_sub_status_label.config(text="\u25cf  Subscription info saved.", fg=GREEN)

    def _check_license_key_format(self):
        from app.accounts.subscription import check_license_key_format

        looks_valid, message = check_license_key_format(self.acct_sub_license_key.get_str())
        self.acct_sub_status_label.config(text=message, fg=(GREEN if looks_valid else AMBER))

    def _account_deactivate_license(self):
        from tkinter import messagebox

        from app.licensing import client as _license_client

        if not messagebox.askyesno(
            "Deactivate this device",
            "This frees this license's device slot on the license server so it can be "
            "activated on a different machine. This app will need to be reactivated the next "
            "time it's launched. Continue?",
        ):
            return
        _ok, message = _license_client.deactivate()
        self.acct_license_status_label.config(text=message, fg=GREEN)
        messagebox.showinfo("Deactivated", f"{message}\n\nT58 will now close -- reactivate it the next time you launch it.")
        self.root.destroy()

    def _account_change_password(self):
        from app.accounts.settings import hash_password, load_account_settings, save_account_settings

        new_password = self.acct_new_password.get_str()
        confirm_password = self.acct_confirm_password.get_str()
        if new_password != confirm_password:
            self.acct_security_status.config(
                text="New password and confirmation didn't match -- nothing was changed.", fg=RED,
            )
            return
        settings = load_account_settings()
        settings.password_hash = hash_password(new_password)  # blank -> "" -> lock removed
        save_account_settings(settings)
        self.acct_new_password.var.set("")
        self.acct_confirm_password.var.set("")
        if new_password:
            self.acct_security_status.config(text="\u25cf  Local app lock password set.", fg=GREEN)
            self.acct_password_status_label.config(text="\u25cf  Set", fg=GREEN)
        else:
            self.acct_security_status.config(text="\u25cf  Local app lock password cleared.", fg=GREEN)
            self.acct_password_status_label.config(text="\u25cb  Not set", fg=TEXT_DIM)

    def _account_log_out(self):
        from app.accounts.settings import load_account_settings

        # The desktop app has no login session of its own -- unlike the web
        # app (which gates browser access behind session["t58_unlocked"]
        # for Mobile Access), there's nothing here to actually log out of.
        # Stated plainly rather than pretending this button does something
        # it can't, matching this app's own honest-about-its-limits style.
        if load_account_settings().has_password:
            self.acct_security_status.config(
                text="The desktop app has no login session to log out of. Your local app lock "
                     "password still gates Mobile Access (the web app running on this machine) "
                     "the same as before.",
                fg=TEXT_MUTED,
            )
        else:
            self.acct_security_status.config(
                text="No lock password is set, so there's nothing to log out of.", fg=TEXT_MUTED,
            )

    def _account_delete(self):
        if not messagebox.askyesno(
            "Delete account",
            "Delete local account data? This clears your name, username, email, company, and "
            "lock password on this computer. Strategies, datasets, and reports are NOT "
            "affected. This can't be undone.",
        ):
            return
        from app.accounts.settings import AccountSettings, save_account_settings

        save_account_settings(AccountSettings())
        self.acct_display_name.var.set("")
        self.acct_username.var.set("")
        self.acct_email.var.set("")
        self.acct_company.var.set("")
        self.acct_new_password.var.set("")
        self.acct_confirm_password.var.set("")
        self.acct_password_status_label.config(text="\u25cb  Not set", fg=TEXT_DIM)
        self.acct_profile_status.config(text="", fg=TEXT_MUTED)
        self.acct_security_status.config(text="\u25cf  Local account data deleted.", fg=GREEN)

    def _account_check_for_updates(self):
        from app.accounts.app_info import check_for_updates

        result = check_for_updates()
        if not result.configured:
            self.acct_update_status.config(text=result.error, fg=TEXT_MUTED)
            self._acct_pending_update = None
            self.acct_download_update_btn.configure(state="disabled")
        elif result.checked and result.update_available:
            self.acct_update_status.config(
                text=f"A newer version is available: {result.latest_version}."
                     + (f"  {result.release_url}" if result.release_url else ""),
                fg=AMBER,
            )
            # v7: an update is available -- enable the DOWNLOAD UPDATE
            # button so the user can fetch it checksum-verified.
            self._acct_pending_update = result
            self.acct_download_update_btn.configure(state="normal")
        elif result.checked:
            self.acct_update_status.config(
                text=f"You're up to date (latest release: {result.latest_version}).", fg=GREEN,
            )
            self._acct_pending_update = None
            self.acct_download_update_btn.configure(state="disabled")
        else:
            self.acct_update_status.config(text=result.error, fg=RED)
            self._acct_pending_update = None
            self.acct_download_update_btn.configure(state="disabled")

    def _account_download_update(self):
        # v7: wire the DOWNLOAD UPDATE button to the real download path --
        # download_update() enforces the SHA-256 checksum (and refuses the
        # download if the release publishes no checksum), then this offers
        # to open the download folder. The app NEVER auto-installs or
        # auto-executes anything -- the user opens/installs it themselves.
        from app.accounts.app_info import download_update

        pending = self._acct_pending_update
        if pending is None:
            messagebox.showinfo(
                "Download update",
                "No update is pending -- click CHECK FOR UPDATES first.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Download update",
            f"Download version {pending.latest_version} from the GitHub release?\n\n"
            "The download is verified against the release's published SHA-256 checksum; "
            "the download is refused if the checksum is missing or does not match. "
            "Nothing is installed automatically -- you will choose whether to open the file yourself.",
            parent=self.root,
        ):
            return

        self.acct_update_status.config(
            text=f"Downloading {pending.latest_version}... (verifying checksum)", fg=AMBER,
        )
        self.acct_download_update_btn.configure(state="disabled")

        def _work():
            result = download_update(confirmed=True)
            self.root.after(0, lambda: self._account_download_finished(result))

        import threading
        threading.Thread(target=_work, daemon=True).start()

    def _account_download_finished(self, result):
        # Runs back on the UI thread after the download thread completes.
        from pathlib import Path

        if result.ok:
            self.acct_update_status.config(
                text=f"Update {result.version} downloaded and checksum-verified: {result.path}",
                fg=GREEN,
            )
            if messagebox.askyesno(
                "Download complete",
                f"Version {result.version} is downloaded and its checksum matched the release.\n\n"
                f"File: {result.path}\n\n"
                "Open the download folder so you can run the installer/update yourself?",
                parent=self.root,
            ):
                folder = str(Path(result.path).parent)
                try:
                    import os as _os
                    _os.startfile(folder)  # Windows only -- the desktop app ships as a Windows .exe
                except Exception:  # noqa: BLE001
                    messagebox.showinfo("Download folder", f"Open this folder yourself:\n{folder}", parent=self.root)
        else:
            self.acct_update_status.config(text=f"Download failed: {result.error}", fg=RED)
            messagebox.showerror("Download failed", result.error or "Unknown error.", parent=self.root)
            # Let them retry without re-running the check.
            self.acct_download_update_btn.configure(state="normal")

    def _save_account_settings(self):
        from app.web.notifications import NotificationSettings, load_notification_settings, save_notification_settings

        entered_password = self.acct_smtp_password.get_str().strip()
        entered_telegram_token = self.acct_telegram_token.get_str().strip()
        existing = load_notification_settings()
        settings = NotificationSettings(
            notify_email=self.acct_notify_email.get_str().strip(),
            smtp_host=self.acct_smtp_host.get_str().strip(),
            smtp_port=self.acct_smtp_port.get_int(587),
            smtp_username=self.acct_smtp_username.get_str().strip(),
            smtp_password=(entered_password or existing.smtp_password),
            smtp_from=self.acct_smtp_from.get_str().strip(),
            email_enabled=self.acct_email_enabled.get(),
            discord_webhook_url=self.acct_discord_webhook.get_str().strip(),
            telegram_bot_token=(entered_telegram_token or existing.telegram_bot_token),
            telegram_chat_id=self.acct_telegram_chat_id.get_str().strip(),
        )
        save_notification_settings(settings)
        self.acct_smtp_password.var.set("")
        self.acct_telegram_token.var.set(settings.telegram_bot_token)
        self.acct_status.config(text="●  Saved.", fg=GREEN)

    def _refresh_profile_picture_preview(self, filename: str) -> None:
        """Best-effort thumbnail via Pillow (see the UPLOAD button's own
        comment above for why Pillow rather than bare Tkinter PhotoImage).
        Never raises -- an unreadable or missing file just clears the
        preview back to a plain placeholder square instead of a crash."""
        from app.data.storage import get_app_base_dir

        self._acct_pic_photo = None  # keep a reference so Tk doesn't garbage-collect it
        path = None
        if filename:
            path = get_app_base_dir() / "data" / "config" / filename
            if not path.exists():
                path = None
        if path is None:
            self.acct_pic_preview_label.config(image="", text="No\npicture", fg=TEXT_DIM, font=_safe_font(8))
            return
        try:
            from PIL import Image, ImageTk

            img = Image.open(path).convert("RGB")
            img.thumbnail((64, 64))
            photo = ImageTk.PhotoImage(img)
            self._acct_pic_photo = photo
            self.acct_pic_preview_label.config(image=photo, text="")
        except Exception:
            # Pillow missing, or a format/file it can't read -- still tell
            # the person a picture IS saved, just can't preview it here.
            self.acct_pic_preview_label.config(image="", text="(saved,\nno preview)", fg=TEXT_DIM, font=_safe_font(8))

    def _account_upload_profile_picture(self):
        import shutil

        from app.accounts.settings import load_account_settings, save_account_settings
        from app.data.storage import get_app_base_dir

        path = filedialog.askopenfilename(
            title="Choose a profile picture",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        src = Path(path)
        ext = src.suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            messagebox.showwarning("Unsupported file", "Choose a PNG, JPEG, GIF, or WEBP image.")
            return
        settings = load_account_settings()
        pic_dir = get_app_base_dir() / "data" / "config"
        pic_dir.mkdir(parents=True, exist_ok=True)
        # Same convention as the web app's own upload route: exactly one
        # profile_picture.<ext> at a time -- clear any other extension
        # left over from a previous upload before writing the new one.
        for other_ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            (pic_dir / f"profile_picture{other_ext}").unlink(missing_ok=True)
        filename = f"profile_picture{ext}"
        shutil.copyfile(src, pic_dir / filename)
        settings.profile_picture_filename = filename
        save_account_settings(settings)
        self._refresh_profile_picture_preview(filename)
        self.acct_profile_status.config(text="\u25cf  Profile picture updated.", fg=GREEN)

    def _account_remove_profile_picture(self):
        from app.accounts.settings import load_account_settings, save_account_settings
        from app.data.storage import get_app_base_dir

        settings = load_account_settings()
        if settings.profile_picture_filename:
            (get_app_base_dir() / "data" / "config" / settings.profile_picture_filename).unlink(missing_ok=True)
            settings.profile_picture_filename = ""
            save_account_settings(settings)
        self._refresh_profile_picture_preview("")
        self.acct_profile_status.config(text="\u25cf  Profile picture removed.", fg=GREEN)

    def _test_notification_channel(self, channel: str):
        """Fires an actual test message through the exact same
        send path a real job-finished notification uses (see
        app.web.notifications), using whatever is CURRENTLY SAVED --
        mirrors the web app's own /settings/notifications/test route
        so desktop and web report identically for the same saved
        settings. These channels are fire-and-forget, so success here
        means the request didn't fail synchronously, not confirmed
        delivery."""
        from app.web.notifications import load_notification_settings

        self.acct_notif_test_status.config(text=f"Testing {channel}...", fg=AMBER)
        self.root.update_idletasks()
        settings = load_notification_settings()

        def run():
            if channel == "discord":
                if not settings.discord_is_usable:
                    ok, message = False, "No Discord webhook URL saved yet."
                else:
                    try:
                        import json
                        import urllib.request
                        payload = json.dumps(
                            {"content": "T58 -- test notification. If you see this, Discord is connected."},
                        ).encode("utf-8")
                        req = urllib.request.Request(
                            settings.discord_webhook_url.strip(), data=payload, method="POST",
                            headers={"Content-Type": "application/json"},
                        )
                        urllib.request.urlopen(req, timeout=10)
                        ok, message = True, "Test message sent to Discord."
                    except Exception as exc:  # noqa: BLE001
                        ok, message = False, f"Discord webhook request failed: {exc}"
            elif channel == "telegram":
                if not settings.telegram_is_usable:
                    ok, message = False, "Telegram needs both a bot token and a chat ID saved."
                else:
                    try:
                        import json
                        import urllib.parse
                        import urllib.request
                        url = f"https://api.telegram.org/bot{settings.telegram_bot_token.strip()}/sendMessage"
                        data = urllib.parse.urlencode({
                            "chat_id": settings.telegram_chat_id.strip(),
                            "text": "T58 -- test notification. If you see this, Telegram is connected.",
                        }).encode("utf-8")
                        req = urllib.request.Request(url, data=data, method="POST")
                        resp = urllib.request.urlopen(req, timeout=10)
                        body = json.loads(resp.read().decode("utf-8"))
                        if not body.get("ok"):
                            ok, message = False, f"Telegram API rejected the request: {body.get('description', 'unknown error')}"
                        else:
                            ok, message = True, "Test message sent to Telegram."
                    except Exception as exc:  # noqa: BLE001
                        ok, message = False, f"Telegram request failed: {exc}"
            elif channel == "email":
                if not settings.is_usable:
                    ok, message = False, "Email notifications aren't fully configured yet."
                else:
                    try:
                        from app.web.notifications import _send_email_notification
                        _send_email_notification(settings, "Test Notification", "connection test", None)
                        ok, message = True, "Test email queued -- check your inbox in a moment."
                    except Exception as exc:  # noqa: BLE001
                        ok, message = False, f"Sending the test email failed: {exc}"
            else:
                ok, message = False, f"Unknown channel '{channel}'."

            def _finish():
                self.acct_notif_test_status.config(text=message, fg=(GREEN if ok else RED))
            # Direct call rather than self.root.after(0, _finish) --
            # see the identical fix (and full explanation) in
            # _search_auto_build_ensemble's own comment above.
            _finish()

        threading.Thread(target=run, daemon=True).start()

    # -----------------------------------------------------------------------
    # Data Integrity Check button -- desktop parity with the same
    # pre-flight gate Run & Report / Full Pipeline / Quick Optimize /
    # Search Lab / Evolution Lab all run automatically on the web app
    # before a job starts (see app.validation.integrity_check's own
    # module docstring for exactly what it checks and why). On desktop
    # this is an on-demand button rather than an automatic gate -- each
    # tab already has its own Start/Run button with its own established
    # flow, and inserting an automatic block there is a bigger behavior
    # change than a button a person can press when they actually want to
    # check their data first. Always run with strategy=None (checking
    # data/timeframe/account alone) -- this is a fully supported,
    # documented mode of run_integrity_check, not a shortcut: Search Lab
    # and Evolution Lab explore many strategy families with no single
    # "current strategy" to check in the first place (this is exactly
    # what the web app's own "family_named" mode does), and using the
    # same data-only check uniformly across all four tabs, rather than a
    # deeper per-tab strategy extraction for only some of them, is a
    # correctness/consistency trade worth making here.
    # -----------------------------------------------------------------------
    def _check_data_integrity(self, context: "RunContextPanel"):
        log_lines: list[str] = []
        df = context.load_dataframe(log_fn=log_lines.append)
        if df is None:
            messagebox.showwarning("Data Integrity Check", "\n".join(log_lines) or "No data loaded yet.")
            return
        try:
            risk = context.build_risk_config()
            rules = context.build_prop_rules()
            report = run_integrity_check(df, None, risk, rules, data_label=context.instrument_label())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Data Integrity Check", f"Couldn't run the integrity check: {exc}")
            return
        win = Toplevel(self.root)
        win.title(f"Data Integrity Check -- {context.key}")
        win.configure(bg=PANEL)
        win.geometry("640x520")
        Label(
            win, text="Checks data/timeframe/account only, not a specific strategy's own lookahead "
            "safety (this tab doesn't have one fixed strategy to check) -- run this before a long "
            "search/optimization to catch a corrupt dataset or unsupportable timeframe early.",
            bg=PANEL, fg=TEXT_DIM, font=_safe_font(9), wraplength=600, justify="left",
        ).pack(anchor="w", padx=12, pady=(12, 8))
        text = Text(win, bg=PANEL_3, fg=TEXT, font=(MONO, 10), relief="flat", bd=0, wrap="word")
        text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        text.insert("1.0", report.render())
        text.config(state="disabled")
