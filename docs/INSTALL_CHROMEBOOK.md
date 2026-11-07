# Install on a Chromebook

1. Turn on Linux: **Settings > About ChromeOS > Linux development environment > Turn on**.
2. Download the **Chromebook** zip (most Chromebooks) or **Chromebook-ARM** (ARM models, e.g. MediaTek/Snapdragon). Not sure? In the Linux terminal run `uname -m`: `x86_64` = Chromebook, `aarch64` = Chromebook-ARM.
3. Move the zip into **My files > Linux files**, then in the Linux terminal:
   ```
   unzip T58-*-Chromebook*.zip -d t58 && cd t58 && bash install_chromebook.sh
   ```
4. The installer adds a **T58** launcher to the ChromeOS app drawer and opens the app in your browser.

Needs Debian 12 (the default Linux container). If the app does not start, run `./T58-Web-App` from the terminal to see the error.
