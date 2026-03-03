import threading
import time
from typing import Callable, Optional

import pyperclip
from pynput import keyboard


class MacTextSelectionListener:
    """Listen for a global hotkey, capture selected text, and replace corrected text."""

    def __init__(self, logger, shortcut: str = "<alt>+<shift>"):
        # Keep logger for detailed debug output in terminal.
        self.logger = logger
        # Save user-configurable hotkey string.
        self.shortcut = shortcut
        # Controller allows us to send Cmd+C and Cmd+V key events programmatically.
        self.controller = keyboard.Controller()
        # Mutex prevents overlapping correction actions on repeated hotkey presses.
        self.lock = threading.Lock()

        # Parse and register the shortcut handler.
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(self.shortcut),
            self._on_activate,
        )

        # Listener captures every key press/release globally.
        self.listener = keyboard.Listener(
            on_press=self._for_canonical(self.hotkey.press),
            on_release=self._for_canonical(self.hotkey.release),
        )

        # Callback injected by run.py to execute correction business logic.
        self.on_shortcut_callback: Optional[Callable[[], None]] = None

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Attach external callback executed when the shortcut is detected."""
        self.on_shortcut_callback = callback

    def start(self) -> None:
        """Start listening and block current thread forever."""
        self.logger.info("Starting listener with shortcut: %s", self.shortcut)
        self.listener.start()
        self.listener.join()

    def _for_canonical(self, func):
        """Normalize key events to canonical format required by HotKey helper."""
        return lambda key: func(self.listener.canonical(key))

    def _on_activate(self) -> None:
        """Triggered when shortcut is pressed."""
        self.logger.info("Shortcut detected")

        # Use lock to ignore re-entrant events while one correction is running.
        if not self.lock.acquire(blocking=False):
            self.logger.warning("Correction already running; this trigger is ignored")
            return

        try:
            if self.on_shortcut_callback:
                self.on_shortcut_callback()
            else:
                self.logger.warning("No callback registered for shortcut event")
        finally:
            self.lock.release()

    def get_selected_text(self) -> str:
        """Copy currently selected text using Cmd+C and read it from clipboard."""
        try:
            sentinel_prefix = "__TYPE_CORRECTION_SENTINEL__"
            previous_clipboard = pyperclip.paste()
            self.logger.info("Clipboard snapshot captured before Cmd+C")

            # Allow user to release shortcut keys (Cmd+Shift+Z) before sending Cmd+C.
            time.sleep(0.08)

            # Write a unique sentinel first, so we can reliably detect clipboard refresh.
            sentinel = f"{sentinel_prefix}{time.time_ns()}__"
            pyperclip.copy(sentinel)

            # Simulate Cmd+C to copy highlighted text from active application.
            with self.controller.pressed(keyboard.Key.cmd):
                self.controller.press("c")
                self.controller.release("c")

            # Poll clipboard for a short period until it changes from sentinel.
            selected_text = ""
            deadline = time.time() + 0.8
            while time.time() < deadline:
                current_clipboard = pyperclip.paste()
                # Ignore current sentinel and stale sentinels from previous failed attempts.
                current_text = current_clipboard.strip() if current_clipboard else ""
                if current_text and not current_text.startswith(sentinel_prefix):
                    selected_text = current_clipboard
                    break
                time.sleep(0.02)

            # If clipboard has no useful text, return empty string.
            if not selected_text or not selected_text.strip():
                self.logger.warning("No text selected or copy command did not update clipboard")
                self.logger.warning("Check Accessibility/Input Monitoring permissions and app copy support")
                pyperclip.copy(previous_clipboard)
                return ""

            # If clipboard is unchanged, selected text might still be valid; log for visibility.
            if selected_text == previous_clipboard:
                self.logger.info("Clipboard unchanged after Cmd+C; using current clipboard text")

            self.logger.info("Selected text: %r", selected_text)
            return selected_text
        except Exception as error:
            self.logger.error("Failed to capture selected text: %s", error)
            return ""

    def replace_selected_text(self, new_text: str) -> bool:
        """Replace highlighted text by pasting corrected content with Cmd+V."""
        if not new_text:
            self.logger.warning("Replacement text is empty; skipping paste")
            return False

        try:
            # Put corrected text in clipboard.
            pyperclip.copy(new_text)
            self.logger.info("Corrected text copied to clipboard")

            # Simulate Cmd+V to replace currently selected text in active app.
            with self.controller.pressed(keyboard.Key.cmd):
                self.controller.press("v")
                self.controller.release("v")

            self.logger.info("Pasted corrected text into active application")
            return True
        except Exception as error:
            self.logger.error("Failed to replace selected text: %s", error)
            return False
