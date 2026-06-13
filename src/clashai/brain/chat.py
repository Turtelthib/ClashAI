# clashai/brain/chat.py
# BrainChatMixin — clan chat acknowledgements + command polling.

import time


class BrainChatMixin:
    """Reads clan chat commands and posts the AI's acknowledgements."""

    def _send_chat_ack(self, target, before=True, result=None):
        """Sends a message in the clan chat."""
        try:
            # Open the chat
            if self._chat_monitor.open_chat(
                    self._classify_screen, self._models):
                time.sleep(0.3)

                if before:
                    self._chat_monitor.send_chat_message(
                        f"IA - jattaque le {target}"
                    )
                else:
                    stars = result.get('stars', 0)
                    pct = result.get('percentage', 0)
                    self._chat_monitor.send_chat_message(
                        f"IA - {target} fait {stars}e {pct}pct"
                    )

                # Close the chat
                self._chat_monitor.close_chat()
        except Exception as e:
            if self.verbose:
                print(f" WARNING: Erreur envoi chat: {e}")

    def _check_clan_chat(self):
        """
        Opens the chat, reads commands, closes the chat.
        Like a player glancing at the chat between attacks.

        Returns:
            commands: list of detected commands
        """
        if self.verbose:
            print("\n  Vérification du chat clan...")

        self._last_chat_check = time.time()
        self._attacks_since_chat_check = 0

        # Open the chat
        if not self._chat_monitor.open_chat(self._classify_screen, self._models):
            if self.verbose:
                print(" WARNING: Unable to open chat")
            return []

        time.sleep(0.5)

        # Read commands
        img = self._adb_screenshot()
        commands = []
        if img is not None:
            commands = self._chat_monitor.check_once(img)

        # Close the chat
        self._chat_monitor.close_chat()

        if commands and self.verbose:
            print(f"  {len(commands)} commande(s) trouvée(s)")

        return commands
