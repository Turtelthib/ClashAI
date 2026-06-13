# clashai/social/chat/monitor.py
# ClanChatMonitor — open/read/parse/close the clan chat + monitoring loop.

import re
import time

import cv2
import numpy as np

from clashai.social.chat.constants import (
    CHAT_ZONE_LEFT, CHAT_ZONE_RIGHT, CHAT_ZONE_TOP, CHAT_ZONE_BOTTOM,
    _get_chat_button_pos, DEFAULT_BOT_NAME, MONITOR_INTERVAL,
)
from clashai.social.chat.adb_io import _adb_screenshot, _adb_tap
from clashai.social.chat.ocr import _ocr_read
from clashai.social.chat.parser import parse_command


class ClanChatMonitor:
    """
    Monitors the clan chat and detects commands.

    The chat itself is the source of truth:

    - If the chat contains "@mini PEKKA 7" but NO "[IA]" mentioning #7
      → new command, the AI must attack
    - If the chat contains "@mini PEKKA 7" AND "[IA] J'attaque le #7"
      → command already handled, ignored

    No JSON file, no snapshot, no OCR timestamp.
    The AI simply checks whether it has already replied to the message.
    """

    BOT_PREFIX = "IA -"

    def __init__(self, bot_name=DEFAULT_BOT_NAME, verbose=True):
        self.bot_name = bot_name
        self.verbose = verbose
        self._running = False

    def send_chat_message(self, message):
        """
        Sends a message in the clan chat via ADB.
        The chat MUST be open before calling this method.
        """
        import subprocess

        try:
            from clashai.navigation.calibrate_ui import get_position
            chat_input_pos = get_position('chat_input')
        except (ImportError, Exception):
            chat_input_pos = (300, 1010)

        _adb_tap(chat_input_pos[0], chat_input_pos[1])
        time.sleep(0.5)

        # ADB input text: spaces become %s
        # Special shell characters must be escaped
        safe_text = message.replace(' ', '%s')
        # Keep only ADB-safe characters
        safe_text = ''.join(c for c in safe_text if c.isalnum() or c in "!?.,-%s")

        try:
            # Phase C.1: routed through clashai.adb.ADBClient.
            from clashai.adb import get_client
            get_client().input_text(safe_text)
        except Exception as e:
            if self.verbose:
                print(f" WARNING: Text input error: {e}")
            return

        try:
            from clashai.navigation.calibrate_ui import get_position
            send_pos = get_position('chat_send')
        except (ImportError, Exception):
            send_pos = (490, 1010)

        _adb_tap(send_pos[0], send_pos[1])
        time.sleep(0.5)

        if self.verbose:
            print(f"  Message envoyé : {message}")

    def mark_executed(self, command):
        """Compatibility — the Brain calls this but the real filter is the chat."""
        pass

    def check_once(self, screenshot_pil=None):
        """
        Checks the chat and returns commands not yet handled.

        Logic based on message ORDER (top = oldest, bottom = most recent):

        1. Iterates over all chat lines
        2. For each target #N, records the position of the LAST command
           (@mini PEKKA N) and the LAST reply ([IA] #N)
        3. If the command is BELOW the reply → new command
           (someone requested again after the bot's reply)
        4. If the reply is BELOW the command → already handled
        5. If no reply at all → new command

        Returns:
            new_commands: list of commands to execute
        """
        if screenshot_pil is None:
            screenshot_pil = _adb_screenshot()
        if screenshot_pil is None:
            return []

        img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

        chat_zone = img_cv[CHAT_ZONE_TOP:CHAT_ZONE_BOTTOM,
                           CHAT_ZONE_LEFT:CHAT_ZONE_RIGHT]

        lines = _ocr_read(chat_zone)

        if self.verbose and lines:
            print(f"  OCR : {len(lines)} lignes lues")

        # Step 1: record the position of each bot reply
        # Bot messages start with "IA -" or "IA" followed by "jattaque" or "fait"
        last_ack_position = {}
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            is_bot_msg = (
                line_lower.startswith('ia -') or
                line_lower.startswith('ia ') and ('jattaque' in line_lower or 'fait' in line_lower) or
                self.BOT_PREFIX.lower() in line_lower
            )
            if is_bot_msg:
                match = re.search(r'(\d{1,2})', line)
                if match:
                    num = int(match.group(1))
                    if 1 <= num <= 50:
                        last_ack_position[num] = i

        if self.verbose and last_ack_position:
            print(f" AI replies found: "
                  f"{dict(sorted(last_ack_position.items()))}")

        # Step 2: record the position of each @mini PEKKA N command
        last_cmd_position = {}
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Skip bot messages
            if (line_lower.startswith('ia -') or
                line_lower.startswith('ia ') and ('jattaque' in line_lower or 'fait' in line_lower)):
                continue
            cmd = parse_command(line, self.bot_name)
            if cmd is not None and cmd['type'] == 'attack':
                cmd['raw_text'] = line
                last_cmd_position[cmd['target']] = (i, cmd)

        # Step 3: compare positions
        new_commands = []
        for target, (cmd_pos, cmd) in last_cmd_position.items():
            ack_pos = last_ack_position.get(target)

            if ack_pos is None:
                # No [IA] reply → new command
                new_commands.append(cmd)
                if self.verbose:
                    print(f"  #{target}: no [IA] reply → new")
            elif cmd_pos > ack_pos:
                # Command AFTER the reply → new request
                new_commands.append(cmd)
                if self.verbose:
                    print(f"  #{target}: command (line {cmd_pos}) "
                          f"after [IA] (line {ack_pos}) → new")
            else:
                # Reply AFTER the command → already handled
                if self.verbose:
                    print(f"  #{target}: [IA] (line {ack_pos}) "
                          f"after command (line {cmd_pos}) → already done")

        # Also parse non-attack commands (stop, status)
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if (line_lower.startswith('ia -') or
                line_lower.startswith('ia ') and ('jattaque' in line_lower or 'fait' in line_lower)):
                continue
            cmd = parse_command(line, self.bot_name)
            if cmd is not None and cmd['type'] != 'attack':
                cmd['raw_text'] = line
                new_commands.append(cmd)

        return new_commands

    def open_chat(self, classify_screen_fn, models):
        """
        Opens the clan chat from the village.

        Args:
            classify_screen_fn: screen classification function
            models: perception models

        Returns:
            success: bool
        """
        # Check that we are at the village
        img = _adb_screenshot()
        if img is None:
            return False

        state, conf = classify_screen_fn(img, models)

        if state == 'chat_clan':
            return True

        if state != 'village_home':
            if self.verbose:
                print(f" WARNING: Not at the village (state: {state}), cannot open chat")
            return False

        # Tap the chat button
        _adb_tap(*_get_chat_button_pos())
        time.sleep(1.5)

        # Verify
        img = _adb_screenshot()
        if img is None:
            return False
        state, conf = classify_screen_fn(img, models)

        if state == 'chat_clan':
            if self.verbose:
                print("  Clan chat opened")
            return True
        else:
            if self.verbose:
                print(f" WARNING: Chat not opened (state: {state})")
            return False

    def close_chat(self):
        """Closes the chat by tapping outside."""
        try:
            from clashai.navigation.calibrate_ui import get_position
            pos = get_position('chat_close_tap')
        except ImportError:
            pos = (1400, 400)
        _adb_tap(pos[0], pos[1])
        time.sleep(0.5)
        _adb_tap(960, 400)
        time.sleep(0.5)

    def monitor_loop(self, classify_screen_fn, models, callback=None,
                     interval=MONITOR_INTERVAL):
        """
        Chat monitoring loop.

        Each cycle:
        1. Open the chat
        2. Screenshot + OCR
        3. Parse commands
        4. Close the chat
        5. Execute the callback if a command is found
        6. Wait for the interval

        Args:
            classify_screen_fn: screen classification function
            models: perception models
            callback: callable(command_dict) — called for each command
            interval: seconds between each check
        """
        self._running = True

        if self.verbose:
            print("\n Chat monitoring started")
            print(f" Bot name: @{self.bot_name}")
            print(f" Interval: {interval}s")

        while self._running:
            try:
                # 1. Open the chat
                if self.open_chat(classify_screen_fn, models):
                    time.sleep(0.5)

                    # 2. Screenshot the chat
                    img = _adb_screenshot()
                    if img is not None:
                        # 3. Detect commands
                        commands = self.check_once(img)

                        # 4. Close the chat
                        self.close_chat()

                        # 5. Execute commands
                        for cmd in commands:
                            if self.verbose:
                                print(f"\n Executing: {cmd}")

                            if cmd['type'] == 'stop':
                                if self.verbose:
                                    print(" Stop monitoring requested")
                                self._running = False
                                return

                            if callback:
                                callback(cmd)
                    else:
                        self.close_chat()
                else:
                    # Not at village? Wait longer
                    time.sleep(10)

            except KeyboardInterrupt:
                if self.verbose:
                    print("\n Monitoring stopped (Ctrl+C)")
                self._running = False
                return

            except Exception as e:
                if self.verbose:
                    print(f" WARNING: Monitoring error: {e}")

            # 6. Wait
            time.sleep(interval)

    def stop(self):
        """Stops the monitoring loop."""
        self._running = False
