# scripts/rl/clan_chat_monitor.py
# Clan chat monitoring and command parsing for ClashAI.
#
# The bot monitors the clan chat at regular intervals.
# When a message containing a command is detected, it executes it.
#
# Supported commands:
# @mini_pekka 3 → Attack target #3 in CW
# @mini_pekka attack 5 → Attack target #5 in CW
# @mini_pekka stop → Stop monitoring
# @mini_pekka status → The AI replies with its status (via a friendly challenge or nothing)
#
# OCR method:
# EasyOCR is used (better than Tesseract on game fonts).
# Fallback to pytesseract if EasyOCR is not installed.
#
# Usage:
# monitor = ClanChatMonitor(bot_name='mini_pekka')
# monitor.start(models) # Infinite monitoring loop
#
# One-shot usage:
# monitor = ClanChatMonitor(bot_name='mini_pekka')
# commands = monitor.check_once(screenshot_pil)

import sys
import re
import time
import cv2
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# Re-imported from clashai/config/screen.py (Phase A).
from clashai.config import ADB_WIDTH, ADB_HEIGHT  # noqa: E402

# Chat zone on screen (when the chat is open)
# The chat occupies approximately the left half of the screen
# BOTTOM = 980 to capture messages at the very bottom (before the input bar)
CHAT_ZONE_LEFT = 0
CHAT_ZONE_RIGHT = 850
CHAT_ZONE_TOP = 60
CHAT_ZONE_BOTTOM = 980

# Button to open the chat — loaded from ui_positions.json
# Calibrated via: python scripts/rl/calibrate_ui.py
def _get_chat_button_pos():
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position('chat_open')
    except ImportError:
        return (47, 400)

# Brain/chat orchestrator constants re-imported from clashai/config/brain.py (Phase A).
from clashai.config import (
    MONITOR_INTERVAL, DEFAULT_BOT_NAME,
    MAX_COMMAND_AGE_MINUTES, MAX_HISTORY,
)  # noqa: E402


# =============================================================================
# ADB FUNCTIONS
# =============================================================================

# Re-exported from the canonical implementation in game_loop (Phase B.1).
# That version routes through WGC (fast, occlusion-proof) with ADB fallback.
from clashai.navigation.game_loop import adb_screenshot as _adb_screenshot  # noqa: E402


def _adb_tap(x, y, delay=0.1):
    """Phase C.1: routed through clashai.adb.ADBClient."""
    from clashai.adb import get_client
    get_client().tap(x, y, delay=delay)


# =============================================================================
# OCR ENGINE
# =============================================================================

_ocr_engine = None
_ocr_type = None


def _init_ocr():
    """Initializes the OCR engine (EasyOCR preferred, Tesseract fallback)."""
    global _ocr_engine, _ocr_type

    if _ocr_engine is not None:
        return _ocr_engine, _ocr_type

    # Try EasyOCR
    try:
        import easyocr
        _ocr_engine = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
        _ocr_type = 'easyocr'
        print(" OCR initialized: EasyOCR (fr+en)")
        return _ocr_engine, _ocr_type
    except ImportError:
        pass

    # Try pytesseract
    try:
        import pytesseract
        _ocr_engine = pytesseract
        _ocr_type = 'tesseract'
        print(" OCR initialized: Tesseract")
        return _ocr_engine, _ocr_type
    except ImportError:
        pass

    print("WARNING: No OCR engine available!")
    print(" Install: pip install easyocr")
    print(" Or: pip install pytesseract")
    _ocr_type = None
    return None, None


def _ocr_read(img_cv):
    """
    Reads text from a BGR image.

    Returns:
        lines: list of str (detected text lines)
    """
    engine, etype = _init_ocr()
    if engine is None:
        return []

    if etype == 'easyocr':
        # EasyOCR returns a list of (bbox, text, confidence)
        results = engine.readtext(img_cv, paragraph=False)
        lines = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and len(text.strip()) > 0:
                lines.append(text.strip())
        return lines

    elif etype == 'tesseract':
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        text = engine.image_to_string(gray, lang='fra+eng')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines

    return []


# =============================================================================
# COMMAND PARSER
# =============================================================================

def parse_command(text, bot_name=DEFAULT_BOT_NAME):
    """
    Parses a text line to detect a command.

    Accepted formats:
        @mini_pekka 3
        @mini_pekka attack 3
        @mini_pekka attaque 3
        @mini_pekka stop
        @mini_pekka status
        @mini pekka 3 (with space)
        mini_pekka 3 (without @)

    Returns:
        command: dict or None
            {'type': 'attack', 'target': 3}
            {'type': 'stop'}
            {'type': 'status'}
    """
    text_lower = text.lower().strip()

    # Normalize the bot name (handles spaces, underscores, @)
    bot_patterns = [
        f'@{bot_name}',
        f'@{bot_name.replace("_", " ")}',
        bot_name,
        bot_name.replace('_', ' '),
    ]

    found = False
    remaining = text_lower
    for pattern in bot_patterns:
        if pattern in text_lower:
            # Extract what follows the mention
            idx = text_lower.index(pattern)
            remaining = text_lower[idx + len(pattern):].strip()
            found = True
            break

    if not found:
        return None

    # Parse the command
    remaining = remaining.strip()

    # "stop"
    if remaining in ('stop', 'arret', 'arrête', 'pause'):
        return {'type': 'stop'}

    # "status"
    if remaining in ('status', 'état', 'etat', 'info'):
        return {'type': 'status'}

    # "reset" — forget already executed commands (new CW)
    if remaining in ('reset', 'clear', 'oublie', 'nouveau', 'new'):
        return {'type': 'reset'}

    # "attack 3" or "attaque 3" or just "3"
    attack_match = re.match(r'(?:attack|attaque|atk|att)?\s*(\d+)', remaining)
    if attack_match:
        target = int(attack_match.group(1))
        if 1 <= target <= 50:
            return {'type': 'attack', 'target': target}

    return None


def parse_all_commands(lines, bot_name=DEFAULT_BOT_NAME, **kwargs):
    """
    Parses all lines and returns the detected commands.

    Returns:
        commands: list of dicts
    """
    commands = []
    for line in lines:
        cmd = parse_command(line, bot_name)
        if cmd is not None:
            cmd['raw_text'] = line
            commands.append(cmd)
    return commands


def parse_timestamp(text):
    """
    Parses a CoC chat timestamp and returns the age in minutes.

    CoC formats:
        "A l'instant" → 0  (in-game French string for "right now")
        "a l'instant" → 0
        "1min" → 1
        "5min" → 5
        "1h 22min" → 82
        "14h 8min" → 848
        "1h" → 60
        "2j" → 2880
        "1j 3h" → 1620

    Handles common OCR errors:
        "IImin" → 11min, "l1min" → 11min, "Ih" → 1h

    Returns:
        int (minutes) or None if not a timestamp
    """
    text_raw = text.strip()
    
    # Clean OCR errors BEFORE lowercasing
    # If the text looks like a timestamp (contains h, j, min),
    # replace I/l/| with 1 in numeric positions
    if re.search(r'[hHjJmM]', text_raw):
        # Replace I/l/| with 1 when in a numeric context
        text_raw = re.sub(r'[Il|](?=[Il|\dhHjJmM])', '1', text_raw)
        # "O" → "0" 
        text_raw = re.sub(r'(?<=\d)O', '0', text_raw)
        text_raw = re.sub(r'O(?=\d)', '0', text_raw)
    
    text = text_raw.lower()
    
    # CoC in-game string for "right now": "a l'instant" / "instant"
    if 'instant' in text:
        return 0
    
    total_minutes = 0
    found = False
    
    # Days
    day_match = re.search(r'(\d+)\s*j', text)
    if day_match:
        total_minutes += int(day_match.group(1)) * 24 * 60
        found = True

    # Hours
    hour_match = re.search(r'(\d+)\s*h', text)
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60
        found = True

    # Minutes
    min_match = re.search(r'(\d+)\s*min', text)
    if min_match:
        total_minutes += int(min_match.group(1))
        found = True
    
    if found:
        return total_minutes
    
    return None


# =============================================================================
# CHAT MONITOR
# =============================================================================

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


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClashAI Chat Monitor")
    parser.add_argument('--test-ocr', action='store_true',
                        help="Test OCR on the current chat")
    parser.add_argument('--test-parse', action='store_true',
                        help="Test command parsing")
    parser.add_argument('--bot-name', type=str, default=DEFAULT_BOT_NAME)
    args = parser.parse_args()

    if args.test_parse:
        print("Test command parsing\n")

        test_lines = [
            "@mini_pekka 3",
            "@mini_pekka attack 5",
            "@mini_pekka attaque 12",
            "@mini pekka 7",
            "mini_pekka 3",
            "@mini_pekka stop",
            "@mini_pekka status",
            "hey les gars ça va ?",
            "quelqu'un pour GdC ?",
            "@mini_pekka",
            "@mini_pekka abc",
            "@mini_pekka 0",
            "@mini_pekka 99",
        ]

        for line in test_lines:
            cmd = parse_command(line, args.bot_name)
            status = f"→ {cmd}" if cmd else "→ (ignored)"
            print(f" '{line}' {status}")

    elif args.test_ocr:
        print("Test OCR on the current chat\n")

        img = _adb_screenshot()
        if img is None:
            print("ERROR: Unable to capture the screen")
            sys.exit(1)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        chat_zone = img_cv[CHAT_ZONE_TOP:CHAT_ZONE_BOTTOM,
                           CHAT_ZONE_LEFT:CHAT_ZONE_RIGHT]

        # Save for debug
        cv2.imwrite('debug_chat_zone.png', chat_zone)
        print(" Chat zone saved: debug_chat_zone.png")

        lines = _ocr_read(chat_zone)
        print(f"\n  {len(lines)} lines detected:")
        for i, line in enumerate(lines):
            cmd = parse_command(line, args.bot_name)
            marker = " ← COMMAND" if cmd else ""
            print(f" [{i:2d}] {line}{marker}")

        if lines:
            commands = parse_all_commands(lines, args.bot_name)
            if commands:
                print(f"\n Commands found: {commands}")
            else:
                print(f"\n No @{args.bot_name} command found")

    else:
        print("Usage:")
        print(" --test-parse Test command parsing")
        print(" --test-ocr Test OCR on the current chat")