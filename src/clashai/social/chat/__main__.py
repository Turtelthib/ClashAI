# clashai/social/chat/__main__.py
# Test CLI: `uv run python -m clashai.social.chat --test-parse | --test-ocr`

import sys
import argparse

import cv2
import numpy as np

from clashai.social.chat.constants import (
    DEFAULT_BOT_NAME,
    CHAT_ZONE_LEFT, CHAT_ZONE_RIGHT, CHAT_ZONE_TOP, CHAT_ZONE_BOTTOM,
)
from clashai.social.chat.adb_io import _adb_screenshot
from clashai.social.chat.ocr import _ocr_read
from clashai.social.chat.parser import parse_command, parse_all_commands


def main():
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


if __name__ == "__main__":
    main()
