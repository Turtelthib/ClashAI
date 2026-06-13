# clashai/social/chat/
# Clan chat monitoring + command parsing (Phase 3 split of clan_chat_monitor.py).
#
# Modules:
#   constants.py — chat zone geometry, button pos, config re-exports
#   adb_io.py    — _adb_screenshot (WGC-routed) + _adb_tap
#   ocr.py       — _init_ocr / _ocr_read (EasyOCR → Tesseract fallback)
#   parser.py    — parse_command / parse_all_commands / parse_timestamp
#   monitor.py   — ClanChatMonitor (open/read/parse/close + loop)
#   __main__.py  — test CLI (--test-parse / --test-ocr)
#
# Public API re-exported so callers keep using:
#   from clashai.social.clan_chat_monitor import ClanChatMonitor, _init_ocr

from clashai.social.chat.constants import (
    CHAT_ZONE_LEFT, CHAT_ZONE_RIGHT, CHAT_ZONE_TOP, CHAT_ZONE_BOTTOM,
    _get_chat_button_pos,
    ADB_WIDTH, ADB_HEIGHT,
    MONITOR_INTERVAL, DEFAULT_BOT_NAME,
    MAX_COMMAND_AGE_MINUTES, MAX_HISTORY,
)
from clashai.social.chat.adb_io import _adb_screenshot, _adb_tap
from clashai.social.chat.ocr import _init_ocr, _ocr_read
from clashai.social.chat.parser import (
    parse_command, parse_all_commands, parse_timestamp,
)
from clashai.social.chat.monitor import ClanChatMonitor

__all__ = [
    'ClanChatMonitor',
    '_init_ocr', '_ocr_read',
    'parse_command', 'parse_all_commands', 'parse_timestamp',
    '_adb_screenshot', '_adb_tap', '_get_chat_button_pos',
    'CHAT_ZONE_LEFT', 'CHAT_ZONE_RIGHT', 'CHAT_ZONE_TOP', 'CHAT_ZONE_BOTTOM',
    'ADB_WIDTH', 'ADB_HEIGHT',
    'MONITOR_INTERVAL', 'DEFAULT_BOT_NAME',
    'MAX_COMMAND_AGE_MINUTES', 'MAX_HISTORY',
]
