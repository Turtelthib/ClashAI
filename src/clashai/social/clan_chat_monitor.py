# clashai/social/clan_chat_monitor.py
# Back-compat shim — implementation moved to the `chat/` package (Phase 3
# split). Re-exports the public API so existing imports keep working:
#   from clashai.social.clan_chat_monitor import ClanChatMonitor, _init_ocr

from clashai.social.chat import (  # noqa: F401
    ClanChatMonitor,
    _init_ocr, _ocr_read,
    parse_command, parse_all_commands, parse_timestamp,
    _adb_screenshot, _adb_tap, _get_chat_button_pos,
    CHAT_ZONE_LEFT, CHAT_ZONE_RIGHT, CHAT_ZONE_TOP, CHAT_ZONE_BOTTOM,
    ADB_WIDTH, ADB_HEIGHT,
    MONITOR_INTERVAL, DEFAULT_BOT_NAME,
    MAX_COMMAND_AGE_MINUTES, MAX_HISTORY,
)

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
