# clashai/social/chat/constants.py
# Chat zone geometry, button positions, and re-exported config constants.

# Re-imported from clashai/config (Phase A) — kept importable for back-compat.
from clashai.config import (  # noqa: F401
    ADB_WIDTH, ADB_HEIGHT,
    MONITOR_INTERVAL, DEFAULT_BOT_NAME,
    MAX_COMMAND_AGE_MINUTES, MAX_HISTORY,
)

# Chat zone on screen (when the chat is open).
# The chat occupies roughly the left half; BOTTOM=980 captures messages at
# the very bottom (just above the input bar).
CHAT_ZONE_LEFT = 0
CHAT_ZONE_RIGHT = 850
CHAT_ZONE_TOP = 60
CHAT_ZONE_BOTTOM = 980


def _get_chat_button_pos():
    """Chat-open button position — calibrated via calibrate_ui, with fallback."""
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position('chat_open')
    except ImportError:
        return (47, 400)
