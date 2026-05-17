# clashai/config/window.py
# Emulator window detection : title keywords + exclusion list.
#
# Used by `screen_capture.find_emulator_bbox()` and `zoom_control` to find
# the right HWND in the global window list. Previously defined in two
# separate modules, consolidated here.

# -----------------------------------------------------------------------------
# Emulator title keywords (case-insensitive substring match)
# -----------------------------------------------------------------------------
# A window whose title contains any of these is a candidate emulator
# window. We keep the keyword broad enough to catch all common Android
# emulators on Windows.
#
# NOTE: 'Clash of Clans' is intentionally excluded — it would match VS
# Code window titles when a CoC screenshot is open in the editor.
EMULATOR_WINDOW_KEYWORDS = [
    'Google Play', 'play games',
    'LDPlayer', 'LD Player', 'LDMultiPlayer',
    'BlueStacks',
    'MuMu',
    'Nox',
    'MEmu',
]


# -----------------------------------------------------------------------------
# Excluded title substrings (case-insensitive)
# -----------------------------------------------------------------------------
# Windows whose title contains one of these are NOT the emulator even if
# they match an EMULATOR_WINDOW_KEYWORDS substring — typically editors and
# browsers showing the keyword as document text (e.g. "Fix Google Play
# emulator - VS Code").
EXCLUDED_TITLE_SUBSTRINGS = [
    'Visual Studio Code',
    '- Mozilla Firefox',
    '- Google Chrome',
    '- Microsoft Edge',
    '- Brave',
    '- Opera',
    '- Vivaldi',
    'Discord',
    'Slack',
    'GitHub Desktop',
    'Notepad',
    'Sublime Text',
    'JetBrains',
    'PyCharm',
    'IntelliJ',
]


def title_is_excluded(title: str) -> bool:
    """True if `title` matches any EXCLUDED_TITLE_SUBSTRINGS (case-insensitive)."""
    t = title.lower()
    return any(s.lower() in t for s in EXCLUDED_TITLE_SUBSTRINGS)
