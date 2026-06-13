# clashai/social/chat/parser.py
# Command + timestamp parsing for clan chat messages.

import re

from clashai.social.chat.constants import DEFAULT_BOT_NAME


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
