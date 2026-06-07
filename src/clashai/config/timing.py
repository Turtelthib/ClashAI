# clashai/config/timing.py
# Centralised timing constants used across the codebase.
#
# Two families:
#   1. Step delays inside an episode (DELAY_*) — micro-pauses between ADB
#      taps so the emulator actually processes the input.
#   2. Navigation/lifecycle waits (WAIT_*) — coarser pauses between menus
#      or while waiting for the game to load a screen.
#   3. ADB-specific delays (ADB_DELAY_*) — kept distinct because they apply
#      to raw ADB calls regardless of which env/agent is using them.

# -----------------------------------------------------------------------------
# Step delays (combat / deploy)
# -----------------------------------------------------------------------------

# Time to wait after switching to a different troop slot in the bottom bar.
DELAY_SWITCH_TROOP = 0.10  # V4.3 value (V3 used 0.15)

# Time between tapping deployment positions when placing the same troop.
DELAY_DEPLOY = 0.05

# Two flavours of wait actions an agent can pick.
DELAY_WAIT_SHORT = 0.5
DELAY_WAIT_LONG = 2.0
# V3 kept a third flavour for combat-only pauses; preserved for back-compat.
DELAY_WAIT_COMBAT = 2.5

# Wait between perception observations. V4.3 dropped this from 2.5 → 0.15
# because PerceptionThread pre-computes asynchronously.
DELAY_OBSERVE = 0.15

# After activating a hero ability, give the game a moment to register it.
DELAY_ABILITY = 0.3


# -----------------------------------------------------------------------------
# Navigation / lifecycle waits
# -----------------------------------------------------------------------------

# After the attack screen loads, the village still has falling decorations
# / animations. Wait before YOLO so detections aren't disturbed.
WAIT_DECORATIONS = 3.0

# Hard cap on combat duration. CoC battles last 3 minutes (180s); a bit of
# slack to account for cinematics.
WAIT_BATTLE_MAX = 195.0

# Polling interval while waiting for combat to end.
WAIT_BATTLE_CHECK = 5.0

# How long the result screen stays before we navigate away.
WAIT_RESULT_SCREEN = 5.0

# Generic wait between menu navigation steps.
WAIT_NAVIGATION = 1.5

# Matchmaking loading time before "Lancer l'attaque" becomes available.
WAIT_MATCHMAKING = 4.0


# -----------------------------------------------------------------------------
# ADB call delays (apply to every ADB invocation regardless of caller)
# -----------------------------------------------------------------------------

ADB_DELAY_TAP = 0.07
ADB_DELAY_SCREENSHOT = 0.2
ADB_DELAY_NAVIGATION = 1.0
ADB_DELAY_MATCHMAKING = 3.0
