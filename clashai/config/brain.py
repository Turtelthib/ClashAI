# clashai/config/brain.py
# Constants used by the brain orchestrator and social sub-agents.
#
# These govern WHEN agents run (intervals, cooldowns) and in what ORDER
# (priorities). Centralised here so V5 multi-agents schedulers and the
# future web dashboard can read a single source of truth.

# -----------------------------------------------------------------------------
# Brain orchestrator timing
# -----------------------------------------------------------------------------

# How often (seconds) the brain checks the clan chat for commands.
CHAT_CHECK_INTERVAL = 45

# Minimum / maximum delay between farm attacks. Brain picks a random value
# in this range to look less bot-like.
IDLE_BETWEEN_ATTACKS = 20
IDLE_BETWEEN_ATTACKS_MAX = 60

# How many farm attacks happen between two clan-chat checks.
ATTACKS_BEFORE_CHAT_CHECK = 2


# -----------------------------------------------------------------------------
# Agent priorities (higher = preempts lower)
# -----------------------------------------------------------------------------
# Used by the upcoming AgentScheduler (V5.1+). Listed in decreasing order
# of urgency.

PRIORITY_GDC_COMMAND = 100   # explicit clan war commands from chat
PRIORITY_FARM_ATTACK = 10    # normal farming
PRIORITY_IDLE = 0            # nothing to do


# -----------------------------------------------------------------------------
# Default bot identity (read from clan chat)
# -----------------------------------------------------------------------------
DEFAULT_BOT_NAME = 'mini_pekka'


# -----------------------------------------------------------------------------
# Clan castle (request troops)
# -----------------------------------------------------------------------------

# Time (seconds) between two automatic troop requests. CoC throttles
# requests so spamming would do nothing.
REQUEST_COOLDOWN = 15 * 60   # 15 minutes


# -----------------------------------------------------------------------------
# Clan chat monitor
# -----------------------------------------------------------------------------

# How often (seconds) to poll the chat for new commands when running.
MONITOR_INTERVAL = 30.0

# Commands older than this are discarded (avoids replaying stale orders
# after a long idle).
MAX_COMMAND_AGE_MINUTES = 10

# Rolling history of last-seen messages (for dedup).
MAX_HISTORY = 20
