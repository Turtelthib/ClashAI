# clashai/brain/
# ClashAI Brain — the single brain of the AI player (Phase 3 split of brain.py).
#
# One program, one player, one account. The Brain decides what to do at every
# moment like a human looking at their phone: check chat, farm, run CW commands.
#
# Implementation split into domain mixins:
#   core.py       — lifecycle: init, load_modules, start, shutdown
#   loop.py       — main decision loop + task dispatch + chat-check timing
#   farm.py       — farm attacks + CC troop request + attack-episode runner
#   war.py        — Clan War attacks on a target
#   chat.py       — clan chat acknowledgements + command polling
#   navigation.py — return-to-village + human-like pauses
#   app.py        — assembled ClashBrain class + main()
#
# Entry point `clashai-brain = "clashai.brain:main"` resolves to app.main.

from clashai.brain.app import ClashBrain, main

__all__ = ['ClashBrain', 'main']
