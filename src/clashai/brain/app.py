# clashai/brain/app.py
# ClashBrain — assembles the domain mixins into the single AI brain + CLI main().

import argparse

from clashai.config import DEFAULT_BOT_NAME
from clashai.brain.core import BrainCoreMixin
from clashai.brain.loop import BrainLoopMixin
from clashai.brain.farm import BrainFarmMixin
from clashai.brain.war import BrainWarMixin
from clashai.brain.chat import BrainChatMixin
from clashai.brain.navigation import BrainNavigationMixin


class ClashBrain(
    BrainCoreMixin,
    BrainLoopMixin,
    BrainFarmMixin,
    BrainWarMixin,
    BrainChatMixin,
    BrainNavigationMixin,
):
    """
    The single brain of ClashAI.

    Manages the whole account like a real player:
    - Farm attacks (multiplayer, for resources)
    - CW commands (from clan chat)
    - Human-like behavior (random pauses, zooms)
    - Robust navigation (always knows how to return to the village)

    Future: village management (upgrades, donations, harvesting)

    Implementation is split across domain mixins (Phase 3):
      core / loop / farm / war / chat / navigation.
    """


def main():
    """Console entry point (`clashai-brain` / `python -m clashai.brain`)."""
    parser = argparse.ArgumentParser(
        description="ClashAI Brain — IA autonome pour Clash of Clans"
    )
    parser.add_argument(
        '--mode', type=str, default='auto',
        choices=['farm', 'gdc', 'auto'],
        help="farm=attaques multi, gdc=attend commandes clan, auto=tout"
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help="Nombre max d'attaques farm (défaut: infini)"
    )
    parser.add_argument(
        '--bot-name', type=str, default=DEFAULT_BOT_NAME,
        help="Nom du bot pour les commandes clan"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Moins de logs"
    )

    args = parser.parse_args()

    brain = ClashBrain(
        mode=args.mode,
        bot_name=args.bot_name,
        verbose=not args.quiet,
    )
    brain.start(max_episodes=args.episodes)


if __name__ == "__main__":
    main()
