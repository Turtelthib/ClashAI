# clashai/brain/app.py
# ClashBrain — assembles the domain mixins into the single AI brain + CLI main().

import argparse

from clashai.brain.core import BrainCoreMixin
from clashai.brain.loop import BrainLoopMixin
from clashai.brain.navigation import BrainNavigationMixin
from clashai.config import DEFAULT_BOT_NAME


class ClashBrain(
    BrainCoreMixin,
    BrainLoopMixin,
    BrainNavigationMixin,
):
    """
    The single brain of ClashAI.

    V5.1: the brain is an orchestrator over the AgentScheduler. The actual
    "doing" (farm, war, chat, clan castle) lives in the agents
    (clashai/agents/), not here. The brain only:
      - core      : lifecycle + loads models/agents/scheduler/Brain
      - loop       : tick = world → Brain.decide → scheduler.run → stats
      - navigation : ensure-at-village recovery + human-like pauses

    The farm/war/chat mixins were removed in V5.1 Étape B (superseded by
    CombatAgent / GdCAgent / ChatAgent / ClanCastleAgent).
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
    parser.add_argument(
        '--no-llm', action='store_true',
        help="Désactive le cerveau LLM et force l'heuristique. Par défaut le "
             "LLM est actif (c'est le but du projet) ; il retombe de toute "
             "façon sur l'heuristique si Ollama n'est pas là."
    )
    parser.add_argument(
        '--llm-model', type=str, default=None,
        help="Modèle Ollama (défaut: mistral)"
    )

    args = parser.parse_args()

    brain = ClashBrain(
        mode=args.mode,
        bot_name=args.bot_name,
        verbose=not args.quiet,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
    )
    brain.start(max_episodes=args.episodes)


if __name__ == "__main__":
    main()
