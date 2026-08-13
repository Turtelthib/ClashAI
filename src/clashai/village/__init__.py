# clashai/village/ — gestion du village (V5.2).
#
# Logique métier "village" pilotée par l'agent règles VillageAgent :
#   - collector.py : récolte des ressources (or / élixir / élixir noir) via le
#     CNN UI (classes recolter_*). Increment 1.
#   - (à venir) upgrades.py : file d'amélioration (murs → défenses → ressources),
#     labo. Increments suivants.
#
# Découplé de l'agent (agents/village_agent.py) comme social/clan_castle l'est de
# ClanCastleAgent : le wrapper gère l'ordonnancement, ce package fait le travail.

from clashai.village.collector import VillageCollector

__all__ = ['VillageCollector']
