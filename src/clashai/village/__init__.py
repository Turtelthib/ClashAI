# clashai/village/ — gestion du village (V5.2).
#
# Logique métier "village" pilotée par l'agent règles VillageAgent :
#   - collector.py : récolte des ressources (or / élixir / élixir noir) via le
#     CNN UI (classes recolter_*). Increment 1.
#   - upgrader.py  : amélioration d'un bâtiment (gating ouvriers → confirmation
#     → affordabilité), avec le garde-fou anti-gemmes partagé. Increment 2.
#   - lab.py       : recherches au laboratoire, réutilise `confirm_step`. Incr. 3.
#
# Découplé de l'agent (agents/village_agent.py) comme social/clan_castle l'est de
# ClanCastleAgent : le wrapper gère l'ordonnancement, ce package fait le travail.

from clashai.village.collector import VillageCollector
from clashai.village.lab import LabCandidate, VillageLab
from clashai.village.upgrader import UpgradeResult, VillageUpgrader

__all__ = ['VillageCollector', 'VillageUpgrader', 'UpgradeResult',
           'VillageLab', 'LabCandidate']
