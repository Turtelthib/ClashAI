# clashai/clan_games/ — agent jeux de clan (V5.2).
#
# L'agent ne JOUE pas les défis, il les CHOISIT : la progression s'incrémente
# toute seule pendant que CombatAgent farme.
#
#   reader.py   perception pure  : écran -> Defi structurés (aucun tap)
#   catalog.py  logique pure     : description -> sémantique + scoring  (à venir)
#   selector.py geste            : ouvrir -> choisir -> Commencer        (à venir)

from clashai.clan_games.reader import ClanGamesReader, Defi

__all__ = ['ClanGamesReader', 'Defi']
