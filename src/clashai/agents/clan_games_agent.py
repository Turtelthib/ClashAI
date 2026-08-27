# clashai/agents/clan_games_agent.py
# ClanGamesAgent — choisit les défis des jeux de clan (V5.2).
#
# L'agent ne JOUE pas les défis : la progression s'incrémente pendant que
# CombatAgent farme. Son travail est de SÉLECTIONNER le bon défi, puis de
# laisser le bot vivre sa vie.
#
# Conséquence directe sur l'ordonnancement : il doit tourner **rarement** et
# **céder le sol** à l'attaque. Un défi engagé n'a plus besoin de lui jusqu'à ce
# qu'il soit terminé — d'où le long cooldown.
#
# Sûr par défaut : `confirmer=False` fait tout le travail sauf le tap final.
# Engager un défi est irréversible (le rejeter a une pénalité en jeu), donc
# comme VillageUpgrader et VillageLab, il faut le demander explicitement.

import time

from clashai.agents.base import AgentResult, BaseAgent
from clashai.clan_games.reader import RACCOURCI, TENTE


class ClanGamesAgent(BaseAgent):
    """
    Choisit un défi des jeux de clan — événement à DURÉE LIMITÉE, les points non
    gagnés sont perdus ; ne coûte ni troupes ni ressources, et le défi progresse
    ensuite tout seul pendant les attaques.

    ⚠️ La première ligne de ce docstring est **ce que le LLM lit** pour arbitrer
    (`llm_brain.build_prompt`). Vécu : elle disait « Engage le meilleur défi des
    jeux de clan quand le bot est au village » — vrai, mais elle ne disait pas
    l'ENJEU, et le modèle a préféré `combat` (« utiliser l'élixir pour gagner
    des ressources »). Sans le caractère limité dans le temps ni la gratuité, il
    n'avait aucune raison de choisir autre chose. Une description d'agent est
    une interface pour le modèle, pas un résumé pour l'humain.

    can_run : au village ET une entrée des jeux est visible (leur présence EST
              le signal « les jeux sont actifs » — hors période, le jeu
              n'affiche ni raccourci ni tente).
    run     : ouvrir -> croiser -> choisir -> engager (si confirmer).
    """

    name = 'clan_games'
    # Entre village (15) et clan_castle (20). Passe avant l'attaque quand son
    # cooldown est écoulé — choisir un défi AVANT d'attaquer vaut mieux
    # qu'après, puisque l'attaque est justement ce qui le fera progresser.
    priority = 16
    # Long : une fois un défi engagé, il n'y a plus rien à décider avant qu'il
    # soit fini. Le cooldown rend le sol à CombatAgent, qui est ce qui fait
    # AVANCER le défi. Trop court, l'agent rouvrirait le menu pour rien.
    cooldown_seconds = 30 * 60

    def __init__(self, selector=None, screenshot_fn=None, tap_fn=None,
                 confirmer=False, verbose=True, **kwargs):
        super().__init__(**kwargs)
        if selector is None:
            from clashai.clan_games.selector import ClanGamesSelector
            selector = ClanGamesSelector(verbose=verbose)
        self._selector = selector
        self._screenshot_fn = screenshot_fn
        self._tap_fn = tap_fn
        self._confirmer = confirmer
        # Mémorisé entre deux runs : une fois le plafond atteint, plus rien à
        # gagner de la semaine. Inutile de rouvrir le menu toutes les 30 min.
        self._plafond_atteint = False
        self._world = {}

    def _io(self):
        """Résout les I/O ADB canoniques (screenshot routé WGC) en lazy."""
        if self._screenshot_fn is None or self._tap_fn is None:
            from clashai.navigation import game_loop as gl
            self._screenshot_fn = self._screenshot_fn or gl.adb_screenshot
            self._tap_fn = self._tap_fn or gl.adb_tap
        return self._screenshot_fn, self._tap_fn

    def can_run(self, world):
        if self._plafond_atteint:
            return False
        if not world.get('on_village_home', False):
            return False

        # Les jeux sont-ils actifs ? On lit le cache du `world` (aucune I/O
        # ici : can_run doit rester gratuit). Le RACCOURCI est le signal
        # fiable — la tente n'apparaît que si la caméra est zoomée dessus,
        # donc son absence ne prouve rien.
        boutons = world.get('buttons') or {}
        self._world = world
        return RACCOURCI in boutons or TENTE in boutons

    def run(self):
        start = time.time()
        screenshot_fn, tap_fn = self._io()

        from clashai.clan_games.catalog import capacites_depuis_world
        capacites = capacites_depuis_world(self._world)

        statut, details = self._selector.engager_le_meilleur(
            screenshot_fn, tap_fn, capacites, confirmer=self._confirmer,
        )

        if statut == 'plafond_atteint':
            self._plafond_atteint = True

        # `aucun_defi_sur` n'est PAS un échec : c'est le refus voulu quand rien
        # n'est certain. Un run qui ne trouve rien de sûr a fait son travail.
        # `engagement_non_confirme`, LUI, en est un : on a tapé sans pouvoir
        # constater le résultat, et ça doit remonter comme tel.
        ok = statut in ('ok', 'choisi_non_engage', 'deja_engage',
                        'plafond_atteint', 'aucun_defi_sur')

        return AgentResult(
            ok=ok,
            duration_s=time.time() - start,
            data={'statut': statut, **details},
            error=None if ok else statut,
        )
