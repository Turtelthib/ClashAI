# clashai/agents/village_agent.py
# VillageAgent — agent règles de gestion du village (V5.2, increment 1).
#
# Increment 1 : RÉCOLTE des ressources via le CNN UI (VillageCollector).
# À venir : file d'amélioration (murs → défenses → ressources) + labo.
#
# Pattern identique à ClanCastleAgent : le wrapper gère l'ordonnancement
# (priorité + cooldown + can_run), la logique vit dans clashai/village/.

import time

from clashai.agents.base import AgentResult, BaseAgent


class VillageAgent(BaseAgent):
    """
    Gère le village quand le bot est oisif à la maison.

    can_run : on est sur village_home.
    run     : récolte les ressources visibles (VillageCollector).
    """

    name = 'village'
    # Entre clan_castle (20) et combat (10) : la collecte passe avant une attaque
    # quand son cooldown est écoulé, mais reste sous la demande de renfort.
    priority = 15
    # Les collecteurs se remplissent lentement — inutile de scruter en boucle.
    # Le cooldown rend le sol à CombatAgent (prio 10) entre deux récoltes.
    cooldown_seconds = 5 * 60

    def __init__(self, collector=None, screenshot_fn=None, tap_fn=None,
                 verbose=True, **kwargs):
        super().__init__(**kwargs)
        if collector is None:
            from clashai.village.collector import VillageCollector
            collector = VillageCollector(verbose=verbose)
        self._collector = collector
        self._screenshot_fn = screenshot_fn
        self._tap_fn = tap_fn

    def _io(self):
        """Résout les I/O ADB canoniques (screenshot routé WGC) en lazy."""
        if self._screenshot_fn is None or self._tap_fn is None:
            from clashai.navigation import game_loop as gl
            self._screenshot_fn = self._screenshot_fn or gl.adb_screenshot
            self._tap_fn = self._tap_fn or gl.adb_tap
        return self._screenshot_fn, self._tap_fn

    def can_run(self, world):
        return world.get('on_village_home', False)

    def run(self):
        start = time.time()
        screenshot_fn, tap_fn = self._io()
        collected = self._collector.collect(screenshot_fn, tap_fn)
        return AgentResult(
            ok=True,
            duration_s=time.time() - start,
            data={'collected': collected},
        )


# =============================================================================
# Démo hors-ligne / smoke test (sans émulateur ni modèle)
# =============================================================================

if __name__ == "__main__":
    from clashai.agents.scheduler import AgentScheduler

    class _FakeDetector:
        """Détecteur bidon : 2 or + 1 élixir à récolter."""
        def detect_raw(self, img):
            from clashai.perception.ui_detector import Detection
            mk = lambda name, x: Detection(name, 0, x, 400, 40, 40, 0.9)
            return {
                'recolter_or': [mk('recolter_or', 300), mk('recolter_or', 600)],
                'recolter_elixir': [mk('recolter_elixir', 900)],
            }

    from clashai.village.collector import VillageCollector

    taps = []
    collector = VillageCollector(detector=_FakeDetector(), verbose=True)
    agent = VillageAgent(
        collector=collector,
        screenshot_fn=lambda: object(),         # frame non-None
        tap_fn=lambda x, y: taps.append((x, y)),
    )

    print("VillageAgent offline demo\n")
    sched = AgentScheduler()
    sched.register(agent)

    # 1. Pas au village → pas choisi
    picked = sched.pick({'on_village_home': False})
    print(f"1. pas au village   -> picked={picked}")
    assert picked is None

    # 2. Au village + cooldown prêt → choisi
    picked = sched.pick({'on_village_home': True})
    print(f"2. village + prêt   -> picked={picked.name if picked else None}")
    assert picked is agent

    # 3. Run → récolte les 3 ressources
    result = sched.run(picked)
    print(f"3. run              -> ok={result.ok} data={result.data} taps={taps}")
    assert result.ok
    assert result.data['collected'] == 3
    assert taps == [(300, 400), (600, 400), (900, 400)]

    # 4. Cooldown → plus choisi
    picked = sched.pick({'on_village_home': True})
    print(f"4. village + cd     -> picked={picked}")
    assert picked is None

    print("\nOffline demo OK — récolte 3 ressources, cooldown respecté")
