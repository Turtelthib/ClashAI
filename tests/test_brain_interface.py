"""Le seam Brain : HeuristicBrain.decide(world) -> agent.

Reflete la demo offline de brain/interface.py (bloc __main__, laisse intact).

Ces tests verrouillent le contrat que devra respecter LocalLLMBrain (V5.3) :
meme entree (le dict world), meme sortie (un BaseAgent eligible ou None).
"""

from conftest import no_hardware

from clashai.agents import AgentScheduler, ClanCastleAgent, CombatAgent, GdCAgent
from clashai.brain.interface import Brain, HeuristicBrain


def _brain_with_three_agents(fake_cc):
    sched = AgentScheduler()
    combat = no_hardware(CombatAgent(models=None))
    gdc = no_hardware(GdCAgent(models=None))
    cc = ClanCastleAgent(
        manager=fake_cc,
        screenshot_fn=lambda: None,
        tap_fn=lambda *a, **k: None,
    )
    for agent in (combat, gdc, cc):
        sched.register(agent)
    return HeuristicBrain(sched), gdc


def test_heuristic_brain_is_a_brain():
    assert issubclass(HeuristicBrain, Brain)


def test_clan_castle_wins_when_no_war_target(fake_cc, village_auto):
    brain, _ = _brain_with_three_agents(fake_cc)
    assert brain.decide(village_auto).name == 'clan_castle'


def test_war_target_outranks_clan_castle(fake_cc, village_auto):
    brain, gdc = _brain_with_three_agents(fake_cc)
    gdc.enqueue_target(5)
    assert brain.decide(village_auto).name == 'gdc'


def test_farm_mode_away_from_village_leaves_only_combat(fake_cc):
    brain, _ = _brain_with_three_agents(fake_cc)
    decided = brain.decide({'mode': 'farm', 'on_village_home': False})
    assert decided.name == 'combat'


def test_decide_returns_none_when_nothing_is_eligible(fake_cc):
    """Un world vide ne doit pas lever : le Brain rend None, la boucle attend."""
    fake_cc._next = 900.0
    brain, _ = _brain_with_three_agents(fake_cc)
    assert brain.decide({'on_village_home': True, 'mode': 'gdc'}) is None
