"""Priorites, cooldowns et gating par mode du AgentScheduler.

Reflete les demos offline de agents/clan_castle_agent.py, agents/combat_agent.py
et agents/gdc_agent.py (blocs __main__, laisses intacts).

Rappel des priorites declarees : chat 30 > gdc 25 > clan_castle 20 > combat 10.
"""

from conftest import no_hardware

from clashai.agents.clan_castle_agent import ClanCastleAgent
from clashai.agents.combat_agent import CombatAgent
from clashai.agents.gdc_agent import GdCAgent


def _cc_agent(manager):
    return ClanCastleAgent(
        manager=manager,
        screenshot_fn=lambda: None,
        tap_fn=lambda *a, **k: None,
    )


# ---------------------------------------------------------------------------
# ClanCastleAgent : le cycle world -> can_run -> pick -> run
# ---------------------------------------------------------------------------

def test_cc_not_picked_when_away_from_village(scheduler, fake_cc):
    scheduler.register(_cc_agent(fake_cc))
    assert scheduler.pick({'on_village_home': False}) is None


def test_cc_picked_when_at_village_and_off_cooldown(scheduler, fake_cc):
    agent = _cc_agent(fake_cc)
    scheduler.register(agent)
    assert scheduler.pick({'on_village_home': True}) is agent


def test_cc_run_fires_the_request_then_goes_on_cooldown(scheduler, fake_cc):
    agent = _cc_agent(fake_cc)
    scheduler.register(agent)

    result = scheduler.run(scheduler.pick({'on_village_home': True}))

    assert result.ok
    assert fake_cc.calls == 1
    # Le manager a pose son cooldown -> l'agent n'est plus eligible.
    assert scheduler.pick({'on_village_home': True}) is None


# ---------------------------------------------------------------------------
# Priorite et gating par mode
# ---------------------------------------------------------------------------

def test_clan_castle_preempts_combat(scheduler, fake_cc, village_farm):
    scheduler.register(_cc_agent(fake_cc))
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))

    assert scheduler.pick(village_farm).name == 'clan_castle'


def test_combat_is_the_default_once_clan_castle_is_on_cooldown(
    scheduler, fake_cc, village_farm
):
    fake_cc._next = 900.0
    scheduler.register(_cc_agent(fake_cc))
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))

    assert scheduler.pick(village_farm).name == 'combat'


def test_gdc_mode_gates_combat_off(scheduler, fake_cc):
    fake_cc._next = 900.0
    scheduler.register(_cc_agent(fake_cc))
    scheduler.register(no_hardware(CombatAgent(models=None, use_heuristic=True)))

    # combat est bride en mode gdc, le chateau est en cooldown -> rien a faire.
    assert scheduler.pick({'on_village_home': True, 'mode': 'gdc'}) is None


# ---------------------------------------------------------------------------
# GdCAgent : la file de cibles de guerre
# ---------------------------------------------------------------------------

def test_combat_runs_while_no_war_target_is_queued(scheduler, village_auto):
    scheduler.register(no_hardware(GdCAgent(models=None)))
    scheduler.register(no_hardware(CombatAgent(models=None)))

    assert scheduler.pick(village_auto).name == 'combat'


def test_queued_war_target_preempts_combat(scheduler, village_auto):
    gdc = no_hardware(GdCAgent(models=None))
    scheduler.register(gdc)
    scheduler.register(no_hardware(CombatAgent(models=None)))

    assert gdc.enqueue_target(5)
    assert scheduler.pick(village_auto).name == 'gdc'
    assert gdc.pending() == [5]


def test_enqueue_rejects_duplicates_and_out_of_range(scheduler):
    gdc = no_hardware(GdCAgent(models=None))

    assert gdc.enqueue_target(5)
    assert not gdc.enqueue_target(5), "un doublon doit etre ignore"
    assert not gdc.enqueue_target(99), "hors bornes (1-50) doit etre ignore"
    assert gdc.pending() == [5]


def test_farm_mode_gates_gdc_off_even_with_a_target(scheduler):
    gdc = no_hardware(GdCAgent(models=None))
    scheduler.register(gdc)
    scheduler.register(no_hardware(CombatAgent(models=None)))
    gdc.enqueue_target(5)

    picked = scheduler.pick({'on_village_home': True, 'mode': 'farm'})
    assert picked.name == 'combat'


# ---------------------------------------------------------------------------
# status() : le contrat lu par le futur dashboard
# ---------------------------------------------------------------------------

def test_status_is_json_safe(scheduler, fake_cc, village_auto):
    scheduler.register(_cc_agent(fake_cc))
    scheduler.register(no_hardware(CombatAgent(models=None)))

    import json
    json.dumps(scheduler.status())  # ne doit pas lever
