"""Flux multi-agents complet : commande chat -> file GdC -> routage scheduler.

Reflete la demo offline de agents/chat_agent.py (bloc __main__, laisse intact).

C'est le chemin qui deviendra le canal d'entree du LocalLLMBrain (V5.4) : une
instruction en langage naturel arrive par le chat de clan et se transforme en
action d'un autre agent.
"""

import time

from conftest import FakeChatMonitor, no_hardware

from clashai.agents.chat_agent import ChatAgent
from clashai.agents.combat_agent import CombatAgent
from clashai.agents.gdc_agent import GdCAgent


def _wire(commands):
    """Monte scheduler + 3 agents, le chat cable sur la file du GdC."""
    from clashai.agents.scheduler import AgentScheduler

    sched = AgentScheduler()
    gdc = no_hardware(GdCAgent(models=None))
    combat = no_hardware(CombatAgent(models=None))
    monitor = FakeChatMonitor(commands)
    chat = ChatAgent(
        monitor=monitor,
        on_attack=gdc.enqueue_target,
        screenshot_fn=lambda: object(),          # non-None pour que check_once tourne
        classify_screen_fn=lambda *a, **k: ('village_home', 1.0),
        cooldown_seconds=0.0,
    )
    for agent in (chat, gdc, combat):
        sched.register(agent)
    return sched, chat, gdc, monitor


def test_chat_has_the_highest_priority(village_auto):
    sched, _, _, _ = _wire([{'type': 'attack', 'target': 5}])
    assert sched.pick(village_auto).name == 'chat'


def test_attack_command_lands_in_the_gdc_queue(village_auto):
    sched, _, gdc, _ = _wire([{'type': 'attack', 'target': 5}])

    result = sched.run(sched.pick(village_auto))

    assert result.ok
    assert result.data['commands']
    assert gdc.pending() == [5]


def test_gdc_takes_over_once_chat_is_on_cooldown(village_auto):
    sched, chat, gdc, _ = _wire([{'type': 'attack', 'target': 5}])
    sched.run(sched.pick(village_auto))

    # On force le cooldown du chat pour observer le pick suivant.
    chat._last_run_at = time.time()
    chat.cooldown_seconds = 999

    assert sched.pick(village_auto).name == 'gdc'
    assert gdc.pending() == [5]


def test_no_command_leaves_the_queue_empty(village_auto):
    sched, _, gdc, _ = _wire([])

    result = sched.run(sched.pick(village_auto))

    assert result.ok
    assert gdc.pending() == []
