"""Fakes partages par les tests d'agents.

Aucun de ces doubles ne touche a ADB, au GPU ou a un fichier de poids : les agents
recoivent leurs dependances par injection (callables / manager), ce qui rend le
scheduler testable tel quel.
"""

import pytest

from clashai.agents.base import AgentResult
from clashai.agents.scheduler import AgentScheduler


def no_hardware(agent, **data):
    """Neutralise le run() d'un agent qui toucherait ADB/GPU.

    Indispensable, et pas seulement par prudence : `CombatAgent.run()` et
    `GdCAgent.run()` lancent un vrai episode d'attaque et **bloquent** en
    attendant l'emulateur. Sans ce garde-fou, une regression de priorite ne fait
    pas echouer la suite -- elle la fige (constate en mutant combat.priority
    10 -> 99 : pytest a tourne 2 minutes sans rendre la main).

    On ne remplace QUE run(). `priority`, `cooldown_seconds` et `can_run()`
    restent ceux de la vraie classe, donc c'est bien le vrai ordonnancement
    qui est teste.
    """
    agent.run = lambda: AgentResult(ok=True, duration_s=0.0, data=dict(data))
    return agent


class FakeClanCastleManager:
    """Double de ClanCastleManager — pas d'ADB, pas de YOLO.

    `time_until_next_request()` pilote le cooldown vu par ClanCastleAgent ;
    `request_if_needed()` simule la demande puis pose 15 min de cooldown.
    """

    def __init__(self, next_request_in=0.0):
        self._next = next_request_in
        self.calls = 0

    def time_until_next_request(self):
        return self._next

    def request_if_needed(self, screenshot_fn, tap_fn):
        self.calls += 1
        self._next = 900.0


class FakeChatMonitor:
    """Double de ClanChatMonitor — rejoue une liste de commandes figee."""

    def __init__(self, commands):
        self._commands = commands
        self.replies = []

    def open_chat(self, classify_fn, models):
        return True

    def check_once(self, img):
        return self._commands

    def close_chat(self):
        pass

    def send_chat_message(self, msg):
        self.replies.append(msg)


@pytest.fixture
def scheduler():
    return AgentScheduler()


@pytest.fixture
def fake_cc():
    return FakeClanCastleManager()


@pytest.fixture
def village_farm():
    """World minimal : au village, mode farm."""
    return {'on_village_home': True, 'mode': 'farm'}


@pytest.fixture
def village_auto():
    """World minimal : au village, mode auto (tous les agents eligibles)."""
    return {'on_village_home': True, 'mode': 'auto'}
