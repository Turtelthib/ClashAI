"""File d'instructions entre la console et la boucle du bot (V5.3, etape 5.3.3a).

L'invariant structurant : un seul thread tape sur le jeu. La console DEPOSE,
la boucle RETIRE. Et les garde-fous du registre sont reappliques a l'execution,
pas seulement a la saisie.
"""

import threading

import pytest

from clashai.brain.commands import Command, CommandQueue, run_next
from clashai.brain.tools import SOURCE_ADMIN, SOURCE_CLAN, Tool, ToolRegistry


def _registry(calls):
    reg = ToolRegistry()
    reg.register(Tool('etat', 'lit', fn=lambda: calls.append('etat') or {'ok': 1},
                      parameters={'type': 'object', 'properties': {}}))
    reg.register(Tool('recolter', 'agit', fn=lambda: calls.append('recolter') or {},
                      parameters={'type': 'object', 'properties': {}}, acts=True))
    reg.register(Tool('attaquer', 'agit, depense',
                      fn=lambda: calls.append('attaquer') or {},
                      parameters={'type': 'object', 'properties': {}},
                      acts=True, spends=True))
    return reg


# ---------------------------------------------------------------------------
# Depot et ordre
# ---------------------------------------------------------------------------

def test_put_returns_a_numbered_command():
    q = CommandQueue()
    first, second = q.put('etat'), q.put('etat')
    assert isinstance(first, Command)
    assert (first.id, second.id) == (1, 2)
    assert first.created_at > 0


def test_fifo_within_a_source():
    q = CommandQueue()
    q.put('a'), q.put('b'), q.put('c')
    assert [q.pop_next().tool for _ in range(3)] == ['a', 'b', 'c']


def test_admin_goes_before_clan_even_if_it_arrived_later():
    """L'admin passe devant le chat de clan quand les deux attendent."""
    q = CommandQueue()
    q.put('clan_1', source=SOURCE_CLAN)
    q.put('admin_1', source=SOURCE_ADMIN)
    q.put('clan_2', source=SOURCE_CLAN)
    assert [q.pop_next().tool for _ in range(3)] == ['admin_1', 'clan_1', 'clan_2']


def test_pending_lists_commands_in_execution_order():
    q = CommandQueue()
    q.put('clan', source=SOURCE_CLAN)
    q.put('admin', source=SOURCE_ADMIN)
    assert [c.tool for c in q.pending()] == ['admin', 'clan']
    assert [c.tool for c in q.pending(SOURCE_CLAN)] == ['clan']


def test_an_empty_queue_pops_none():
    assert CommandQueue().pop_next() is None


def test_len_counts_every_source():
    q = CommandQueue()
    q.put('a'), q.put('b', source=SOURCE_CLAN)
    assert len(q) == 2


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError):
        CommandQueue().put('etat', source='inconnu')


def test_args_are_copied_not_shared():
    """Modifier le dict d'origine apres coup ne doit pas changer la commande."""
    args = {'troupes': ['ballon']}
    cmd = CommandQueue().put('donner', args)
    args['troupes'] = ['golem']
    assert cmd.args == {'troupes': ['ballon']}


# ---------------------------------------------------------------------------
# Annulation
# ---------------------------------------------------------------------------

def test_cancel_empties_only_the_given_source():
    q = CommandQueue()
    q.put('a'), q.put('b'), q.put('c', source=SOURCE_CLAN)
    assert q.cancel(SOURCE_ADMIN) == 2
    assert [c.tool for c in q.pending()] == ['c']


def test_cancel_on_an_empty_queue_returns_zero():
    assert CommandQueue().cancel() == 0


# ---------------------------------------------------------------------------
# Execution par la boucle du bot
# ---------------------------------------------------------------------------

def test_run_next_on_an_empty_queue_does_nothing():
    calls = []
    assert run_next(CommandQueue(), _registry(calls)) is None
    assert calls == []


def test_run_next_executes_one_command_and_reports_it():
    calls = []
    q = CommandQueue()
    q.put('recolter'), q.put('etat')
    cmd, result = run_next(q, _registry(calls))
    assert cmd.tool == 'recolter' and result.ok
    assert calls == ['recolter']          # UNE seule commande
    assert len(q) == 1


def test_a_confirmed_spending_command_runs():
    calls = []
    q = CommandQueue()
    q.put('attaquer', confirmed=True)
    _cmd, result = run_next(q, _registry(calls))
    assert result.ok and calls == ['attaquer']


# ---------------------------------------------------------------------------
# INVARIANT : les garde-fous sont reappliques A L'EXECUTION
# ---------------------------------------------------------------------------

def test_an_unconfirmed_spending_command_is_refused_at_execution():
    calls = []
    q = CommandQueue()
    q.put('attaquer', confirmed=False)
    _cmd, result = run_next(q, _registry(calls))
    assert not result.ok and 'confirmation' in result.error
    assert calls == []                    # jamais execute


def test_a_clan_action_that_slipped_into_the_queue_is_refused():
    """La console ne met pas en file une action venue du clan ; mais si ca
    arrivait, l'execution la refuserait quand meme."""
    calls = []
    q = CommandQueue()
    q.put('recolter', source=SOURCE_CLAN)
    _cmd, result = run_next(q, _registry(calls))
    assert not result.ok
    assert calls == []


# ---------------------------------------------------------------------------
# Issue et ecouteur
# ---------------------------------------------------------------------------

def test_completion_is_recorded_and_listener_called():
    seen = []
    calls = []
    q = CommandQueue(listener=lambda cmd, res: seen.append((cmd.tool, res.ok)))
    q.put('etat')
    run_next(q, _registry(calls))
    assert seen == [('etat', True)]
    assert [c.tool for c, _r in q.done] == ['etat']


def test_a_crashing_listener_does_not_break_the_bot_loop():
    def boom(cmd, res):
        raise RuntimeError('affichage casse')

    calls = []
    q = CommandQueue(listener=boom)
    q.put('etat')
    assert run_next(q, _registry(calls)) is not None   # ne leve pas
    assert len(q.done) == 1


def test_the_done_history_is_capped():
    calls = []
    q = CommandQueue(history_size=3)
    reg = _registry(calls)
    for _ in range(10):
        q.put('etat')
        run_next(q, reg)
    assert len(q.done) == 3


def test_set_listener_replaces_the_listener():
    seen = []
    q = CommandQueue()
    q.set_listener(lambda cmd, res: seen.append(cmd.id))
    q.put('etat')
    run_next(q, _registry([]))
    assert seen == [1]


# ---------------------------------------------------------------------------
# Concurrence : la console et la boucle sont deux threads
# ---------------------------------------------------------------------------

def test_concurrent_puts_lose_nothing_and_never_reuse_an_id():
    q = CommandQueue()

    def worker():
        for _ in range(200):
            q.put('etat')

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [c.id for c in q.pending()]
    assert len(ids) == 1600 and len(set(ids)) == 1600


def test_concurrent_pop_never_hands_the_same_command_twice():
    q = CommandQueue()
    for _ in range(1000):
        q.put('etat')
    taken, lock = [], threading.Lock()

    def worker():
        while True:
            cmd = q.pop_next()
            if cmd is None:
                return
            with lock:
                taken.append(cmd.id)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 1000 and len(set(taken)) == 1000
