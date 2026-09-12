"""Console operateur : une ligne tapee -> un appel d'outil (V5.3, etape 5.3.3a).

Aucun terminal : saisie, confirmation, discussion et « ce que fait le bot » sont
injectes. Les regles testees :
  - LECTURE -> tout de suite ;  ACTION -> en file ;  DEPENSE -> o/n d'abord ;
  - les noms de troupes sont refuses A LA SAISIE, rien n'est mis en file ;
  - `handle` ne leve jamais.
"""

from clashai.brain.commands import CommandQueue, run_next
from clashai.brain.console import (
    SLASH_TOOLS,
    ConsoleSession,
    format_completion,
    format_result,
)
from clashai.brain.tools import SOURCE_ADMIN, Tool, ToolRegistry, ToolResult

KNOWN = frozenset({'ballon', 'yeti', 'dragon', 'bebe_dragon', 'golem_glace'})
_NO = {'type': 'object', 'properties': {}}


def _setup(answers=(), chat=None, busy=None, confirm_spending=True,
           registry_extra=True):
    calls = []
    reg = ToolRegistry()
    reg.register(Tool('etat_du_village', 'lit',
                      fn=lambda: {'ecran': 'village_home', 'ressources': None},
                      parameters=dict(_NO)))
    if registry_extra:
        reg.register(Tool('recolter_ressources', 'agit',
                          fn=lambda: calls.append('recolte') or {},
                          parameters=dict(_NO), acts=True))
        reg.register(Tool('lancer_attaque', 'agit, depense',
                          fn=lambda: calls.append('attaque') or {},
                          parameters=dict(_NO), acts=True, spends=True))
        reg.register(Tool('donner_troupes', 'agit',
                          fn=lambda troupes=None: calls.append(('dons', troupes)) or {},
                          parameters={'type': 'object',
                                      'properties': {'troupes': {'type': 'array'}}},
                          acts=True))
        reg.register(Tool('lancer_recherche', 'agit, depense',
                          fn=lambda troupe=None: calls.append(('labo', troupe)) or {},
                          parameters={'type': 'object',
                                      'properties': {'troupe': {'type': 'string'}}},
                          acts=True, spends=True))
    queue = CommandQueue()
    asked = []
    pending_answers = list(answers)

    def ask(prompt):
        asked.append(prompt)
        return pending_answers.pop(0) if pending_answers else False

    session = ConsoleSession(reg, queue, chat_fn=chat, ask_yes_no=ask,
                             busy_fn=lambda: busy,
                             confirm_spending=confirm_spending,
                             known_troops=KNOWN)
    return session, queue, reg, calls, asked


# ---------------------------------------------------------------------------
# Lecture : tout de suite
# ---------------------------------------------------------------------------

def test_a_read_tool_runs_immediately_and_is_not_queued():
    s, q, _reg, _calls, _asked = _setup()
    out = s.handle('/etat')
    assert out[0] == '[ok] etat_du_village'
    assert any('village_home' in line for line in out)
    assert len(q) == 0


def test_an_unreadable_value_is_shown_as_such_never_as_zero():
    s, *_ = _setup()
    assert any("non lisible d'ici" in line for line in s.handle('/etat'))


# ---------------------------------------------------------------------------
# Action : en file, jamais executee par la console
# ---------------------------------------------------------------------------

def test_an_action_is_queued_not_executed():
    s, q, _reg, calls, asked = _setup()
    out = s.handle('/recolte')
    assert calls == []                              # la console ne tape pas
    assert len(q) == 1 and q.pending()[0].tool == 'recolter_ressources'
    assert out[0].startswith('[en file] #1 recolter_ressources')
    assert asked == []                              # gratuit : pas de o/n


def test_the_queued_action_is_executed_by_the_bot_loop():
    s, q, reg, calls, _asked = _setup()
    s.handle('/recolte')
    run_next(q, reg)
    assert calls == ['recolte']


def test_the_message_says_what_the_bot_is_busy_with():
    s, *_ = _setup(busy='combat')
    assert "je termine d'abord : combat" in s.handle('/recolte')[0]


def test_the_message_says_how_many_commands_are_ahead():
    s, *_ = _setup()
    s.handle('/recolte')
    assert '1 avant elle' in s.handle('/recolte')[0]


# ---------------------------------------------------------------------------
# Depense : o/n d'abord
# ---------------------------------------------------------------------------

def test_spending_asks_before_queuing():
    s, q, _reg, _calls, asked = _setup(answers=[True])
    s.handle('/attaque')
    assert len(asked) == 1 and 'attaque' in asked[0]
    assert q.pending()[0].confirmed is True


def test_answering_no_queues_nothing():
    s, q, _reg, calls, _asked = _setup(answers=[False])
    assert s.handle('/attaque') == ["[annulé] rien n'est parti"]
    assert len(q) == 0 and calls == []


def test_without_confirmation_mode_nothing_is_asked_but_it_is_confirmed():
    s, q, _reg, _calls, asked = _setup(confirm_spending=False)
    s.handle('/attaque')
    assert asked == []
    assert q.pending()[0].confirmed is True


def test_a_confirmed_attack_actually_runs_in_the_bot_loop():
    s, q, reg, calls, _asked = _setup(answers=[True])
    s.handle('/attaque')
    _cmd, result = run_next(q, reg)
    assert result.ok and calls == ['attaque']


# ---------------------------------------------------------------------------
# Noms de troupes : refuses A LA SAISIE
# ---------------------------------------------------------------------------

def test_donation_names_are_normalized_before_queuing():
    s, q, *_ = _setup()
    s.handle('/dons ballons yétis')
    assert q.pending()[0].args == {'troupes': ['ballon', 'yeti']}


def test_an_unknown_troop_is_refused_at_input_and_nothing_is_queued():
    s, q, *_ = _setup()
    out = s.handle('/dons dragonn')
    assert out[0].startswith('[refus]') and 'dragon' in out[0]
    assert len(q) == 0


def test_quantities_are_refused_at_input():
    s, q, *_ = _setup()
    assert '5.3.4' in s.handle('/dons 3 ballons')[0]
    assert len(q) == 0


def test_a_donation_with_only_filler_words_is_refused():
    s, q, *_ = _setup()
    assert 'aucune troupe' in s.handle('/dons de la')[0]
    assert len(q) == 0


def test_donation_without_names_is_free_choice():
    s, q, *_ = _setup()
    s.handle('/dons')
    assert q.pending()[0].args == {}


def test_research_name_is_resolved_and_confirmation_asked():
    s, q, _reg, _calls, asked = _setup(answers=[True])
    s.handle('/labo bébés dragons')
    assert q.pending()[0].args == {'troupe': 'bebe_dragon'}
    assert len(asked) == 1


def test_an_unknown_research_name_is_refused_before_any_question():
    """On ne demande pas « tu confirmes ? » pour une troupe qui n'existe pas."""
    s, q, _reg, _calls, asked = _setup(answers=[True])
    assert s.handle('/labo licorne')[0].startswith('[refus]')
    assert asked == [] and len(q) == 0


def test_a_command_without_argument_refuses_one():
    s, q, *_ = _setup()
    assert "ne prend pas d'argument" in s.handle('/recolte vite')[0]
    assert len(q) == 0


# ---------------------------------------------------------------------------
# Commandes de la console
# ---------------------------------------------------------------------------

def test_file_lists_pending_commands():
    s, *_ = _setup(answers=[True])
    assert s.handle('/file') == ['(file vide)']
    s.handle('/recolte')
    s.handle('/dons ballon')
    out = s.handle('/file')
    assert out[0].startswith('2 commande(s)')
    assert any('donner_troupes' in line and 'ballon' in line for line in out)


def test_annuler_empties_the_admin_queue():
    s, q, *_ = _setup()
    s.handle('/recolte')
    s.handle('/recolte')
    assert '2 commande(s)' in s.handle('/annuler')[0]
    assert len(q) == 0
    assert s.handle('/annuler') == ['(file déjà vide)']


def test_outils_shows_what_each_tool_does():
    s, *_ = _setup()
    out = '\n'.join(s.handle('/outils'))
    assert 'etat_du_village' in out and '[lit]' in out
    assert '[agit, dépense]' in out and '/attaque' in out


def test_a_tool_missing_in_this_mode_is_reported():
    s, *_ = _setup(registry_extra=False)
    assert 'pas disponible' in s.handle('/attaque')[0]


def test_an_unknown_command_points_to_help():
    s, *_ = _setup()
    assert '/aide' in s.handle('/danse')[0]


def test_help_lists_every_slash_command():
    s, *_ = _setup()
    text = '\n'.join(s.handle('/aide'))
    assert all(command in text for command in SLASH_TOOLS)


def test_quit_sets_the_flag():
    s, *_ = _setup()
    s.handle('/quit')
    assert s.wants_quit


def test_an_empty_line_does_nothing():
    s, q, *_ = _setup()
    assert s.handle('   ') == [] and len(q) == 0


def test_commands_are_case_insensitive():
    s, q, *_ = _setup()
    s.handle('/RECOLTE')
    assert len(q) == 1


# ---------------------------------------------------------------------------
# Texte libre -> discussion
# ---------------------------------------------------------------------------

def test_free_text_goes_to_the_brain():
    seen = []
    s, *_ = _setup(chat=lambda text: seen.append(text) or "J'ai 2 ouvriers.")
    assert s.handle('combien d ouvriers ?') == ["cerveau > J'ai 2 ouvriers."]
    assert seen == ['combien d ouvriers ?']


def test_free_text_without_brain_points_to_help():
    s, *_ = _setup()
    assert '/aide' in s.handle('salut')[0]


def test_an_unavailable_brain_is_said_plainly():
    s, *_ = _setup(chat=lambda text: None)
    assert 'indisponible' in s.handle('salut')[0]


def test_free_text_never_queues_an_action():
    """En 5.3.3a, la discussion ne peut pas agir : seul un /outil le peut."""
    s, q, *_ = _setup(chat=lambda text: 'ok, je lance une attaque !')
    s.handle('lance une attaque')
    assert len(q) == 0


# ---------------------------------------------------------------------------
# INVARIANT : handle ne leve jamais
# ---------------------------------------------------------------------------

def test_a_crashing_brain_becomes_a_line():
    def boom(text):
        raise ConnectionError('Ollama absent')

    s, *_ = _setup(chat=boom)
    assert s.handle('salut')[0].startswith('[erreur console]')


def test_a_crashing_confirmation_queues_nothing():
    s, q, reg, *_ = _setup()

    def boom(prompt):
        raise EOFError('stdin ferme')

    s._ask = boom
    assert s.handle('/attaque')[0].startswith('[erreur console]')
    assert len(q) == 0


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------

def test_format_result_ok_and_refusal():
    assert format_result(ToolResult(ok=True, tool='x', data={'a': 1})) == [
        '[ok] x', '   a : 1']
    assert format_result(ToolResult(ok=False, tool='x', error='non')) == [
        '[refus] x : non']


def test_format_completion():
    q = CommandQueue()
    cmd = q.put('lancer_attaque')
    ok = format_completion(cmd, ToolResult(ok=True, tool='lancer_attaque',
                                           data={'stars': 2, 'percentage': 71}))
    assert ok == '[fini] #1 lancer_attaque : ok - stars=2 | percentage=71'
    ko = format_completion(cmd, ToolResult(ok=False, tool='lancer_attaque',
                                           error='Echec: attack episode failed'))
    assert 'REFUS - Echec: attack episode failed' in ko


def test_output_stays_within_the_windows_codepage():
    """Un « ✔ » ou « → » fait planter print quand la sortie est redirigee."""
    s, *_ = _setup(answers=[True], chat=lambda t: 'ok')
    lines = []
    for line in ('/aide', '/etat', '/recolte', '/attaque', '/dons dragonn',
                 '/file', '/outils', '/annuler', 'salut', '/danse'):
        lines += s.handle(line)
    for line in lines:
        line.encode('cp1252')          # leve UnicodeEncodeError sinon


def test_admin_commands_are_queued_as_admin():
    s, q, *_ = _setup()
    s.handle('/recolte')
    assert q.pending()[0].source == SOURCE_ADMIN
