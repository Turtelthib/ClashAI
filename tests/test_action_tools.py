"""Outils qui AGISSENT sur le jeu (V5.3, etape 5.3.3a).

Tout est injecte : agents, dons et labo sont des faux. On teste la traduction
(noms de troupes, resultats d'agents, statuts du labo) et les verrous — en
particulier qu'AUCUN confirm_decider n'est jamais passe au labo.
"""

import pytest

from clashai.agents.base import AgentResult
from clashai.brain.action_tools import (
    AGENT_TOOLS,
    Echec,
    Refus,
    donatable_troop_names,
    make_donate_fn,
    make_research_fn,
    normalize_name,
    parse_troop_list,
    prepare_args,
    register_action_tools,
    research_outcome,
    resolve_troop_name,
    suggest_troop_names,
)
from clashai.brain.tools import SOURCE_CLAN, ToolRegistry
from clashai.village.lab import LabCandidate
from clashai.village.upgrader import UpgradeResult

KNOWN = frozenset({
    'ballon', 'yeti', 'dragon', 'bebe_dragon', 'electro_dragon', 'sorciere',
    'sorcier', 'golem', 'golem_glace', 'chauve_souris', 'pekka', 'archere',
    'barbare', 'zap',
})


# ---------------------------------------------------------------------------
# Noms donnables
# ---------------------------------------------------------------------------

TT = [
    {'name': 'ballon', 'role': 'ranged'},
    {'name': 'roi', 'role': 'hero'},
    {'name': 'sorciere_ruine', 'role': 'clean'},
    {'name': 'zap', 'role': 'spell'},
]


def test_heroes_are_never_donatable():
    assert 'roi' not in donatable_troop_names(TT, cnn_names={'ballon', 'roi', 'zap'})


def test_names_are_crossed_with_the_troop_bar_cnn():
    """Un nom absent du CNN ne peut jamais correspondre a une carte du pop-up."""
    assert donatable_troop_names(TT, cnn_names={'ballon', 'zap'}) == {'ballon', 'zap'}


def test_unreadable_cnn_classes_fall_back_to_the_registry():
    assert donatable_troop_names(TT, cnn_names=set()) == {
        'ballon', 'sorciere_ruine', 'zap'}


def test_the_real_registry_gives_sensible_names():
    names = donatable_troop_names()
    assert {'ballon', 'yeti', 'dragon'} <= names
    assert 'roi' not in names and 'reine' not in names


# ---------------------------------------------------------------------------
# Normalisation et resolution — jamais d'approximation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('raw, expected', [
    ('Yéti', 'yeti'),
    ('BÉBÉ-DRAGON', 'bebe_dragon'),
    ("golem de glace", 'golem_glace'),
    ('  ballon,  ', 'ballon'),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize('raw, expected', [
    ('ballon', 'ballon'),
    ('ballons', 'ballon'),
    ('yétis', 'yeti'),
    ('bébés dragons', 'bebe_dragon'),
    ('électro-dragons', 'electro_dragon'),
    ('chauves-souris', 'chauve_souris'),     # seul le 1er mot porte le pluriel
    ('golems de glace', 'golem_glace'),
    ('sorcières', 'sorciere'),
    ('sorciers', 'sorcier'),                 # ne glisse pas vers « sorciere »
    ('PEKKA', 'pekka'),
])
def test_resolve_troop_name(raw, expected):
    assert resolve_troop_name(raw, KNOWN) == expected


@pytest.mark.parametrize('raw', ['dragonn', 'licorne', '', None, 'roi'])
def test_unknown_or_misspelled_names_are_refused(raw):
    """« dragonn » n'est PAS arrondi a « dragon » : on refuse, on suggere."""
    assert resolve_troop_name(raw, KNOWN) is None


def test_suggestions_are_only_suggestions():
    assert 'dragon' in suggest_troop_names('dragonn', KNOWN)
    assert resolve_troop_name('dragonn', KNOWN) is None


# ---------------------------------------------------------------------------
# Liste de troupes en texte libre
# ---------------------------------------------------------------------------

def test_parse_a_simple_list():
    assert parse_troop_list('ballons yétis', KNOWN) == (['ballon', 'yeti'], None)


def test_parse_prefers_the_longest_known_group():
    names, error = parse_troop_list('bébé dragon, golem de glace, golem', KNOWN)
    assert error is None
    assert names == ['bebe_dragon', 'golem_glace', 'golem']


def test_parse_removes_duplicates_and_keeps_order():
    assert parse_troop_list('yeti ballon yétis', KNOWN) == (['yeti', 'ballon'], None)


def test_parse_refuses_an_unknown_name_with_a_suggestion():
    names, error = parse_troop_list('ballons dragonn', KNOWN)
    assert names == []
    assert 'dragonn' in error and 'dragon' in error


def test_parse_refuses_quantities_instead_of_ignoring_them():
    """« 3 ballons » lu comme « ballons » donnerait des ballons sans limite."""
    names, error = parse_troop_list('3 ballons', KNOWN)
    assert names == [] and '5.3.4' in error


def test_parse_an_empty_text():
    assert parse_troop_list('', KNOWN) == ([], None)


# ---------------------------------------------------------------------------
# prepare_args — a la saisie ET a l'execution
# ---------------------------------------------------------------------------

def test_prepare_donation_args_normalizes_names():
    args, error = prepare_args('donner_troupes', {'troupes': ['Ballons', 'yétis']}, KNOWN)
    assert error is None and args['troupes'] == ['ballon', 'yeti']


def test_prepare_donation_args_refuses_any_unknown_name():
    _args, error = prepare_args('donner_troupes', {'troupes': ['ballon', 'licorne']}, KNOWN)
    assert 'licorne' in error


def test_prepare_research_args():
    assert prepare_args('lancer_recherche', {'troupe': 'dragons'}, KNOWN) == (
        {'troupe': 'dragon'}, None)


def test_prepare_leaves_other_tools_untouched():
    assert prepare_args('lancer_attaque', {}, KNOWN) == ({}, None)


# ---------------------------------------------------------------------------
# Enregistrement
# ---------------------------------------------------------------------------

def _ok_agent(calls, data=None):
    def run_agent(name):
        calls.append(name)
        return AgentResult(ok=True, duration_s=2.345, data=dict(data or {}))
    return run_agent


def _full_registry(calls, donate=None, research=None, **kw):
    reg = ToolRegistry()
    register_action_tools(
        reg, run_agent=_ok_agent(calls),
        donate_fn=donate or (lambda wanted=None: {'wanted': wanted}),
        research_fn=research or (lambda troupe=None: UpgradeResult('ok', price=100)),
        known_troops=KNOWN, **kw)
    return reg


def test_every_action_tool_is_registered():
    reg = _full_registry([])
    assert set(reg.names()) == {
        'recolter_ressources', 'lancer_attaque', 'demander_renforts',
        'donner_troupes', 'lancer_recherche'}


def test_every_action_tool_acts_and_only_attack_and_research_spend():
    reg = _full_registry([])
    assert all(reg.get(n).acts for n in reg.names())
    assert {n for n in reg.names() if reg.get(n).spends} == {
        'lancer_attaque', 'lancer_recherche'}


def test_the_clan_sees_none_of_the_action_tools():
    assert _full_registry([]).names(SOURCE_CLAN) == []


def test_a_missing_dependency_means_no_tool():
    reg = ToolRegistry()
    assert register_action_tools(reg) == []
    assert reg.names() == []


def test_available_agents_filters_agent_tools():
    reg = ToolRegistry()
    register_action_tools(reg, run_agent=_ok_agent([]),
                          available_agents={'village'})
    assert reg.names() == ['recolter_ressources']


def test_agent_tools_map_to_the_right_agents():
    assert {t[0]: t[1] for t in AGENT_TOOLS} == {
        'recolter_ressources': 'village',
        'lancer_attaque': 'combat',
        'demander_renforts': 'clan_castle',
    }


# ---------------------------------------------------------------------------
# Outils adosses a un agent
# ---------------------------------------------------------------------------

def test_harvest_runs_the_village_agent_and_reports_its_data():
    calls = []
    reg = ToolRegistry()
    register_action_tools(reg, run_agent=_ok_agent(calls, {'collected': 3}))
    res = reg.call('recolter_ressources')
    assert res.ok and calls == ['village']
    assert res.data == {'collected': 3, 'duree_s': 2.3}


def test_attack_needs_confirmation_then_runs_the_combat_agent():
    calls = []
    reg = _full_registry(calls)
    assert not reg.call('lancer_attaque').ok and calls == []
    assert reg.call('lancer_attaque', confirm=True).ok and calls == ['combat']


def test_a_failing_agent_becomes_a_readable_failure():
    reg = ToolRegistry()
    register_action_tools(
        reg, run_agent=lambda name: AgentResult(ok=False, duration_s=1.0,
                                                error='attack episode failed'))
    res = reg.call('recolter_ressources')
    assert not res.ok and res.error == 'Echec: attack episode failed'


def test_an_absent_agent_is_a_refusal():
    reg = ToolRegistry()
    register_action_tools(reg, run_agent=lambda name: None)
    res = reg.call('demander_renforts')
    assert not res.ok and 'Refus' in res.error and 'clan_castle' in res.error


# ---------------------------------------------------------------------------
# Dons
# ---------------------------------------------------------------------------

def test_donation_without_troops_leaves_the_choice_to_the_game():
    seen = []
    reg = _full_registry([], donate=lambda wanted=None: seen.append(wanted) or {})
    assert reg.call('donner_troupes').ok
    assert seen == [None]


def test_donation_passes_normalized_names():
    seen = []
    reg = _full_registry([], donate=lambda wanted=None: seen.append(wanted) or {})
    assert reg.call('donner_troupes', {'troupes': ['Ballons', 'yétis']}).ok
    assert seen == [{'ballon', 'yeti'}]


def test_an_unknown_troop_is_refused_before_touching_the_game():
    seen = []
    reg = _full_registry([], donate=lambda wanted=None: seen.append(wanted) or {})
    res = reg.call('donner_troupes', {'troupes': ['licorne']})
    assert not res.ok and 'licorne' in res.error
    assert seen == []


def test_a_chat_that_does_not_open_is_a_readable_failure():
    def donate(wanted=None):
        raise RuntimeError("impossible d'ouvrir le chat de clan")

    res = _full_registry([], donate=donate).call('donner_troupes')
    assert not res.ok and res.error.startswith('Echec:')


# ---------------------------------------------------------------------------
# Recherche au labo
# ---------------------------------------------------------------------------

def test_research_needs_confirmation():
    assert not _full_registry([]).call('lancer_recherche').ok


def test_a_successful_research_reports_its_price():
    res = _full_registry([]).call('lancer_recherche', {'troupe': 'dragons'},
                                  confirm=True)
    assert res.ok and res.data == {'statut': 'ok', 'troupe': 'dragon', 'prix': 100}


@pytest.mark.parametrize('status, fragment', [
    ('busy', 'occupé'),
    ('lab_not_found', 'introuvable'),
    ('cant_afford', 'ressources'),
    ('need_decision', 'aucune dépense'),
    ('declined', 'aucune dépense'),
])
def test_every_non_ok_lab_status_is_a_motivated_refusal(status, fragment):
    reg = _full_registry([], research=lambda troupe=None: UpgradeResult(status))
    res = reg.call('lancer_recherche', confirm=True)
    assert not res.ok and fragment in res.error


def test_a_named_research_not_on_screen_says_which_troop():
    with pytest.raises(Refus, match='dragon'):
        research_outcome(UpgradeResult('nothing_upgradable'), 'dragon')


def test_cant_afford_mentions_the_price_when_known():
    with pytest.raises(Refus, match='9000000'):
        research_outcome(UpgradeResult('cant_afford', price=9_000_000))


def test_an_unknown_research_troop_never_reaches_the_lab():
    seen = []
    reg = _full_registry(
        [], research=lambda troupe=None: seen.append(troupe) or UpgradeResult('ok'))
    assert not reg.call('lancer_recherche', {'troupe': 'licorne'}, confirm=True).ok
    assert seen == []


# ---------------------------------------------------------------------------
# Fabriques branchees sur les vrais modules
# ---------------------------------------------------------------------------

class _FakeLab:
    def __init__(self, candidates=()):
        self._candidates = list(candidates)
        self.kwargs = None
        self.picked = 'jamais appele'

    def research(self, screenshot_fn, tap_fn, models, **kwargs):
        self.kwargs = kwargs
        choose = kwargs.get('choose')
        self.picked = choose(self._candidates) if choose else None
        return UpgradeResult('ok', price=1)


def test_research_fn_never_passes_a_confirm_decider():
    """LE verrou anti-gemmes de cet outil : sans decideur, le labo ne confirme
    que sur preuve d'affordabilite. Un decideur remplacait la preuve avant le
    12 sept. 2026."""
    lab = _FakeLab()
    make_research_fn(lab, None, None, {})('dragon')
    assert lab.kwargs.get('confirm_decider') is None
    assert 'confirm_decider' not in lab.kwargs


def test_research_fn_picks_the_named_troop():
    cards = [LabCandidate('barbare', 1, 1, 10), LabCandidate('dragon', 2, 2, 99)]
    lab = _FakeLab(cards)
    make_research_fn(lab, None, None, {})('dragon')
    assert lab.picked.name == 'dragon'


def test_research_fn_picks_nothing_if_the_troop_is_absent():
    lab = _FakeLab([LabCandidate('barbare', 1, 1, 10)])
    make_research_fn(lab, None, None, {})('dragon')
    assert lab.picked is None


def test_research_fn_without_troop_lets_the_lab_choose():
    lab = _FakeLab()
    make_research_fn(lab, None, None, {})()
    assert lab.kwargs.get('choose') is None


def test_donate_fn_uses_the_donation_flow(monkeypatch):
    seen = {}

    def fake_flow(manager, monitor, screenshot_fn, tap_fn, classify_fn, models,
                  wanted=None, max_requests=None):
        seen.update(wanted=wanted, max_requests=max_requests, manager=manager)
        return {'dons': 1}

    from clashai.social import donation_flow
    monkeypatch.setattr(donation_flow, 'donate_visible_requests', fake_flow)
    donate = make_donate_fn('mgr', 'mon', None, None, None, {}, max_requests=2)
    assert donate(wanted={'ballon'}) == {'dons': 1}
    assert seen == {'wanted': {'ballon'}, 'max_requests': 2, 'manager': 'mgr'}


def test_refus_and_echec_read_well_in_error_messages():
    """Le registre formate « NomException: message » : ces noms se lisent en
    francais dans la console."""
    assert issubclass(Refus, Exception) and issubclass(Echec, Exception)
    assert not issubclass(Refus, RuntimeError)
