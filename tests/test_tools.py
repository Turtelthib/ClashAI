"""Registre d'outils du cerveau (V5.3, increment 5.3.2).

L'invariant qui structure tout ce fichier : **la securite est dans `call()`, pas
dans le prompt**. On ne teste jamais « le modele a-t-il refuse ? » — on teste que
le registre refuse, quoi que le modele demande.

Mesure du 19 aout 2026 : une consigne donnee a un 7B s'applique de travers.
Une instruction est une suggestion ; une verification n'en est pas une.
"""

import pytest

from clashai.brain.tools import (
    SOURCE_ADMIN,
    SOURCE_CLAN,
    Tool,
    ToolRegistry,
    ToolResult,
    build_read_only_registry,
)


def _tool(name='lire', acts=False, spends=False, fn=None, parameters=None):
    return Tool(
        name=name,
        description=f"outil {name}",
        fn=fn or (lambda **kw: {'appele': True, **kw}),
        parameters=parameters or {'type': 'object', 'properties': {}},
        acts=acts,
        spends=spends,
    )


def _reg(*tools):
    reg = ToolRegistry()
    for t in tools or (_tool(),):
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

def test_a_registered_tool_can_be_called():
    reg = _reg()
    res = reg.call('lire')
    assert res.ok and res.data['appele'] is True


def test_registering_the_same_name_twice_is_a_programming_error():
    reg = _reg()
    with pytest.raises(ValueError):
        reg.register(_tool())


def test_an_unknown_tool_lists_what_is_available():
    """Le message repart au modele : il doit lui permettre de se corriger."""
    res = _reg().call('conquerir_le_monde')
    assert not res.ok
    assert 'inconnu' in res.error and 'lire' in res.error


def test_the_schema_is_in_the_tool_calling_format():
    fn = _reg().get('lire').schema()
    assert fn['type'] == 'function'
    assert fn['function']['name'] == 'lire'
    assert 'description' in fn['function']
    assert fn['function']['parameters']['type'] == 'object'


# ---------------------------------------------------------------------------
# GARDE-FOU 1 : autorite par source
#
# Un membre du clan peut ecrire « @mini_pekka lance une attaque ». C'est du
# texte. Il ne peut pas agir parce qu'aucun outil d'action ne lui est
# accessible — pas parce que le modele decline poliment.
# ---------------------------------------------------------------------------

def test_the_clan_cannot_call_a_tool_that_acts():
    reg = _reg(_tool('attaquer', acts=True))
    res = reg.call('attaquer', source=SOURCE_CLAN)
    assert not res.ok and 'opérateur' in res.error


def test_the_admin_can_call_a_tool_that_acts():
    reg = _reg(_tool('attaquer', acts=True))
    assert reg.call('attaquer', source=SOURCE_ADMIN).ok


def test_the_clan_can_still_call_a_read_only_tool():
    """Le chat de clan reste un VRAI chat : lire et discuter est permis."""
    reg = _reg(_tool('etat', acts=False))
    assert reg.call('etat', source=SOURCE_CLAN).ok


def test_an_acting_tool_is_not_even_shown_to_the_clan():
    """Inutile de l'inviter a essayer pour refuser ensuite."""
    reg = _reg(_tool('etat'), _tool('attaquer', acts=True))
    assert reg.names(SOURCE_CLAN) == ['etat']
    assert reg.names(SOURCE_ADMIN) == ['attaquer', 'etat']
    assert len(reg.schemas(SOURCE_CLAN)) == 1


def test_a_tool_refused_by_source_is_never_executed():
    """Le refus doit precéder l'appel, pas le suivre."""
    calls = []
    reg = _reg(_tool('attaquer', acts=True,
                     fn=lambda **kw: calls.append(1) or {}))
    reg.call('attaquer', source=SOURCE_CLAN)
    assert calls == []


# ---------------------------------------------------------------------------
# GARDE-FOU 2 : depenser exige une confirmation
#
# Ecrit UNE FOIS ici, donc valable pour tout outil ajoute ensuite sans qu'on ait
# a y repenser. C'est le socle du principe anti-gemmes.
# ---------------------------------------------------------------------------

def test_a_spending_tool_is_refused_without_confirmation():
    reg = _reg(_tool('ameliorer', spends=True))
    res = reg.call('ameliorer')
    assert not res.ok and 'confirmation' in res.error


def test_a_spending_tool_runs_once_confirmed():
    reg = _reg(_tool('ameliorer', spends=True))
    assert reg.call('ameliorer', confirm=True).ok


def test_an_unconfirmed_spending_tool_is_never_executed():
    """Le point entier : aucune depense ne part sans accord explicite."""
    calls = []
    reg = _reg(_tool('ameliorer', spends=True,
                     fn=lambda **kw: calls.append(1) or {}))
    reg.call('ameliorer')
    assert calls == []


def test_confirmation_does_not_bypass_source_authority():
    """Deux garde-fous independants : confirmer n'est pas devenir admin."""
    reg = _reg(_tool('ameliorer', acts=True, spends=True))
    res = reg.call('ameliorer', source=SOURCE_CLAN, confirm=True)
    assert not res.ok and 'opérateur' in res.error


def test_a_read_only_tool_needs_no_confirmation():
    assert _reg(_tool('etat')).call('etat').ok


# ---------------------------------------------------------------------------
# GARDE-FOU 3 : arguments valides
#
# Un 7B invente des parametres et en oublie. On refuse plutot que d'appeler avec
# n'importe quoi.
# ---------------------------------------------------------------------------

_PARAMS = {
    'type': 'object',
    'properties': {'batiment': {'type': 'string'},
                   'quantite': {'type': 'integer'}},
    'required': ['batiment'],
}


def _typed(**kw):
    return _reg(_tool('cible', parameters=_PARAMS, **kw))


def test_a_missing_required_argument_is_refused():
    res = _typed().call('cible', {'quantite': 2})
    assert not res.ok and 'batiment' in res.error


def test_an_invented_argument_is_refused():
    res = _typed().call('cible', {'batiment': 'canon', 'couleur': 'rouge'})
    assert not res.ok and 'couleur' in res.error


def test_a_wrong_type_is_refused():
    res = _typed().call('cible', {'batiment': 'canon', 'quantite': 'trois'})
    assert not res.ok and 'quantite' in res.error


def test_a_boolean_is_not_an_integer():
    """`True` vaut 1 en Python : sans garde explicite, il passerait pour un
    nombre et l'outil recevrait n'importe quoi."""
    res = _typed().call('cible', {'batiment': 'canon', 'quantite': True})
    assert not res.ok and 'booléen' in res.error


def test_valid_arguments_reach_the_tool():
    res = _typed().call('cible', {'batiment': 'canon', 'quantite': 3})
    assert res.ok and res.data['batiment'] == 'canon' and res.data['quantite'] == 3


def test_an_optional_argument_may_be_omitted():
    assert _typed().call('cible', {'batiment': 'canon'}).ok


def test_invalid_arguments_never_reach_the_tool():
    calls = []
    reg = _reg(_tool('cible', parameters=_PARAMS,
                     fn=lambda **kw: calls.append(kw) or {}))
    reg.call('cible', {})
    assert calls == []


# ---------------------------------------------------------------------------
# INVARIANT : un outil qui casse ne casse pas le bot
# ---------------------------------------------------------------------------

def test_an_exception_becomes_a_result_not_a_crash():
    def boom(**kw):
        raise RuntimeError('ADB deconnecte')

    res = _reg(_tool('lire', fn=boom)).call('lire')
    assert not res.ok
    assert 'RuntimeError' in res.error and 'ADB' in res.error


def test_a_non_dict_return_is_wrapped():
    """Un outil peut rendre une valeur simple sans casser le contrat."""
    res = _reg(_tool('lire', fn=lambda **kw: 42)).call('lire')
    assert res.ok and res.data == {'resultat': 42}


def test_tool_result_is_truthy_only_when_ok():
    assert bool(ToolResult(ok=True, tool='x'))
    assert not bool(ToolResult(ok=False, tool='x', error='non'))


# ---------------------------------------------------------------------------
# Journal — servira au tableau de bord et a la memoire d'actions (5.3.5)
# ---------------------------------------------------------------------------

def test_every_call_is_journalled_including_refusals():
    reg = _reg(_tool('etat'), _tool('attaquer', acts=True))
    reg.call('etat')
    reg.call('attaquer', source=SOURCE_CLAN)
    reg.call('inexistant')
    assert [e['tool'] for e in reg.journal] == ['etat', 'attaquer', 'inexistant']
    assert [e['ok'] for e in reg.journal] == [True, False, False]


def test_the_journal_records_who_asked():
    reg = _reg(_tool('etat'))
    reg.call('etat', source=SOURCE_CLAN)
    assert reg.journal[-1]['source'] == SOURCE_CLAN


def test_the_journal_is_capped():
    reg = ToolRegistry(journal_size=3)
    reg.register(_tool('etat'))
    for _ in range(10):
        reg.call('etat')
    assert len(reg.journal) == 3


def test_last_returns_the_most_recent_calls():
    reg = _reg(_tool('etat'))
    for _ in range(4):
        reg.call('etat')
    assert len(reg.last(2)) == 2


def test_the_journal_is_a_copy():
    reg = _reg(_tool('etat'))
    reg.call('etat')
    reg.journal.clear()
    assert len(reg.journal) == 1


# ---------------------------------------------------------------------------
# Les outils de LECTURE
# ---------------------------------------------------------------------------

WORLD = {
    'screen_state': 'village_home',
    'buildings': [1, 2, 3],
    'troop_positions': {'dragon': (1, 2, 0.9), 'barbare': (3, 4, 0.8)},
    'readings': {
        'resources': {'or': 100},
        'builders': {'libres': 4, 'total': 5},
        'lab_libre': False,
        'recoltes': {'or': 5},
        'dons_en_attente': 2,
    },
}


def test_the_read_only_registry_acts_on_nothing():
    """Tout l'interet de l'increment : on peut le brancher sans risque."""
    reg = build_read_only_registry(lambda: WORLD)
    assert all(not reg.get(n).acts and not reg.get(n).spends
               for n in reg.names())


def test_etat_du_village_reports_the_readings():
    reg = build_read_only_registry(lambda: WORLD)
    d = reg.call('etat_du_village').data
    assert d['ressources'] == {'or': 100}
    assert d['ouvriers'] == {'libres': 4, 'total': 5}
    assert d['collecteurs_pleins'] == {'or': 5}
    assert d['demandes_de_dons'] == 2
    assert d['batiments_detectes'] == 3


def test_an_unreadable_value_is_none_not_zero():
    """None = « pas lisible d'ici ». Rendre 0 serait affirmer qu'il n'y a rien."""
    reg = build_read_only_registry(lambda: {'screen_state': 'combat'})
    d = reg.call('etat_du_village').data
    assert d['ressources'] is None and d['demandes_de_dons'] is None


def test_listing_troops_is_sorted_and_names_only():
    reg = build_read_only_registry(lambda: WORLD)
    assert reg.call('lister_troupes_disponibles').data['troupes'] == \
        ['barbare', 'dragon']


def test_the_read_only_tools_survive_an_empty_world():
    reg = build_read_only_registry(lambda: None)
    assert reg.call('etat_du_village').ok
    assert reg.call('lister_troupes_disponibles').data['troupes'] == []


def test_the_world_is_read_at_call_time_not_at_build_time():
    """Sinon le cerveau repondrait avec l'etat du demarrage."""
    box = {'w': {'screen_state': 'a'}}
    reg = build_read_only_registry(lambda: box['w'])
    box['w'] = {'screen_state': 'b'}
    assert reg.call('etat_du_village').data['ecran'] == 'b'
