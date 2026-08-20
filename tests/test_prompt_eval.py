"""Banc d'essai des prompts (V5.3, increment 5.3.0).

Le banc mesure le MODELE ; ces tests mesurent le BANC. Aucun Ollama : `ask` est
un faux qui rend des reponses choisies, y compris des reponses pieges.

Un banc qui se trompe est pire que pas de banc : il donne une confiance fausse.
"""

import pytest

from clashai.brain.prompt_eval import (
    SUITE,
    WORLD_LU,
    WORLD_NON_LU,
    Case,
    digits_of,
    run_case,
    run_suite,
    says_all_of,
    says_no_number,
    says_number,
    says_number_or_stays_vague,
    says_numbers,
)


def _case(check, phrasings=('q1', 'q2'), world=None):
    return Case(intent='test', world=world or {}, phrasings=list(phrasings),
                check=check)


# ---------------------------------------------------------------------------
# says_number : la donnee juste compte, meme mal formatee
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('answer', [
    "Vous avez 18549 Elixir noir.",
    "Tu as 18 549 elixir noir.",         # separateur de milliers
    "Il te reste 18.549 d'elixir noir.",
    "1 8549",                            # Mistral coupe parfois au mauvais endroit
    "Or : 2235125, elixir noir : 18549",  # noye au milieu d'autres nombres
])
def test_a_correct_value_counts_however_it_is_formatted(answer):
    """La typographie du modele n'est pas le sujet : la donnee est juste."""
    assert says_number(18549)(answer)


@pytest.mark.parametrize('answer', [
    "Je ne sais pas.",
    "Vous avez 2235125 d'or.",           # le bon format, la mauvaise valeur
    "",
    None,
])
def test_a_wrong_or_absent_value_fails(answer):
    assert not says_number(18549)(answer)


def test_digits_of_ignores_everything_but_digits():
    assert digits_of("Or : 2 235 125 !") == '2235125'
    assert digits_of(None) == ''


# ---------------------------------------------------------------------------
# says_no_number : detecter un montant invente
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('answer', [
    "Je ne sais pas, la valeur n'est pas lue.",
    "Cette information n'est pas disponible.",
    "Je n'ai pas acces a ce compteur.",
])
def test_a_real_refusal_passes(answer):
    assert says_no_number()(answer)


@pytest.mark.parametrize('answer', [
    "Tu as 15568 or.",                   # l'hallucination d'origine
    "Environ 2 235 125.",
    "Il te reste 999 elixir noir.",
])
def test_an_invented_amount_is_caught(answer):
    assert not says_no_number()(answer)


def test_a_small_number_is_not_an_invented_amount():
    """« 4 ouvriers » ne doit pas etre pris pour un montant invente : le seuil
    est a 3 chiffres, sinon le banc crierait au loup en permanence."""
    assert says_no_number()("Il te reste 4 ouvriers, mais je ne sais pas l'or.")


# ---------------------------------------------------------------------------
# says_all_of
# ---------------------------------------------------------------------------

def test_says_all_of_requires_every_fragment():
    check = says_all_of('2235125', '18549')
    assert check("Or 2235125 et elixir noir 18549")
    assert not check("Or 2235125 seulement")


# ---------------------------------------------------------------------------
# Execution et score
# ---------------------------------------------------------------------------

def test_score_is_a_rate_over_phrasings():
    """Le coeur du banc : une formulation qui marche ne prouve rien."""
    answers = {'q1': '18549', 'q2': 'je ne sais pas'}
    result = run_case(_case(says_number(18549), ('q1', 'q2')),
                      lambda q, w: answers[q])
    assert (result.passed, result.total) == (1, 2)
    assert result.score == 0.5
    assert [r.phrasing for r in result.failures] == ['q2']


def test_repeat_multiplies_the_draws():
    """Un modele bavarde a temperature 0.7 : --repeat mesure la stabilite."""
    result = run_case(_case(says_number(1), ('q1',)), lambda q, w: '1', repeat=3)
    assert result.total == 3 and result.passed == 3


def test_the_world_of_the_case_is_the_one_passed_to_ask():
    """Sinon le banc testerait un autre etat que celui qu'il annonce."""
    seen = []
    run_case(_case(says_number(1), ('q1',), world=WORLD_LU),
             lambda q, w: seen.append(w) or '1')
    assert seen == [WORLD_LU]


def test_chat_history_is_not_shared_between_phrasings():
    """`ask` recoit les formulations independamment : c'est au runner de ne pas
    les enchainer. On verifie qu'aucun etat ne fuit entre deux appels."""
    calls = []
    run_case(_case(says_number(1), ('q1', 'q2')),
             lambda q, w: calls.append(q) or '1')
    assert calls == ['q1', 'q2']


# ---------------------------------------------------------------------------
# INVARIANT : un cerveau indisponible ECHOUE, il ne casse pas le banc
# ---------------------------------------------------------------------------

def test_an_exception_counts_as_a_failure_not_a_crash():
    def boom(q, w):
        raise ConnectionError('Ollama absent')

    result = run_case(_case(says_number(1)), boom)
    assert result.passed == 0 and result.total == 2
    assert 'ConnectionError' in result.failures[0].answer


def test_a_none_answer_counts_as_a_failure():
    result = run_case(_case(says_number(1)), lambda q, w: None)
    assert result.passed == 0


def test_none_never_passes_even_a_refusal_check():
    """Piege : `says_no_number()(None)` est vrai (pas de chiffre dans None).
    Le runner doit exiger une reponse NON VIDE, sinon un LLM eteint obtiendrait
    un sans-faute sur tous les cas de retenue."""
    result = run_case(_case(says_no_number()), lambda q, w: None)
    assert result.passed == 0


# ---------------------------------------------------------------------------
# Rapport global
# ---------------------------------------------------------------------------

def test_report_aggregates_every_case():
    suite = [_case(says_number(1), ('a',)), _case(says_number(2), ('b',))]
    report = run_suite(suite, lambda q, w: '1')
    assert (report.passed, report.total) == (1, 2)
    assert report.score == 0.5
    assert report.meets(0.5) and not report.meets(0.9)


def test_on_case_fires_once_per_case():
    seen = []
    run_suite([_case(says_number(1), ('a',))] * 3, lambda q, w: '1',
              on_case=seen.append)
    assert len(seen) == 3


def test_an_empty_suite_does_not_divide_by_zero():
    report = run_suite([], lambda q, w: 'x')
    assert report.score == 0.0 and report.total == 0


# ---------------------------------------------------------------------------
# La suite de reference
# ---------------------------------------------------------------------------

def test_every_intent_is_unique():
    names = [c.intent for c in SUITE]
    assert len(names) == len(set(names))


def test_the_suite_covers_both_directions():
    """Un banc qui ne teste qu'un sens laisse passer la sur-correction : c'est
    exactement comme ca que le refus est devenu le reflexe par defaut."""
    assert any('/rappel' in c.intent for c in SUITE)
    assert any('/retenue' in c.intent for c in SUITE)


@pytest.mark.parametrize('intent', ['ressources.or', 'ressources.elixir_noire',
                                    'ouvriers'])
def test_sensitive_readings_are_tested_in_both_directions(intent):
    got = {c.intent for c in SUITE}
    assert f'{intent}/rappel' in got and f'{intent}/retenue' in got


def test_retenue_cases_have_no_readings_at_all():
    """Sinon `says_no_number` crierait au loup sur une valeur legitime."""
    for c in SUITE:
        if '/retenue' in c.intent:
            assert not c.world.get('readings'), c.intent


def test_rappel_cases_do_have_readings():
    for c in SUITE:
        if '/rappel' in c.intent:
            assert c.world.get('readings'), c.intent


def test_both_worlds_show_the_same_buttons():
    """Le piege d'origine : le modele deduisait un montant du seul NOM du bouton
    `compteur_or`. Les deux mondes doivent donc afficher les memes boutons —
    seules les LECTURES changent."""
    assert WORLD_LU['buttons'] == WORLD_NON_LU['buttons']
    assert 'compteur_or' in WORLD_NON_LU['buttons']


def test_every_case_has_several_phrasings():
    """Une seule formulation par intention ramenerait le banc a l'essai manuel
    qui avait conclu « ca marche » la veille du bug."""
    for c in SUITE:
        assert len(c.phrasings) >= 2, c.intent


def test_every_case_explains_what_it_protects():
    for c in SUITE:
        assert c.why, c.intent


# ---------------------------------------------------------------------------
# says_numbers : le banc doit normaliser AVANT de comparer
#
# Bug du banc lui-meme (19 aout 2026) : `says_all_of('2235125', ...)` notait 0/2
# une reponse pourtant parfaite, parce que le modele ecrit « 2 235 125 ». Un banc
# qui se trompe donne une confiance fausse.
# ---------------------------------------------------------------------------

def test_says_numbers_ignores_thousand_separators():
    check = says_numbers(2235125, 2399904, 18549)
    assert check("Or : 2 235 125  Élixir : 2 399 904  Élixir noir : 18 549")
    assert check("2235125, 2399904 et 18549")


def test_says_numbers_needs_every_value():
    check = says_numbers(2235125, 18549)
    assert not check("Or : 2 235 125 seulement")


def test_says_all_of_stays_for_text_only():
    """Il reste utile pour du texte, pas pour des nombres."""
    assert says_all_of('ouvrier', 'libre')("4 ouvriers libres")


# ---------------------------------------------------------------------------
# says_number_or_stays_vague : les questions fermees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('answer', [
    "Oui.",                              # correct sans chiffre : acceptable
    "Oui, tu peux.",
    "Oui, tu as 4 ouvriers libres.",     # correct avec le bon chiffre
    "Oui, 4 sur 5 sont libres.",
])
def test_a_closed_question_may_answer_without_a_number(answer):
    assert says_number_or_stays_vague(4)(answer)


@pytest.mark.parametrize('answer', [
    "Oui, tu as 1 ouvrier disponible.",  # le raté mesure le 19 aout
    "Non, tu as 0 ouvrier.",
])
def test_a_wrong_count_still_fails_a_closed_question(answer):
    assert not says_number_or_stays_vague(4)(answer)
