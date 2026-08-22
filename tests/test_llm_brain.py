"""LocalLLMBrain : le cerveau LLM (V5.3), avec repli heuristique systematique.

Le client Ollama est INJECTE : aucun serveur, aucun modele, aucun GPU. On teste
la logique de decision, pas la qualite du LLM.

L'invariant CRITIQUE : **le LLM ne doit JAMAIS bloquer le bot**. Ollama absent,
timeout, JSON casse, agent invente -> on retombe sur l'heuristique. Un cerveau
indisponible degrade l'intelligence, jamais la disponibilite.
"""

import pytest

from clashai.agents.base import AgentResult, BaseAgent
from clashai.agents.scheduler import AgentScheduler
from clashai.brain.llm_brain import LocalLLMBrain


class _Agent(BaseAgent):
    """Agent minimal : eligible ou non, sans aucune E/S."""

    def __init__(self, name, priority, ready=True):
        super().__init__()
        self.name = name
        self.priority = priority
        self._ready = ready

    def can_run(self, world):
        return self._ready

    def run(self):
        return AgentResult(ok=True, duration_s=0.0)


class _Client:
    """Faux client Ollama : rend une reponse figee, ou leve."""

    def __init__(self, content=None, boom=None):
        self._content = content
        self._boom = boom
        self.calls = 0

    def chat(self, **kw):
        self.calls += 1
        if self._boom:
            raise self._boom
        return {'message': {'content': self._content}}


def _brain(content=None, boom=None, agents=(('combat', 10), ('village', 15)),
           **kw):
    sched = AgentScheduler()
    for name, prio in agents:
        sched.register(_Agent(name, prio))
    kw.setdefault('min_interval_s', 0.0)     # pas d'attente dans les tests
    return LocalLLMBrain(sched, client=_Client(content, boom),
                         verbose=False, **kw), sched


WORLD = {'mode': 'auto', 'on_village_home': True, 'screen_state': 'village_home'}


# ---------------------------------------------------------------------------
# Le LLM decide
# ---------------------------------------------------------------------------

def test_llm_choice_is_followed():
    brain, _ = _brain('{"agent": "combat", "raison": "ressources pleines"}')
    assert brain.decide(WORLD).name == 'combat'
    assert brain.stats['llm'] == 1
    assert brain.last_reason == 'ressources pleines'


def test_llm_choice_wins_over_priority():
    """Tout l'interet : le LLM peut preferer `combat` (prio 10) a `village` (15),
    ce que l'heuristique ne ferait jamais."""
    brain, _ = _brain('{"agent": "combat"}')
    assert brain.decide(WORLD).name == 'combat'


def test_json_wrapped_in_text_is_still_parsed():
    """Un LLM local bavarde parfois autour du JSON malgre format=json."""
    brain, _ = _brain('Voici ma reponse :\n{"agent": "village"}\nVoila.')
    assert brain.decide(WORLD).name == 'village'


# ---------------------------------------------------------------------------
# INVARIANT : ne jamais bloquer le bot
# ---------------------------------------------------------------------------

def test_ollama_unavailable_falls_back_to_heuristic():
    brain, _ = _brain(boom=ConnectionError('Ollama pas lance'))
    picked = brain.decide(WORLD)
    assert picked is not None and picked.name == 'village'   # prio la + haute
    assert brain.stats['errors'] == 1 and brain.stats['fallback'] == 1


def test_broken_json_falls_back():
    brain, _ = _brain('je ne sais pas trop')
    assert brain.decide(WORLD).name == 'village'
    assert brain.stats['fallback'] == 1


def test_hallucinated_agent_falls_back():
    """Le LLM invente un agent -> on n'invente rien, on retombe."""
    brain, _ = _brain('{"agent": "conquerir_le_monde"}')
    assert brain.decide(WORLD).name == 'village'
    assert brain.stats['fallback'] == 1


def test_llm_cannot_pick_an_ineligible_agent():
    """Le cerveau choisit PARMI ce qui est jouable : il ne contourne ni les
    cooldowns ni les can_run."""
    sched = AgentScheduler()
    sched.register(_Agent('village', 15, ready=True))
    sched.register(_Agent('combat', 10, ready=False))     # non eligible
    brain = LocalLLMBrain(sched, client=_Client('{"agent": "combat"}'),
                          verbose=False, min_interval_s=0.0)
    assert brain.decide(WORLD).name == 'village'          # pas 'combat'


def test_null_agent_means_idle_via_fallback():
    """`{"agent": null}` = le LLM ne veut rien faire -> l'heuristique tranche."""
    brain, _ = _brain('{"agent": null, "raison": "rien d urgent"}')
    assert brain.decide(WORLD).name == 'village'


# ---------------------------------------------------------------------------
# Economie d'appels
# ---------------------------------------------------------------------------

def test_no_llm_call_when_nothing_is_eligible():
    """Rien a faire -> inutile de reveiller le LLM."""
    sched = AgentScheduler()
    sched.register(_Agent('combat', 10, ready=False))
    client = _Client('{"agent": "combat"}')
    brain = LocalLLMBrain(sched, client=client, verbose=False, min_interval_s=0.0)
    assert brain.decide(WORLD) is None
    assert client.calls == 0


def test_no_llm_call_when_only_one_candidate():
    """Un seul agent jouable : il n'y a rien a arbitrer."""
    sched = AgentScheduler()
    sched.register(_Agent('combat', 10))
    client = _Client('{"agent": "combat"}')
    brain = LocalLLMBrain(sched, client=client, verbose=False, min_interval_s=0.0)
    assert brain.decide(WORLD).name == 'combat'
    assert client.calls == 0


def test_llm_is_not_called_on_every_tick():
    """Le scheduler boucle serre : on n'interroge le cerveau qu'a intervalle."""
    brain, _ = _brain('{"agent": "combat"}', min_interval_s=3600)
    first = brain.decide(WORLD)
    second = brain.decide(WORLD)
    assert first.name == 'combat'          # 1er tick : le LLM parle
    assert second.name == 'village'        # ensuite : heuristique
    assert brain._client.calls == 1


# ---------------------------------------------------------------------------
# Le contexte envoye au LLM
# ---------------------------------------------------------------------------

def test_prompt_carries_screen_buttons_and_agents():
    brain, sched = _brain('{"agent": "combat"}')
    world = dict(WORLD, buttons={'attaquer': (1, 2, 0.9), 'donner': (3, 4, 0.8)},
                 buildings=[1, 2, 3])
    prompt = brain.build_prompt(world, sched.agents)
    assert 'village_home' in prompt
    assert 'attaquer' in prompt and 'donner' in prompt
    assert 'combat' in prompt and 'village' in prompt
    assert '3' in prompt                    # nombre de batiments


def test_prompt_survives_an_empty_world():
    brain, sched = _brain('{"agent": "combat"}')
    assert brain.build_prompt({}, sched.agents)      # ne leve pas


@pytest.mark.parametrize('content', [None, 42, '', '{}', '[]', '{"agent": 7}'])
def test_degenerate_responses_never_crash(content):
    brain, _ = _brain(content)
    assert brain.decide(WORLD) is not None            # toujours une decision


# ---------------------------------------------------------------------------
# Back-off : ne pas attendre un serveur absent a chaque intervalle
#
# Mesure reelle : Ollama non lance = ~6 s bloquees par tentative. Sans back-off,
# le bot y passerait ~20 % de son temps pour retomber sur l'heuristique.
# ---------------------------------------------------------------------------

def test_retry_interval_grows_after_failures():
    brain, _ = _brain(boom=ConnectionError('serveur absent'), min_interval_s=30.0)
    assert brain._retry_interval() == 30.0          # au depart : normal
    brain._consecutive_errors = 1
    assert brain._retry_interval() == 60.0
    brain._consecutive_errors = 3
    assert brain._retry_interval() == 240.0


def test_retry_interval_is_capped():
    brain, _ = _brain(min_interval_s=30.0)
    brain._consecutive_errors = 99
    assert brain._retry_interval() == 600.0         # plafond, pas d'explosion


def test_a_failure_delays_the_next_attempt():
    """2e decision juste apres un echec : on n'appelle PAS le serveur."""
    brain, _ = _brain(boom=ConnectionError('absent'), min_interval_s=30.0)
    brain.decide(WORLD)
    assert brain._client.calls == 1
    brain.decide(WORLD)
    assert brain._client.calls == 1, "ne doit pas retenter immediatement"


def test_success_resets_the_backoff():
    """`ollama serve` lance entre-temps -> on repart a la cadence normale."""
    brain, _ = _brain('{"agent": "combat"}', min_interval_s=0.0)
    brain._consecutive_errors = 4
    assert brain.decide(WORLD).name == 'combat'
    assert brain._consecutive_errors == 0


# ---------------------------------------------------------------------------
# Mode DISCUSSION (console operateur, pas le chat du clan)
# ---------------------------------------------------------------------------

def test_chat_returns_the_answer_and_remembers():
    brain, _ = _brain('Je vois le village, tout est calme.')
    assert brain.chat('tu vois quoi ?') == 'Je vois le village, tout est calme.'
    # la question ET la reponse sont memorisees -> vraie conversation
    assert len(brain._chat_history) == 2
    brain.chat('et maintenant ?')
    assert len(brain._chat_history) == 4


def test_chat_history_is_capped():
    """Un 7B local a une fenetre modeste : la memoire reste courte."""
    from clashai.brain.llm_brain import CHAT_HISTORY_TURNS
    brain, _ = _brain('ok')
    for i in range(CHAT_HISTORY_TURNS + 5):
        brain.chat(f'question {i}')
    assert len(brain._chat_history) == 2 * CHAT_HISTORY_TURNS


def test_reset_chat_forgets_the_conversation():
    brain, _ = _brain('ok')
    brain.chat('salut')
    brain.reset_chat()
    assert brain._chat_history == []


def test_chat_sends_the_game_state():
    """Le LLM doit VOIR l'etat du jeu, sinon il ne peut que broder."""
    captured = {}

    class _Spy:
        def chat(self, **kw):
            captured['messages'] = kw['messages']
            return {'message': {'content': 'ok'}}

    from clashai.agents.scheduler import AgentScheduler
    brain = LocalLLMBrain(AgentScheduler(), client=_Spy(), verbose=False)
    brain.chat('tu vois quoi ?', {'screen_state': 'village_home',
                                  'buttons': {'attaquer': (1, 2, 0.9)}})
    joined = ' '.join(m['content'] for m in captured['messages'])
    assert 'village_home' in joined and 'attaquer' in joined


def test_chat_returns_none_when_llm_is_unavailable():
    """L'appelant doit pouvoir dire 'Ollama ne repond pas' plutot que rester muet."""
    brain, _ = _brain(boom=ConnectionError('serveur absent'))
    brain.verbose = False
    assert brain.chat('salut') is None


def test_describe_world_tolerates_an_empty_world():
    brain, _ = _brain('ok')
    assert brain.describe_world({})          # ne leve pas


# ---------------------------------------------------------------------------
# Prechauffage : payer le chargement GPU UNE fois, au demarrage
#
# Mesure reelle : 1er appel ~21 s (chargement du modele), appels suivants
# 0,1-0,6 s. Sans prechauffage, la premiere decision depasse le timeout et part
# a l'heuristique pour rien.
# ---------------------------------------------------------------------------

def test_warmup_reports_success_and_clears_backoff():
    brain, _ = _brain('ok')
    brain._consecutive_errors = 3
    assert brain.warmup() is True
    assert brain._consecutive_errors == 0


def test_warmup_failure_is_not_fatal():
    """Ollama absent : warmup renvoie False, ne leve pas, et le bot continue."""
    brain, _ = _brain(boom=ConnectionError('absent'))
    assert brain.warmup() is False
    assert brain.decide(WORLD) is not None          # l'heuristique prend le relais


def test_warmup_keeps_the_model_resident():
    """`keep_alive` evite qu'Ollama decharge le modele entre deux decisions —
    sinon on repaie le demarrage a froid."""
    captured = {}

    class _Spy:
        def chat(self, **kw):
            captured.update(kw)
            return {'message': {'content': 'ok'}}

    from clashai.agents.scheduler import AgentScheduler
    from clashai.brain.llm_brain import KEEP_ALIVE
    brain = LocalLLMBrain(AgentScheduler(), client=_Spy(), verbose=False)
    brain.warmup()
    assert captured.get('keep_alive') == KEEP_ALIVE


# ---------------------------------------------------------------------------
# Anti-invention : nommer explicitement ce qu'on ne sait pas
#
# Cas reel (19 aout) : le monde ne contenait pas les ressources, le LLM voyait
# seulement le NOM du bouton `compteur_or`... et a repondu "tu as 15568 or".
# Un 7B comble les trous. Le seul remede fiable est de nommer l'ignorance.
# ---------------------------------------------------------------------------

def test_missing_readings_are_announced_as_unknown():
    brain, _ = _brain('ok')
    desc = brain.describe_world({'screen_state': 'village_home',
                                 'buttons': {'compteur_or': (1, 2, 0.9)}})
    assert 'NON LUES' in desc, "les ressources absentes doivent etre annoncees"
    assert 'NON LUS' in desc                      # ouvriers
    assert 'inconnu' in desc                      # laboratoire


def test_real_readings_are_shown_verbatim():
    brain, _ = _brain('ok')
    desc = brain.describe_world({
        'readings': {'resources': {'or': 4261458},
                     'builders': {'libres': 5, 'total': 5},
                     'lab_libre': True}})
    assert '4261458' in desc
    assert '- ouvriers libres : 5' in desc
    assert 'libre' in desc
    assert 'NON LUES' not in desc


def test_buttons_are_labelled_as_names_not_values():
    """`compteur_or` est un NOM d'element detecte, pas un montant : le prompt
    doit le dire, sinon le modele le lit comme une donnee."""
    brain, _ = _brain('ok')
    desc = brain.describe_world({'buttons': {'compteur_or': (1, 2, 0.9)}})
    assert 'noms techniques' in desc


def test_decision_prompt_also_carries_the_readings():
    """La decision doit voir les vraies valeurs, pas seulement le chat."""
    brain, sched = _brain('{"agent": "combat"}')
    prompt = brain.build_prompt({'readings': {'resources': {'or': 42}}},
                                sched.agents)
    assert '42' in prompt and 'Agents éligibles' in prompt


# ---------------------------------------------------------------------------
# Le prompt est une interface pour un MODELE, pas un dump de dict
#
# Bug reel (19 aout 2026) : avec la ligne unique
#   « ressources : elixir = 2399904, elixir_noire = 18549, or = 2235125 »
# Mistral 7B repondait « les seules ressources indiquees sont l'elixir et l'or ».
# `elixir_noire` CONTIENT `elixir`, et `or` est aussi un mot francais.
# ---------------------------------------------------------------------------

def _desc_resources(brain, resources):
    return brain.describe_world({'readings': {'resources': resources}})


def test_each_resource_gets_its_own_line_in_french():
    brain, _ = _brain('ok')
    desc = _desc_resources(brain, {'or': 2235125, 'elixir': 2399904,
                                   'elixir_noire': 18549})
    assert '- or : 2235125' in desc
    assert '- élixir : 2399904' in desc
    assert '- élixir noir : 18549' in desc


def test_technical_resource_keys_never_reach_the_prompt():
    """`elixir_noire` est un nom d'API : il ne doit pas etre montre au modele."""
    brain, _ = _brain('ok')
    desc = _desc_resources(brain, {'or': 1, 'elixir': 2, 'elixir_noire': 3})
    assert 'elixir_noire' not in desc


def test_resources_keep_the_hud_order_not_alphabetical():
    """Or, elixir, elixir noir — l'ordre du HUD. L'ordre alphabetique mettait
    `elixir` avant `elixir_noire`, ce qui favorisait justement la fusion."""
    brain, _ = _brain('ok')
    desc = _desc_resources(brain, {'elixir_noire': 3, 'elixir': 2, 'or': 1})
    assert desc.index('- or :') < desc.index('- élixir :') < desc.index('- élixir noir :')


def test_an_unknown_resource_is_still_reported():
    """Une ressource future (non libellee) doit apparaitre plutot que
    disparaitre silencieusement du prompt."""
    brain, _ = _brain('ok')
    desc = _desc_resources(brain, {'or': 1, 'gemme': 414})
    assert '414' in desc


def test_partial_resources_do_not_invent_the_missing_ones():
    brain, _ = _brain('ok')
    desc = _desc_resources(brain, {'or': 7})
    assert '- or : 7' in desc
    assert 'élixir' not in desc


# ---------------------------------------------------------------------------
# Le refus doit avoir un declencheur LITTERAL, pas une ambiance
#
# Sur-correction mesuree : a force de marteler « n'invente jamais », le modele
# repondait « le montant d'elixir noir est NON LUE » alors que la valeur ETAIT
# presente. Le prompt doit donc (a) affirmer que les chiffres font autorite,
# (b) restreindre le refus au marqueur exact.
# ---------------------------------------------------------------------------

def test_chat_prompt_asserts_the_numbers_are_authoritative():
    from clashai.brain.llm_brain import CHAT_SYSTEM_PROMPT
    assert 'AUTORITÉ' in CHAT_SYSTEM_PROMPT


def test_chat_prompt_restricts_refusal_to_the_literal_marker():
    from clashai.brain.llm_brain import CHAT_SYSTEM_PROMPT
    assert 'NON LUE' in CHAT_SYSTEM_PROMPT
    assert 'QUE si' in CHAT_SYSTEM_PROMPT     # le refus est conditionne


def test_chat_prompt_decouples_buttons_from_values():
    """Le modele deduisait « le compteur n'est pas dans les boutons, donc je ne
    connais pas la valeur ». Les deux listes sont independantes."""
    from clashai.brain.llm_brain import CHAT_SYSTEM_PROMPT
    assert 'AUCUNE incidence' in CHAT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Comptes : recoltes disponibles et dons en attente (increment 5.3.1)
#
# `buttons` ne garde qu'UNE detection par classe ; le NOMBRE etait jete alors
# qu'il est deja calcule. Une capture reelle : 5 recolter_or, 6 recolter_elixir,
# 3 recolter_elixir_noire.
# ---------------------------------------------------------------------------

def test_each_collector_gets_its_own_line():
    """La liste a virgules « 5 or, 6 elixir, 3 elixir noir » reproduisait le bug
    d'origine : a « combien de collecteurs d'or ? » le modele repondait « six »,
    le compte de l'ELIXIR. Une ligne par ressource, comme pour les stocks."""
    brain, _ = _brain('ok')
    desc = brain.describe_world({'readings': {
        'recoltes': {'or': 5, 'elixir': 6, 'elixir_noire': 3}}})
    assert "- collecteurs d'or pleins, prêts à récolter : 5" in desc
    assert "- collecteurs d'élixir pleins, prêts à récolter : 6" in desc
    assert "- collecteurs d'élixir noir pleins, prêts à récolter : 3" in desc
    assert 'elixir_noire' not in desc          # jamais de cle technique


def test_collectors_keep_the_hud_order():
    brain, _ = _brain('ok')
    desc = brain.describe_world({'readings': {
        'recoltes': {'elixir_noire': 3, 'or': 5}}})
    assert desc.index("collecteurs d'or") < desc.index("collecteurs d'élixir noir")


def test_pending_donations_say_what_they_mean():
    """« demandes de dons en attente : 2 » seul faisait repondre « aucun » a
    « combien de membres attendent des troupes ? ». Le libelle doit porter le
    SENS, pas seulement le nom du champ."""
    brain, _ = _brain('ok')
    desc = brain.describe_world({'readings': {'dons_en_attente': 2}})
    assert 'demandes de dons en attente' in desc and ': 2' in desc
    assert 'troupes' in desc


def test_nothing_to_collect_stays_silent():
    """« 0 collecteur plein » est du bruit : une ligne absente ne peut pas etre
    lue comme une valeur inventee, une ligne a zero invite au commentaire."""
    brain, _ = _brain('ok')
    desc = brain.describe_world({'readings': {'recoltes': {},
                                              'dons_en_attente': 0}})
    assert 'collecteurs' not in desc and 'dons en attente' not in desc


def test_an_unknown_collector_key_is_still_reported():
    """Une ressource future doit apparaitre plutot que disparaitre du prompt."""
    brain, _ = _brain('ok')
    desc = brain.describe_world({'readings': {'recoltes': {'or': 1, 'gemme': 2}}})
    assert "collecteurs d'or pleins, prêts à récolter : 1" in desc
    assert "collecteurs d'gemme pleins, prêts à récolter : 2" in desc


def test_builders_are_split_into_two_unambiguous_lines():
    """« 4 libres sur 5 » = deux nombres sur une ligne : le modele attrapait le
    dernier et repondait « 5 ouvriers libres »."""
    brain, _ = _brain('ok')
    desc = brain.describe_world({'readings': {
        'builders': {'libres': 4, 'total': 5}}})
    assert '- ouvriers libres : 4' in desc
    assert '- ouvriers au total : 5' in desc
    assert 'libres sur' not in desc
