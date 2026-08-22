# clashai/brain/prompt_eval.py
# Banc d'essai des prompts (V5.3, incrément 5.3.0).
#
# POURQUOI CE FICHIER EXISTE
# --------------------------
# Le 19 août 2026, un correctif validé à la main était cassé — pas dans le code,
# dans la façon dont le modèle LISAIT le prompt. Deux propriétés de ce bug le
# rendent impossible à attraper autrement :
#
#   1. les tests unitaires ne le voient pas : le code était juste, c'est
#      l'interprétation du modèle qui était fausse ;
#   2. il était INTERMITTENT SELON LA FORMULATION — « combien j'ai d'élixir
#      noir ? » marchait, « combien exactement ? » échouait. Un essai manuel
#      unique concluait donc « ça marche ».
#
# Toute la V5.3 consiste à faire interpréter du français à un 7B. Cette classe de
# défaut sera le mode d'échec dominant, pas les bugs de code. D'où ce banc : on
# mesure un TAUX sur plusieurs formulations, jamais un essai.
#
# LES DEUX SENS
# -------------
# Un correctif qui gagne d'un côté en perdant de l'autre n'est pas un correctif.
# Chaque intention se teste donc dans les deux sens :
#
#   RAPPEL  : la valeur est lue      -> il doit la donner
#   RETENUE : la valeur n'est pas lue -> il doit refuser, sans inventer
#
# Corriger l'hallucination avait cassé le rappel (3/6). Corriger le rappel
# pouvait rouvrir l'hallucination. Seule la mesure croisée le dit.
#
# ARCHITECTURE
# ------------
# Ce module ne connaît NI Ollama NI LocalLLMBrain : il reçoit un `ask(question,
# world) -> str|None`. Tout est donc testable avec un faux modèle, et le banc
# peut servir plus tard à comparer deux modèles ou deux prompts.
#
# ÉTAT DE RÉFÉRENCE — Mistral 7B, 19 août 2026 : **98/102 (96 %)** à --repeat 3,
# après l'incrément 5.3.1.
#
# ⚠️ LEÇON PRINCIPALE : sur 3 « défauts du modèle » identifiés, 2 étaient des
# défauts de MON PROMPT. Avant d'accuser le 7B, relire la ligne qu'on lui donne.
#
#   ✅ CORRIGÉ — « est-ce que j'ai UN ouvrier ? » → « Oui, tu as 1 ouvrier ».
#      J'avais noté « il recopie le déterminant de la question ». FAUX : la ligne
#      « ouvriers : 4 libres sur 5 » portait DEUX nombres, et il attrapait le
#      mauvais. Séparée en « ouvriers libres : 4 » + « ouvriers au total : 5 » :
#      0/3 → 6/6.
#   ✅ CORRIGÉ — « combien de collecteurs d'or ? » → « Six » (le compte de
#      l'élixir). La liste à virgules « 5 or, 6 élixir, 3 élixir noir » refaisait
#      le bug d'origine. Une ligne par ressource : 5/6 → 6/6.
#   ⏳ OUVERT — `dons_en_attente`, formulation INDIRECTE. « il y a combien de
#      demandes de dons ? » → 3/3 parfait. « combien de membres attendent des
#      troupes ? » → 0/3, avec trois réponses différentes (« Quatre », « Aucun »,
#      « 3 ») : il devine. Le modèle ne fait pas le pont entre la paraphrase et
#      la ligne. Un libellé plus explicite a déjà fait passer ce cas de 4/6 à
#      9/10 en isolé, mais l'effet ne tient pas quand le prompt s'allonge.
#   ⏳ OUVERT — `ressources.or/retenue`, formulation ELLIPTIQUE. « mon stock d'or
#      stp » sur un world vide → « Or : 12345 », un nombre bouche-trou (1/9).
#      La question directe (« combien j'ai d'or ? ») tient parfaitement.
#
# ⚠️ NE PAS AJUSTER LE PROMPT JUSQU'À CE QU'UNE FORMULATION PRÉCISE PASSE : un
# banc qu'on optimise ne mesure plus rien. On corrige une CAUSE (format ambigu,
# liste à virgules, clé technique), jamais un cas.
#
# ⚠️ Tentative ratée, à ne pas refaire : ajouter « tout chiffre que tu cites
# vient de l'état, jamais de la question » ne corrige rien et n'améliore rien.
#
# ⚠️ TENSION MESURÉE : plus le prompt s'allonge, plus les formulations courtes ou
# indirectes se dégradent. Chaque champ ajouté au `world` a donc un coût — à
# surveiller au banc quand on en ajoutera d'autres (coûts d'upgrade, etc.).

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Vérificateurs de réponse
# ---------------------------------------------------------------------------

# Un nombre d'au moins 3 chiffres, tolérant les séparateurs que les modèles
# ajoutent spontanément ("2 235 125", "2.235.125", "18,549").
_BIG_NUMBER = re.compile(r'\d[\d\s.,  ]{2,}')


# Petits nombres écrits en toutes lettres. Mesuré : le modèle répond « Six
# collecteurs d'or sont prêts » — sans conversion, le banc raterait une réponse
# JUSTE écrite en français.
#
# ⚠️ Sont volontairement ABSENTS :
#   - « un »/« une » : des articles avant d'être des nombres (« un ouvrier »,
#     « d'une part »). Les convertir créerait des succès fantômes partout où le
#     chiffre attendu est 1.
#   - « aucun »/« aucune » : « je n'ai AUCUNE information » est un refus tout à
#     fait correct, pas une affirmation de zéro. Les compter comme un chiffre
#     ferait échouer les meilleurs refus. On tolère donc « aucun collecteur »,
#     qui sur-affirme un peu — un faux positif coûte plus cher au banc qu'une
#     indulgence.
_WORD_NUMBERS = {
    'zero': 0, 'zéro': 0,
    'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5, 'six': 6, 'sept': 7,
    'huit': 8, 'neuf': 9, 'dix': 10, 'onze': 11, 'douze': 12,
}
_WORD_RE = re.compile(r'\b(' + '|'.join(_WORD_NUMBERS) + r')\b', re.IGNORECASE)


def digits_of(text):
    """Tous les chiffres du texte, concaténés.

    Volontairement brutal : Mistral coupe parfois un nombre au mauvais endroit
    (« 1 8549 » pour 18549). En comparant sur la suite des chiffres, une valeur
    correctement rapportée mais mal formatée compte comme un succès — ce qui est
    le bon jugement : la donnée est juste, seule la typographie déraille.

    Les petits nombres écrits en toutes lettres sont convertis d'abord (voir
    `_WORD_NUMBERS`) : « six collecteurs » doit compter comme « 6 collecteurs ».
    """
    text = _WORD_RE.sub(lambda m: str(_WORD_NUMBERS[m.group(1).lower()]),
                        text or '')
    return re.sub(r'\D', '', text)


def says_number(expected):
    """Vérificateur : la réponse contient ce nombre."""
    needle = str(expected)

    def check(answer):
        return needle in digits_of(answer)

    return check


def says_no_number():
    """Vérificateur : aucun montant inventé.

    À n'utiliser que sur un `world` SANS lectures : là, tout nombre à 3 chiffres
    ou plus ne peut venir que du modèle.
    """

    def check(answer):
        return not _BIG_NUMBER.search(answer or '')

    return check


def says_numbers(*expected):
    """Vérificateur : tous ces nombres apparaissent.

    ⚠️ NE PAS utiliser `says_all_of` pour des nombres. Le banc a commencé par
    faire cette erreur et notait 0/2 une réponse pourtant parfaite : le modèle
    écrit « 2 235 125 », le vérificateur cherchait « 2235125 ». Un banc qui se
    trompe donne une confiance fausse — pire que pas de banc.
    """

    def check(answer):
        digits = digits_of(answer)
        return all(str(n) in digits for n in expected)

    return check


def says_no_digit():
    """Vérificateur strict : AUCUN chiffre.

    `says_no_number` tolère les petits nombres (seuil à 3 chiffres) pour ne pas
    crier au loup sur « 4 ouvriers ». Mais quand la valeur attendue EST un petit
    compte — collecteurs prêts, demandes de dons — inventer « 5 » passerait
    inaperçu. Sur un `world` totalement vide, tout chiffre est une invention.
    """

    def check(answer):
        return not digits_of(answer)

    return check


def says_all_of(*needles):
    """Vérificateur : tous ces fragments TEXTE apparaissent (casse ignorée).

    Pour des nombres, utiliser `says_numbers` (voir ci-dessus).
    """

    def check(answer):
        low = (answer or '').lower()
        return all(n.lower() in low for n in needles)

    return check


def says_number_or_stays_vague(expected):
    """Vérificateur pour les questions FERMÉES (« est-ce que j'ai un ouvrier ? »).

    Un simple « oui » est une réponse correcte : exiger le chiffre serait
    injuste. En revanche, s'il avance un chiffre, il doit être le bon —
    « Oui, tu as 1 ouvrier » alors qu'il y en a 4 reste une erreur factuelle.
    """

    def check(answer):
        digits = digits_of(answer)
        return str(expected) in digits if digits else True

    return check


# ---------------------------------------------------------------------------
# Cas
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """Une intention, plusieurs façons de la formuler, un critère de réussite."""

    intent: str                       # 'ressources.elixir_noire/rappel'
    world: dict
    phrasings: List[str]
    check: Callable[[Optional[str]], bool]
    why: str = ''                     # ce que ce cas protège


@dataclass
class PhrasingResult:
    phrasing: str
    answer: Optional[str]
    ok: bool


@dataclass
class CaseResult:
    case: Case
    results: List[PhrasingResult] = field(default_factory=list)

    @property
    def passed(self):
        return sum(r.ok for r in self.results)

    @property
    def total(self):
        return len(self.results)

    @property
    def score(self):
        return (self.passed / self.total) if self.total else 0.0

    @property
    def failures(self):
        return [r for r in self.results if not r.ok]


@dataclass
class Report:
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def passed(self):
        return sum(c.passed for c in self.cases)

    @property
    def total(self):
        return sum(c.total for c in self.cases)

    @property
    def score(self):
        return (self.passed / self.total) if self.total else 0.0

    def meets(self, threshold):
        return self.score >= threshold


def run_case(case, ask, repeat=1):
    """Joue toutes les formulations d'un cas. Aucune exception ne remonte.

    Un `ask` qui lève ou rend None compte comme un échec : un cerveau
    indisponible est un cerveau qui ne passe pas le banc, pas un banc cassé.
    """
    out = CaseResult(case=case)
    for phrasing in case.phrasings:
        for _ in range(max(1, repeat)):
            try:
                answer = ask(phrasing, case.world)
            except Exception as e:
                answer = None
                out.results.append(PhrasingResult(
                    phrasing, f'[{type(e).__name__}]', False))
                continue
            ok = bool(answer) and case.check(answer)
            out.results.append(PhrasingResult(phrasing, answer, ok))
    return out


def run_suite(suite, ask, repeat=1, on_case=None):
    """Joue toute la suite. `on_case(CaseResult)` permet d'afficher au fil de l'eau."""
    report = Report()
    for case in suite:
        result = run_case(case, ask, repeat=repeat)
        report.cases.append(result)
        if on_case:
            on_case(result)
    return report


# ---------------------------------------------------------------------------
# La suite de référence
# ---------------------------------------------------------------------------

_READINGS = {
    'resources': {'or': 2235125, 'elixir': 2399904, 'elixir_noire': 18549},
    'builders': {'libres': 4, 'total': 5},
    'lab_libre': False,
    # Comptes ajoutés en 5.3.1 — repris d'une capture réelle (19 août 2026).
    'recoltes': {'or': 5, 'elixir': 6, 'elixir_noire': 3},
    'dons_en_attente': 2,
}

# Boutons réellement détectés sur une capture de village (19 août 2026). On les
# garde dans TOUS les mondes, y compris ceux sans lecture : c'est justement le
# piège d'origine — le modèle voyait `compteur_or` et en déduisait un montant.
_BUTTONS = {
    'attaquer': (1, 2, 0.96), 'compteur_or': (3, 4, 0.91),
    'compteur_elixir': (5, 6, 0.90), 'compteur_elixir_noire': (7, 8, 0.93),
    'nombre_ouvrier': (9, 10, 0.80), 'place_labo': (11, 12, 0.92),
}

WORLD_LU = {
    'screen_state': 'village_home', 'mode': 'auto', 'buildings': [],
    'buttons': dict(_BUTTONS), 'readings': _READINGS,
}

# Même écran, mêmes boutons, mais RIEN n'a pu être lu.
WORLD_NON_LU = {
    'screen_state': 'village_home', 'mode': 'auto', 'buildings': [],
    'buttons': dict(_BUTTONS), 'readings': {},
}


SUITE = [
    # ---- RAPPEL : la valeur est là, il doit la donner ----------------------
    Case(
        intent='ressources.or/rappel',
        world=WORLD_LU,
        phrasings=[
            "combien j'ai d'or ?",
            "mon stock d'or stp",
            "il me reste combien d'or ?",
            "or ?",
        ],
        check=says_number(2235125),
        why="L'or a toujours marché — c'est le témoin. S'il tombe, "
            "la régression est générale, pas spécifique à une ressource.",
    ),
    Case(
        intent='ressources.elixir/rappel',
        world=WORLD_LU,
        phrasings=[
            "combien j'ai d'elixir ?",
            "mon elixir rose il est a combien ?",
            "il me reste combien d'elixir ?",
        ],
        check=says_number(2399904),
        why="L'élixir doit rester distinct de l'élixir NOIR : c'est la "
            "confusion qui a produit le bug du 19 août.",
    ),
    Case(
        intent='ressources.elixir_noire/rappel',
        world=WORLD_LU,
        phrasings=[
            "combien j'ai d'elixir noire ?",
            "combien d'elixir noir ai-je exactement ?",
            "j'ai combien d'elixir noir la ?",
            "il me reste combien de dark elixir ?",
            "mon stock d'elixir noir stp",
            "elixir noir ?",
        ],
        check=says_number(18549),
        why="LE cas du bug. `elixir_noire` contient `elixir` et `or` est un mot "
            "français : en une seule ligne de clés techniques, le modèle "
            "fusionnait et n'annonçait que deux ressources.",
    ),
    Case(
        intent='ressources.toutes/rappel',
        world=WORLD_LU,
        phrasings=[
            "liste moi mes 3 ressources",
            "fais le point sur mes ressources",
        ],
        check=says_numbers(2235125, 2399904, 18549),
        why="Les trois doivent coexister dans UNE réponse — c'est le test le "
            "plus dur de la distinction entre élixir et élixir noir.",
    ),
    Case(
        intent='ouvriers/rappel',
        world=WORLD_LU,
        phrasings=[
            "j'ai combien d'ouvriers libres ?",
            "mes ouvriers ?",
            "combien d'ouvriers sont dispo ?",
        ],
        check=says_numbers(4),
        why="Un ratio, pas un montant : autre format de lecture, autre risque.",
    ),
    Case(
        intent='ouvriers.question_fermee/rappel',
        world=WORLD_LU,
        phrasings=[
            "est-ce que j'ai un ouvrier de dispo ?",
            "je peux lancer une amelioration la ?",
        ],
        check=says_number_or_stays_vague(4),
        why="Question fermée : « oui » suffit. Mais s'il avance un chiffre, il "
            "doit être le bon — mesuré le 19 août : « Oui, tu as 1 ouvrier » "
            "alors que le world en annonçait 4.",
    ),

    Case(
        intent='recoltes/rappel',
        world=WORLD_LU,
        phrasings=[
            "combien de collecteurs d'or sont prets ?",
            "j'ai combien de collecteurs d'or a recolter ?",
        ],
        check=says_numbers(5),
        why="Un COMPTE, pas un montant : `buttons` ne gardait qu'une detection "
            "par classe et jetait le nombre, pourtant deja calcule.",
    ),
    Case(
        intent='dons_en_attente/rappel',
        world=WORLD_LU,
        phrasings=[
            "il y a combien de demandes de dons ?",
            "combien de membres attendent des troupes ?",
        ],
        check=says_numbers(2),
        why="Ce compte decidera si l'agent dons vaut la peine d'etre lance.",
    ),

    # ---- RETENUE : rien n'est lu, il ne doit RIEN inventer ------------------
    Case(
        intent='ressources.or/retenue',
        world=WORLD_NON_LU,
        phrasings=[
            "combien j'ai d'or ?",
            "mon stock d'or stp",
            "il me reste combien d'or ?",
        ],
        check=says_no_number(),
        why="L'hallucination d'origine : « tu as 15568 or », inventé à partir du "
            "seul NOM du bouton `compteur_or`.",
    ),
    Case(
        intent='ressources.elixir_noire/retenue',
        world=WORLD_NON_LU,
        phrasings=[
            "combien d'elixir noir ai-je exactement ?",
            "il me reste combien de dark elixir ?",
            "elixir noir ?",
        ],
        check=says_no_number(),
        why="Le pendant du cas de rappel : réparer le rappel ne doit pas "
            "rouvrir la porte à l'invention.",
    ),
    Case(
        intent='comptes/retenue',
        world=WORLD_NON_LU,
        phrasings=[
            "combien de collecteurs d'or sont prets ?",
            "il y a combien de demandes de dons ?",
        ],
        check=says_no_digit(),
        why="Un petit compte invente (« il y en a 5 ») passerait sous le seuil "
            "de `says_no_number` : ici, sur un world vide, TOUT chiffre ment.",
    ),
    Case(
        intent='ouvriers/retenue',
        world=WORLD_NON_LU,
        phrasings=[
            "j'ai combien d'ouvriers libres ?",
            "mes ouvriers ?",
        ],
        check=says_no_number(),
        why="Un ouvrier inventé enverrait le bot tenter une amélioration "
            "impossible.",
    ),
]
