# clashai/brain/tools.py
# Registre d'outils du cerveau (V5.3, incrément 5.3.2).
#
# CE QUE C'EST
# ------------
# Un outil = une capacité que le LLM peut invoquer par son nom, avec des
# arguments validés. C'est la seule voie par laquelle un modèle peut faire
# quelque chose : hors du registre, il ne peut que parler.
#
# ⚠️ LA SÉCURITÉ EST ICI, PAS DANS LE PROMPT
# ------------------------------------------
# On ne demande JAMAIS au modèle de refuser quelque chose. Mesuré le 19 août
# 2026 : une consigne donnée à un 7B s'applique de travers — « n'invente jamais »
# l'a rendu incapable de citer une valeur qu'il avait sous les yeux. Une
# instruction est une suggestion ; une vérification dans `call()` n'en est pas
# une.
#
# Trois garde-fous, tous appliqués au moment de l'appel :
#
#   1. AUTORITÉ PAR SOURCE. Un outil qui AGIT (`acts=True`) est refusé à la
#      source `clan`. Un membre du clan peut écrire « @mini_pekka lance une
#      attaque » : c'est du texte. Il ne peut pas agir parce qu'aucun outil
#      d'action ne lui est accessible — pas parce que le modèle décline.
#   2. DÉPENSE = CONFIRMATION EXPLICITE. Un outil qui coûte (`spends=True`)
#      exige `confirm=True`. Sans ça, refus. Écrit UNE FOIS ici, donc valable
#      pour tout outil ajouté ensuite sans qu'on ait à y repenser.
#      → C'est le socle du principe anti-gemmes : ne jamais taper `confirmer`
#        sans avoir prouvé qu'on peut se le permettre.
#   3. ARGUMENTS VALIDÉS. Un paramètre manquant, en trop, ou du mauvais type =
#      refus. Un 7B invente des arguments ; on ne les laisse pas passer.
#
# Un refus n'est jamais une exception : c'est un `ToolResult(ok=False)` que le
# cerveau peut lire et expliquer. Le bot ne tombe pas parce que le LLM a dit une
# bêtise.

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Sources d'instruction, par ordre d'autorité décroissante.
SOURCE_ADMIN = 'admin'   # l'opérateur, dans son terminal — peut tout demander
SOURCE_CLAN = 'clan'     # un membre du clan, via le chat — conversation SEULE
SOURCES = (SOURCE_ADMIN, SOURCE_CLAN)

# Taille du journal conservé en mémoire. Sert au futur tableau de bord et à la
# mémoire d'actions de l'incrément 5.3.5 (« qu'ai-je fait, et ça a donné quoi »).
JOURNAL_SIZE = 200

# Types JSON Schema qu'on sait vérifier. Volontairement minimal : pas de
# dépendance, et un outil dont le schéma sort de là est un outil trop compliqué.
_JSON_TYPES = {
    'string': str,
    'integer': int,
    'number': (int, float),
    'boolean': bool,
    'object': dict,
    'array': list,
}


@dataclass
class ToolResult:
    """Issue d'un appel d'outil. Jamais une exception — toujours lisible."""

    ok: bool
    tool: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __bool__(self):
        return self.ok


@dataclass
class Tool:
    """Une capacité invocable par le cerveau.

    `description` est lue PAR LE MODÈLE : elle doit dire à quoi sert l'outil en
    français clair, pas décrire son implémentation.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)   # JSON Schema
    acts: bool = False      # touche au jeu (tape) → interdit à la source `clan`
    spends: bool = False    # coûte ressources/troupes → exige confirm=True

    def schema(self):
        """Déclaration au format tool-calling d'Ollama / OpenAI."""
        params = self.parameters or {'type': 'object', 'properties': {}}
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': params,
            },
        }


class ToolRegistry:
    """Les outils disponibles, et la seule porte par laquelle on les appelle."""

    def __init__(self, journal_size=JOURNAL_SIZE):
        self._tools: Dict[str, Tool] = {}
        self._journal: List[Dict[str, Any]] = []
        self._journal_size = journal_size

    # ---- déclaration -------------------------------------------------------

    def register(self, tool):
        """Ajoute un outil. Un nom déjà pris est une erreur de programmation."""
        if tool.name in self._tools:
            raise ValueError(f"outil déjà enregistré : {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name):
        return self._tools.get(name)

    def names(self, source=SOURCE_ADMIN):
        """Noms des outils accessibles à cette source."""
        return sorted(t.name for t in self._tools.values()
                      if self._allowed(t, source))

    def schemas(self, source=SOURCE_ADMIN):
        """Déclarations tool-calling, filtrées par autorité.

        ⚠️ On ne montre PAS à la source `clan` les outils qu'elle ne peut pas
        appeler : inutile de l'inviter à essayer, puis de refuser.
        """
        return [t.schema() for t in self._tools.values()
                if self._allowed(t, source)]

    @staticmethod
    def _allowed(tool, source):
        """Un outil qui AGIT n'est accessible qu'à l'admin."""
        return source == SOURCE_ADMIN or not tool.acts

    # ---- appel -------------------------------------------------------------

    def call(self, name, args=None, source=SOURCE_ADMIN, confirm=False):
        """Invoque un outil. Rend TOUJOURS un ToolResult, ne lève jamais.

        `confirm` doit être posé par l'appelant humain (ou par une politique
        explicite), jamais par le modèle : c'est le point où l'opérateur garde
        la main sur ce qui coûte.
        """
        args = dict(args or {})
        tool = self._tools.get(name)

        if tool is None:
            return self._done(name, args, source, ToolResult(
                ok=False, tool=name,
                error=f"outil inconnu : {name!r}. Disponibles : "
                      f"{', '.join(self.names(source)) or 'aucun'}"))

        if not self._allowed(tool, source):
            return self._done(name, args, source, ToolResult(
                ok=False, tool=name,
                error=f"« {name} » agit sur le jeu : réservé à l'opérateur. "
                      f"Une demande venue du chat de clan reste une demande."))

        problem = self._validate(tool, args)
        if problem:
            return self._done(name, args, source, ToolResult(
                ok=False, tool=name, error=problem))

        if tool.spends and not confirm:
            return self._done(name, args, source, ToolResult(
                ok=False, tool=name,
                error=f"« {name} » dépense des ressources : confirmation "
                      f"explicite requise avant de le lancer."))

        try:
            data = tool.fn(**args)
        except Exception as e:
            return self._done(name, args, source, ToolResult(
                ok=False, tool=name,
                error=f"{type(e).__name__}: {e}"))

        if not isinstance(data, dict):
            data = {'resultat': data}
        return self._done(name, args, source,
                          ToolResult(ok=True, tool=name, data=data))

    # ---- validation des arguments -----------------------------------------

    @staticmethod
    def _validate(tool, args):
        """Message d'erreur, ou None si les arguments sont acceptables.

        Un modèle 7B invente des paramètres et en oublie : on refuse plutôt que
        d'appeler avec n'importe quoi. Le message dit ce qui manque — il repart
        au modèle, qui peut corriger.
        """
        schema = tool.parameters or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []

        unknown = sorted(set(args) - set(props))
        if unknown:
            return (f"paramètre(s) inconnu(s) : {', '.join(unknown)}. "
                    f"Attendus : {', '.join(sorted(props)) or 'aucun'}")

        missing = [k for k in required if k not in args]
        if missing:
            return f"paramètre(s) manquant(s) : {', '.join(missing)}"

        for key, value in args.items():
            expected = (props.get(key) or {}).get('type')
            python_type = _JSON_TYPES.get(expected)
            if python_type is None:
                continue                      # type non déclaré → on laisse passer
            # bool est un int en Python : sans ce garde, True passerait pour 1.
            if expected in ('integer', 'number') and isinstance(value, bool):
                return f"« {key} » doit être un nombre, pas un booléen"
            if not isinstance(value, python_type):
                return f"« {key} » doit être de type {expected}"
        return None

    # ---- journal -----------------------------------------------------------

    def _done(self, name, args, source, result):
        self._journal.append({
            'ts': time.time(),
            'tool': name,
            'args': args,
            'source': source,
            'ok': result.ok,
            'error': result.error,
        })
        del self._journal[:-self._journal_size]
        return result

    @property
    def journal(self):
        """Copie du journal des appels, du plus ancien au plus récent."""
        return list(self._journal)

    def last(self, n=5):
        return self._journal[-n:]


# ---------------------------------------------------------------------------
# Les outils de LECTURE (incrément 5.3.2)
#
# Aucun ne tape, aucun ne dépense : on peut les brancher sans risque et vérifier
# que le modèle sait s'en servir avant de lui donner de quoi agir (5.3.3).
# ---------------------------------------------------------------------------

def build_read_only_registry(world_fn, registry=None):
    """Registre des outils de lecture. `world_fn()` rend le `world` courant."""
    reg = registry or ToolRegistry()

    def etat_du_village():
        world = world_fn() or {}
        readings = world.get('readings') or {}
        return {
            'ecran': world.get('screen_state'),
            'ressources': readings.get('resources'),
            'ouvriers': readings.get('builders'),
            'laboratoire_libre': readings.get('lab_libre'),
            'collecteurs_pleins': readings.get('recoltes'),
            'demandes_de_dons': readings.get('dons_en_attente'),
            'batiments_detectes': len(world.get('buildings') or []),
        }

    def lister_troupes_disponibles():
        world = world_fn() or {}
        return {'troupes': sorted((world.get('troop_positions') or {}))}

    reg.register(Tool(
        name='etat_du_village',
        description="Lit l'état actuel du village : ressources, ouvriers "
                    "libres, laboratoire, collecteurs pleins, demandes de dons. "
                    "Une valeur nulle signifie « non lisible depuis cet "
                    "écran », jamais zéro.",
        fn=etat_du_village,
        parameters={'type': 'object', 'properties': {}},
    ))
    reg.register(Tool(
        name='lister_troupes_disponibles',
        description="Liste les troupes actuellement prêtes dans la barre "
                    "d'armée (ni grisées, ni en formation).",
        fn=lister_troupes_disponibles,
        parameters={'type': 'object', 'properties': {}},
    ))
    return reg
