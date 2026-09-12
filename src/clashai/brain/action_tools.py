# clashai/brain/action_tools.py
# Outils qui AGISSENT sur le jeu (V5.3, étape 5.3.3a).
#
# 5.3.2 a posé le registre et ses trois garde-fous (autorité, dépense, arguments)
# avec des outils de lecture. Ici, les outils qui TAPENT : récolte, attaque,
# renforts, dons, recherche au labo. Aucun geste n'est nouveau : chaque outil
# enveloppe un agent ou un module déjà validé en réel.
#
# ⚠️ OÙ S'EXÉCUTENT-ILS ? Jamais dans la console. La console les MET EN FILE ;
# c'est la boucle du bot — le seul thread qui tape — qui les exécute entre deux
# agents (`commands.run_next`, branché en 5.3.3b).
#
# ⚠️ DEUX VERROUS PAR DÉPENSE, indépendants :
#   1. l'INTENTION — le `o/n` de l'opérateur, porté par `confirm=True` ;
#   2. la PREUVE — le module vérifie prix lu + ressource identifiée + solde.
# Aucun outil ne passe de `confirm_decider` : jusqu'au 12 sept. 2026, un décideur
# REMPLAÇAIT la preuve (voir TROUBLESHOOTING « Un décideur pouvait confirmer
# sans preuve »). Vouloir dépenser n'est pas pouvoir payer.
#
# ⚠️ NOMS DE TROUPES : validés contre le registre (`troops.json`, héros exclus)
# croisé avec les classes du CNN de la barre. Accents, casse et pluriels sont
# normalisés (« yétis » -> `yeti`), mais JAMAIS d'approximation : « dragonn » est
# refusé, avec des suggestions. Donner la mauvaise troupe à un membre ne rend
# service à personne.

import itertools
import re
import unicodedata

from clashai.brain.tools import Tool

# Mots vides retirés avant de comparer (« golem de glace » -> `golem_glace`).
_STOPWORDS = frozenset({'de', 'du', 'des', 'd', 'la', 'le', 'les', 'l', 'et'})

# Un nom de troupe fait au plus 4 mots : borne le découpage glouton et le nombre
# de variantes singulier/pluriel essayées (2^4 = 16).
MAX_NAME_WORDS = 4


class Refus(Exception):
    """L'outil n'a rien fait, et dit pourquoi (troupe inconnue, labo occupé…)."""


class Echec(Exception):
    """L'action a été tentée mais n'a pas abouti."""


# ---------------------------------------------------------------------------
# Noms de troupes
# ---------------------------------------------------------------------------

def donatable_troop_names(troop_types=None, cnn_names=None):
    """Noms qu'on peut donner ou rechercher : registre sans héros ∩ CNN barre.

    Si les classes du CNN sont illisibles (ensemble vide), on garde le registre
    seul plutôt que de tout refuser.
    """
    if troop_types is None:
        from clashai.combat.troop_registry import load_troop_types
        troop_types = load_troop_types()
    names = {t['name'] for t in troop_types if t.get('role') != 'hero'}
    if cnn_names is None:
        from clashai.combat.troop_registry import cnn_class_names
        cnn_names = cnn_class_names()
    cnn = set(cnn_names or ())
    return (names & cnn) if cnn else names


def normalize_name(raw):
    """« Bébés-Dragons » -> 'bebes_dragons' : minuscules, sans accents ni mots vides."""
    text = unicodedata.normalize('NFKD', str(raw or ''))
    text = text.encode('ascii', 'ignore').decode('ascii').lower()
    words = [w for w in re.split(r'[^a-z0-9]+', text)
             if w and w not in _STOPWORDS]
    return '_'.join(words)


def _singular(word):
    return word[:-1] if len(word) > 3 and word[-1] in 'sx' else word


def resolve_troop_name(raw, known):
    """Nom officiel correspondant à `raw`, ou None. Jamais d'approximation.

    Essaie le nom tel quel, puis chaque combinaison singulier/pluriel des mots :
    « chauves-souris » -> `chauve_souris` (seul le premier mot porte le pluriel).
    """
    key = normalize_name(raw)
    if not key:
        return None
    if key in known:
        return key
    words = key.split('_')
    if len(words) > MAX_NAME_WORDS:
        return None
    for variant in itertools.product(*[(w, _singular(w)) for w in words]):
        candidate = '_'.join(variant)
        if candidate in known:
            return candidate
    return None


def suggest_troop_names(raw, known, limit=4):
    """Noms connus commençant comme `raw` — une SUGGESTION, jamais appliquée."""
    key = normalize_name(raw)
    if not key:
        return []
    head = _singular(key.split('_')[0])[:3]
    return sorted(n for n in known if n.startswith(head))[:limit]


def unknown_troop_message(raw, known):
    hints = suggest_troop_names(raw, known)
    message = f"troupe inconnue : « {raw} »"
    if hints:
        message += f" — tu voulais peut-être : {', '.join(hints)} ?"
    return message


def parse_troop_list(text, known):
    """(noms, erreur | None) depuis un texte libre : « bébés dragons, ballons ».

    Découpage glouton : le groupe de mots le plus long qui forme une troupe
    connue passe d'abord (« golem de glace » avant « golem »). Doublons retirés,
    ordre conservé.

    ⚠️ Un nombre dans le texte est REFUSÉ, pas ignoré : « 3 ballons » compris
    comme « ballons » donnerait des ballons sans limite. Les quantités arrivent
    en 5.3.4.
    """
    key = normalize_name(text)
    tokens = key.split('_') if key else []
    if any(t.isdigit() for t in tokens):
        return [], ("les quantités ne sont pas encore gérées (5.3.4) : donne "
                    "seulement les noms, ex. /dons ballons yétis")
    names, unknown = [], []
    i = 0
    while i < len(tokens):
        for size in range(min(MAX_NAME_WORDS, len(tokens) - i), 0, -1):
            name = resolve_troop_name('_'.join(tokens[i:i + size]), known)
            if name is not None:
                if name not in names:
                    names.append(name)
                i += size
                break
        else:
            unknown.append(tokens[i])
            i += 1
    if unknown:
        return [], ' ; '.join(unknown_troop_message(t, known) for t in unknown)
    return names, None


def prepare_args(tool_name, args, known):
    """(arguments normalisés, erreur | None).

    Appelé DEUX fois : à la saisie (la console refuse tout de suite, avant de
    mettre en file) et à l'exécution (défense en profondeur : un appel venu
    d'ailleurs — le LLM en 5.3.3c — est validé de la même façon).
    """
    args = dict(args or {})
    if tool_name == 'donner_troupes' and args.get('troupes'):
        raw = args['troupes']
        items = [raw] if isinstance(raw, str) else list(raw)
        names, errors = [], []
        for item in items:
            name = resolve_troop_name(item, known)
            if name is None:
                errors.append(unknown_troop_message(item, known))
            elif name not in names:
                names.append(name)
        if errors:
            return args, ' ; '.join(errors)
        args['troupes'] = names
    elif tool_name == 'lancer_recherche' and args.get('troupe'):
        name = resolve_troop_name(args['troupe'], known)
        if name is None:
            return args, unknown_troop_message(args['troupe'], known)
        args['troupe'] = name
    return args, None


# ---------------------------------------------------------------------------
# Outils adossés à un agent existant
# ---------------------------------------------------------------------------

# (outil, agent, dépense ?, description lue PAR LE MODÈLE)
AGENT_TOOLS = (
    ('recolter_ressources', 'village', False,
     "Récolte l'or, l'élixir et l'élixir noir des collecteurs pleins du "
     "village. Gratuit et rapide."),
    ('lancer_attaque', 'combat', True,
     "Lance UNE attaque multijoueur complète : recherche d'adversaire, "
     "déploiement, retour au village. Dépense les troupes de l'armée et dure "
     "plusieurs minutes."),
    ('demander_renforts', 'clan_castle', False,
     "Demande des troupes au château de clan (renforts envoyés par les "
     "membres). Gratuit."),
)

DONATE_PARAMS = {
    'type': 'object',
    'properties': {
        'troupes': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': "Troupes à donner EXCLUSIVEMENT, ex. [\"ballon\", "
                           "\"yeti\"]. Absent = celles que le jeu autorise.",
        },
    },
}

RESEARCH_PARAMS = {
    'type': 'object',
    'properties': {
        'troupe': {
            'type': 'string',
            'description': "Troupe ou sort à améliorer, ex. \"dragon\". "
                           "Absent = la recherche la moins chère.",
        },
    },
}

_RESEARCH_REFUSALS = {
    'busy': "le laboratoire est déjà occupé",
    'lab_not_found': "laboratoire introuvable à l'écran",
    'menu_not_found': "bouton « rechercher » introuvable",
    'nothing_upgradable': "aucune recherche disponible",
    'cant_afford': "pas assez de ressources",
    'need_decision': "prix ou ressource illisible : annulé par sécurité, "
                     "aucune dépense",
    'declined': "annulé par la politique de dépense, aucune dépense",
}


def _no_params():
    return {'type': 'object', 'properties': {}}


def _agent_tool(run_agent, agent_name):
    """Outil qui exécute un agent et traduit son AgentResult."""

    def run():
        result = run_agent(agent_name)
        if result is None:
            raise Refus(f"agent « {agent_name} » absent dans ce mode")
        if not result.ok:
            raise Echec(result.error or f"« {agent_name} » a échoué sans détail")
        data = dict(result.data or {})
        data['duree_s'] = round(float(result.duration_s or 0.0), 1)
        return data

    return run


def _donate_tool(donate_fn, known):
    def donner_troupes(troupes=None):
        wanted = None
        if troupes:
            args, error = prepare_args('donner_troupes', {'troupes': troupes}, known)
            if error:
                raise Refus(error)
            wanted = set(args['troupes'])
        try:
            return donate_fn(wanted=wanted)
        except RuntimeError as e:
            raise Echec(str(e)) from e

    return donner_troupes


def research_outcome(result, name=None):
    """UpgradeResult du labo -> données (ok) ou Refus motivé (tout le reste)."""
    status = getattr(result, 'status', None)
    if status == 'ok':
        return {'statut': 'ok', 'troupe': name or '(la moins chère)',
                'prix': getattr(result, 'price', None)}
    if status == 'nothing_upgradable' and name:
        raise Refus(f"« {name} » n'est pas améliorable maintenant "
                    f"(ou pas visible sur l'écran du labo)")
    message = _RESEARCH_REFUSALS.get(status, f"statut inattendu : {status}")
    if status == 'cant_afford' and getattr(result, 'price', None):
        message += f" (prix {result.price})"
    raise Refus(message)


def _research_tool(research_fn, known):
    def lancer_recherche(troupe=None):
        name = None
        if troupe:
            args, error = prepare_args('lancer_recherche', {'troupe': troupe}, known)
            if error:
                raise Refus(error)
            name = args['troupe']
        return research_outcome(research_fn(troupe=name), name)

    return lancer_recherche


def register_action_tools(registry, *, run_agent=None, donate_fn=None,
                          research_fn=None, known_troops=None,
                          available_agents=None):
    """Enregistre les outils qui agissent. Rend la liste des noms enregistrés.

    Une dépendance absente = outil NON enregistré, plutôt qu'un outil qui
    échouerait à chaque appel : la console dira « pas disponible dans ce mode ».

    Args:
        run_agent: (nom d'agent) -> AgentResult | None (None = agent absent).
        donate_fn: (wanted=set|None) -> dict (voir `make_donate_fn`).
        research_fn: (troupe=str|None) -> UpgradeResult (voir `make_research_fn`).
        known_troops: noms autorisés ; par défaut `donatable_troop_names()`.
        available_agents: noms des agents présents ; None = tous.
    """
    registered = []
    if run_agent is not None:
        for tool_name, agent_name, spends, description in AGENT_TOOLS:
            if available_agents is not None and agent_name not in available_agents:
                continue
            registry.register(Tool(
                name=tool_name, description=description,
                fn=_agent_tool(run_agent, agent_name),
                parameters=_no_params(), acts=True, spends=spends))
            registered.append(tool_name)

    if (donate_fn is not None or research_fn is not None) and known_troops is None:
        known_troops = donatable_troop_names()
    known = frozenset(known_troops or ())

    if donate_fn is not None:
        registry.register(Tool(
            name='donner_troupes',
            description="Ouvre le chat de clan, répond aux demandes de troupes "
                        "visibles (5 au plus) puis referme. Gratuit : onglet des "
                        "dons normaux uniquement, jamais celui payé en gemmes.",
            fn=_donate_tool(donate_fn, known),
            parameters=DONATE_PARAMS, acts=True, spends=False))
        registered.append('donner_troupes')

    if research_fn is not None:
        registry.register(Tool(
            name='lancer_recherche',
            description="Lance une recherche au laboratoire. Dépense de "
                        "l'élixir ou de l'élixir noir, et n'achète QUE si le "
                        "prix est lu et le solde suffisant.",
            fn=_research_tool(research_fn, known),
            parameters=RESEARCH_PARAMS, acts=True, spends=True))
        registered.append('lancer_recherche')

    return registered


# ---------------------------------------------------------------------------
# Fabriques branchées sur les vrais modules (utilisées en 5.3.3b)
# ---------------------------------------------------------------------------

def make_research_fn(lab, screenshot_fn, tap_fn, models):
    """research_fn(troupe) adossée à VillageLab.

    ⚠️ AUCUN `confirm_decider` n'est passé : la confirmation dépend uniquement de
    la preuve d'affordabilité. L'accord de l'opérateur a déjà été vérifié en
    amont par le registre (`spends=True` -> `confirm=True`).
    """
    def research(troupe=None):
        choose = None
        if troupe:
            def choose(candidates):
                match = [c for c in candidates if c.name == troupe]
                return match[0] if match else None
        return lab.research(screenshot_fn, tap_fn, models, choose=choose)

    return research


def make_donate_fn(manager, monitor, screenshot_fn, tap_fn, classify_fn, models,
                   max_requests=None):
    """donate_fn(wanted) adossée au parcours de dons validé en V5.2."""
    from clashai.social.donation_flow import DEFAULT_MAX_REQUESTS

    limit = DEFAULT_MAX_REQUESTS if max_requests is None else max_requests

    def donate(wanted=None):
        from clashai.social import donation_flow
        return donation_flow.donate_visible_requests(
            manager, monitor, screenshot_fn, tap_fn, classify_fn, models,
            wanted=wanted, max_requests=limit)

    return donate
