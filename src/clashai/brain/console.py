# clashai/brain/console.py
# Console opérateur : une ligne tapée -> un appel d'outil (V5.3, étape 5.3.3a).
#
# RÈGLE DE CONDUITE
# -----------------
#   - outil qui LIT       -> exécuté tout de suite (il lit le cache de perception,
#                            ne tape rien, ne peut rien entrelacer) ;
#   - outil qui AGIT      -> MIS EN FILE, exécuté par la boucle du bot entre deux
#                            agents (`commands.run_next`) ;
#   - outil qui DÉPENSE   -> « o/n » d'abord ; « n » = rien ne part.
#
# La console ne fait aucune E/S par elle-même : saisie, confirmation, discussion
# et « qu'est-ce que le bot fait en ce moment » sont injectés. On la teste donc
# sans terminal, et 5.3.3b la branchera sur le vrai stdin du bot.
#
# Les noms de troupes sont validés À LA SAISIE : l'opérateur apprend tout de
# suite que « dragonn » n'existe pas, au lieu de le découvrir quand la file se
# vide, parfois plusieurs minutes plus tard.
#
# ⚠️ Caractères : on se limite à ceux du codepage Windows (cp1252). Un « ✔ » ou
# une flèche « → » fait planter `print` quand la sortie est redirigée.

from clashai.brain.action_tools import (
    donatable_troop_names,
    parse_troop_list,
    resolve_troop_name,
    unknown_troop_message,
)
from clashai.brain.tools import SOURCE_ADMIN

# Commande -> outil. L'ordre sert aussi à l'aide.
SLASH_TOOLS = {
    '/etat': 'etat_du_village',
    '/troupes': 'lister_troupes_disponibles',
    '/recolte': 'recolter_ressources',
    '/attaque': 'lancer_attaque',
    '/renforts': 'demander_renforts',
    '/dons': 'donner_troupes',
    '/labo': 'lancer_recherche',
}

# Outils qui acceptent un argument libre après la commande.
_WITH_ARGUMENT = frozenset({'donner_troupes', 'lancer_recherche'})

# Ce qu'on demande de confirmer, en clair.
CONFIRM_LABELS = {
    'lancer_attaque': "lancer une attaque (dépense les troupes de ton armée)",
    'lancer_recherche': "lancer une recherche au labo "
                        "(dépense de l'élixir ou de l'élixir noir)",
}

HELP = """Commandes :
  /etat              état du village                      (immédiat)
  /troupes           troupes prêtes                       (immédiat)
  /recolte           récolter les ressources              -> en file
  /attaque           lancer une attaque           (o/n)   -> en file
  /renforts          demander des renforts                -> en file
  /dons [troupes]    donner des troupes, ex. /dons ballons yétis -> en file
  /labo [troupe]     lancer une recherche         (o/n)   -> en file
  /file              commandes en attente
  /annuler           vider ta file
  /outils            outils disponibles
  /quit              quitter
Texte libre : discussion avec le cerveau."""


def _ask_on_stdin(prompt):
    try:
        return input(prompt).strip().lower() in ('o', 'oui', 'y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False


def _fmt(value):
    if value is None:
        return "non lisible d'ici"
    if isinstance(value, dict):
        return ', '.join(f"{k}={_fmt(v)}" for k, v in value.items()) or '-'
    if isinstance(value, (list, tuple, set)):
        return ', '.join(str(v) for v in value) or '-'
    return str(value)


def format_result(result):
    """ToolResult -> lignes lisibles."""
    if not result.ok:
        return [f"[refus] {result.tool} : {result.error}"]
    lines = [f"[ok] {result.tool}"]
    lines += [f"   {key} : {_fmt(value)}" for key, value in result.data.items()]
    return lines


def format_completion(cmd, result):
    """Une ligne quand la boucle du bot a terminé une commande."""
    head = f"#{cmd.id} {cmd.tool}"
    if not result.ok:
        return f"[fini] {head} : REFUS - {result.error}"
    items = [f"{k}={_fmt(v)}" for k, v in list(result.data.items())[:4]]
    return f"[fini] {head} : ok - " + (' | '.join(items) if items else 'fait')


class ConsoleSession:
    """Traduit ce que tape l'opérateur. `handle(ligne)` ne lève jamais."""

    def __init__(self, registry, queue, *, chat_fn=None, ask_yes_no=None,
                 busy_fn=None, confirm_spending=True, known_troops=None):
        self._registry = registry
        self._queue = queue
        self._chat_fn = chat_fn
        self._ask = ask_yes_no or _ask_on_stdin
        self._busy_fn = busy_fn
        self._confirm_spending = confirm_spending
        self._known = frozenset(known_troops) if known_troops is not None else None
        self.wants_quit = False

    # ---- point d'entrée ----------------------------------------------------

    def handle(self, line):
        """Lignes à afficher en réponse. Une erreur devient une ligne, jamais
        une exception : la console ne doit pas mourir sur une faute de frappe."""
        try:
            return self._handle((line or '').strip())
        except Exception as e:
            return [f"[erreur console] {type(e).__name__}: {e}"]

    def _handle(self, text):
        if not text:
            return []
        if not text.startswith('/'):
            return self._chat(text)

        command, _, rest = text.partition(' ')
        command, rest = command.lower(), rest.strip()

        if command in ('/quit', '/exit', '/q'):
            self.wants_quit = True
            return ["À plus !"]
        if command in ('/aide', '/help', '/?'):
            return HELP.splitlines()
        if command == '/outils':
            return self._list_tools()
        if command == '/file':
            return self._list_pending()
        if command == '/annuler':
            n = self._queue.cancel(SOURCE_ADMIN)
            return [f"[ok] {n} commande(s) retirée(s) de la file" if n
                    else "(file déjà vide)"]

        tool_name = SLASH_TOOLS.get(command)
        if tool_name is None:
            return [f"Commande inconnue : {command} - tape /aide"]
        args, error = self._args_for(command, tool_name, rest)
        if error:
            return [f"[refus] {error}"]
        return self._run_tool(tool_name, args, text)

    # ---- arguments ---------------------------------------------------------

    def _known_troops(self):
        if self._known is None:
            self._known = frozenset(donatable_troop_names())
        return self._known

    def _args_for(self, command, tool_name, rest):
        """(args, erreur | None) — validés AVANT toute mise en file."""
        if not rest:
            return {}, None
        if tool_name not in _WITH_ARGUMENT:
            return None, f"{command} ne prend pas d'argument"
        known = self._known_troops()
        if tool_name == 'donner_troupes':
            names, error = parse_troop_list(rest, known)
            if error:
                return None, error
            if not names:
                return None, f"aucune troupe reconnue dans « {rest} »"
            return {'troupes': names}, None
        name = resolve_troop_name(rest, known)
        if name is None:
            return None, unknown_troop_message(rest, known)
        return {'troupe': name}, None

    # ---- exécution ---------------------------------------------------------

    def _run_tool(self, tool_name, args, text):
        tool = self._registry.get(tool_name)
        if tool is None:
            return [f"[refus] « {tool_name} » n'est pas disponible dans ce mode"]

        if not tool.acts:
            # Lecture : tout de suite, rien à entrelacer.
            return format_result(
                self._registry.call(tool_name, args, source=SOURCE_ADMIN))

        confirmed = False
        if tool.spends:
            if self._confirm_spending:
                label = CONFIRM_LABELS.get(tool_name, tool_name)
                if not self._ask(f"Confirmer : {label} ? (o/n) "):
                    return ["[annulé] rien n'est parti"]
            confirmed = True

        ahead = len(self._queue)
        cmd = self._queue.put(tool_name, args, source=SOURCE_ADMIN,
                              confirmed=confirmed, text=text)
        busy = self._busy_fn() if self._busy_fn else None
        when = (f"je termine d'abord : {busy}" if busy
                else "je m'en occupe au prochain créneau")
        position = f", {ahead} avant elle" if ahead else ""
        return [f"[en file] #{cmd.id} {tool_name}{position} - {when}"]

    def _chat(self, text):
        if self._chat_fn is None:
            return ["(discussion indisponible ici - tape /aide pour les commandes)"]
        answer = self._chat_fn(text)
        if not answer:
            return ["cerveau > [indisponible] Ollama ne répond pas."]
        return [f"cerveau > {answer}"]

    # ---- inspection --------------------------------------------------------

    def _list_pending(self):
        pending = self._queue.pending()
        if not pending:
            return ["(file vide)"]
        lines = [f"{len(pending)} commande(s) en attente :"]
        for cmd in pending:
            args = f" {_fmt(cmd.args)}" if cmd.args else ""
            lines.append(f"   #{cmd.id} {cmd.tool}{args} ({cmd.source})")
        return lines

    def _list_tools(self):
        aliases = {tool: command for command, tool in SLASH_TOOLS.items()}
        lines = ["Outils disponibles :"]
        for name in self._registry.names(SOURCE_ADMIN):
            tool = self._registry.get(name)
            nature = 'lit' if not tool.acts else (
                'agit, dépense' if tool.spends else 'agit')
            alias = aliases.get(name, '')
            lines.append(f"   {alias:<10} {name:<28} [{nature}]")
        return lines
