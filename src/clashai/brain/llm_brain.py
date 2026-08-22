# clashai/brain/llm_brain.py
# LocalLLMBrain — le cerveau LLM local (V5.3).
#
# Même contrat que HeuristicBrain : `decide(world) -> agent | None`. Aucun agent
# n'a besoin de savoir qu'un LLM existe — c'est tout l'intérêt du seam `Brain`
# posé en V5.1.
#
# Division du travail (figée en Session 15) :
#   - le LLM décide QUOI/QUAND (quel agent lancer, selon l'état du village, les
#     ressources, les demandes du clan) ;
#   - les agents exécutent et rapportent ;
#   - le RL garde le micro temps réel (le LLM est bien trop lent pour ça).
#
# ⚠️ PRINCIPE NON NÉGOCIABLE : LE LLM NE DOIT JAMAIS BLOQUER LE BOT.
# Ollama pas lancé, modèle absent, timeout, JSON invalide, agent inexistant,
# agent non éligible → on retombe **silencieusement** sur `HeuristicBrain`. Un
# cerveau indisponible dégrade l'intelligence, jamais la disponibilité. C'est
# aussi ce qui permet de le brancher sans risque avant qu'il soit bon.

import json
import time

from clashai.brain.interface import Brain, HeuristicBrain

# Modèle par défaut : Mistral 7B Instruct (labo 🇫🇷, Apache 2.0, FR natif,
# tool-calling, ~4.5 Go en Q4 → tient sur le GPU à côté des CNN).
DEFAULT_MODEL = 'mistral'
DEFAULT_HOST = 'http://localhost:11434'

# Au-delà, on n'attend pas : l'heuristique reprend la main. Le bot doit rester
# réactif même si Ollama rame. Mesuré sur Mistral 7B : une fois le modèle chargé,
# une décision prend **0,1 à 0,6 s** — 20 s est donc très large.
DEFAULT_TIMEOUT_S = 20.0

# Le PREMIER appel charge le modèle sur le GPU : ~21 s mesurées, soit juste
# au-dessus du timeout. On paie donc ce coût UNE FOIS au démarrage (`warmup`),
# avec un délai généreux, plutôt que de le subir au milieu d'une décision.
WARMUP_TIMEOUT_S = 180.0

# Ollama décharge un modèle inutilisé au bout de 5 min par défaut : le rechargement
# retomberait alors dans le cas « démarrage à froid ». On le garde résident plus
# longtemps que nos intervalles de décision.
KEEP_ALIVE = '30m'

# Un appel LLM par tick serait absurde (le scheduler tourne en boucle serrée) :
# on ne consulte le cerveau qu'à cet intervalle, l'heuristique assure entre-temps.
DEFAULT_MIN_INTERVAL_S = 30.0

# Back-off quand le serveur ne répond pas. Sans lui, un Ollama non lancé coûte
# ~6 s d'attente À CHAQUE intervalle (mesuré) : le bot passerait son temps à
# attendre un serveur absent, pour retomber sur l'heuristique de toute façon.
# On double l'attente à chaque échec consécutif, jusqu'à ce plafond ; le premier
# succès remet tout à zéro (l'utilisateur a lancé `ollama serve` entre-temps).
BACKOFF_MAX_S = 600.0

SYSTEM_PROMPT = """Tu es le cerveau d'un bot Clash of Clans. À chaque tour, on te
donne l'état du jeu et la liste des agents disponibles. Tu choisis UN agent à
lancer, ou aucun si rien n'est pertinent.

Règles :
- Réponds UNIQUEMENT par un objet JSON : {"agent": "<nom>", "raison": "<courte>"}
- "<nom>" doit être un nom de la liste fournie, ou null pour ne rien faire.
- Ne choisis jamais un agent marqué non éligible.
- Sois économe : attaquer coûte des troupes, améliorer coûte des ressources.
"""

# Prompt du mode DISCUSSION (`chat()`), distinct de celui de la décision : ici on
# veut une réponse en français à un humain, pas un JSON à parser.
CHAT_SYSTEM_PROMPT = """Tu es le cerveau d'un bot Clash of Clans qui joue en
autonomie. Tu discutes avec ton opérateur dans un terminal — c'est une
conversation privée entre vous deux, pas le chat du clan.

Tu reçois l'état RÉEL du jeu tel que la perception le voit. Règles :
- Réponds en français, court et concret.
- L'état ci-dessous fait AUTORITÉ. Toute ligne qui porte un chiffre est une
  mesure fiable et à jour : donne-la directement, sans réserve ni condition.
- Ne réponds « je ne sais pas » QUE si la ligne concernée porte littéralement la
  mention « NON LUE ». Dans ce cas précis, n'invente aucun chiffre.
- Les « boutons visibles » sont des noms techniques d'éléments détectés, pas des
  valeurs, et n'ont AUCUNE incidence sur les chiffres ci-dessus.
- Tu peux donner ton avis sur ce qu'il faudrait faire et pourquoi.
"""

# Nombre de tours conservés dans la conversation. Un modèle 7B local a une
# fenêtre de contexte modeste : mieux vaut une mémoire courte et fiable qu'un
# historique qui déborde et fait dérailler les réponses.
CHAT_HISTORY_TURNS = 8

# Libellés FRANÇAIS des ressources, dans l'ordre du HUD.
#
# ⚠️ NE JAMAIS remettre les clés techniques dans le prompt. Mesuré sur Mistral 7B :
# la ligne « ressources : elixir = 2399904, elixir_noire = 18549, or = 2235125 »
# lui fait répondre « les seules ressources indiquées sont l'élixir et l'or ».
# `elixir_noire` CONTIENT `elixir`, et `or` est aussi un mot français : le modèle
# fusionne les deux premières et n'en compte plus que deux. Une ressource par
# ligne, en français lisible, supprime l'ambiguïté — vérifié sur 6 formulations.
#
# Règle générale : le prompt est une interface pour un modèle, pas un dump de
# dict. Un nom d'API n'y a pas sa place s'il ressemble à un autre.
RESOURCE_LABELS = {
    'or': 'or',
    'elixir': 'élixir',
    'elixir_noire': 'élixir noir',
}


class LocalLLMBrain(Brain):
    """Cerveau LLM local (Ollama), avec repli heuristique systématique."""

    def __init__(self, scheduler, model=DEFAULT_MODEL, host=DEFAULT_HOST,
                 timeout_s=DEFAULT_TIMEOUT_S,
                 min_interval_s=DEFAULT_MIN_INTERVAL_S,
                 client=None, verbose=True):
        self._scheduler = scheduler
        self._fallback = HeuristicBrain(scheduler)
        self._model = model
        self._host = host
        self._timeout_s = timeout_s
        self._min_interval_s = min_interval_s
        self._client = client              # injectable (tests / autre runtime)
        self._injected = client is not None
        self._last_call = 0.0
        self._consecutive_errors = 0
        self.verbose = verbose
        # Télémétrie — lisible par le futur dashboard.
        self.stats = {'llm': 0, 'fallback': 0, 'errors': 0}
        self.last_reason = None
        self._chat_history = []            # mode discussion (voir chat())

    # ---- contrat Brain -----------------------------------------------------

    def decide(self, world):
        """Choisit l'agent à lancer. Retombe sur l'heuristique en cas de souci."""
        eligible = self._eligible(world)
        if not eligible:
            return None                     # rien à faire, inutile de réveiller le LLM

        if len(eligible) == 1:
            # Un seul candidat : il n'y a rien à arbitrer.
            return eligible[0]

        now = time.time()
        if now - self._last_call < self._retry_interval():
            return self._fallback.decide(world)

        self._last_call = now
        name = self._ask(world, eligible)
        if name is None:
            self.stats['fallback'] += 1
            return self._fallback.decide(world)

        chosen = next((a for a in eligible if a.name == name), None)
        if chosen is None:
            # Le LLM a nommé un agent inconnu ou non éligible : on n'invente pas.
            self._log(f"agent '{name}' non éligible → heuristique")
            self.stats['fallback'] += 1
            return self._fallback.decide(world)

        self.stats['llm'] += 1
        return chosen

    def _retry_interval(self):
        """Intervalle avant le prochain appel, allongé après des échecs."""
        if not self._consecutive_errors:
            return self._min_interval_s
        return min(self._min_interval_s * (2 ** self._consecutive_errors),
                   BACKOFF_MAX_S)

    # ---- interrogation du LLM ---------------------------------------------

    def _ask(self, world, eligible):
        """Nom d'agent choisi par le LLM, ou None si indisponible/invalide."""
        client = self._get_client()
        if client is None:
            self._consecutive_errors += 1
            return None
        try:
            raw = client.chat(
                model=self._model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': self.build_prompt(world, eligible)},
                ],
                format='json',
                options={'temperature': 0.2},
                keep_alive=KEEP_ALIVE,
            )
            content = raw['message']['content']
        except Exception as e:
            self.stats['errors'] += 1
            self._consecutive_errors += 1
            self._log(f"LLM indisponible ({type(e).__name__}) → heuristique "
                      f"(nouvel essai dans {self._retry_interval():.0f}s)")
            return None

        self._consecutive_errors = 0      # le serveur répond : on repart normal

        name, reason = self._parse(content)
        self.last_reason = reason
        if reason:
            self._log(f"décision : {name or 'rien'} — {reason}")
        return name

    @staticmethod
    def _parse(content):
        """(agent, raison) depuis la réponse ; (None, None) si illisible.

        Un LLM local rend parfois du texte autour du JSON malgré `format=json` :
        on récupère le premier objet plutôt que d'échouer sur un préambule.
        """
        if not isinstance(content, str):
            return None, None
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            i, j = content.find('{'), content.rfind('}')
            if i < 0 or j <= i:
                return None, None
            try:
                data = json.loads(content[i:j + 1])
            except (ValueError, TypeError):
                return None, None
        if not isinstance(data, dict):
            return None, None
        name = data.get('agent')
        return (name if isinstance(name, str) else None), data.get('raison')

    # ---- préchauffage -------------------------------------------------------

    def warmup(self, timeout_s=WARMUP_TIMEOUT_S):
        """Charge le modèle sur le GPU. True si le cerveau est opérationnel.

        À appeler UNE FOIS au démarrage : sans ça, le tout premier appel paie
        ~21 s de chargement — au-dessus du timeout de décision, donc la première
        décision partirait systématiquement à l'heuristique pour rien.

        Ne lève jamais : un échec signifie simplement « pas de LLM », et le
        repli heuristique fait son travail.
        """
        client = self._get_client()
        if client is None:
            return False
        # Client DÉDIÉ au préchauffage : celui des décisions a le timeout court
        # (20 s), qui est justement celui que le chargement dépasse.
        # ⚠️ Un client INJECTÉ (tests, autre runtime) est réutilisé tel quel :
        # en fabriquer un vrai ici ferait parler les tests au serveur réel.
        warm_client = client
        if not self._injected:
            try:
                import ollama
                warm_client = ollama.Client(host=self._host, timeout=timeout_s)
            except Exception:
                warm_client = client
        try:
            warm_client.chat(model=self._model,
                             messages=[{'role': 'user', 'content': 'ok'}],
                             keep_alive=KEEP_ALIVE,
                             options={'num_predict': 1})
        except Exception as e:
            self._log(f"préchauffage impossible ({type(e).__name__}) — "
                      f"heuristique en attendant")
            self._consecutive_errors += 1
            return False
        self._consecutive_errors = 0
        self._log(f"modèle « {self._model} » chargé et prêt")
        return True

    # ---- mode DISCUSSION (terminal opérateur) ------------------------------

    def chat(self, question, world=None):
        """Répond à l'opérateur, en voyant l'état réel du jeu. str | None.

        Rien à voir avec `decide()` : ici on veut du texte pour un humain, pas
        un agent à lancer. Le chat du clan (V5.4) est encore autre chose — ceci
        est une console privée entre l'opérateur et le cerveau.

        None = LLM indisponible ; à l'appelant de le dire à l'utilisateur.
        """
        client = self._get_client()
        if client is None:
            return None

        messages = [{'role': 'system', 'content': CHAT_SYSTEM_PROMPT}]
        if world:
            messages.append({
                'role': 'system',
                'content': 'État du jeu :\n' + self.describe_world(world),
            })
        messages.extend(self._chat_history)
        messages.append({'role': 'user', 'content': question})

        try:
            raw = client.chat(model=self._model, messages=messages,
                              options={'temperature': 0.7},
                              keep_alive=KEEP_ALIVE)
            answer = raw['message']['content']
        except Exception as e:
            self.stats['errors'] += 1
            self._consecutive_errors += 1
            return None if not self.verbose else f"[LLM indisponible : {type(e).__name__}]"

        self._consecutive_errors = 0
        if isinstance(answer, str):
            self._chat_history.append({'role': 'user', 'content': question})
            self._chat_history.append({'role': 'assistant', 'content': answer})
            # On garde une fenêtre courte (voir CHAT_HISTORY_TURNS).
            del self._chat_history[:-2 * CHAT_HISTORY_TURNS]
            return answer
        return None

    def reset_chat(self):
        """Oublie la conversation en cours (pas l'état du jeu)."""
        self._chat_history.clear()

    def describe_world(self, world):
        """État du jeu en texte lisible — partagé par la décision et le chat.

        ⚠️ Les valeurs NON LUES sont annoncées comme telles, explicitement. Sans
        ça, le modèle comble les trous : avec seulement le *nom* du bouton
        `compteur_or` sous les yeux, un 7B a inventé « 15568 or ». Nommer
        l'ignorance est le seul moyen fiable d'éviter ça.
        """
        readings = world.get('readings') or {}
        buttons = world.get('buttons') or {}

        lines = [
            f"- écran : {world.get('screen_state') or 'inconnu'}",
            f"- mode : {world.get('mode', 'auto')}",
            f"- bâtiments détectés : {len(world.get('buildings') or [])}",
        ]

        res = readings.get('resources')
        if res:
            for key, label in RESOURCE_LABELS.items():
                if key in res:
                    lines.append(f"- {label} : {res[key]}")
            for key in sorted(set(res) - set(RESOURCE_LABELS)):
                lines.append(f"- {key} : {res[key]}")   # ressource future
        else:
            lines.append("- ressources : NON LUES (ne pas les inventer)")

        b = readings.get('builders')
        if b:
            # DEUX lignes, pas « 4 libres sur 5 » : sur ce format, le modèle
            # attrapait le dernier nombre et répondait « 5 ouvriers libres ».
            # Deux nombres sur une ligne = une ligne ambiguë.
            lines.append(f"- ouvriers libres : {b['libres']}")
            lines.append(f"- ouvriers au total : {b['total']}")
        else:
            lines.append("- ouvriers : NON LUS (ne pas les inventer)")

        # Récoltes et dons : des COMPTES, pas des montants. On ne les annonce que
        # s'il y en a — « 0 collecteur plein » est du bruit, et une absence de
        # ligne ne peut pas être lue comme une valeur inventée.
        # UNE LIGNE PAR RESSOURCE. La liste à virgules « 5 or, 6 élixir, 3 élixir
        # noir » reproduisait le bug d'origine : à « combien de collecteurs d'or
        # ? » le modèle répondait « six » — le compte de l'élixir. Même cause,
        # même remède.
        recoltes = readings.get('recoltes') or {}
        for key in list(RESOURCE_LABELS) + sorted(set(recoltes) - set(RESOURCE_LABELS)):
            if recoltes.get(key):
                label = RESOURCE_LABELS.get(key, key)
                lines.append(f"- collecteurs d'{label} pleins, prêts à récolter : "
                             f"{recoltes[key]}")

        dons = readings.get('dons_en_attente')
        if dons:
            # Le libellé dit le SENS, pas seulement le nom du champ. Avec
            # « demandes de dons en attente : 2 » seul, le modèle répondait
            # « aucun » à « combien de membres attendent des troupes ? » : il ne
            # faisait pas le lien. Nommer les deux formes coûte six mots.
            lines.append(f"- demandes de dons en attente "
                         f"(membres du clan qui réclament des troupes) : {dons}")

        if 'lab_libre' in readings:
            lines.append("- laboratoire : "
                         + ("libre" if readings['lab_libre'] else "occupé"))
        else:
            lines.append("- laboratoire : état inconnu")

        if buttons:
            lines.append("- boutons visibles (noms techniques, pas des valeurs) : "
                         + ', '.join(sorted(buttons)[:25]))
        troops = world.get('troop_positions') or {}
        if troops:
            lines.append(f"- troupes prêtes : {', '.join(sorted(troops))}")
        return chr(10).join(lines)

    # ---- construction du contexte -----------------------------------------

    def build_prompt(self, world, eligible):
        """Le `world` en texte compact — c'est TOUT ce que le LLM voit."""
        lines = [self.describe_world(world), "", "Agents éligibles :"]
        for a in eligible:
            doc = (a.__doc__ or '').strip().splitlines()
            lines.append(f"- {a.name} (priorité {a.priority}) : "
                         f"{doc[0] if doc else 'sans description'}")
        return chr(10).join(lines)

    # ---- interne -----------------------------------------------------------

    def _eligible(self, world):
        """Agents que le scheduler accepterait — le LLM ne peut pas passer outre.

        Le cerveau choisit PARMI ce qui est jouable ; il ne contourne ni les
        cooldowns ni les `can_run`. Une mauvaise décision du LLM reste donc
        toujours une décision *valide*.
        """
        return [a for a in self._scheduler.agents if a.is_ready(world)]

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._host,
                                             timeout=self._timeout_s)
            except Exception as e:
                self.stats['errors'] += 1
                self._log(f"Ollama non disponible ({type(e).__name__}) — "
                          f"`uv add ollama` + `ollama serve` ; heuristique en attendant")
                self._client = False       # False = testé et absent
        return self._client or None

    def _log(self, msg):
        if self.verbose:
            from clashai.config.logging import pp
            pp(f" LLM: {msg}", tag='ok')
