# clashai/brain/commands.py
# File d'instructions entre la console et la boucle du bot (V5.3, étape 5.3.3a).
#
# POURQUOI UNE FILE
# -----------------
# La console tourne dans son propre thread (elle attend ce que tape l'opérateur).
# La boucle du bot, elle, est le SEUL endroit qui tape sur l'émulateur : le
# scheduler exécute un agent à la fois, par construction. Si la console
# exécutait elle-même un outil qui agit, ses taps s'entrelaceraient avec ceux de
# l'agent en cours — une attaque et une ouverture de chat mélangées.
#
# Donc : la console DÉPOSE une commande, la boucle la RETIRE entre deux agents
# (`run_next`). Un seul thread touche au jeu, toujours.
#
# PRIORITÉ
# --------
# Deux sources d'instructions, par ordre d'autorité (`tools.SOURCES`) : l'admin
# passe devant le clan quand les deux attendent. À l'intérieur d'une même
# source, premier arrivé, premier servi.
#
# ⚠️ LES GARDE-FOUS SONT RÉAPPLIQUÉS À L'EXÉCUTION
# ----------------------------------------------
# `run_next` passe par `ToolRegistry.call`, qui revérifie autorité, dépense et
# arguments. Une commande `clan` d'action qui se glisserait dans la file par
# erreur serait refusée au moment de s'exécuter, pas seulement à la saisie.

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict

from clashai.brain.tools import SOURCE_ADMIN, SOURCES

# Commandes terminées gardées en mémoire (pour `/file`, le futur tableau de bord
# et la mémoire d'actions de 5.3.5).
DONE_HISTORY = 50


@dataclass
class Command:
    """Une instruction en attente d'exécution par la boucle du bot."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    source: str = SOURCE_ADMIN
    confirmed: bool = False       # l'opérateur a dit « oui » à la dépense
    text: str = ''                # la ligne d'origine, pour le journal
    id: int = 0
    created_at: float = 0.0


class CommandQueue:
    """File thread-safe : la console dépose, la boucle du bot retire."""

    def __init__(self, listener=None, history_size=DONE_HISTORY):
        self._lock = threading.Lock()
        self._pending = {source: deque() for source in SOURCES}
        self._ids = itertools.count(1)
        self._listener = listener
        self._done = deque(maxlen=history_size)

    # ---- côté console ------------------------------------------------------

    def put(self, tool, args=None, source=SOURCE_ADMIN, confirmed=False, text=''):
        """Dépose une commande et la rend (avec son numéro)."""
        if source not in SOURCES:
            raise ValueError(f"source inconnue : {source!r}")
        with self._lock:
            cmd = Command(tool=tool, args=dict(args or {}), source=source,
                          confirmed=bool(confirmed), text=text,
                          id=next(self._ids), created_at=time.time())
            self._pending[source].append(cmd)
        return cmd

    def cancel(self, source=SOURCE_ADMIN):
        """Vide la file d'une source. Rend le nombre de commandes retirées."""
        with self._lock:
            n = len(self._pending[source])
            self._pending[source].clear()
        return n

    # ---- côté boucle du bot ------------------------------------------------

    def pop_next(self):
        """Prochaine commande — admin d'abord — ou None si la file est vide."""
        with self._lock:
            for source in SOURCES:
                if self._pending[source]:
                    return self._pending[source].popleft()
        return None

    def complete(self, cmd, result):
        """Enregistre l'issue d'une commande et prévient l'écouteur.

        Un écouteur qui plante (affichage cassé) ne doit pas faire tomber la
        boucle du bot : on l'isole.
        """
        with self._lock:
            self._done.append((cmd, result))
        listener = self._listener
        if listener is not None:
            try:
                listener(cmd, result)
            except Exception:
                pass

    def set_listener(self, listener):
        self._listener = listener

    # ---- inspection --------------------------------------------------------

    def pending(self, source=None):
        """Commandes en attente, dans l'ordre où elles seront exécutées."""
        with self._lock:
            sources = SOURCES if source is None else (source,)
            return [cmd for s in sources for cmd in self._pending[s]]

    @property
    def done(self):
        """(Command, ToolResult) terminées, de la plus ancienne à la plus récente."""
        with self._lock:
            return list(self._done)

    def __len__(self):
        with self._lock:
            return sum(len(q) for q in self._pending.values())


def run_next(queue, registry):
    """Exécute UNE commande en attente. Rend (Command, ToolResult), ou None.

    ⚠️ À appeler UNIQUEMENT depuis la boucle du bot, le seul thread qui tape.
    Les garde-fous du registre (autorité, dépense, arguments) sont réappliqués
    ici, à l'exécution — pas seulement au moment de la saisie.
    """
    cmd = queue.pop_next()
    if cmd is None:
        return None
    result = registry.call(cmd.tool, cmd.args, source=cmd.source,
                           confirm=cmd.confirmed)
    queue.complete(cmd, result)
    return cmd, result
