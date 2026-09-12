# src/tools/debug/console_demo.py
# Démo de la console opérateur + file d'instructions (V5.3, étape 5.3.3a).
#
# SANS ÉMULATEUR : les agents, les dons et le labo sont SIMULÉS. Rien ne touche
# au jeu. On vérifie la mécanique : lecture immédiate, actions en file, « o/n »
# sur les dépenses, noms de troupes validés à la saisie, file vidée par une
# « boucle du bot » qui tourne dans un autre thread — exactement comme en 5.3.3b.
#
# Usage :
#   uv run python -m tools.debug.console_demo
#   uv run python -m tools.debug.console_demo --occupe 15        # bot déjà en attaque
#   uv run python -m tools.debug.console_demo --sans-confirmation
#   uv run python -m tools.debug.console_demo --llm              # texte libre -> Mistral
#
# À essayer : /etat, /attaque, /dons ballons yétis, /dons dragonn, /dons 3 ballons,
#             /labo bébés dragons (deux fois : le labo sera occupé), /file, /annuler

import argparse
import random
import threading
import time

MONDE_SIMULE = {
    'screen_state': 'village_home',
    'buildings': [1] * 42,
    'troop_positions': {'dragon': (1, 2, .9), 'ballon': (3, 4, .8),
                        'sorciere': (5, 6, .85)},
    'readings': {
        'resources': {'or': 2235125, 'elixir': 2399904, 'elixir_noire': 18549},
        'builders': {'libres': 4, 'total': 5},
        'lab_libre': True,
        'recoltes': {'or': 5, 'elixir': 6, 'elixir_noire': 3},
    },
}

# Durée simulée de chaque agent (secondes).
DUREES = {'combat': 6.0, 'village': 1.5, 'clan_castle': 1.0}


def main():
    ap = argparse.ArgumentParser(description="Démo console opérateur (simulée)")
    ap.add_argument('--occupe', type=float, default=0.0,
                    help="Le bot est déjà en attaque pendant N secondes au "
                         "démarrage : tes actions attendront la fin.")
    ap.add_argument('--sans-confirmation', action='store_true',
                    help="Ne demande pas o/n avant une dépense.")
    ap.add_argument('--llm', action='store_true',
                    help="Le texte libre part vers le cerveau (Ollama requis).")
    args = ap.parse_args()

    from clashai.agents.base import AgentResult
    from clashai.brain.action_tools import register_action_tools
    from clashai.brain.commands import CommandQueue, run_next
    from clashai.brain.console import ConsoleSession, format_completion
    from clashai.brain.tools import build_read_only_registry
    from clashai.village.upgrader import UpgradeResult

    print_lock = threading.Lock()

    def say(line):
        with print_lock:
            print(line, flush=True)

    state = {'busy': None, 'labo_occupe': False}

    def world_fn():
        return MONDE_SIMULE

    # ---- faux agents (simulés : aucun tap) ---------------------------------
    def run_agent(name):
        state['busy'] = name
        try:
            say(f"   [bot] {name} en cours...")
            time.sleep(DUREES.get(name, 1.0))
            data = {}
            if name == 'combat':
                data = {'stars': random.randint(1, 3),
                        'percentage': random.randint(55, 100)}
            elif name == 'village':
                data = {'collected': 3}
            return AgentResult(ok=True, duration_s=DUREES.get(name, 1.0),
                               data=data)
        finally:
            state['busy'] = None

    def donate_fn(wanted=None):
        state['busy'] = 'dons'
        try:
            say("   [bot] ouverture du chat, dons en cours...")
            time.sleep(1.5)
            return {'demandes_vues': 2, 'demandes_servies': 2, 'dons': 8,
                    'troupes_voulues': sorted(wanted) if wanted else None}
        finally:
            state['busy'] = None

    def research_fn(troupe=None):
        if state['labo_occupe']:
            return UpgradeResult('busy')
        state['labo_occupe'] = True
        return UpgradeResult('ok', price=450_000)

    # ---- registre + file ---------------------------------------------------
    registry = build_read_only_registry(world_fn)
    register_action_tools(registry, run_agent=run_agent, donate_fn=donate_fn,
                          research_fn=research_fn)
    queue = CommandQueue(listener=lambda cmd, res: say(format_completion(cmd, res)))

    # ---- « boucle du bot » : le SEUL thread qui exécute les actions --------
    stop = threading.Event()

    def bot_loop():
        if args.occupe:
            state['busy'] = 'combat (attaque autonome)'
            say(f"   [bot] attaque autonome en cours ({args.occupe:.0f} s)...")
            time.sleep(args.occupe)
            state['busy'] = None
            say("   [bot] attaque autonome terminée")
        while not stop.is_set():
            if len(queue):
                run_next(queue, registry)
            else:
                time.sleep(0.2)

    threading.Thread(target=bot_loop, name='BotLoopSimule', daemon=True).start()

    # ---- discussion (optionnelle) ------------------------------------------
    chat_fn = None
    if args.llm:
        from clashai.agents.scheduler import AgentScheduler
        from clashai.brain.llm_brain import LocalLLMBrain
        brain = LocalLLMBrain(AgentScheduler(), verbose=False)
        say("Préchauffage du cerveau...")
        if brain.warmup():
            def chat_fn(text):
                return brain.chat(text, world_fn())
        else:
            say("Ollama ne répond pas : discussion désactivée.")

    session = ConsoleSession(registry, queue, chat_fn=chat_fn,
                             busy_fn=lambda: state['busy'],
                             confirm_spending=not args.sans_confirmation)

    say("")
    say("=" * 64)
    say(" Console opérateur - DÉMO SIMULÉE (aucun tap sur le jeu)")
    say("=" * 64)
    for line in session.handle('/aide'):
        say(line)
    say("")

    while not session.wants_quit:
        try:
            line = input("admin > ")
        except (EOFError, KeyboardInterrupt):
            break
        for out in session.handle(line):
            say(out)

    stop.set()
    say("Démo terminée.")


if __name__ == "__main__":
    main()
