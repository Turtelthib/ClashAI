# src/tools/debug/llm_chat.py
# Console de discussion avec le cerveau LLM (V5.3).
#
# Conversation PRIVÉE entre l'opérateur et le cerveau, dans un terminal — rien à
# voir avec le chat du clan (V5.4). Le LLM voit l'état RÉEL du jeu : écran
# courant, boutons visibles, troupes prêtes… donc on peut lui demander « tu vois
# quoi là ? » ou « qu'est-ce que tu ferais ? » et juger ses réponses.
#
# Usage :
#   uv run python -m tools.debug.llm_chat                # avec le jeu (perception)
#   uv run python -m tools.debug.llm_chat --sans-jeu     # discussion seule, démarrage instantané
#   uv run python -m tools.debug.llm_chat --model mistral-nemo
#
# Prérequis : Ollama installé et lancé (`ollama serve`) + `ollama pull mistral`.
# Sans lui, la console le dit clairement au lieu de rester muette.

import argparse

BANNER = """
============================================================
 ClashAI — discussion avec le cerveau
 Commandes : /etat  (ce qu'il voit)   /oubli  (vider la mémoire)
             /quit  (sortir)
============================================================
"""


def main():
    ap = argparse.ArgumentParser(description="Discuter avec le cerveau LLM")
    ap.add_argument('--model', default=None, help="Modèle Ollama (défaut: mistral)")
    ap.add_argument('--sans-jeu', action='store_true',
                    help="Ne charge pas la perception : discussion pure, "
                         "démarrage instantané, mais il ne voit pas le jeu.")
    args = ap.parse_args()

    from clashai.agents import AgentScheduler
    from clashai.brain.llm_brain import DEFAULT_MODEL, LocalLLMBrain

    models = None
    if not args.sans_jeu:
        print("Chargement de la perception (quelques secondes)…")
        try:
            from clashai.navigation import game_loop as gl
            from clashai.perception.ui_detector import install
            install(verbose=False)
            models = gl.load_models()
        except Exception as e:
            print(f"Perception indisponible ({type(e).__name__}) → discussion "
                  f"sans le jeu. Lance l'émulateur, ou utilise --sans-jeu.")
            models = None

    brain = LocalLLMBrain(AgentScheduler(), model=args.model or DEFAULT_MODEL,
                          verbose=False)

    def world():
        if not models:
            return {}
        from clashai.agents.world import build_world
        return build_world(models)

    print(BANNER)
    print(f"Modèle : {brain._model}"
          f"{'  (sans le jeu)' if not models else ''}\n")
    # Le 1er appel charge le modèle sur le GPU : autant le faire maintenant,
    # sinon c'est la première question qui attend ~20 s pour rien.
    print('Préchauffage du modèle…', flush=True)
    if brain.warmup():
        print('Prêt.')
    else:
        print('Ollama ne répond pas — vérifie qu il tourne (icône barre '
              'système) et que `ollama pull mistral` est terminé.')
    print()

    while True:
        try:
            q = input("toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ('/quit', '/exit', '/q'):
            break
        if q == '/oubli':
            brain.reset_chat()
            print("(mémoire de conversation vidée)\n")
            continue
        if q == '/etat':
            w = world()
            print(brain.describe_world(w) if w else "(pas de perception chargée)")
            print()
            continue

        answer = brain.chat(q, world())
        if answer is None:
            print("\ncerveau > [indisponible] Ollama ne répond pas.")
            print("          Vérifie : `ollama serve` lancé, et "
                  "`ollama pull mistral` terminé.\n")
        else:
            print(f"\ncerveau > {answer}\n")

    print("À plus !")


if __name__ == "__main__":
    main()
