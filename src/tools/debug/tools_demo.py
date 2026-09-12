# src/tools/debug/tools_demo.py
# Démo du registre d'outils du cerveau (V5.3, incrément 5.3.2).
#
# Montre CE QUE LE LLM POURRA FAIRE, et surtout ce qu'il ne pourra pas — les
# garde-fous étant dans le code, ils se démontrent sans modèle ni émulateur.
#
# LECTURE SEULE : les outils livrés en 5.3.2 ne tapent rien et ne dépensent rien.
#
# Usage :
#   uv run python -m tools.debug.tools_demo              # sur monde simulé
#   uv run python -m tools.debug.tools_demo --jeu        # sur le vrai jeu
#   uv run python -m tools.debug.tools_demo --garde-fous # démontre les refus

import argparse
import json

MONDE_SIMULE = {
    'screen_state': 'village_home',
    'buildings': [1] * 42,
    'troop_positions': {'dragon': (1, 2, .9), 'ballon': (3, 4, .8),
                        'sorciere': (5, 6, .85)},
    'readings': {
        'resources': {'or': 2235125, 'elixir': 2399904, 'elixir_noire': 18549},
        'builders': {'libres': 4, 'total': 5},
        'lab_libre': False,
        'recoltes': {'or': 5, 'elixir': 6, 'elixir_noire': 3},
    },
}


def main():
    ap = argparse.ArgumentParser(description="Démo du registre d'outils")
    ap.add_argument('--jeu', action='store_true',
                    help="Utilise la perception réelle au lieu du monde simulé.")
    ap.add_argument('--garde-fous', action='store_true',
                    help="Démontre les trois refus (autorité, dépense, arguments).")
    args = ap.parse_args()

    from clashai.brain.tools import (
        SOURCE_ADMIN,
        SOURCE_CLAN,
        Tool,
        build_read_only_registry,
    )

    if args.jeu:
        print("Chargement de la perception (quelques secondes)…")
        from clashai.agents.world import build_world
        from clashai.navigation import game_loop as gl
        from clashai.perception.ui_detector import install
        install(verbose=False)
        models = gl.load_models()

        def world_fn():
            return build_world(models)
    else:
        def world_fn():
            return MONDE_SIMULE

    reg = build_read_only_registry(world_fn)

    print("\n=== OUTILS DISPONIBLES ===")
    print(f"  pour l'ADMIN (toi)        : {', '.join(reg.names(SOURCE_ADMIN))}")
    print(f"  pour le CLAN (les membres): {', '.join(reg.names(SOURCE_CLAN))}")

    print("\n=== APPELS ===")
    for name in reg.names():
        res = reg.call(name)
        print(f"\n  {name}() ->  {'OK' if res.ok else 'REFUS'}")
        print('   ' + json.dumps(res.data, ensure_ascii=False,
                                 indent=2).replace(chr(10), chr(10) + '   '))

    if args.garde_fous:
        # Outils factices : ils n'existent que pour montrer les refus. Aucun
        # d'eux ne touche au jeu — leur fonction ne fait rien.
        reg.register(Tool(name='lancer_attaque',
                          description="Lance une attaque (démo).",
                          fn=lambda **kw: {'lance': True},
                          parameters={'type': 'object', 'properties': {}},
                          acts=True, spends=True))
        reg.register(Tool(
            name='donner_troupes',
            description="Donne des troupes (démo).",
            fn=lambda **kw: {'donne': True},
            parameters={'type': 'object',
                        'properties': {'troupe': {'type': 'string'},
                                       'quantite': {'type': 'integer'}},
                        'required': ['troupe']},
            acts=True))

        print("\n\n=== LES TROIS GARDE-FOUS ===")
        print("\n(ils sont dans le CODE, pas dans le prompt : le modèle n'a")
        print(" jamais l'occasion de refuser, il n'a simplement pas la main)\n")

        cas = [
            ("1. AUTORITÉ — un membre du clan demande une attaque",
             ('lancer_attaque', {}, SOURCE_CLAN, False)),
            ("   le même membre lit l'état : autorisé",
             ('etat_du_village', {}, SOURCE_CLAN, False)),
            ("2. DÉPENSE — l'admin attaque sans confirmer",
             ('lancer_attaque', {}, SOURCE_ADMIN, False)),
            ("   l'admin confirme",
             ('lancer_attaque', {}, SOURCE_ADMIN, True)),
            ("3. ARGUMENTS — le modèle invente un paramètre",
             ('donner_troupes', {'troupe': 'ballon', 'couleur': 'rouge'},
              SOURCE_ADMIN, False)),
            ("   il oublie l'obligatoire",
             ('donner_troupes', {'quantite': 3}, SOURCE_ADMIN, False)),
            ("   il se trompe de type",
             ('donner_troupes', {'troupe': 'ballon', 'quantite': 'trois'},
              SOURCE_ADMIN, False)),
            ("   arguments corrects",
             ('donner_troupes', {'troupe': 'ballon', 'quantite': 3},
              SOURCE_ADMIN, False)),
        ]
        for titre, (name, a, src, confirm) in cas:
            res = reg.call(name, a, source=src, confirm=confirm)
            print(f"{titre}")
            print(f"   -> {'OK' if res.ok else 'REFUS : ' + res.error}\n")

        print("=== JOURNAL ===")
        print("(tout est tracé, y compris les refus — base du tableau de bord)")
        for e in reg.last(6):
            mark = 'ok   ' if e['ok'] else 'REFUS'
            print(f"  [{mark}] {e['source']:<5} {e['tool']}")


if __name__ == "__main__":
    main()
