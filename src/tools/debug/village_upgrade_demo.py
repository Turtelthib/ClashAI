# src/tools/debug/village_upgrade_demo.py
# Démo de test codée en dur pour l'exécuteur d'upgrade (V5.2, increment 2).
#
# En attendant que le LLM décide QUOI améliorer, ce petit script vérifie que le
# FLUX marche en vrai : on donne une position de bâtiment en dur, on lit les
# capteurs (ouvriers / ressources), et on lance upgrade_building.
#
# SÛR PAR DÉFAUT : sans --confirm, le décideur d'affordabilité n'est pas fourni →
# la démo va jusqu'à l'écran de confirmation puis ANNULE (aucune dépense). Idéal
# pour valider l'ouverture du bâtiment + la détection du bouton `Améliorer` sans
# risquer un vrai upgrade. Ajoute --confirm pour réellement confirmer (dépense !).
#
# Usage :
#   # émulateur branché, sur l'écran du village :
#   uv run python -m tools.debug.village_upgrade_demo --x 900 --y 500
#   uv run python -m tools.debug.village_upgrade_demo --x 900 --y 500 --confirm
#
# --x/--y = position ADB (1920x1080) du bâtiment à tenter d'améliorer.

import argparse


def main():
    ap = argparse.ArgumentParser(description="Démo upgrade bâtiment (village)")
    ap.add_argument('--x', type=int, required=True, help="X ADB du bâtiment")
    ap.add_argument('--y', type=int, required=True, help="Y ADB du bâtiment")
    ap.add_argument('--confirm', action='store_true',
                    help="Confirme réellement l'upgrade (DÉPENSE). Sinon annule.")
    args = ap.parse_args()

    from clashai.navigation import game_loop as gl
    from clashai.perception.ui_detector import install
    from clashai.village.upgrader import VillageUpgrader

    # Branche le CNN UI (comme au démarrage du bot) pour que reader/upgrader le
    # réutilisent via get_detector().
    install(verbose=True)

    up = VillageUpgrader(verbose=True)

    # Capteurs (contexte que le LLM lira) ------------------------------------
    img = gl.adb_screenshot()
    if img is None:
        print("ERREUR: pas de capture (émulateur branché ? bon écran ?)")
        return
    builders = up.free_builders(img)
    resources = up.resources(img)
    print(f"\nOuvriers libres : {builders}")
    print(f"Ressources      : {resources}")
    print("(None / {} = classes compteur_*/nombre_ouvrier pas encore dans le "
          "CNN UI — re-train nécessaire)\n")

    # Décideur : --confirm force la confirmation, sinon on laisse la sécurité
    # annuler (pas de resource_type ni de décideur → statut need_decision).
    decider = (lambda price, res: True) if args.confirm else None

    result = up.upgrade_building(
        (args.x, args.y), gl.adb_screenshot, gl.adb_tap,
        confirm_decider=decider,
    )
    print(f"\nRésultat : {result.status}"
          f" | prix={result.price} | ressources={result.resources}"
          f" | ouvriers={result.builders}")


if __name__ == "__main__":
    main()
