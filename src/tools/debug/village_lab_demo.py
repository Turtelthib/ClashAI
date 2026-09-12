# src/tools/debug/village_lab_demo.py
# Démo de test pour les recherches au laboratoire (V5.2, increment 3).
#
# SÛR PAR DÉFAUT : sans --confirm, la démo va jusqu'à l'écran de confirmation
# puis ANNULE, même si la recherche est payable (statut `declined`). Idéal pour
# vérifier que le labo est trouvé, que le menu s'ouvre et que les troupes
# améliorables sont bien reconnues.
#
# Avec --confirm : dépense, mais SEULEMENT sur preuve d'affordabilité. Un prix
# illisible annule. (Avant le 12 sept. 2026, --confirm passait `lambda: True`,
# qui confirmait même un prix illisible, et le mode « sûr » confirmait une
# recherche prouvée payable.)
#
# Usage (émulateur branché, sur l'écran du village) :
#   uv run python -m tools.debug.village_lab_demo
#   uv run python -m tools.debug.village_lab_demo --scan        # lecture seule
#   uv run python -m tools.debug.village_lab_demo --confirm     # DÉPENSE
#   uv run python -m tools.debug.village_lab_demo --troupe dragon

import argparse


def main():
    ap = argparse.ArgumentParser(description="Démo recherche laboratoire")
    ap.add_argument('--confirm', action='store_true',
                    help="Confirme réellement la recherche (DÉPENSE). Sinon annule.")
    ap.add_argument('--troupe', default=None,
                    help="Nom CNN de la troupe à améliorer (ex. dragon). "
                         "Défaut : la moins chère.")
    ap.add_argument('--scan', action='store_true',
                    help="N'agit PAS : capture l'écran courant et liste ce que "
                         "le CNN y reconnaît (à lancer sur l'écran du labo).")
    ap.add_argument('--debug-dir', default='debug_lab',
                    help="Dossier des captures annotées. Vide pour désactiver.")
    args = ap.parse_args()

    from clashai.navigation import game_loop as gl
    from clashai.perception.ui_detector import install
    from clashai.village.lab import VillageLab

    install(verbose=True)
    models = gl.load_models()
    lab = VillageLab(verbose=True, debug_dir=args.debug_dir or None)

    # --- mode scan : diagnostic pur, aucun tap ------------------------------
    if args.scan:
        img = gl.adb_screenshot()
        if img is None:
            print("ERREUR: pas de capture (émulateur branché ?)")
            return
        print("\nCartes reconnues sur l'écran courant :")
        for c in lab.list_candidates(img, models):
            print(f"   {c}  conf={c.conf:.2f}")
        print("\n(vide = soit tu n'es pas sur l'écran du labo, soit le CNN de la "
              "barre de troupes ne reconnaît pas ces vignettes)")
        print(f"Labo libre : {lab.is_free(img)}   "
              f"(True = aucune recherche en cours)")
        if args.debug_dir:
            import os
            out = lab.annotate_scan(
                img, models, os.path.join(args.debug_dir, 'lab_scan.png'))
            if out:
                print("")
                print(f"Capture annotée -> {out}")
                print("  vert = améliorable (avec le prix lu) "
                      "· gris = non améliorable")
        return

    # --- flux complet -------------------------------------------------------
    img = gl.adb_screenshot()
    if img is None:
        print("ERREUR: pas de capture (émulateur branché ? bon écran ?)")
        return
    print(f"\nLabo libre : {lab.is_free(img)}")
    print(f"Position du labo : {lab.find_lab(img, models)}")

    choose = None
    if args.troupe:
        def choose(cands):
            match = [c for c in cands if c.name == args.troupe]
            return match[0] if match else None

    # Sans --confirm : refuse tout (annulation garantie). Avec --confirm : aucun
    # décideur → confirmation uniquement sur preuve d'affordabilité.
    decider = None if args.confirm else (lambda price, res: False)

    result = lab.research(gl.adb_screenshot, gl.adb_tap, models,
                          choose=choose, confirm_decider=decider)
    print(f"\nRésultat : {result.status} | prix={result.price} "
          f"| ressources={result.resources}")
    if result.status == 'declined':
        print("  (mode sûr : la recherche était PROUVÉE payable, annulée "
              "volontairement — relance avec --confirm pour dépenser)")
    elif result.status == 'need_decision':
        print("  (prix ou ressource illisible : annulé par sécurité, même "
              "avec --confirm)")
    if args.debug_dir:
        print(f"\nCaptures annotées dans : {args.debug_dir}/")
        print("  lab_1_village.png / lab_2_menu.png / lab_3_grille.png")


if __name__ == "__main__":
    main()
