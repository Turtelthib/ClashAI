# src/tools/debug/clan_games_demo.py
# Démo de l'agent jeux de clan — teste le flux SANS passer par le cerveau.
#
# Pourquoi ce fichier existe
# --------------------------
# Vécu : `--mode farm --jeux-clan-confirmer` a tourné 3,5 min, le LLM a pris UNE
# décision (`combat`), et l'agent n'a jamais eu de seconde chance. L'agent était
# pourtant bien éligible — mais on ne pouvait pas le savoir depuis les logs.
#
# Diagnostiquer un agent en attendant qu'un arbitre non déterministe veuille
# bien le choisir, c'est ingérable. Ici on l'appelle DIRECTEMENT.
#
# Usage (émulateur branché, bot AU VILLAGE) :
#   uv run python -m tools.debug.clan_games_demo --scan   # 0 tap : que voit-on ?
#   uv run python -m tools.debug.clan_games_demo          # flux complet, SANS engager
#   uv run python -m tools.debug.clan_games_demo --confirmer   # engage vraiment

import argparse
import os


def scanner(reader, img):
    """Diagnostic pur : ce que le CNN voit, sans toucher à rien."""
    from clashai.clan_games.reader import RACCOURCI, TENTE

    raw = reader._det().detect_raw(img)

    print("\n--- entrées vers les jeux de clan ---")
    for classe in (RACCOURCI, TENTE):
        dets = raw.get(classe, [])
        if dets:
            print(f"  {classe:16s} " +
                  ' '.join(f'{d.conf:.2f}@({d.x},{d.y})' for d in dets))
        else:
            print(f"  {classe:16s} absent")
    entree = reader.entree(img, raw=raw)
    print(f"  -> retenue : {entree if entree else 'AUCUNE (agent inactif)'}")

    print("\n--- fenêtre des jeux ---")
    print(f"  menu ouvert     : {reader.menu_ouvert(raw=raw)}")
    score = reader.score_personnel(img, raw=raw)
    print(f"  score personnel : {score if score else '(non lu)'}")

    grille = reader.lire_grille(img, raw=raw)
    print(f"  cartes          : {len(grille)}")
    for d in grille:
        etat = 'EN COURS' if d.active else 'grisée' if d.grisee else 'dispo'
        print(f"     ({d.x:4d},{d.y:4d})  {str(d.points):>5} pts  {etat}")


def main():
    from clashai.paths import DATA_DIR

    ap = argparse.ArgumentParser(
        description="Démo de l'agent jeux de clan (sûr par défaut : n'engage rien).")
    ap.add_argument('--scan', action='store_true',
                    help="N'agit PAS : montre ce que le CNN voit sur l'écran courant.")
    ap.add_argument('--confirmer', action='store_true',
                    help="Autorise le tap 'Commencer'. IRRÉVERSIBLE.")
    ap.add_argument('--debug-dir',
                    default=os.path.join(DATA_DIR, 'captures', 'jeux_clan', 'demo'),
                    help="Dossier des captures. Vide pour désactiver.")
    args = ap.parse_args()

    from clashai.clan_games.catalog import Catalogue, capacites_depuis_world
    from clashai.clan_games.reader import ClanGamesReader
    from clashai.clan_games.selector import ClanGamesSelector
    from clashai.navigation import game_loop as gl
    from clashai.perception.ui_detector import install

    install(verbose=True)
    reader = ClanGamesReader()

    if args.scan:
        img = gl.adb_screenshot()
        if img is None:
            print("ERREUR: pas de capture (émulateur branché ?)")
            return
        scanner(reader, img)
        return

    # Capacités : la barre de troupes du moment décide des défis « avec unité ».
    # Ici on charge les modèles pour la lire ; dans le bot, elle vient du `world`.
    models = gl.load_models()
    img = gl.adb_screenshot()
    troop_bar = []
    try:
        bar = (models or {}).get('troop_bar_detector')
        if bar is not None and img is not None:
            troop_bar = bar.detect(img)
    except Exception as e:
        print(f"WARNING: barre de troupes illisible ({e}) — aucun défi « avec unité »")

    capacites = capacites_depuis_world({'troop_bar': troop_bar})
    print(f"\nTroupes disponibles : "
          f"{', '.join(sorted(capacites.unites_disponibles)) or '(aucune lue)'}")

    selector = ClanGamesSelector(reader=reader, catalogue=Catalogue(),
                                 verbose=True, debug_dir=args.debug_dir or None)

    print("=" * 70)
    print(" AGENT JEUX DE CLAN — flux complet")
    print(f" confirmer = {args.confirmer}"
          f"{'  (Commencer SERA tapé)' if args.confirmer else '  (aucun engagement)'}")
    print("=" * 70)

    statut, details = selector.engager_le_meilleur(
        gl.adb_screenshot, gl.adb_tap, capacites, confirmer=args.confirmer,
    )

    print(f"\n{'=' * 70}")
    print(f" statut : {statut}")
    for k, v in details.items():
        print(f"   {k:12s} : {v}")
    print("=" * 70)

    if statut == 'choisi_non_engage':
        print("\nLe défi ci-dessus est celui qui SERAIT engagé.")
        print("Relance avec --confirmer si le choix te convient.")
    elif statut == 'aucun_defi_sur':
        print("\nAucun défi n'est faisable avec certitude -> on n'engage rien.")
        print("C'est le comportement voulu, pas une panne.")
    elif statut == 'menu_introuvable':
        print("\nEntrée des jeux non détectée. Vérifie avec --scan que le bot")
        print("est bien au village et que les jeux de clan sont actifs.")


if __name__ == '__main__':
    main()
