# src/tools/debug/clan_games_reader_demo.py
# Vérifie ClanGamesReader — HORS LIGNE sur les captures, ou en direct.
#
# Ne tape JAMAIS : le reader n'a aucune méthode qui agit. Le mode `--live`
# photographie l'écran courant et le lit, rien de plus.
#
# Usage :
#   # sur les captures sauvegardées (aucun émulateur requis)
#   uv run python -m tools.debug.clan_games_reader_demo
#   uv run python -m tools.debug.clan_games_reader_demo --fichier menu_defis
#
#   # sur l'écran courant (émulateur branché, à lancer sur la fenêtre des jeux)
#   uv run python -m tools.debug.clan_games_reader_demo --live

import argparse
import glob
import os


def afficher(nom, img, reader):
    from clashai.clan_games.reader import ClanGamesReader

    raw = reader._det().detect_raw(img)

    print(f"\n{'=' * 66}\n {nom}\n{'=' * 66}")

    ent = reader.entree(img)
    print(f"  entree village      : {ent if ent else chr(45)}")
    print(f"  menu ouvert         : {reader.menu_ouvert(raw=raw)}")

    score = reader.score_personnel(img, raw=raw)
    print(f"  score personnel     : {score if score else '(non lu)'}")

    grille = reader.lire_grille(img, raw=raw)
    print(f"  cartes              : {len(grille)}")
    for d in grille:
        etat = ('EN COURS' if d.active
                else 'grisée' if d.grisee
                else 'dispo')
        pts = d.points if d.points is not None else '?'
        prog = f"  progression={d.progression}" if d.progression else ''
        print(f"     ({d.x:4d},{d.y:4d})  {pts:>5} pts  {etat:9s} conf={d.conf:.2f}{prog}")

    actif = ClanGamesReader.defi_actif(grille)
    if actif:
        print(f"  -> défi en cours à ({actif.x},{actif.y}), progression {actif.progression}")

    pop = reader.lire_popup(img, raw=raw)
    if pop['ouvert']:
        print(f"  pop-up              : engagé={pop['engage']}  commencer={pop['commencer']}")
        print(f"     description : {pop['description']}")


def main():
    from clashai.paths import DATA_DIR

    ap = argparse.ArgumentParser(description="Vérifie ClanGamesReader (lecture seule).")
    ap.add_argument('--dossier', default=os.path.join(DATA_DIR, 'captures', 'jeux_clan'))
    ap.add_argument('--fichier', help="Une seule capture (nom sans _raw.png).")
    ap.add_argument('--live', action='store_true', help="Lire l'écran courant.")
    args = ap.parse_args()

    from PIL import Image

    from clashai.clan_games.reader import ClanGamesReader
    reader = ClanGamesReader()

    if args.live:
        from clashai.navigation import game_loop as gl
        img = gl.adb_screenshot()
        if img is None:
            print("ERREUR: pas de capture (émulateur branché ?)")
            return
        afficher('ÉCRAN COURANT', img, reader)
        return

    if args.fichier:
        chemins = [os.path.join(args.dossier, f'{args.fichier}_raw.png')]
    else:
        chemins = sorted(glob.glob(os.path.join(args.dossier, '*_raw.png')))

    if not chemins:
        print(f"Aucune capture dans {args.dossier}")
        print("Lance d'abord : uv run python -m tools.debug.clan_games_capture")
        return

    for p in chemins:
        if not os.path.exists(p):
            print(f"Introuvable : {p}")
            continue
        nom = os.path.basename(p).replace('_raw.png', '')
        afficher(nom, Image.open(p).convert('RGB'), reader)


if __name__ == '__main__':
    main()
