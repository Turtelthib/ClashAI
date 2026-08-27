# src/tools/debug/clan_games_croiser.py
# Croise les 8 défis : ouvre chaque carte, LIT sa description, passe à la suivante.
#
# Pourquoi cet outil existe
# -------------------------
# La grille ne porte AUCUN texte (icône + points seulement). La phrase qui dit
# quoi faire — « Gagnez une étoile en combat multijoueur en utilisant au moins
# 1 Dragon. » — n'existe que dans le pop-up de détail. Elle est la seule source
# de sens pour le catalogue (`configs/clan_games.json`), donc il faut ouvrir les
# cartes une par une. C'est aussi le prototype de `selector.lister_defis()`.
#
# ⚠️⚠️ SÉCURITÉ — CE SCRIPT TAPE. Trois garde-fous, dans cet ordre :
#
#   1. LISTE BLANCHE DE POINTS. Un tap n'est émis que sur (a) le centre d'une
#      `carte_defi` détectée à l'instant, ou (b) le point neutre de fermeture.
#      Aucune autre coordonnée n'est atteignable — il n'y a pas de branche du
#      code qui tape autre chose.
#   2. VETO GÉOMÉTRIQUE. Avant CHAQUE tap, on vérifie que le point ne tombe pas
#      dans la boîte de `commencer_defi` ni de `rejeter`. Le pop-up recouvre la
#      grille : sans ce veto, taper « la carte 5 » d'après une lecture périmée
#      pourrait atterrir sur COMMENCER. Le tap est annulé, pas ajusté.
#   3. RELECTURE À CHAQUE TOUR. La grille est re-détectée avant chaque carte —
#      on n'agit jamais sur une position vieille d'une seconde.
#
# Accepter un défi est IRRÉVERSIBLE (le rejeter a une pénalité en jeu). D'où le
# `--confirmer` obligatoire : sans lui, le script montre ce qu'il ferait et sort.
#
# Usage (émulateur branché, fenêtre des jeux de clan DÉJÀ OUVERTE) :
#   uv run python -m tools.debug.clan_games_croiser              # simulation, 0 tap
#   uv run python -m tools.debug.clan_games_croiser --confirmer  # croise pour de vrai

import argparse
import json
import os
import time

# Délais (s). Le pop-up s'ouvre avec une animation ; trop court = on lit l'écran
# d'avant et on croit que la carte n'a pas de description.
_D_POPUP = 1.2
_D_FERMETURE = 0.8

# Point neutre de fermeture : le panneau d'illustration à GAUCHE de la fenêtre
# (le barbare). Volontairement au-dessus du bandeau de score, qui porte un « + »
# (achat) qu'on ne veut approcher sous aucun prétexte.
POINT_NEUTRE = (370, 450)


def _interdit(x, y, raw, reader):
    """Le point tombe-t-il sur un bouton qui ENGAGE ou REJETTE un défi ?"""
    from clashai.clan_games.reader import COMMENCER, REJETER
    for cls in (COMMENCER, REJETER):
        for d in raw.get(cls, []):
            x1, y1, x2, y2 = reader._boite(d)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return cls
    return None


def croiser(reader, screenshot_fn, tap_fn, confirmer, debug_dir=None):
    """Ouvre chaque carte, lit sa description. Renvoie la liste des défis lus."""
    img = screenshot_fn()
    if img is None:
        print("ERREUR: pas de capture (émulateur branché ?)")
        return []

    if not reader.menu_ouvert(screenshot_pil=img):
        print("ERREUR: la fenêtre des jeux de clan n'est pas ouverte.")
        print("        Ouvre-la toi-même, puis relance.")
        return []

    grille = reader.lire_grille(img)
    print(f"\n{len(grille)} carte(s) détectée(s).")
    score = reader.score_personnel(img)
    if score:
        print(f"Score personnel : {score[0]} / {score[1]}")

    if not confirmer:
        print("\nSIMULATION (--confirmer absent) — voici les taps qui SERAIENT émis :")
        for i, d in enumerate(grille, 1):
            pts = d.points if d.points is not None else '?'
            print(f"   {i}. carte ({d.x}, {d.y})  {pts} pts")
        print(f"   puis fermeture neutre en {POINT_NEUTRE}")
        print("\nAucun tap émis. Relance avec --confirmer pour croiser réellement.")
        return []

    resultats = []
    for i in range(len(grille)):
        print(f"\n--- carte {i + 1}/{len(grille)} ---")

        # 3. RELECTURE : on repart d'une frame fraîche à chaque tour.
        img = screenshot_fn()
        if img is None:
            print("   pas de capture, on arrête")
            break

        raw = reader._det().detect_raw(img)

        # Un pop-up encore ouvert recouvre la grille -> on ferme d'abord.
        if reader.lire_popup(img, raw=raw)['ouvert']:
            tap_fn(*POINT_NEUTRE)
            time.sleep(_D_FERMETURE)
            img = screenshot_fn()
            raw = reader._det().detect_raw(img)

        cartes = reader.lire_grille(img, raw=raw)
        if i >= len(cartes):
            print("   carte absente de cette frame (grille plus courte), on arrête")
            break
        carte = cartes[i]

        # 2. VETO : on n'ajuste pas un tap suspect, on l'annule.
        faute = _interdit(carte.x, carte.y, raw, reader)
        if faute:
            print(f"   ANNULÉ : ({carte.x},{carte.y}) tombe sur '{faute}' — on ne tape pas.")
            continue

        tap_fn(carte.x, carte.y)
        time.sleep(_D_POPUP)

        img = screenshot_fn()
        if img is None:
            print("   pas de capture après le tap")
            continue

        pop = reader.lire_popup(img)
        desc = pop['description']
        print(f"   points      : {carte.points if carte.points is not None else '?'}")
        print(f"   engagé      : {pop['engage']}")
        print(f"   description : {desc if desc else '(non lue)'}")

        resultats.append({
            'index': i + 1,
            'x': carte.x, 'y': carte.y,
            'points': carte.points,
            'grisee': carte.grisee,
            'active': carte.active,
            'progression': carte.progression,
            'engage': pop['engage'],
            'description': desc,
        })

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            img.save(os.path.join(debug_dir, f'defi_{i + 1:02d}.png'))

    # Fermeture finale.
    tap_fn(*POINT_NEUTRE)
    return resultats


def main():
    from clashai.paths import DATA_DIR

    sortie = os.path.join(DATA_DIR, 'captures', 'jeux_clan', 'defis.json')

    ap = argparse.ArgumentParser(
        description="Croise les défis pour en lire les descriptions (prototype du selector).")
    ap.add_argument('--confirmer', action='store_true',
                    help="Émet réellement les taps. Sans ce drapeau : simulation.")
    ap.add_argument('--out', default=sortie, help=f"JSON de sortie (défaut : {sortie})")
    ap.add_argument('--debug-dir', default=os.path.join(DATA_DIR, 'captures', 'jeux_clan', 'defis'),
                    help="Dossier des captures de pop-up. Vide pour désactiver.")
    args = ap.parse_args()

    from clashai.clan_games.reader import ClanGamesReader
    from clashai.navigation import game_loop as gl
    from clashai.perception.ui_detector import install

    install(verbose=True)
    reader = ClanGamesReader()

    print("=" * 70)
    print(" CROISEMENT DES DÉFIS")
    print(" Ne tape QUE : le centre d'une carte détectée, et le point neutre.")
    print(" `commencer_defi` et `rejeter` sont sous veto géométrique.")
    print("=" * 70)

    resultats = croiser(
        reader, gl.adb_screenshot, gl.adb_tap,
        confirmer=args.confirmer,
        debug_dir=args.debug_dir or None,
    )

    if not resultats:
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f" {len(resultats)} défi(s) lu(s) -> {args.out}")
    lues = sum(1 for r in resultats if r['description'])
    print(f" {lues} description(s) exploitable(s) pour le catalogue.")
    print("=" * 70)


if __name__ == '__main__':
    main()
