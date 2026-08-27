# src/tools/debug/clan_games_capture.py
# Incrément 0 des jeux de clan : CAPTURER, pas agir.
#
# LECTURE SEULE ABSOLUE — ce script ne tape JAMAIS. C'est toi qui navigues dans
# le jeu ; lui ne fait que photographier ce que tu lui montres. Aucun risque de
# gemme, aucun défi accepté par accident (accepter un défi est irréversible :
# l'abandonner a une pénalité en jeu).
#
# Pourquoi un dump OCR et pas seulement des PNG
# ---------------------------------------------
# La question qui décide de TOUTE l'archi de l'agent est : « EasyOCR lit-il les
# titres des cartes de défi de façon fiable ? ». Si oui, on n'a besoin d'AUCUNE
# nouvelle classe CNN (l'OCR rend bbox + texte, donc géométrie ET sémantique).
# Si non, il faut labéliser et re-train — ce qui ne rentre pas dans la fenêtre
# des jeux de clan. Chaque capture produit donc :
#
#   <etape>_raw.png    la frame brute (rejouable indéfiniment, hors ligne)
#   <etape>_ocr.json   [{bbox, texte, conf}, ...]  <- la vraie réponse
#   <etape>_ocr.png    la frame avec les boîtes OCR dessinées (lecture humaine)
#   <etape>_cnn.png    ce que le CNN UI actuel croit voir (--cnn, facultatif)
#
# Les PNG bruts sont l'actif durable : les jeux de clan durent ~1 semaine, le
# code non. Une fois capturés, reader.py se développe et se teste hors ligne.
#
# Usage (émulateur branché) :
#   uv run python -m tools.debug.clan_games_capture           # guidé, toutes les étapes
#   uv run python -m tools.debug.clan_games_capture --etape menu_defis
#   uv run python -m tools.debug.clan_games_capture --liste
#   uv run python -m tools.debug.clan_games_capture --cnn     # + annotation CNN UI
#
#   # RAFALE — pour constituer le dataset de labeling (raw seulement, pas d'OCR)
#   uv run python -m tools.debug.clan_games_capture --rafale 120 --intervalle 2

import argparse
import json
import os
import time

# Séquence guidée. (clé, consigne affichée). L'ordre suit la navigation réelle.
ETAPES = [
    ('village_home',
     "Va sur ton VILLAGE (vue normale), avec l'icône d'entrée des jeux de clan "
     "visible à l'écran."),
    ('menu_defis',
     "Ouvre les JEUX DE CLAN. Laisse la grille de défis telle qu'elle s'affiche "
     "(sans scroller)."),
    ('menu_defis_scroll',
     "Fais défiler la grille vers le BAS pour montrer d'autres défis."),
    ('popup_defi',
     "Tape UN défi pour ouvrir son pop-up de détail (celui avec la description "
     "et le bouton pour l'accepter). ATTENTION : NE L'ACCEPTE PAS."),
    ('defi_en_cours',
     "Si tu as déjà un défi ACCEPTÉ en cours, montre-le (avec sa barre de "
     "progression). Sinon : tape 'skip'."),
    ('recompenses',
     "Ouvre le panneau des RÉCOMPENSES / paliers de points du clan."),
    ('village_home_defi_actif',
     "Reviens au village AVEC un défi actif (widget de progression visible). "
     "Sinon : tape 'skip'."),
]

_ETAPES_MAP = dict(ETAPES)


# =============================================================================
# OCR — EasyOCR direct, pour garder les BBOX
# =============================================================================
# social/chat/ocr.py existe déjà mais jette les bbox (il ne rend que des lignes
# de texte). Ici les positions sont le produit principal : c'est ce qui dira où
# taper une carte. On appelle donc EasyOCR directement.

_reader = None


def _ocr_reader():
    global _reader
    if _reader is None:
        import easyocr
        # Texte du jeu 100 % français. GPU laissé au CNN/YOLO : ce script est
        # un one-shot, le CPU suffit largement.
        _reader = easyocr.Reader(['fr'], gpu=False, verbose=False)
    return _reader


def _ocr_dump(img_pil):
    """[{bbox:[[x,y]x4], texte, conf}] — coordonnées en pixels de l'image."""
    import numpy as np
    results = _ocr_reader().readtext(np.asarray(img_pil.convert('RGB')),
                                     paragraph=False)
    out = []
    for bbox, texte, conf in results:
        out.append({
            'bbox': [[int(p[0]), int(p[1])] for p in bbox],
            'texte': texte.strip(),
            'conf': round(float(conf), 3),
        })
    # Ordre de lecture : haut->bas, puis gauche->droite.
    out.sort(key=lambda r: (r['bbox'][0][1], r['bbox'][0][0]))
    return out


def _draw_ocr(img_pil, lignes, out_path):
    """Dessine les boîtes OCR + le texte lu, pour vérification à l'œil."""
    from PIL import ImageDraw
    img = img_pil.convert('RGB').copy()
    d = ImageDraw.Draw(img)
    for r in lignes:
        pts = [tuple(p) for p in r['bbox']]
        d.polygon(pts + [pts[0]], outline=(0, 255, 0))
        d.text((pts[0][0], max(0, pts[0][1] - 12)),
               f"{r['texte']} ({r['conf']:.2f})", fill=(255, 255, 0))
    img.save(out_path)


# =============================================================================
# CAPTURE
# =============================================================================

def capturer(cle, out_dir, avec_cnn=False, detector=None):
    """Photographie l'écran courant et écrit raw/ocr.json/ocr.png (+ cnn.png)."""
    from clashai.navigation import game_loop as gl

    img = gl.adb_screenshot()
    if img is None:
        print("   ERREUR: pas de capture (émulateur branché ? fenêtre visible ?)")
        return False

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, cle)

    img.save(f'{base}_raw.png')
    print(f"   {cle}_raw.png          {img.size[0]}x{img.size[1]}")

    lignes = _ocr_dump(img)
    with open(f'{base}_ocr.json', 'w', encoding='utf-8') as f:
        json.dump(lignes, f, ensure_ascii=False, indent=2)
    _draw_ocr(img, lignes, f'{base}_ocr.png')
    print(f"   {cle}_ocr.json         {len(lignes)} zones de texte lues")

    # Aperçu immédiat : c'est ce qui répond « l'OCR tient-il ? » sans attendre.
    for r in lignes[:12]:
        print(f"      {r['conf']:.2f}  {r['texte']}")
    if len(lignes) > 12:
        print(f"      ... (+{len(lignes) - 12} autres, tout est dans le .json)")

    if avec_cnn and detector is not None:
        try:
            detector.annotate(img, f'{base}_cnn.png')
            print(f"   {cle}_cnn.png          (ce que le CNN UI actuel voit)")
        except Exception as e:
            print(f"   WARNING: annotation CNN impossible ({e})")

    return True


def rafale(n, intervalle, out_dir):
    """N captures BRUTES espacées de `intervalle` s — dataset de labeling.

    Pas d'OCR ici : à 2 s d'intervalle, EasyOCR ne suivrait pas, et pour
    labéliser on ne veut que des PNG. Tu navigues librement pendant ce
    temps (scroll de la grille, ouverture/fermeture de pop-ups, aller-retour
    village) : plus les captures sont VARIÉES, moins le modèle apprend une
    position par cœur.
    """
    from clashai.navigation import game_loop as gl

    os.makedirs(out_dir, exist_ok=True)
    horodatage = time.strftime('%Y%m%d_%H%M%S')
    print(f"\nRafale : {n} captures toutes les {intervalle} s "
          f"(~{n * intervalle // 60} min). Ctrl+C pour arrêter.")
    print("Navigue dans le jeu pendant ce temps — varie les écrans et les scrolls.\n")

    faits = 0
    try:
        for i in range(1, n + 1):
            img = gl.adb_screenshot()
            if img is None:
                print(f"  [{i}/{n}] pas de capture, on continue")
            else:
                nom = f'rafale_{horodatage}_{i:04d}.png'
                img.save(os.path.join(out_dir, nom))
                faits += 1
                print(f"  [{i}/{n}] {nom}")
            if i < n:
                time.sleep(intervalle)
    except KeyboardInterrupt:
        print("\n  (interrompu)")
    return faits


def main():
    from clashai.paths import DATA_DIR

    defaut = os.path.join(DATA_DIR, 'captures', 'jeux_clan')

    ap = argparse.ArgumentParser(
        description="Capture les écrans des jeux de clan (LECTURE SEULE, ne tape jamais).")
    ap.add_argument('--out', default=defaut, help=f"Dossier de sortie (défaut : {defaut})")
    ap.add_argument('--etape', help="Ne capturer qu'une étape (voir --liste).")
    ap.add_argument('--liste', action='store_true', help="Affiche les étapes et sort.")
    ap.add_argument('--cnn', action='store_true',
                    help="Ajoute l'annotation du CNN UI actuel (charge le modèle).")
    ap.add_argument('--rafale', type=int, metavar='N',
                    help="Mode dataset : N captures brutes d'affilée (pas d'OCR).")
    ap.add_argument('--intervalle', type=float, default=2.0,
                    help="Secondes entre deux captures en rafale (défaut 2).")
    args = ap.parse_args()

    if args.liste:
        print("\nÉtapes de capture :\n")
        for cle, consigne in ETAPES:
            print(f"  {cle:26s} {consigne}")
        return

    if args.etape and args.etape not in _ETAPES_MAP:
        print(f"Étape inconnue : {args.etape!r}. Voir --liste.")
        return

    if args.rafale:
        out = os.path.join(args.out, 'dataset')
        n = rafale(args.rafale, args.intervalle, out)
        print(f"\n{n} capture(s) brute(s) dans {out}")
        print("Prêtes à labéliser (LabelMe / Roboflow).")
        return

    detector = None
    if args.cnn:
        from clashai.perception.ui_detector import UIDetector
        detector = UIDetector(verbose=True)

    print("=" * 72)
    print(" CAPTURE JEUX DE CLAN — incrément 0")
    print(" LECTURE SEULE : ce script ne tape jamais. N'ACCEPTE AUCUN DÉFI.")
    print(f" Sortie : {args.out}")
    print("=" * 72)

    etapes = ([(args.etape, _ETAPES_MAP[args.etape])] if args.etape else ETAPES)

    faits = 0
    for i, (cle, consigne) in enumerate(etapes, 1):
        print(f"\n[{i}/{len(etapes)}] {cle}")
        print(f"   -> {consigne}")
        rep = input("   Entrée pour capturer  ('skip' pour passer, 'q' pour quitter) : ")
        rep = rep.strip().lower()
        if rep in ('q', 'quit', 'quitter'):
            break
        if rep in ('s', 'skip', 'passer'):
            print("   (passé)")
            continue
        if capturer(cle, args.out, avec_cnn=args.cnn, detector=detector):
            faits += 1

    print(f"\n{'=' * 72}")
    print(f" {faits} capture(s) dans {args.out}")
    print(" Regarde les *_ocr.png : si les titres des défis sont bien encadrés")
    print(" et bien lus, l'agent se construit SANS nouveau modèle.")
    print("=" * 72)


if __name__ == '__main__':
    main()
