# src/tools/debug/widgets_demo.py
# Diagnostic des widgets chiffrés (compteurs, ouvriers, labo) — V5.3.
#
# À QUOI ÇA SERT : quand le cerveau répond « je ne sais pas » pour une ressource,
# il y a exactement DEUX causes possibles, et elles se corrigent à des endroits
# opposés :
#
#   1. le CNN UI n'a pas détecté le widget       -> problème de DATASET
#      (classe rare, mal labélisée, ou sous le seuil de confiance)
#   2. le widget est détecté mais le nombre est refusé -> problème de LECTURE
#      (segmentation des glyphes, garde-fou d'homogénéité)
#
# Cet outil tranche : il affiche, widget par widget, la détection ET la lecture,
# et sauve les crops pour qu'on VOIE ce que le digit CNN a reçu.
#
# LECTURE SEULE : aucun tap, aucune dépense. Sans danger.
#
# Usage (émulateur branché, sur l'écran du village) :
#   uv run python -m tools.debug.widgets_demo
#   uv run python -m tools.debug.widgets_demo --conf 0.15   # abaisse le seuil CNN
#   uv run python -m tools.debug.widgets_demo --debug-dir ""  # sans les crops

import argparse
import os


def main():
    ap = argparse.ArgumentParser(
        description="Diagnostic des widgets chiffrés (lecture seule)")
    ap.add_argument('--conf', type=float, default=None,
                    help="Seuil de confiance du CNN UI. Abaisse-le (ex. 0.15) "
                         "pour savoir si un widget manquant est SOUS le seuil "
                         "ou carrément invisible pour le modèle.")
    ap.add_argument('--debug-dir', default='debug_widgets',
                    help="Dossier des crops + capture annotée. Vide = désactivé.")
    args = ap.parse_args()

    from clashai.navigation import game_loop as gl
    from clashai.perception import digit_reader
    from clashai.perception.ui_detector import UIDetector
    from clashai.perception.widget_reader import (
        BUILDERS_CLASS,
        LAB_CLASS,
        RESOURCE_CLASSES,
        WidgetReader,
    )

    det = UIDetector(conf=args.conf, verbose=True) if args.conf is not None \
        else UIDetector(verbose=True)

    img = gl.adb_screenshot()
    if img is None:
        print("ERREUR: pas de capture (émulateur branché ?)")
        return

    raw = det.detect_raw(img)
    reader = WidgetReader(detector=_Cached(raw))

    out_dir = args.debug_dir or None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nSeuil CNN utilisé : {det.conf:.2f}")
    print(f"Classes détectées à l'écran : {len(raw)}")
    print("")
    print(f"{'widget':<26} {'détecté':<20} {'lu':<14} {'glyphes'}")
    print("-" * 72)

    targets = [(k, c) for k, c in RESOURCE_CLASSES.items()]
    targets += [('ouvriers', BUILDERS_CLASS), ('labo', LAB_CLASS)]

    for key, cls in targets:
        dets = raw.get(cls) or []
        if not dets:
            print(f"{key:<26} {'NON — absent':<20} {'-':<14} -")
            continue

        d = dets[0]
        seen = f"oui conf={d.conf:.2f}"
        crop = reader._widget_crop(img, d)
        if crop is None:
            print(f"{key:<26} {seen:<20} {'crop vide':<14} -")
            continue

        glyphs = digit_reader.segment_widget_glyphs(crop)
        if cls in (BUILDERS_CLASS, LAB_CLASS):
            value, conf = digit_reader.read_widget_ratio(crop)
            shown = f"{value[0]}/{value[1]}" if value else "REFUSÉ"
        else:
            value, conf = digit_reader.read_widget_number(crop)
            shown = str(value) if value is not None else "REFUSÉ"
        if value is not None:
            shown += f" ({conf:.2f})"

        print(f"{key:<26} {seen:<20} {shown:<14} {len(glyphs)}")

        if out_dir:
            crop.save(os.path.join(out_dir, f"{key}.png"))

    if out_dir:
        det.annotate(img, os.path.join(out_dir, 'ecran_annote.png'))
        print(f"\nCrops + capture annotée -> {out_dir}/")

    print("""
Comment lire ce tableau
  « NON — absent »  le CNN UI ne voit pas ce widget. Relance avec --conf 0.15 :
                    s'il apparaît, c'est un manque de DATASET (classe à renforcer),
                    pas un bug de code.
  « REFUSÉ »        le widget est trouvé mais le nombre est jugé peu fiable, donc
                    écarté volontairement (on ne devine jamais). Regarde le crop
                    dans le dossier de debug : glyphes collés, icône prise pour un
                    chiffre, ou crop trop serré.
  glyphes           nombre de formes isolées. Il doit valoir le nombre de CHIFFRES
                    (2 pour un ratio "1/6" : le '/' n'est pas un glyphe).
""")


class _Cached:
    """Rejoue des détections déjà calculées (évite une 2e inférence UI)."""

    def __init__(self, raw):
        self._raw = raw

    def detect_raw(self, _img):
        return self._raw


if __name__ == "__main__":
    main()
