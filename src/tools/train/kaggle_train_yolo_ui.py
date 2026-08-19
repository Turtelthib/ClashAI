# kaggle_train_yolo_ui.py
# Entraînement YOLO "boutons / éléments UI" — version KAGGLE, standalone.
# Détecte tout seul la structure (Roboflow) + lit les classes du data.yaml.
#
# ⚠️ Config DIFFÉRENTE du script troupes :
#   - imgsz ÉLEVÉ (1280) : les boutons se différencient par leur TEXTE → il
#     faut de la résolution pour que le modèle "voie" les lettres.
#   - fliplr=0 (CRITIQUE) : pas de miroir horizontal (sinon le texte est
#     retourné = le modèle apprend n'importe quoi). Idem flipud/rotation.
#   - augmentations DOUCES : l'UI est régulière (couleurs/positions stables),
#     pas besoin des grosses augmentations des troupes (mixup/copy_paste off,
#     mosaic modéré, hsv/scale/translate faibles).
#
# ── À FAIRE sur Kaggle ───────────────────────────────────────────────────────
#   1. Upload ton dataset UI (train/images, train/labels, valid/…, data.yaml)
#      en Dataset Kaggle, ajoute-le au notebook.  2. GPU T4.
#   3. Règle DATASET_DIR.  4. !python kaggle_train_yolo_ui.py
#   → Résultat : /kaggle/working/yolo_ui_best.pt  → place-le en weights/yolo_ui.pt
# ─────────────────────────────────────────────────────────────────────────────

import os
import subprocess
import sys

# ════════════════════════════ CONFIG (à ajuster) ════════════════════════════
DATASET_DIR = "/kaggle/input/dataset-ui"   # <-- CHEMIN de ton dataset UI monté

MODEL    = "yolo26m.pt"   # bon choix (alt : "yolo11m.pt")
EPOCHS   = 120            # early-stop (patience) coupe tout seul si ça plafonne
BATCH    = -1             # -1 = auto (ultralytics prend le max qui rentre). OOM → 8
IMG_SIZE = 1280           # ÉLEVÉ pour lire le texte des boutons (1024 si trop lent/OOM)
OUT_DIR  = "/kaggle/working"
# ═════════════════════════════════════════════════════════════════════════════


def _find(base, candidates):
    for c in candidates:
        if os.path.isdir(os.path.join(base, c)):
            return c
    return None



def _check_gpu():
    """Refuse de demarrer sans GPU. Sans accelerateur, Kaggle bascule sur CPU :
    l'entrainement devient inutilisable ET la RAM sature -> le kernel meurt
    ~50 s apres << Starting training >>, sans aucune trace Python. Vecu en vrai,
    et le message d'erreur ne dit rien d'utile -> on verifie explicitement."""
    try:
        import torch
    except ImportError:
        print("ATTENTION: torch indisponible, impossible de verifier le GPU.")
        return True
    if torch.cuda.is_available():
        print(f"GPU OK : {torch.cuda.get_device_name(0)}")
        return True
    print("=" * 70)
    print("ARRET : AUCUN GPU DETECTE.")
    print("  Kaggle -> panneau de droite -> Session options -> Accelerator")
    print("  -> choisir 'GPU T4 x2' (ou P100), puis relancer.")
    print("  Sans GPU : entrainement inutilisable + kernel tue faute de RAM.")
    print("=" * 70)
    return False


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"],
                   check=False)
    import yaml
    from ultralytics import YOLO

    if not _check_gpu():
        return

    if not os.path.isdir(DATASET_DIR):
        print(f"ERREUR: DATASET_DIR introuvable : {DATASET_DIR}")
        return

    train_rel = _find(DATASET_DIR, ["train/images", "images/train"])
    val_rel = _find(DATASET_DIR, ["valid/images", "val/images", "images/val", "images/valid"])
    if not train_rel or not val_rel:
        print("ERREUR: dossiers d'images introuvables.")
        print("Contenu :", os.listdir(DATASET_DIR))
        return

    src_yaml = os.path.join(DATASET_DIR, "data.yaml")
    if not os.path.exists(src_yaml):
        print(f"ERREUR: {src_yaml} introuvable.")
        return
    with open(src_yaml, encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        print("ERREUR: aucune classe 'names' dans le data.yaml.")
        return
    print(f"{len(names)} classes détectées | train='{train_rel}' | val='{val_rel}'")

    yaml_path = os.path.join(OUT_DIR, "coc_ui_kaggle.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_DIR}\ntrain: {train_rel}\nval: {val_rel}\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    print(f"\nEntraînement UI : {MODEL} | {EPOCHS} ep | batch {BATCH} | imgsz {IMG_SIZE}\n")
    yolo = YOLO(MODEL)
    yolo.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        project=OUT_DIR,
        name="yolo_ui_train",
        exist_ok=True,
        patience=25,
        save=True, save_period=10, verbose=True, plots=True,
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        box=7.5, cls=1.0, dfl=1.5,
        # ── AUGMENTATIONS UI (douces + pas de miroir/rotation) ──────────────
        fliplr=0.0,        # ⚠️ PAS de miroir horizontal (texte des boutons)
        flipud=0.0,        # pas de retournement vertical
        degrees=0.0,       # pas de rotation
        perspective=0.0, shear=0.0,
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.3,   # léger (couleurs UI stables, fonds variés)
        translate=0.1, scale=0.2,            # léger (UI à résolution fixe)
        mosaic=0.5,        # modéré (aide un peu, pas 1.0 = composites trop artificiels)
        mixup=0.0, copy_paste=0.0,           # off (fusion d'écrans UI = artefacts)
        close_mosaic=10,   # coupe le mosaic sur les 10 dernières epochs (affine)
    )

    best_src = os.path.join(OUT_DIR, "yolo_ui_train", "weights", "best.pt")
    best_dst = os.path.join(OUT_DIR, "yolo_ui_best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ Meilleur modèle : {best_dst}")
        print("   -> télécharge-le et place-le en  weights/yolo_ui.pt")
    else:
        print(f"\n⚠️ best.pt introuvable ({best_src}).")


if __name__ == "__main__":
    main()
