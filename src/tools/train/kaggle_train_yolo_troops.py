# kaggle_train_yolo_troops.py
# Entraînement YOLO "troupes terrain" — version KAGGLE, standalone, sans flags.
# S'adapte tout seul : détecte la structure (Roboflow train/images ou images/train)
# et lit les classes depuis le data.yaml du dataset (rien à maintenir à la main).
#
# ── À FAIRE sur Kaggle ───────────────────────────────────────────────────────
#   1. Upload ton dossier dataset_troops (train/images, train/labels,
#      valid/images, valid/labels, data.yaml) en tant que "Dataset" Kaggle,
#      puis ajoute-le au notebook (panneau Input).
#   2. Notebook Settings → Accelerator → **GPU** (T4).
#   3. Règle DATASET_DIR ci-dessous sur le chemin monté (panneau Input :
#      /kaggle/input/<nom-de-ton-dataset>[/sous-dossier]).
#   4. Lance :   !python kaggle_train_yolo_troops.py
#      (ou colle tout ce fichier dans une cellule de notebook)
#
# ── Résultat ────────────────────────────────────────────────────────────────
#   Meilleur modèle → /kaggle/working/yolo_troops_best.pt
#   Télécharge-le, puis place-le en  weights/yolo_troops.pt  dans ton repo.
# ─────────────────────────────────────────────────────────────────────────────

import os
import subprocess
import sys

# ════════════════════════════ CONFIG (à ajuster) ════════════════════════════
DATASET_DIR = "/kaggle/input/dataset-troops"   # <-- CHEMIN de ton dataset monté

MODEL    = "yolo26m.pt"   # modèle de base auto-téléchargé (alt : "yolo11m.pt")
EPOCHS   = 100
BATCH    = 16             # réduis à 8 si "CUDA out of memory" sur le GPU Kaggle
IMG_SIZE = 832
OUT_DIR  = "/kaggle/working"
# ═════════════════════════════════════════════════════════════════════════════


def _find(base, candidates):
    for c in candidates:
        if os.path.isdir(os.path.join(base, c)):
            return c
    return None


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"],
                   check=False)
    from ultralytics import YOLO
    import yaml

    if not os.path.isdir(DATASET_DIR):
        print(f"ERREUR: DATASET_DIR introuvable : {DATASET_DIR}")
        print("-> Ajuste DATASET_DIR (regarde le panneau Input du notebook).")
        return

    # 1. détecte la structure des images (Roboflow ou images/train)
    train_rel = _find(DATASET_DIR, ["train/images", "images/train"])
    val_rel = _find(DATASET_DIR, ["valid/images", "val/images", "images/val", "images/valid"])
    if not train_rel or not val_rel:
        print("ERREUR: dossiers d'images introuvables sous DATASET_DIR.")
        print("Contenu :", os.listdir(DATASET_DIR))
        return

    # 2. lit les classes depuis le data.yaml du dataset
    src_yaml = os.path.join(DATASET_DIR, "data.yaml")
    if not os.path.exists(src_yaml):
        print(f"ERREUR: {src_yaml} introuvable (le data.yaml du dataset).")
        return
    with open(src_yaml, encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        print("ERREUR: aucune classe 'names' dans le data.yaml.")
        return
    print(f"{len(names)} classes détectées | train='{train_rel}' | val='{val_rel}'")

    # 3. génère un data.yaml avec chemins absolus Kaggle + les classes
    yaml_path = os.path.join(OUT_DIR, "coc_troops_kaggle.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write(f"train: {train_rel}\n")
        f.write(f"val: {val_rel}\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    # 4. entraînement (mêmes hyperparamètres + augmentations que le script local)
    print(f"\nEntraînement : {MODEL} | {EPOCHS} epochs | batch {BATCH} | imgsz {IMG_SIZE}\n")
    yolo = YOLO(MODEL)
    yolo.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        project=OUT_DIR,
        name="yolo_troops_train",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        verbose=True,
        plots=True,
        # optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        # loss weights
        box=7.5,
        cls=1.0,
        dfl=1.5,
        # augmentations (identiques au coc_troops.yaml)
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        degrees=0.0, translate=0.15, scale=0.4,
        fliplr=0.5, flipud=0.0, perspective=0.0, shear=0.0,
        mosaic=1.0, mixup=0.15, copy_paste=0.15,
    )

    # 5. copie le best.pt à un chemin simple à télécharger
    best_src = os.path.join(OUT_DIR, "yolo_troops_train", "weights", "best.pt")
    best_dst = os.path.join(OUT_DIR, "yolo_troops_best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ Meilleur modèle : {best_dst}")
        print("   -> télécharge-le et place-le en  weights/yolo_troops.pt  dans ton repo.")
    else:
        print(f"\n⚠️ best.pt introuvable ({best_src}) — regarde les logs d'entraînement.")


if __name__ == "__main__":
    main()
