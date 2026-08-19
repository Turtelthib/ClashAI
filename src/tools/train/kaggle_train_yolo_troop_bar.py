# kaggle_train_yolo_troop_bar.py
# Entraînement YOLO "barre de troupes" — version KAGGLE, standalone, sans flags.
# S'adapte tout seul : détecte la structure (Roboflow train/images ou images/train)
# et lit les classes depuis le data.yaml du dataset.
#
# Ce modèle détecte les VIGNETTES de troupes/sorts/héros (barre de combat, et
# aussi les cartes de l'écran du laboratoire). L'état grisé n'est PAS une classe :
# il est déduit à l'inférence par saturation HSV (`TroopBarDetector._is_grayed`).
# Seules les machines de siège déployées ont leur classe `_deploye` dédiée, car
# cliquer dessus a une conséquence destructive.
#
# ⚠️ DEUX PIÈGES À CONNAÎTRE AVANT DE DÉPLOYER UN NOUVEAU MODÈLE
#
# 1. PARITÉ imgsz ENTRAÎNEMENT ↔ INFÉRENCE. `troop_bar_detector.YOLO_IMGSZ` doit
#    valoir la MÊME chose que IMG_SIZE ici. Un écart a déjà fait chuter la
#    détection à 0-1 icône sur 9 (Session 13). Si tu changes IMG_SIZE, change
#    aussi YOLO_IMGSZ.
#
# 2. `model_artifacts.json` PILOTE LES DIMENSIONS DE L'OBS RL.
#    `troop_registry.cnn_class_names()` le lit pour dériver `SPELL_NAMES`
#    (registre ∩ classes du CNN) → un sort absent du CNN reste inerte. Donc :
#      - déposer le .pt SANS son `model_artifacts.json` à jour = liste de classes
#        périmée ;
#      - AJOUTER/RETIRER un sort dans le dataset CHANGE `SPELL_NAMES`, donc
#        l'obs et l'espace d'actions → le checkpoint RL ne se recharge plus
#        (`PPOAgentV4.load()` repart de zéro EN SILENCE).
#    Vérifier après déploiement :
#      uv run python -c "from clashai.combat import action_space as A; \
#        from clashai.combat.agent_v4 import constants as C; \
#        print('sorts', A.NUM_SPELLS, 'obs', C.VECTOR_SIZE, 'actions', A.TOTAL_ACTIONS)"
#
# ── À FAIRE sur Kaggle ───────────────────────────────────────────────────────
#   1. Upload le dataset (train/images, train/labels, valid/…, data.yaml) en
#      Dataset Kaggle, puis ajoute-le au notebook (panneau Input).
#   2. Notebook Settings → Accelerator → GPU (T4).
#   3. Règle DATASET_DIR sur le chemin monté.
#   4. Lance :  !python kaggle_train_yolo_troop_bar.py
#
# ── Résultat ────────────────────────────────────────────────────────────────
#   Meilleur modèle → /kaggle/working/yolo_troop_bar_best.pt
#   Le déposer en   weights/yolo_troop_bar/yolo_troop_bar.pt
#   + mettre à jour weights/yolo_troop_bar/model_artifacts.json (cf. piège 2)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import subprocess
import sys

# ════════════════════════════ CONFIG (à ajuster) ════════════════════════════
DATASET_DIR = "/kaggle/input/dataset-troupes-barre"   # <-- chemin du dataset monté

MODEL    = "yolo26m.pt"   # le modèle déployé actuellement est un 26m (~44 Mo)
EPOCHS   = 120
BATCH    = 8              # EXPLICITE, surtout PAS -1 (voir "Kernel died" plus bas)
IMG_SIZE = 1088           # ⚠️ DOIT rester égal à troop_bar_detector.YOLO_IMGSZ
WORKERS  = 2              # Kaggle a peu de RAM : 8 workers (defaut) la saturent
OUT_DIR  = "/kaggle/working"

# -- << Kernel died >> sur Kaggle ------------------------------------------
# Le kernel meurt SANS trace Python (le processus est tue), typiquement ~50 s
# apres << Starting training >>.
#
#   CAUSE N.1, DE LOIN : L'ACCELERATEUR GPU N'EST PAS ACTIVE.
#   Sans GPU, ultralytics bascule sur CPU -> la RAM Kaggle sature et le kernel
#   se fait tuer. Le message d'erreur ne dit rien de tout ca.
#   -> Kaggle, panneau de droite : Session options > Accelerator > GPU T4 x2.
#   `_check_gpu()` ci-dessous refuse maintenant de demarrer dans ce cas.
#
# Si le GPU EST actif et que ca meurt quand meme, c'est la memoire : descendre
# DANS CET ORDRE (le 1er preserve la qualite) :
#   1. BATCH   8 -> 4 -> 2       (BATCH est explicite ici, pas -1 : l'auto-batch
#      choisit son lot en sondant le GPU, autant garder la main a 1088)
#   2. WORKERS 2 -> 0            (0 = chargement dans le process principal)
#   3. MODEL   yolo26m -> yolo26s  (2x plus leger ; l'ancien script tournait en
#      26s avec BATCH=16 a 1088, donc ca passe large)
#   NE PAS baisser IMG_SIZE : il doit rester egal a YOLO_IMGSZ (cf. piege 1).
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
        print("-> Ajuste DATASET_DIR (regarde le panneau Input du notebook).")
        return

    train_rel = _find(DATASET_DIR, ["train/images", "images/train"])
    val_rel = _find(DATASET_DIR, ["valid/images", "val/images", "images/val", "images/valid"])
    if not train_rel or not val_rel:
        print("ERREUR: dossiers d'images introuvables sous DATASET_DIR.")
        print("Contenu :", os.listdir(DATASET_DIR))
        return

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

    yaml_path = os.path.join(OUT_DIR, "coc_troop_bar_kaggle.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_DIR}\ntrain: {train_rel}\nval: {val_rel}\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    print(f"\nEntraînement barre : {MODEL} | {EPOCHS} ep | batch {BATCH} | imgsz {IMG_SIZE}\n")
    if BATCH == -1:
        print("ATTENTION: BATCH = -1 (auto) fait mourir le kernel Kaggle a "
              "cette resolution. Mets une valeur explicite (8, puis 4, puis 2).")
    yolo = YOLO(MODEL)
    yolo.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        project=OUT_DIR,
        name="yolo_troop_bar_train",
        exist_ok=True,
        patience=25,
        save=True, save_period=10, verbose=True, plots=True,
        workers=WORKERS,   # RAM Kaggle (cf. "Kernel died" en tete)
        cache=False,       # ne pas charger le dataset en RAM
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        box=7.5, cls=1.0, dfl=1.5,
        # ── AUGMENTATIONS (vignettes d'UI : régulières, jamais miroir) ───────
        fliplr=0.0,        # ⚠️ PAS de miroir : les icônes ont une orientation
        flipud=0.0,
        degrees=3.0,       # légère rotation seulement
        perspective=0.0, shear=0.0,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,   # ⚠️ garder hsv_s MODÉRÉ : la
        # saturation distingue actif/grisé à l'inférence ; la massacrer à
        # l'entraînement brouillerait ce signal.
        translate=0.1, scale=0.2,
        mosaic=1.0, mixup=0.05, copy_paste=0.0,
        close_mosaic=10,
    )

    best_src = os.path.join(OUT_DIR, "yolo_troop_bar_train", "weights", "best.pt")
    best_dst = os.path.join(OUT_DIR, "yolo_troop_bar_best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        # `model_artifacts.json` accompagne le modèle : c'est LUI que lit
        # troop_registry pour dériver SPELL_NAMES (cf. piège 2 en tête).
        art = os.path.join(OUT_DIR, "model_artifacts.json")
        with open(art, "w", encoding="utf-8") as f:
            json.dump({"names": list(names)}, f, ensure_ascii=False, indent=2)
        print(f"\nMeilleur modèle : {best_dst}")
        print(f"Artefacts       : {art}")
        print("-> place-les en weights/yolo_troop_bar/yolo_troop_bar.pt "
              "et weights/yolo_troop_bar/model_artifacts.json")
        print(f"-> et vérifie que troop_bar_detector.YOLO_IMGSZ == {IMG_SIZE}")
    else:
        print(f"\nATTENTION: best.pt introuvable ({best_src}) — regarde les logs.")


if __name__ == "__main__":
    main()
