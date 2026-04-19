import os
import json
import glob
import random
import shutil

from clashai.paths import PROJECT_ROOT as project_root

CLASSES = [
    'hdv',
    'chateau_clan',
    'tour_enfer_mono',
    'tour_enfer_multiple',
    'aigle_artilleur',
    'catapulte_erratique',
    'arcX_sol',
    'arcX_sol_air',
    'monolithe',
    'tour_archere',
    'canon',
    'mortier',
    'multi_mortier',
    'tour_sorcier',
    'defense_antiaerienne',
    'prop_air',
    'tesla',
    'cabane_ouvrier_arme',
    'canon_ricochet',
    'cracheur_feu',
    'tour_runique_rage',
    'tour_runique_poison',
    'tour_runique_invisible',
    'tour_multi_equipe_rapide',
    'tour_bombe',
    'tour_archere_multiple',
    'tour_multi_equipe_lente',
    'tour_archere_rapide',
    'double_canon',
    'tour_vengeuse',
    'super_tour_sorcier',
    'reserve_or',
    'reserve_elixir',
    'reserve_noire',
    'hall_heros',
    'laboratoire',
    'sort',
    'ressources',
    'gigabombe',
    'tour_runique_seisme',
    'atelier',
    'animalerie',
    'caserne',
    'camps_militaires',
    'forgeron',
    'cabane_assistants',
    'cabane_bob',
]

def convert_labelme_to_yolo(json_path, output_dir, img_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        yolo_txt_path = os.path.join(output_dir, f"{base_filename}.txt")

        with open(yolo_txt_path, 'w') as f_out:
            for shape in data['shapes']:
                label = shape['label']
                if label not in CLASSES:
                    continue
                
                class_id = CLASSES.index(label)
                points = shape['points']
                if len(points) < 2: continue

                xmin = min(points[0][0], points[1][0])
                ymin = min(points[0][1], points[1][1])
                xmax = max(points[0][0], points[1][0])
                ymax = max(points[0][1], points[1][1])
                
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        print(f"Erreur sur {json_path}: {e}")

print("🔨 Début de la préparation du dataset...")

# Paths based on the project root
source_dir = os.path.join(project_root, 'data_source', 'data')
dataset_dir = os.path.join(project_root, 'dataset')

train_img_dir = os.path.join(dataset_dir, 'images', 'train')
train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
val_img_dir = os.path.join(dataset_dir, 'images', 'val')
val_label_dir = os.path.join(dataset_dir, 'labels', 'val')

# Cleanup and Creation
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Retrieve images
all_images = glob.glob(os.path.join(source_dir, '*.jpg'))
all_images.extend(glob.glob(os.path.join(source_dir, '*.png')))

if not all_images:
    print(f"ERROR: ERREUR CRITIQUE : Aucune image trouvée dans {source_dir}")
    print("Vérifiez que vos images sont bien dans 'COCProj/data_source/data'")
    exit(1)

random.shuffle(all_images)

split_index = int(len(all_images) * 0.85)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

print(f"Total: {len(all_images)} | Train: {len(train_images)} | Val: {len(val_images)}")

def process_files(image_list, img_dest, label_dest):
    for img_path in image_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(os.path.dirname(img_path), f"{base_name}.json")
        
        if os.path.exists(json_path):
            shutil.copy(img_path, img_dest)
            convert_labelme_to_yolo(json_path, label_dest, img_path)

process_files(train_images, train_img_dir, train_label_dir)
process_files(val_images, val_img_dir, val_label_dir)

print("Dataset prêt et généré dans 'COCProj/dataset' !")
