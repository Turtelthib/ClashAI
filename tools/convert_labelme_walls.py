# tools/convert_labelme_walls.py
# Wrapper de convert_labelme.py pour le dataset MURS.
#
# Le convert_labelme.py original est configuré pour les troupes (14 classes,
# dossier 'combat_captures'). Ce wrapper le réutilise avec les bons paramètres
# pour les murs (1 classe 'mur', dossier 'data_source/data_walls').
#
# Usage :
#   uv run python tools/convert_labelme_walls.py
#   uv run python tools/convert_labelme_walls.py --input custom/path --output custom/out

import os
import sys
import argparse
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# On réutilise la fonction principale, mais on override le mapping des classes
import tools.convert_labelme as convert_labelme


# Override du mapping de classes pour les murs uniquement
WALL_CLASS_MAP = {
    'mur': 0,
}

# Aliases pour tolérer les variations de nommage
WALL_ALIASES = {
    'Mur': 'mur',
    'MUR': 'mur',
    'rempart': 'mur',
    'Rempart': 'mur',
    'REMPART': 'mur',
    'wall': 'mur',
    'Wall': 'mur',
    'WALL': 'mur',
    'walls': 'mur',
}


def main():
    parser = argparse.ArgumentParser(
        description="Conversion LabelMe → YOLO pour les murs/remparts"
    )
    parser.add_argument(
        '--input', type=str,
        default=os.path.join(project_root, 'data_source', 'data_walls'),
        help="Dossier avec images + .json Labelme (défaut: data_source/data_walls)"
    )
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(project_root, 'dataset_walls'),
        help="Dossier de sortie YOLO (défaut: dataset_walls)"
    )
    parser.add_argument(
        '--split', type=float, default=0.8,
        help="Ratio train/total (défaut: 0.8)"
    )
    args = parser.parse_args()

    # Override temporaire du mapping et des aliases dans convert_labelme
    convert_labelme.CLASS_MAP = WALL_CLASS_MAP
    convert_labelme.ALIASES = WALL_ALIASES

    print(f"\n{'='*60}")
    print("  🧱 Conversion LabelMe → YOLO pour murs")
    print(f"{'='*60}")
    print(f"  Classes  : {list(WALL_CLASS_MAP.keys())}")
    print(f"  Aliases  : {list(WALL_ALIASES.keys())}")
    print(f"{'='*60}\n")

    convert_labelme.process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_split=args.split,
    )


if __name__ == "__main__":
    main()
