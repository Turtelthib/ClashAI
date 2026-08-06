# test_deploy.py
from clashai.navigation import game_loop
from clashai.perception.deploy_zone import (
    get_perimeter_from_buildings,
    save_deploy_debug_image,
)

# 1. Charger + screenshot + YOLO
models = game_loop.load_models()
img = game_loop.adb_screenshot()
buildings = game_loop.analyze_village(img, models)

# 2. Calculer les positions AVEC debug (unpack 4 valeurs au lieu de 3)
positions, center, ok, debug = get_perimeter_from_buildings(
    buildings, num_points=20, return_debug=True,
    screenshot_pil=img,
)

# 3. Générer l'image annotée via la fonction du projet
path = save_deploy_debug_image(
    img, # screenshot PIL
    buildings,
    positions or [],
    center,
    output_dir='.', # enregistre à la racine (au lieu de logs/deploy_zone/)
    episode=None, # pas d'épisode pour un test manuel
    extra_info=f'Test manuel {len(buildings)} batiments',
    rejected_rays=debug.get('rejected_rays') if debug else None,
)

# 4. Résumé console
print(f"\n{'='*60}")
print(f" Centre village : {center}")
print(f" Positions trouvées : {len(positions) if positions else 0}/20")
print(f" Rayons rejetés : {len(debug.get('rejected_rays', [])) if debug else 0}")
print(f" Rayon moyen du hull : {debug.get('mean_radius', 0):.0f}px" if debug else "")
print(f" Image sauvegardée : {path}")
print(f"{'='*60}")
