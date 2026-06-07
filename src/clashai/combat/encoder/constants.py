# clashai/combat/encoder/constants.py
# Building categories, grid channels, defense stats, special class lists.

from clashai.config import GRID_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT


CATEGORIES = {
    'defenses_dangereuses': [
        'tour_enfer_mono', 'tour_enfer_multiple', 'aigle_artilleur',
        'arcX_sol', 'arcX_sol_air', 'catapulte_erratique', 'monolithe',
        'tour_vengeuse', 'gigabombe', 'tour_runique_seisme'
    ],
    'defenses_moyennes': [
        'canon', 'tour_archere', 'mortier', 'multi_mortier', 'tour_sorcier',
        'tour_bombe', 'canon_double', 'canon_ricochet', 'cracheur_feu',
        'super_tour_sorcier', 'tour_archere_multiple', 'tour_archere_rapide',
        'tour_multi_equipe_rapide', 'tour_multi_equipe_lente',
        'cabane_ouvrier_arme'
    ],
    'defenses_anti_aeriennes': [
        'defense_antiaerienne', 'prop_air'
    ],
    'defenses_speciales': [
        'tesla', 'tour_runique_rage', 'tour_runique_poison',
        'tour_runique_invisible'
    ],
    'ressources': [
        'reserve_or', 'reserve_elixir', 'reserve_noire', 'ressources'
    ],
    'hdv': [
        'hdv'
    ],
    'chateau_clan': [
        'chateau_clan'
    ],
    'hall_heros': [
        'hall_heros'
    ],
    'batiments_non_defensifs': [
        'caserne', 'camps_militaires', 'laboratoire', 'atelier',
        'animalerie', 'forgeron', 'cabane_assistants', 'cabane_bob',
        'sort'
    ],
    # Channels 10-11 (danger) computed separately
}

CHANNEL_NAMES = list(CATEGORIES.keys()) + [
    'densite_totale',
    'danger_sol',
    'danger_air',
]

# Reverse mapping: class name → channel index
CLASS_TO_CHANNEL = {}
for channel_idx, (category, classes) in enumerate(CATEGORIES.items()):
    for cls_name in classes:
        CLASS_TO_CHANNEL[cls_name] = channel_idx


NUM_CHANNELS = len(CHANNEL_NAMES)
CELL_WIDTH = SCREEN_WIDTH / GRID_SIZE
CELL_HEIGHT = SCREEN_HEIGHT / GRID_SIZE
NUM_VILLAGE_FEATURES = 20


DEFENSE_STATS = {
    # --- Dangerous defenses ---
    'tour_enfer_mono': {'range': 400, 'dps': 1.0, 'targets': 'both'},
    'tour_enfer_multiple': {'range': 350, 'dps': 0.9, 'targets': 'both'},
    'aigle_artilleur': {'range': 600, 'dps': 0.8, 'targets': 'both'},
    'arcX_sol': {'range': 450, 'dps': 0.85, 'targets': 'ground'},
    'arcX_sol_air': {'range': 450, 'dps': 0.85, 'targets': 'both'},
    'catapulte_erratique': {'range': 500, 'dps': 0.7, 'targets': 'ground'},
    'monolithe': {'range': 400, 'dps': 0.9, 'targets': 'ground'},
    'tour_vengeuse': {'range': 350, 'dps': 0.6, 'targets': 'both'},
    'gigabombe': {'range': 300, 'dps': 0.5, 'targets': 'ground'},
    'tour_runique_seisme': {'range': 350, 'dps': 0.5, 'targets': 'ground'},
    # --- Medium defenses ---
    'canon': {'range': 300, 'dps': 0.3, 'targets': 'ground'},
    'tour_archere': {'range': 350, 'dps': 0.35, 'targets': 'both'},
    'mortier': {'range': 400, 'dps': 0.2, 'targets': 'ground'},
    'multi_mortier': {'range': 350, 'dps': 0.4, 'targets': 'ground'},
    'tour_sorcier': {'range': 350, 'dps': 0.4, 'targets': 'both'},
    'tour_bombe': {'range': 300, 'dps': 0.35, 'targets': 'ground'},
    'canon_double': {'range': 300, 'dps': 0.4, 'targets': 'ground'},
    'canon_ricochet': {'range': 350, 'dps': 0.35, 'targets': 'ground'},
    'cracheur_feu': {'range': 250, 'dps': 0.5, 'targets': 'ground'},
    'super_tour_sorcier': {'range': 350, 'dps': 0.5, 'targets': 'both'},
    'tour_archere_multiple': {'range': 350, 'dps': 0.4, 'targets': 'both'},
    'tour_archere_rapide': {'range': 350, 'dps': 0.45, 'targets': 'both'},
    'tour_multi_equipe_rapide': {'range': 350, 'dps': 0.4, 'targets': 'ground'},
    'tour_multi_equipe_lente': {'range': 350, 'dps': 0.35, 'targets': 'ground'},
    'cabane_ouvrier_arme': {'range': 250, 'dps': 0.2, 'targets': 'ground'},
    # --- Anti-air ---
    'defense_antiaerienne': {'range': 350, 'dps': 0.6, 'targets': 'air'},
    'prop_air': {'range': 300, 'dps': 0.3, 'targets': 'air'},
    # --- Special defenses ---
    'tesla': {'range': 250, 'dps': 0.5, 'targets': 'both'},
    'tour_runique_rage': {'range': 300, 'dps': 0.3, 'targets': 'both'},
    'tour_runique_poison': {'range': 300, 'dps': 0.3, 'targets': 'both'},
    'tour_runique_invisible': {'range': 300, 'dps': 0.2, 'targets': 'both'},
}

# Specific classes for enriched features
INFERNO_CLASSES = ['tour_enfer_mono', 'tour_enfer_multiple']
EAGLE_CLASSES = ['aigle_artilleur']
SCATTER_CLASSES = ['catapulte_erratique']
CC_CLASSES = ['chateau_clan']
