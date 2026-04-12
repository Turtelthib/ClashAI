# clashai/combat/troop_manager.py
# Gestion de la barre de troupes pour ClashAI.
#
# Responsabilités :
#   - Scanner la barre de troupes (TroopFinder + saturation)
#   - Sélectionner une troupe par nom ou par rôle
#   - Traduire rôle V4 → troupe concrète (round-robin)
#   - Cleanup : déployer toutes les troupes restantes
#
# Séparé de l'environnement pour garder des fichiers < 500 lignes.

import time
import cv2
import numpy as np

from clashai.combat.action_space import (
    DEPLOY_ROLES, ROLE_TO_TROOPS, DEPLOY_SECTORS,
    SECTOR_OFFSETS, NUM_POSITIONS,
)


# =============================================================================
#                         CONFIGURATION
# =============================================================================

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SLOT_SATURATION_THRESHOLD = 40   # sat > 40 = troupe colorée (dispo)
DELAY_SWITCH_TROOP = 0.15
DELAY_DEPLOY = 0.08
MAX_CLEANUP_ROUNDS = 5

# Alias pour les templates
TROOP_ALIASES = {
    'lance_buche': ['lance_buche_vide'],
}
ALIAS_MAP = {'lance_buche_vide': 'lance_buche'}


# =============================================================================
#                         TROOP MANAGER
# =============================================================================

class TroopManager:
    """
    Gère la barre de troupes : scan, sélection, deploy par rôle.
    """

    def __init__(self, troop_finder, troop_types, troop_name_to_idx,
                 adb_screenshot_fn, adb_tap_fn, verbose=True):
        """
        Args:
            troop_finder: TroopFinder instance
            troop_types: list[dict] — TROOP_TYPES du V3
            troop_name_to_idx: dict {name: idx}
            adb_screenshot_fn: callable → PIL Image
            adb_tap_fn: callable (x, y) → None
            verbose: bool
        """
        self._finder = troop_finder
        self._troop_types = troop_types
        self._name_to_idx = troop_name_to_idx
        self._screenshot = adb_screenshot_fn
        self._tap = adb_tap_fn
        self.verbose = verbose

        # État
        self._last_troop_name = None
        self._deploy_failed_count = 0

        # Round-robin par rôle (pour V4)
        self._role_cursors = {role: 0 for role in DEPLOY_ROLES}

    def reset(self):
        """Reset pour un nouvel épisode."""
        self._last_troop_name = None
        self._deploy_failed_count = 0
        self._role_cursors = {role: 0 for role in DEPLOY_ROLES}

    # -----------------------------------------------------------------
    #  Sélection par nom
    # -----------------------------------------------------------------

    def select_troop(self, troop_name):
        """Sélectionne une troupe par nom dans la barre."""
        if troop_name == self._last_troop_name:
            return True
        if self._finder.select(troop_name):
            self._last_troop_name = troop_name
            return True
        for alias in TROOP_ALIASES.get(troop_name, []):
            if self._finder.select(alias):
                self._last_troop_name = troop_name
                return True
        if self.verbose:
            print(f"      ⚠️  {troop_name} non trouvé dans la barre")
        self._last_troop_name = None
        return False

    # -----------------------------------------------------------------
    #  Sélection par rôle (V4)
    # -----------------------------------------------------------------

    def select_next_for_role(self, role_name, remaining_troops):
        """
        Sélectionne la prochaine troupe disponible pour un rôle donné.
        Round-robin entre les troupes du même rôle.

        Args:
            role_name: 'tank', 'ranged', 'melee', 'hero', 'siege'
            remaining_troops: array (N,) — compteurs

        Returns:
            (troop_idx, troop_name) ou (None, None)
        """
        candidates = ROLE_TO_TROOPS.get(role_name, [])
        if not candidates:
            return None, None

        cursor = self._role_cursors.get(role_name, 0)
        n = len(candidates)

        # Chercher à partir du cursor (round-robin)
        for offset in range(n):
            i = (cursor + offset) % n
            name = candidates[i]
            if name not in self._name_to_idx:
                continue
            idx = self._name_to_idx[name]
            if remaining_troops[idx] > 0:
                if self.select_troop(name):
                    self._role_cursors[role_name] = (i + 1) % n
                    return idx, name

        return None, None

    # -----------------------------------------------------------------
    #  Saturation check
    # -----------------------------------------------------------------

    def is_slot_active(self, img_cv, x, y):
        """Vérifie si un slot est coloré (dispo) ou grisé (épuisé)."""
        h, w = img_cv.shape[:2]
        ix = int(x * w / SCREEN_WIDTH)
        iy = int(y * h / SCREEN_HEIGHT)

        sample_half_w = int(15 * w / SCREEN_WIDTH)
        sample_top = int(20 * h / SCREEN_HEIGHT)
        sample_bot = int(5 * h / SCREEN_HEIGHT)

        y1 = max(0, iy - sample_top)
        y2 = min(h, iy + sample_bot)
        x1 = max(0, ix - sample_half_w)
        x2 = min(w, ix + sample_half_w)

        region = img_cv[y1:y2, x1:x2]
        if region.size == 0:
            return False

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        avg_sat = float(np.mean(hsv[:, :, 1]))
        return avg_sat > SLOT_SATURATION_THRESHOLD

    # -----------------------------------------------------------------
    #  Rescan barre
    # -----------------------------------------------------------------

    def rescan(self, remaining_troops, read_counts_fn=None):
        """
        Rescanne la barre de troupes avec un screenshot frais.

        Args:
            remaining_troops: array (N,) — sera modifié in-place
            read_counts_fn: optional callable(img, finder) → dict
        """
        img = self._screenshot()
        if img is None:
            return

        self._finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for name_raw, (tx, ty, conf) in self._finder.positions.items():
            name = ALIAS_MAP.get(name_raw, name_raw)
            if name not in self._name_to_idx:
                continue
            idx = self._name_to_idx[name]
            active = self.is_slot_active(img_cv, tx, ty)
            if not active:
                remaining_troops[idx] = 0
            elif remaining_troops[idx] <= 0:
                remaining_troops[idx] = 1.0

        # Troupes disparues du finder
        available = {ALIAS_MAP.get(n, n) for n in self._finder.positions}
        for i, t in enumerate(self._troop_types):
            if t['name'] not in available and remaining_troops[i] > 0:
                if t['role'] != 'spell':
                    remaining_troops[i] = 0

        # OCR compteurs
        if read_counts_fn is not None:
            try:
                real = read_counts_fn(img, self._finder)
                for name, count in real.items():
                    real_name = ALIAS_MAP.get(name, name)
                    if real_name in self._name_to_idx:
                        remaining_troops[self._name_to_idx[real_name]] = float(count)
            except Exception:
                pass

        if self.verbose:
            total = int(np.sum(remaining_troops))
            print(f"      🔄 Rescan barre : {total} troupes restantes")

    # -----------------------------------------------------------------
    #  Cleanup : deploy tout ce qui reste
    # -----------------------------------------------------------------

    def cleanup(self, remaining_troops, deploy_positions, village_center):
        """
        Déploie toutes les troupes encore dans la barre (tap-until-gray).

        Args:
            remaining_troops: array — sera modifié
            deploy_positions: list[(x,y)] — positions de deploy
            village_center: (x,y) — fallback
        """
        if self.verbose:
            print("\n   🧹 Cleanup : rescan de la barre...")

        img = self._screenshot()
        if img is None:
            return

        self._finder.update(img)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        spell_names = {t['name'] for t in self._troop_types if t['role'] == 'spell'}

        # Spread positions
        center_idx = NUM_POSITIONS // 2
        spread = []
        for off in [-2, -1, 0, 1, 2]:
            p = (center_idx + off) % NUM_POSITIONS
            if deploy_positions and p < len(deploy_positions):
                spread.append(deploy_positions[p])
        if not spread:
            spread = [village_center or (960, 500)]

        # Trouver les troupes colorées
        to_deploy = []
        for name_raw, (tx, ty, conf) in self._finder.positions.items():
            name = ALIAS_MAP.get(name_raw, name_raw)
            if name in spell_names or name not in self._name_to_idx:
                continue
            if self.is_slot_active(img_cv, tx, ty):
                idx = self._name_to_idx[name]
                role = self._troop_types[idx]['role']
                count = max(int(remaining_troops[idx]), 1)
                to_deploy.append((name_raw, name, role, count))
                if self.verbose:
                    print(f"      📦 {name} ({role}) x{count}")

        if not to_deploy:
            if self.verbose:
                print("   🧹 Cleanup : rien à déployer")
            return

        total = sum(c for _, _, _, c in to_deploy)
        if self.verbose:
            print(f"      → {total} troupes à déployer")

        # Ordre tactique
        role_order = {'tank': 0, 'ranged': 1, 'melee': 2, 'siege': 3, 'hero': 4}
        to_deploy.sort(key=lambda t: role_order.get(t[2], 99))

        deployed = 0
        for name_raw, name, role, count in to_deploy:
            if not self._finder.select(name_raw):
                continue
            time.sleep(DELAY_SWITCH_TROOP)
            idx = self._name_to_idx[name]
            tx, ty, _ = self._finder.positions[name_raw]
            taps = 0

            for _ in range(MAX_CLEANUP_ROUNDS):
                for tap_i in range(3):
                    pos = spread[(taps + tap_i) % len(spread)]
                    self._tap(pos[0], pos[1])
                    time.sleep(DELAY_DEPLOY)
                taps += 3

                check = self._screenshot()
                if check is None:
                    break
                check_cv = cv2.cvtColor(np.array(check), cv2.COLOR_RGB2BGR)
                if not self.is_slot_active(check_cv, tx, ty):
                    break

            deployed += taps
            remaining_troops[idx] = 0
            self._last_troop_name = None

            if self.verbose:
                status = 'grisé ✅' if taps < MAX_CLEANUP_ROUNDS * 3 else 'max atteint'
                print(f"      🧹 {name} → {taps} taps ({status})")

        if self.verbose:
            remaining = int(np.sum([
                remaining_troops[i] for i, t in enumerate(self._troop_types)
                if t['role'] != 'spell'
            ]))
            print(f"   🧹 Cleanup terminé : {deployed} actions"
                  f" ({remaining} encore en compteur)")

    # -----------------------------------------------------------------
    #  Utilitaire : position depuis secteur V4
    # -----------------------------------------------------------------

    @staticmethod
    def sector_to_position(sector_idx, center_pos):
        """Convertit un secteur V4 en position absolue sur le périmètre."""
        sector_name = DEPLOY_SECTORS[sector_idx]
        offset = SECTOR_OFFSETS[sector_name]
        return (center_pos + offset) % NUM_POSITIONS
