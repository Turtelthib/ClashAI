# scripts/rl/hero_ability.py
# Gestion des capacités spéciales des héros pour ClashAI V3.
#
# Détection DYNAMIQUE des icônes d'ability pendant le combat via
# template matching (même approche que TroopFinder).
#
# Chaque héros a une capacité activable pendant le combat :
#   - Roi des Barbares : Rage Royale (boost dégâts + invocations)
#   - Reine des Archères : Cloak Royal (invisibilité + tir ciblé)
#   - Grand Gardien : Tome Éternel (invincibilité troupes proches)
#   - Championne Royale : Seeking Shield (bouclier + recherche cible)
#
# Note : Max 4 héros en combat dans CoC.
#        Le Prince Gargouille est un pet, pas un héros avec ability.
#
# Setup (une seule fois) :
#   1. python scripts/rl/hero_ability.py --extract
#      → Capture un screenshot mid-combat et sauvegarde la zone héros
#   2. Découpe chaque icône d'ability et sauvegarde :
#      ability_roi.png, ability_reine.png, etc.
#
# Usage dans le code :
#   manager = HeroAbilityManager()
#   manager.scan(screenshot_pil)         # Détecte les icônes présentes
#   manager.activate('roi', adb_tap_fn)  # Tape sur l'icône du roi

import os
import subprocess
import io
import time

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

from clashai.paths import HERO_TEMPLATES_DIR

TEMPLATES_DIR = HERO_TEMPLATES_DIR

# Zone de l'écran où les icônes d'ability apparaissent pendant le combat.
# En combat, APRÈS avoir déployé les héros, leurs icônes d'ability
# apparaissent dans la barre de troupes en bas de l'écran.
# On scanne toute la zone UI du bas pour être sûr de les trouver.
# La barre de troupes est à environ y=930-1080, mais les ability icons
# peuvent apparaître un peu au-dessus → on prend large.
ABILITY_ZONE_TOP = 850
ABILITY_ZONE_BOTTOM = 1080
ABILITY_ZONE_LEFT = 0
ABILITY_ZONE_RIGHT = 1920

# Template matching
MATCH_THRESHOLD = 0.50
MATCH_SCALES = [1.0, 0.9, 1.1, 0.85, 1.15]

# Les 5 héros qui peuvent avoir une ability en combat.
# Au début de chaque attaque, le TroopFinder détecte quels héros sont
# dans la barre → seuls les héros déployés auront leur ability activable.
# Si la championne est en amélioration, elle ne sera pas dans la barre,
# pas déployée, et son ability sera masquée automatiquement.
HERO_NAMES = ['roi', 'reine', 'grand_gardien', 'championne', 'prince_gargouille']
NUM_HEROES = len(HERO_NAMES)  # 5

HERO_ABILITY_NAMES = {
    'roi': 'Rage Royale',
    'reine': 'Cloak Royal',
    'grand_gardien': 'Tome Éternel',
    'championne': 'Seeking Shield',
    'prince_gargouille': 'Visage Noir',
}

# Délai minimum après le déploiement avant de scanner les abilities
# (les icônes n'apparaissent pas instantanément)
DEPLOY_TO_SCAN_DELAY = 5.0

# Cooldown entre deux scans (éviter de spammer les screenshots)
SCAN_COOLDOWN = 2.0


# =============================================================================
#                    FONCTIONS ADB
# =============================================================================

def _adb_screenshot():
    """Capture l'écran et retourne une image PIL."""
    try:
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0 or len(result.stdout) < 100:
            return None
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception:
        return None


def _adb_tap(x, y, delay=0.1):
    """Tap ADB."""
    subprocess.run(["adb", "shell", f"input tap {x} {y}"],
                   capture_output=True, timeout=5)
    time.sleep(delay)


# =============================================================================
#                    TEMPLATE MATCHING
# =============================================================================

def _match_template_multiscale(region, template):
    """
    Template matching multi-échelle.
    
    Returns:
        (best_val, best_loc, best_tw, best_th) ou (0, None, 0, 0)
    """
    best_val = 0
    best_loc = None
    best_tw = 0
    best_th = 0

    for scale in MATCH_SCALES:
        th, tw = template.shape[:2]
        new_h = int(th * scale)
        new_w = int(tw * scale)

        if new_h > region.shape[0] or new_w > region.shape[1]:
            continue
        if new_h < 10 or new_w < 10:
            continue

        resized = cv2.resize(template, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_tw = new_w
            best_th = new_h

    return best_val, best_loc, best_tw, best_th


# =============================================================================
#                    HERO ABILITY MANAGER
# =============================================================================

class HeroAbilityManager:
    """
    Gère la détection et l'activation des capacités héros pendant le combat.
    
    Utilise le template matching pour trouver dynamiquement les icônes
    d'ability sur l'écran — pas de positions hardcodées.
    
    Workflow :
        1. Pendant le deploy, appeler mark_deployed() pour chaque héros posé.
        2. Pendant le combat, appeler scan(screenshot) pour détecter les icônes.
        3. Appeler activate(hero_name) pour taper sur l'icône détectée.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

        # Templates d'icônes d'ability
        self._templates = {}       # nom → image template (numpy BGR)
        self._load_templates()

        # État par épisode
        self._deployed = {}        # nom → bool
        self._deploy_time = {}     # nom → timestamp
        self._activated = {}       # nom → bool
        self._icon_positions = {}  # nom → (x_adb, y_adb, confidence)
        self._last_scan_time = 0

    def _load_templates(self):
        """Charge les templates d'icônes depuis hero_ability_templates/."""
        if not os.path.exists(TEMPLATES_DIR):
            if self.verbose:
                print("⚠️  Dossier hero_ability_templates/ introuvable")
                print("   Lancez : python scripts/rl/hero_ability.py --extract")
            return

        count = 0
        for hero in HERO_NAMES:
            filename = f"ability_{hero}.png"
            path = os.path.join(TEMPLATES_DIR, filename)
            if os.path.exists(path):
                tmpl = cv2.imread(path)
                if tmpl is not None:
                    self._templates[hero] = tmpl
                    count += 1

        if self.verbose:
            if count > 0:
                print(f"📦 {count} templates d'ability chargés : "
                      f"{sorted(self._templates.keys())}")
            else:
                print(f"⚠️  Aucun template d'ability dans {TEMPLATES_DIR}")

    def reset(self):
        """Reset au début d'un nouvel épisode."""
        self._deployed = {name: False for name in HERO_NAMES}
        self._deploy_time = {}
        self._activated = {name: False for name in HERO_NAMES}
        self._icon_positions = {}
        self._last_scan_time = 0

    def mark_deployed(self, hero_name):
        """
        Marque un héros comme déployé sur le terrain.
        Note : le prince_gargouille est ignoré (pas d'ability).
        """
        if hero_name not in HERO_NAMES:
            return

        if not self._deployed.get(hero_name, False):
            self._deployed[hero_name] = True
            self._deploy_time[hero_name] = time.time()

            if self.verbose:
                ability = HERO_ABILITY_NAMES.get(hero_name, '?')
                print(f"      🦸 {hero_name} déployé (ability: {ability})")

    def scan(self, screenshot_pil):
        """
        Scanne un screenshot mid-combat pour détecter les icônes d'ability.
        Met à jour les positions internes.
        
        Args:
            screenshot_pil: PIL Image du combat en cours
            
        Returns:
            found: list de noms des héros dont l'icône est visible
        """
        now = time.time()

        # Cooldown entre les scans
        if now - self._last_scan_time < SCAN_COOLDOWN:
            return list(self._icon_positions.keys())

        self._last_scan_time = now

        if not self._templates:
            return []

        # Convertir et cropper la zone d'ability
        screen = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                       ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]

        found = []

        for hero_name, template in self._templates.items():
            # Ne chercher que les héros déployés et pas encore activés
            if not self._deployed.get(hero_name, False):
                continue
            if self._activated.get(hero_name, False):
                continue

            # Vérifier le délai post-déploiement
            deploy_t = self._deploy_time.get(hero_name, now)
            if now - deploy_t < DEPLOY_TO_SCAN_DELAY:
                continue

            best_val, best_loc, best_tw, best_th = _match_template_multiscale(
                zone, template
            )

            if best_val >= MATCH_THRESHOLD and best_loc is not None:
                # Convertir position zone → coordonnées ADB
                x_adb = ABILITY_ZONE_LEFT + best_loc[0] + best_tw // 2
                y_adb = ABILITY_ZONE_TOP + best_loc[1] + best_th // 2
                self._icon_positions[hero_name] = (x_adb, y_adb, best_val)
                found.append(hero_name)

        if self.verbose and found:
            for name in found:
                x, y, conf = self._icon_positions[name]
                print(f"      🎯 {name} ability détectée "
                      f"à ({x}, {y}) conf={conf:.2f}")

        return found

    def get_available_abilities(self):
        """
        Retourne les héros dont l'ability est disponible.
        
        Conditions :
            - Héros déployé
            - Ability pas encore utilisée
            - Icône détectée par le dernier scan
        """
        available = []
        for name in HERO_NAMES:
            if not self._deployed.get(name, False):
                continue
            if self._activated.get(name, False):
                continue
            if name in self._icon_positions:
                available.append(name)
        return available

    def get_ability_mask(self):
        """
        Masque binaire (4,) pour le masking PPO.
        1.0 = ability disponible et détectée, 0.0 sinon.
        """
        available = set(self.get_available_abilities())
        mask = np.zeros(NUM_HEROES, dtype=np.float32)
        for i, name in enumerate(HERO_NAMES):
            if name in available:
                mask[i] = 1.0
        return mask

    def activate(self, hero_name, adb_tap_fn=None):
        """
        Active la capacité d'un héros en tapant sur son icône détectée.
        
        Args:
            hero_name: str ('roi', 'reine', etc.)
            adb_tap_fn: callable(x, y) — si None, utilise _adb_tap interne
            
        Returns:
            success: bool
        """
        tap_fn = adb_tap_fn or _adb_tap

        if hero_name not in HERO_NAMES:
            if self.verbose:
                print(f"      ⚠️  {hero_name} n'est pas un héros avec ability")
            return False

        if not self._deployed.get(hero_name, False):
            if self.verbose:
                print(f"      ⚠️  {hero_name} pas déployé")
            return False

        if self._activated.get(hero_name, False):
            if self.verbose:
                print(f"      ⚠️  Ability de {hero_name} déjà utilisée")
            return False

        if hero_name not in self._icon_positions:
            if self.verbose:
                print(f"      ⚠️  Icône de {hero_name} pas détectée — "
                      f"relancez scan()")
            return False

        x, y, conf = self._icon_positions[hero_name]

        if self.verbose:
            ability = HERO_ABILITY_NAMES.get(hero_name, '?')
            print(f"      ⚡ {ability} ({hero_name}) "
                  f"→ tap ({x}, {y}) conf={conf:.2f}")

        tap_fn(x, y)
        time.sleep(0.3)

        self._activated[hero_name] = True
        # Retirer de icon_positions pour ne pas re-taper
        del self._icon_positions[hero_name]

        return True

    def get_status_vector(self):
        """
        Vecteur de status (4,) pour l'observation PPO.
        
        0.0  = pas déployé
        0.25 = déployé, icône pas encore cherchée
        0.5  = déployé, icône non trouvée par scan
        0.75 = déployé, ability détectée et disponible
        1.0  = ability activée
        """
        status = np.zeros(NUM_HEROES, dtype=np.float32)
        now = time.time()

        for i, name in enumerate(HERO_NAMES):
            if not self._deployed.get(name, False):
                status[i] = 0.0
            elif self._activated.get(name, False):
                status[i] = 1.0
            elif name in self._icon_positions:
                status[i] = 0.75
            else:
                deploy_t = self._deploy_time.get(name, now)
                if now - deploy_t < DEPLOY_TO_SCAN_DELAY:
                    status[i] = 0.25
                else:
                    status[i] = 0.5

        return status

    def num_deployed(self):
        return sum(1 for v in self._deployed.values() if v)

    def num_activated(self):
        return sum(1 for v in self._activated.values() if v)

    def has_templates(self):
        return len(self._templates) > 0

    # -----------------------------------------------------------------
    #  V4 : positions YOLO des héros sur le champ de bataille
    # -----------------------------------------------------------------
    def update_battlefield_positions(self, hero_positions_named: dict):
        """
        Met à jour les positions des héros sur le terrain via YOLO.
        
        Args:
            hero_positions_named: dict {hero_name: (x, y)} du CombatObserver
        """
        self._battlefield_positions = hero_positions_named
        # Marquer automatiquement comme déployé si YOLO le détecte
        for name in hero_positions_named:
            if name in HERO_NAMES and not self._deployed.get(name, False):
                self.mark_deployed(name)
                if self.verbose:
                    print(f"      🦸 {name} détecté par YOLO → marqué déployé")

    def get_hero_position(self, hero_name):
        """Retourne la position YOLO (x, y) d'un héros, ou None."""
        return getattr(self, '_battlefield_positions', {}).get(hero_name)

    def heroes_near_center(self, village_center, radius=250):
        """Retourne les héros proches du centre village (bon moment pour ability)."""
        import math
        result = []
        positions = getattr(self, '_battlefield_positions', {})
        for name, (hx, hy) in positions.items():
            if name not in HERO_NAMES:
                continue
            dist = math.sqrt((hx - village_center[0])**2 + (hy - village_center[1])**2)
            if dist < radius:
                result.append((name, dist))
        return result


# =============================================================================
#                    OUTILS : EXTRACTION DE TEMPLATES
# =============================================================================

def extract_ability_zone():
    """
    Capture un screenshot mid-combat et sauvegarde la zone
    des icônes d'ability pour le découpage manuel.
    
    ⚠️ IMPORTANT : les héros doivent être DÉPLOYÉS sur le terrain !
    Les icônes d'ability n'apparaissent que quand le héros est en combat,
    pas quand il est encore dans la barre de troupes.
    """
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    print("📸 Capture de la zone d'ability héros...")
    print()
    print("   ⚠️  IMPORTANT : les héros doivent être DÉPLOYÉS !")
    print("   ⚠️  Pas dans la barre de troupes, mais SUR le terrain.")
    print("   ⚠️  Les icônes d'ability n'apparaissent qu'après le déploiement.")
    print()

    img = _adb_screenshot()
    if img is None:
        print("❌ Impossible de capturer l'écran")
        return

    screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = screen.shape[:2]

    full_path = os.path.join(TEMPLATES_DIR, '_combat_full.png')
    cv2.imwrite(full_path, screen)
    print(f"   ✅ Écran complet → {full_path}")

    # Zone de la barre de troupes (bas de l'écran)
    zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                   ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]
    zone_path = os.path.join(TEMPLATES_DIR, '_ability_zone.png')
    cv2.imwrite(zone_path, zone)
    print(f"   ✅ Zone barre (y={ABILITY_ZONE_TOP}-{ABILITY_ZONE_BOTTOM}) → {zone_path}")

    # Zone élargie : toute la moitié basse de l'écran
    bottom_half = screen[h // 2:, :]
    bottom_path = os.path.join(TEMPLATES_DIR, '_bottom_half.png')
    cv2.imwrite(bottom_path, bottom_half)
    print(f"   ✅ Moitié basse (y={h//2}-{h}) → {bottom_path}")

    print("\n📝 Étapes suivantes :")
    print(f"   1. Ouvre {full_path} ou {zone_path}")
    print("   2. Repère les icônes d'ABILITY (portraits des héros déployés)")
    print("      ⚠️ Ce ne sont PAS les cartes de la barre de troupes !")
    print("      ⚠️ Les abilities sont des petits portraits qui apparaissent")
    print("         APRÈS avoir déployé le héros sur le terrain.")
    print(f"   3. Découpe chaque icône et sauvegarde dans {TEMPLATES_DIR}/ :")
    for hero in HERO_NAMES:
        print(f"      ability_{hero}.png")
    print("\n   Si tu ne vois pas d'icônes d'ability, c'est que les héros")
    print("   n'étaient pas encore déployés au moment de la capture !")


def test_scan():
    """Test le scan des abilities sur un screenshot en cours."""
    print("🧪 Test du scan d'ability...\n")

    manager = HeroAbilityManager()
    if not manager.has_templates():
        print("❌ Pas de templates. Lancez --extract et découpez les icônes.")
        return

    img = _adb_screenshot()
    if img is None:
        print("❌ Impossible de capturer l'écran")
        return

    # Sauvegarder le screenshot pour debug
    debug_path = os.path.join(TEMPLATES_DIR, '_test_screenshot.png')
    screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(debug_path, screen)

    # Sauvegarder la zone scannée pour vérifier
    zone = screen[ABILITY_ZONE_TOP:ABILITY_ZONE_BOTTOM,
                   ABILITY_ZONE_LEFT:ABILITY_ZONE_RIGHT]
    zone_path = os.path.join(TEMPLATES_DIR, '_test_zone.png')
    cv2.imwrite(zone_path, zone)

    print(f"   Zone scannée: y={ABILITY_ZONE_TOP}-{ABILITY_ZONE_BOTTOM}, "
          f"x={ABILITY_ZONE_LEFT}-{ABILITY_ZONE_RIGHT}")
    print(f"   Screenshot → {debug_path}")
    print(f"   Zone → {zone_path}")

    # Simuler que tous les héros sont déployés (pour le test)
    for hero in HERO_NAMES:
        manager._deployed[hero] = True
        manager._deploy_time[hero] = time.time() - 30

    found = manager.scan(img)

    if found:
        print(f"\n🎯 Abilities détectées : {found}")
        print(f"   Status : {manager.get_status_vector()}")
        print(f"   Mask   : {manager.get_ability_mask()}")
    else:
        print("\n⚠️  Aucune ability détectée.")
        print("   Causes possibles :")
        print("   1. Les héros ne sont pas déployés (encore dans la barre)")
        print("   2. Les templates ne correspondent pas aux icônes d'ability")
        print("      → les templates doivent être des icônes APRÈS déploiement")
        print("      → pas les cartes de la barre de troupes")
        print(f"   3. Vérifie {zone_path} pour voir ce que le scan voit")


# =============================================================================
#                            MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClashAI Hero Ability Manager")
    parser.add_argument('--extract', action='store_true',
                        help="Capture la zone d'ability pour les templates")
    parser.add_argument('--test', action='store_true',
                        help="Test le scan sur l'écran actuel")
    args = parser.parse_args()

    if args.extract:
        extract_ability_zone()
    elif args.test:
        test_scan()
    else:
        print("🧪 Test HeroAbilityManager (sans ADB)\n")
        manager = HeroAbilityManager(verbose=True)
        manager.reset()

        manager.mark_deployed('roi')
        manager.mark_deployed('reine')
        manager.mark_deployed('grand_gardien')
        manager.mark_deployed('prince_gargouille')  # Ignoré

        print(f"\n   Déployés : {manager.num_deployed()}/4")
        print(f"   Status : {manager.get_status_vector()}")

        for name in ['roi', 'reine']:
            manager._deploy_time[name] = time.time() - 20
            manager._icon_positions[name] = (100, 600, 0.85)

        print("\n   Après scan simulé :")
        print(f"   Disponibles : {manager.get_available_abilities()}")
        print(f"   Mask : {manager.get_ability_mask()}")

        manager.activate('roi', lambda x, y: print(f"   TAP ({x}, {y})"))
        print(f"   Status : {manager.get_status_vector()}")
        print("\n✅ Test terminé !")