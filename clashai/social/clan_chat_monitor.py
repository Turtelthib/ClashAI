# scripts/rl/clan_chat_monitor.py
# Surveillance du chat de clan et parsing de commandes pour ClashAI.
#
# Le bot surveille le chat du clan à intervalles réguliers.
# Quand un message contenant une commande est détecté, il l'exécute.
#
# Commandes supportées :
#   @mini_pekka 3        → Attaquer la cible n°3 en GdC
#   @mini_pekka attack 5 → Attaquer la cible n°5 en GdC
#   @mini_pekka stop     → Arrêter le monitoring
#   @mini_pekka status   → L'IA répond son état (via un défi amical ou rien)
#
# Méthode OCR :
#   On utilise EasyOCR (meilleur que Tesseract sur les polices de jeu).
#   Fallback sur pytesseract si EasyOCR pas installé.
#
# Usage :
#   monitor = ClanChatMonitor(bot_name='mini_pekka')
#   monitor.start(models)  # Boucle infinie de surveillance
#
# Usage ponctuel :
#   monitor = ClanChatMonitor(bot_name='mini_pekka')
#   commands = monitor.check_once(screenshot_pil)

import sys
import re
import time
import subprocess
import io

import cv2
import numpy as np
from PIL import Image


# =============================================================================
#                         CONFIGURATION
# =============================================================================

ADB_WIDTH = 1920
ADB_HEIGHT = 1080

# Zone du chat dans l'écran (quand le chat est ouvert)
# Le chat occupe environ la moitié gauche de l'écran
# BOTTOM = 980 pour capturer les messages tout en bas (avant la barre de saisie)
CHAT_ZONE_LEFT = 0
CHAT_ZONE_RIGHT = 850
CHAT_ZONE_TOP = 60
CHAT_ZONE_BOTTOM = 980

# Bouton pour ouvrir le chat — chargé depuis ui_positions.json
# Calibré via : python scripts/rl/calibrate_ui.py
def _get_chat_button_pos():
    try:
        from clashai.navigation.calibrate_ui import get_position
        return get_position('chat_open')
    except ImportError:
        return (47, 400)  # Fallback

# Intervalle de surveillance (secondes)
MONITOR_INTERVAL = 30.0

# Nom du bot (le @mention à détecter)
DEFAULT_BOT_NAME = 'mini_pekka'

# Âge max d'un message pour être pris en compte (minutes)
# Les messages plus vieux que ça sont ignorés (comme un vrai joueur)
MAX_COMMAND_AGE_MINUTES = 10

# Nombre de messages récents à garder (pour éviter de re-traiter)
MAX_HISTORY = 20


# =============================================================================
#                    FONCTIONS ADB
# =============================================================================

def _adb_screenshot():
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
    subprocess.run(["adb", "shell", f"input tap {x} {y}"],
                   capture_output=True, timeout=5)
    time.sleep(delay)


# =============================================================================
#                    OCR ENGINE
# =============================================================================

_ocr_engine = None
_ocr_type = None


def _init_ocr():
    """Initialise le moteur OCR (EasyOCR prioritaire, Tesseract fallback)."""
    global _ocr_engine, _ocr_type

    if _ocr_engine is not None:
        return _ocr_engine, _ocr_type

    # Essayer EasyOCR
    try:
        import easyocr
        _ocr_engine = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
        _ocr_type = 'easyocr'
        print("📖 OCR initialisé : EasyOCR (fr+en)")
        return _ocr_engine, _ocr_type
    except ImportError:
        pass

    # Essayer pytesseract
    try:
        import pytesseract
        _ocr_engine = pytesseract
        _ocr_type = 'tesseract'
        print("📖 OCR initialisé : Tesseract")
        return _ocr_engine, _ocr_type
    except ImportError:
        pass

    print("⚠️  Aucun moteur OCR disponible !")
    print("   Installez : pip install easyocr")
    print("   Ou       : pip install pytesseract")
    _ocr_type = None
    return None, None


def _ocr_read(img_cv):
    """
    Lit le texte dans une image BGR.
    
    Returns:
        lines: liste de str (lignes de texte détectées)
    """
    engine, etype = _init_ocr()
    if engine is None:
        return []

    if etype == 'easyocr':
        # EasyOCR retourne une liste de (bbox, text, confidence)
        results = engine.readtext(img_cv, paragraph=False)
        lines = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and len(text.strip()) > 0:
                lines.append(text.strip())
        return lines

    elif etype == 'tesseract':
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        text = engine.image_to_string(gray, lang='fra+eng')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines

    return []


# =============================================================================
#                    COMMAND PARSER
# =============================================================================

def parse_command(text, bot_name=DEFAULT_BOT_NAME):
    """
    Parse une ligne de texte pour détecter une commande.
    
    Formats acceptés :
        @mini_pekka 3
        @mini_pekka attack 3
        @mini_pekka attaque 3
        @mini_pekka stop
        @mini_pekka status
        @mini pekka 3  (avec espace)
        mini_pekka 3   (sans @)
    
    Returns:
        command: dict ou None
            {'type': 'attack', 'target': 3}
            {'type': 'stop'}
            {'type': 'status'}
    """
    text_lower = text.lower().strip()

    # Normaliser le nom du bot (gère espaces, underscores, @)
    bot_patterns = [
        f'@{bot_name}',
        f'@{bot_name.replace("_", " ")}',
        bot_name,
        bot_name.replace('_', ' '),
    ]

    found = False
    remaining = text_lower
    for pattern in bot_patterns:
        if pattern in text_lower:
            # Extraire ce qui suit le mention
            idx = text_lower.index(pattern)
            remaining = text_lower[idx + len(pattern):].strip()
            found = True
            break

    if not found:
        return None

    # Parser la commande
    remaining = remaining.strip()

    # "stop"
    if remaining in ('stop', 'arret', 'arrête', 'pause'):
        return {'type': 'stop'}

    # "status"
    if remaining in ('status', 'état', 'etat', 'info'):
        return {'type': 'status'}

    # "reset" — oublier les commandes déjà exécutées (nouvelle GdC)
    if remaining in ('reset', 'clear', 'oublie', 'nouveau', 'new'):
        return {'type': 'reset'}

    # "attack 3" ou "attaque 3" ou juste "3"
    attack_match = re.match(r'(?:attack|attaque|atk|att)?\s*(\d+)', remaining)
    if attack_match:
        target = int(attack_match.group(1))
        if 1 <= target <= 50:  # Sanity check
            return {'type': 'attack', 'target': target}

    return None


def parse_all_commands(lines, bot_name=DEFAULT_BOT_NAME, **kwargs):
    """
    Parse toutes les lignes et retourne les commandes trouvées.
    
    Returns:
        commands: liste de dicts
    """
    commands = []
    for line in lines:
        cmd = parse_command(line, bot_name)
        if cmd is not None:
            cmd['raw_text'] = line
            commands.append(cmd)
    return commands


def parse_timestamp(text):
    """
    Parse un timestamp de chat CoC et retourne l'âge en minutes.
    
    Formats CoC :
        "À l'instant"     → 0
        "à l'instant"     → 0
        "1min"            → 1
        "5min"            → 5
        "1h 22min"        → 82
        "14h 8min"        → 848
        "1h"              → 60
        "2j"              → 2880
        "1j 3h"           → 1620
        
    Gère les erreurs OCR courantes :
        "IImin" → 11min, "l1min" → 11min, "Ih" → 1h
        
    Returns:
        int (minutes) ou None si pas un timestamp
    """
    text_raw = text.strip()
    
    # Nettoyer les erreurs OCR AVANT le lowercase
    # Si le texte ressemble à un timestamp (contient h, j, min),
    # remplacer I/l/| par 1 dans les positions numériques
    if re.search(r'[hHjJmM]', text_raw):
        # Remplacer I/l/| par 1 quand c'est dans un contexte numérique
        text_raw = re.sub(r'[Il|](?=[Il|\dhHjJmM])', '1', text_raw)
        # "O" → "0" 
        text_raw = re.sub(r'(?<=\d)O', '0', text_raw)
        text_raw = re.sub(r'O(?=\d)', '0', text_raw)
    
    text = text_raw.lower()
    
    # "à l'instant" / "a l'instant" / "instant"
    if 'instant' in text:
        return 0
    
    total_minutes = 0
    found = False
    
    # Jours
    day_match = re.search(r'(\d+)\s*j', text)
    if day_match:
        total_minutes += int(day_match.group(1)) * 24 * 60
        found = True
    
    # Heures
    hour_match = re.search(r'(\d+)\s*h', text)
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60
        found = True
    
    # Minutes
    min_match = re.search(r'(\d+)\s*min', text)
    if min_match:
        total_minutes += int(min_match.group(1))
        found = True
    
    if found:
        return total_minutes
    
    return None


# =============================================================================
#                    CHAT MONITOR
# =============================================================================

class ClanChatMonitor:
    """
    Surveille le chat du clan et détecte les commandes.
    
    Le chat lui-même est la source de vérité :
    
    - Si le chat contient "@mini PEKKA 7" mais PAS de "[IA]" mentionnant #7
      → commande nouvelle, l'IA doit attaquer
    - Si le chat contient "@mini PEKKA 7" ET "[IA] J'attaque le #7"
      → commande déjà traitée, ignorée
    
    Pas de fichier JSON, pas de snapshot, pas de timestamp OCR.
    L'IA regarde simplement si elle a déjà répondu au message.
    """

    BOT_PREFIX = "IA -"

    def __init__(self, bot_name=DEFAULT_BOT_NAME, verbose=True):
        self.bot_name = bot_name
        self.verbose = verbose
        self._running = False

    def send_chat_message(self, message):
        """
        Envoie un message dans le chat du clan via ADB.
        Le chat DOIT être ouvert avant d'appeler cette méthode.
        """
        import subprocess

        try:
            from clashai.navigation.calibrate_ui import get_position
            chat_input_pos = get_position('chat_input')
        except (ImportError, Exception):
            chat_input_pos = (300, 1010)

        _adb_tap(chat_input_pos[0], chat_input_pos[1])
        time.sleep(0.5)

        # ADB input text : les espaces deviennent %s
        # Les caractères spéciaux du shell doivent être échappés
        safe_text = message.replace(' ', '%s')
        # Garder seulement les caractères safe pour ADB
        safe_text = ''.join(c for c in safe_text if c.isalnum() or c in "!?.,-%s")

        try:
            subprocess.run(
                ['adb', 'shell', 'input', 'text', safe_text],
                capture_output=True, timeout=5
            )
            time.sleep(0.3)
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Erreur saisie texte: {e}")
            return

        try:
            from clashai.navigation.calibrate_ui import get_position
            send_pos = get_position('chat_send')
        except (ImportError, Exception):
            send_pos = (490, 1010)

        _adb_tap(send_pos[0], send_pos[1])
        time.sleep(0.5)

        if self.verbose:
            print(f"   📤 Message envoyé : {message}")

    def mark_executed(self, command):
        """Compatibilité — le Brain appelle ça mais le vrai filtre c'est le chat."""
        pass

    def check_once(self, screenshot_pil=None):
        """
        Vérifie le chat et retourne les commandes non encore traitées.
        
        Logique basée sur l'ORDRE des messages (haut = ancien, bas = récent) :
        
        1. Parcourt toutes les lignes du chat
        2. Pour chaque cible #N, note la position de la DERNIÈRE commande
           (@mini PEKKA N) et de la DERNIÈRE réponse ([IA] #N)
        3. Si la commande est PLUS BAS que la réponse → nouvelle commande
           (quelqu'un a re-demandé après la réponse du bot)
        4. Si la réponse est PLUS BAS que la commande → déjà traitée
        5. Si pas de réponse du tout → nouvelle commande
        
        Returns:
            new_commands: liste de commandes à exécuter
        """
        if screenshot_pil is None:
            screenshot_pil = _adb_screenshot()
        if screenshot_pil is None:
            return []

        img_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

        chat_zone = img_cv[CHAT_ZONE_TOP:CHAT_ZONE_BOTTOM,
                           CHAT_ZONE_LEFT:CHAT_ZONE_RIGHT]

        lines = _ocr_read(chat_zone)

        if self.verbose and lines:
            print(f"   📖 OCR : {len(lines)} lignes lues")

        # Étape 1 : noter la position de chaque réponse du bot
        # Les messages du bot commencent par "IA -" ou "IA" suivi de "jattaque" ou "fait"
        last_ack_position = {}  # {target_num: line_index}
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            is_bot_msg = (
                line_lower.startswith('ia -') or
                line_lower.startswith('ia ') and ('jattaque' in line_lower or 'fait' in line_lower) or
                self.BOT_PREFIX.lower() in line_lower
            )
            if is_bot_msg:
                match = re.search(r'(\d{1,2})', line)
                if match:
                    num = int(match.group(1))
                    if 1 <= num <= 50:
                        last_ack_position[num] = i

        if self.verbose and last_ack_position:
            print(f"   🤖 Réponses IA trouvées : "
                  f"{dict(sorted(last_ack_position.items()))}")

        # Étape 2 : noter la position de chaque commande @mini PEKKA N
        last_cmd_position = {}
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Skip les messages du bot
            if (line_lower.startswith('ia -') or
                line_lower.startswith('ia ') and ('jattaque' in line_lower or 'fait' in line_lower)):
                continue
            cmd = parse_command(line, self.bot_name)
            if cmd is not None and cmd['type'] == 'attack':
                cmd['raw_text'] = line
                last_cmd_position[cmd['target']] = (i, cmd)

        # Étape 3 : comparer les positions
        new_commands = []
        for target, (cmd_pos, cmd) in last_cmd_position.items():
            ack_pos = last_ack_position.get(target)

            if ack_pos is None:
                # Pas de réponse [IA] → commande nouvelle
                new_commands.append(cmd)
                if self.verbose:
                    print(f"   🆕 #{target} : aucune réponse [IA] → nouvelle")
            elif cmd_pos > ack_pos:
                # Commande APRÈS la réponse → nouvelle demande
                new_commands.append(cmd)
                if self.verbose:
                    print(f"   🆕 #{target} : commande (ligne {cmd_pos}) "
                          f"après [IA] (ligne {ack_pos}) → nouvelle")
            else:
                # Réponse APRÈS la commande → déjà traitée
                if self.verbose:
                    print(f"   ⏭️  #{target} : [IA] (ligne {ack_pos}) "
                          f"après commande (ligne {cmd_pos}) → déjà faite")

        # Aussi parser les commandes non-attack (stop, status)
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if (line_lower.startswith('ia -') or
                line_lower.startswith('ia ') and ('jattaque' in line_lower or 'fait' in line_lower)):
                continue
            cmd = parse_command(line, self.bot_name)
            if cmd is not None and cmd['type'] != 'attack':
                cmd['raw_text'] = line
                new_commands.append(cmd)

        return new_commands

    def open_chat(self, classify_screen_fn, models):
        """
        Ouvre le chat du clan depuis le village.
        
        Args:
            classify_screen_fn: fonction de classification d'écran
            models: modèles de perception
            
        Returns:
            success: bool
        """
        # Vérifier qu'on est au village
        img = _adb_screenshot()
        if img is None:
            return False

        state, conf = classify_screen_fn(img, models)

        if state == 'chat_clan':
            return True  # Déjà ouvert

        if state != 'village_home':
            if self.verbose:
                print(f"   ⚠️  Pas au village (état: {state}), impossible d'ouvrir le chat")
            return False

        # Taper sur le bouton chat
        _adb_tap(*_get_chat_button_pos())
        time.sleep(1.5)

        # Vérifier
        img = _adb_screenshot()
        if img is None:
            return False
        state, conf = classify_screen_fn(img, models)

        if state == 'chat_clan':
            if self.verbose:
                print("   💬 Chat clan ouvert")
            return True
        else:
            if self.verbose:
                print(f"   ⚠️  Chat non ouvert (état: {state})")
            return False

    def close_chat(self):
        """Ferme le chat en tapant en dehors."""
        try:
            from clashai.navigation.calibrate_ui import get_position
            pos = get_position('chat_close_tap')
        except ImportError:
            pos = (1400, 400)
        _adb_tap(pos[0], pos[1])
        time.sleep(0.5)
        _adb_tap(960, 400)
        time.sleep(0.5)

    def monitor_loop(self, classify_screen_fn, models, callback=None,
                     interval=MONITOR_INTERVAL):
        """
        Boucle de surveillance du chat.
        
        À chaque cycle :
        1. Ouvre le chat
        2. Screenshot + OCR
        3. Parse les commandes
        4. Ferme le chat
        5. Exécute le callback si commande trouvée
        6. Attend l'intervalle
        
        Args:
            classify_screen_fn: fonction de classification d'écran
            models: modèles de perception
            callback: callable(command_dict) — appelé pour chaque commande
            interval: secondes entre chaque check
        """
        self._running = True

        if self.verbose:
            print("\n👁️  Monitoring du chat démarré")
            print(f"   Bot name  : @{self.bot_name}")
            print(f"   Intervalle: {interval}s")

        while self._running:
            try:
                # 1. Ouvrir le chat
                if self.open_chat(classify_screen_fn, models):
                    time.sleep(0.5)

                    # 2. Screenshot du chat
                    img = _adb_screenshot()
                    if img is not None:
                        # 3. Détecter les commandes
                        commands = self.check_once(img)

                        # 4. Fermer le chat
                        self.close_chat()

                        # 5. Exécuter les commandes
                        for cmd in commands:
                            if self.verbose:
                                print(f"\n   ⚡ Exécution : {cmd}")

                            if cmd['type'] == 'stop':
                                if self.verbose:
                                    print("   🛑 Arrêt du monitoring demandé")
                                self._running = False
                                return

                            if callback:
                                callback(cmd)
                    else:
                        self.close_chat()
                else:
                    # Pas au village ? Attendre plus longtemps
                    time.sleep(10)

            except KeyboardInterrupt:
                if self.verbose:
                    print("\n   ⛔ Monitoring arrêté (Ctrl+C)")
                self._running = False
                return

            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Erreur monitoring : {e}")

            # 6. Attendre
            time.sleep(interval)

    def stop(self):
        """Arrête la boucle de monitoring."""
        self._running = False


# =============================================================================
#                            TEST
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClashAI Chat Monitor")
    parser.add_argument('--test-ocr', action='store_true',
                        help="Test OCR sur le chat actuel")
    parser.add_argument('--test-parse', action='store_true',
                        help="Test parsing de commandes")
    parser.add_argument('--bot-name', type=str, default=DEFAULT_BOT_NAME)
    args = parser.parse_args()

    if args.test_parse:
        print("🧪 Test parsing de commandes\n")

        test_lines = [
            "@mini_pekka 3",
            "@mini_pekka attack 5",
            "@mini_pekka attaque 12",
            "@mini pekka 7",
            "mini_pekka 3",
            "@mini_pekka stop",
            "@mini_pekka status",
            "hey les gars ça va ?",
            "quelqu'un pour GdC ?",
            "@mini_pekka",
            "@mini_pekka abc",
            "@mini_pekka 0",
            "@mini_pekka 99",
        ]

        for line in test_lines:
            cmd = parse_command(line, args.bot_name)
            status = f"→ {cmd}" if cmd else "→ (ignoré)"
            print(f"   '{line}' {status}")

    elif args.test_ocr:
        print("🧪 Test OCR sur le chat actuel\n")

        img = _adb_screenshot()
        if img is None:
            print("❌ Impossible de capturer l'écran")
            sys.exit(1)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        chat_zone = img_cv[CHAT_ZONE_TOP:CHAT_ZONE_BOTTOM,
                           CHAT_ZONE_LEFT:CHAT_ZONE_RIGHT]

        # Sauvegarder pour debug
        cv2.imwrite('debug_chat_zone.png', chat_zone)
        print("   Zone chat sauvegardée : debug_chat_zone.png")

        lines = _ocr_read(chat_zone)
        print(f"\n   📖 {len(lines)} lignes détectées :")
        for i, line in enumerate(lines):
            cmd = parse_command(line, args.bot_name)
            marker = " ← COMMANDE" if cmd else ""
            print(f"   [{i:2d}] {line}{marker}")

        if lines:
            commands = parse_all_commands(lines, args.bot_name)
            if commands:
                print(f"\n   🎯 Commandes trouvées : {commands}")
            else:
                print(f"\n   Aucune commande @{args.bot_name} trouvée")

    else:
        print("Usage :")
        print("  --test-parse    Test le parsing de commandes")
        print("  --test-ocr      Test OCR sur le chat actuel")