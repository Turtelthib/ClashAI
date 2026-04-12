# scripts/rl/zoom_control.py
# Contrôle du zoom via Windows API (ctypes mouse_event).
#
# pyautogui.scroll ne fonctionne PAS sur l'émulateur Google Play Games.
# En revanche, ctypes.windll.user32.mouse_event avec MOUSEEVENTF_WHEEL
# fonctionne parfaitement.
#
# Usage :
#   from clashai.navigation.zoom_control import zoom_out
#   zoom_out()  # Dézoome au maximum avant chaque attaque

import time
import sys
import ctypes


# =============================================================================
#                         CONFIGURATION
# =============================================================================

# Nom de la fenêtre de l'émulateur (pour la trouver automatiquement)
EMULATOR_WINDOW_KEYWORDS = ['mulateur', 'Google Play', 'play games']

# Fallback si la fenêtre n'est pas trouvée
FALLBACK_CENTER_X = 1334
FALLBACK_CENTER_Y = 764

# Paramètres de scroll
ZOOM_OUT_SCROLLS = 15     # Nombre de scrolls pour un dézoom complet
SCROLL_DELTA = -120        # -120 = 1 cran de molette vers le bas = dézoom
SCROLL_DELAY = 0.08        # Délai entre les scrolls

# Windows API constants
MOUSEEVENTF_WHEEL = 0x0800


# =============================================================================
#                 TROUVER LA FENÊTRE DE L'ÉMULATEUR
# =============================================================================

def _find_emulator_center():
    """
    Trouve le centre de la fenêtre de l'émulateur automatiquement.
    Utilise pygetwindow si disponible, sinon fallback.
    """
    try:
        import pygetwindow as gw
        windows = gw.getAllTitles()

        for keyword in EMULATOR_WINDOW_KEYWORDS:
            matches = [w for w in windows if keyword.lower() in w.lower()]
            if matches:
                win = gw.getWindowsWithTitle(matches[0])[0]
                cx = win.left + win.width // 2
                cy = win.top + win.height // 2
                return cx, cy

    except ImportError:
        pass
    except Exception:
        pass

    return FALLBACK_CENTER_X, FALLBACK_CENTER_Y


# =============================================================================
#                     FONCTIONS PRINCIPALES
# =============================================================================

def zoom_out(scrolls=None):
    """
    Dézoome au maximum en simulant la molette souris via Windows API.

    Args:
        scrolls: nombre de scrolls (défaut: ZOOM_OUT_SCROLLS)

    Returns:
        True si le dézoom a été effectué
    """
    if scrolls is None:
        scrolls = ZOOM_OUT_SCROLLS

    # Trouver le centre de l'émulateur
    center_x, center_y = _find_emulator_center()

    # Sauvegarder la position actuelle du curseur
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    original_x, original_y = pt.x, pt.y

    try:
        # Placer le curseur au centre de l'émulateur
        ctypes.windll.user32.SetCursorPos(center_x, center_y)
        time.sleep(0.2)

        # Scroller pour dézoomer
        for _ in range(scrolls):
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_WHEEL, 0, 0, SCROLL_DELTA, 0
            )
            time.sleep(SCROLL_DELAY)

        # Petit délai pour que le jeu finisse l'animation de zoom
        time.sleep(0.3)

        print(f"   🔍 Dézoom effectué ({scrolls} scrolls)")
        return True

    except Exception as e:
        print(f"   ⚠️  Erreur dézoom : {e}")
        return False

    finally:
        # Remettre le curseur à sa position originale
        ctypes.windll.user32.SetCursorPos(original_x, original_y)


def zoom_in(scrolls=5):
    """Zoome (scroll vers le haut). Utile pour les tests."""
    center_x, center_y = _find_emulator_center()

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    original_x, original_y = pt.x, pt.y

    try:
        ctypes.windll.user32.SetCursorPos(center_x, center_y)
        time.sleep(0.2)

        for _ in range(scrolls):
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_WHEEL, 0, 0, 120, 0  # Positif = zoom in
            )
            time.sleep(SCROLL_DELAY)

        return True
    finally:
        ctypes.windll.user32.SetCursorPos(original_x, original_y)


# =============================================================================
#                            MAIN
# =============================================================================

if __name__ == "__main__":
    if '--test' in sys.argv:
        print("🔍 Test dézoom...")
        zoom_out(scrolls=10)
        print("✅ Terminé ! Vérifie que le jeu a dézoomé.")

    elif '--zoom-in' in sys.argv:
        print("🔍 Test zoom in...")
        zoom_in(scrolls=5)
        print("✅ Terminé !")

    else:
        cx, cy = _find_emulator_center()
        print("zoom_control.py — Dézoom via Windows API")
        print(f"  Centre émulateur : ({cx}, {cy})")
        print(f"  Scrolls          : {ZOOM_OUT_SCROLLS}")
        print()
        print("  --test      Tester le dézoom")
        print("  --zoom-in   Tester le zoom")