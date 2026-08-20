# clashai/agents/world.py
# Build the shared `world` snapshot consumed by every agent's can_run().
#
# The `world` is the SSOT context the AgentScheduler passes to each agent on
# every tick. It is a plain dict (cheap to build, easy to fake in tests) sourced
# from the async PerceptionThread cache — NO blocking screenshot here, so
# can_run() stays cheap (V5.1: "ADB zéro screenshot" for perception).
#
# Keys (raw perception):
#   screen_state      : str | None  — CNN screen classification
#   screen_conf       : float
#   buildings         : list | None — YOLO building detections
#   troop_bar         : list | None — troop bar CNN detections
#   troop_positions   : dict | None — {name: (x,y,conf)} active only
#   buttons           : dict         — {classe CNN | clé calibration: (x,y,conf)}
#                                      boutons visibles à l'écran (CNN UI)
#   buttons_age_s     : float | None — âge de cette détection (None = jamais)
#   readings          : dict         — valeurs LUES à l'écran : resources,
#                                      builders, lab_libre. Une clé ABSENTE =
#                                      non lisible ici (jamais devinée)
#   fresh             : bool         — True if the perception cache was fresh
#   timestamp         : float
# Derived convenience flags:
#   on_village_home   : bool         — screen_state == 'village_home'
# Plus any **flags the caller merges in (e.g. attacks_since_chat_check).

import time

# Documented surface of the snapshot (raw perception keys).
WORLD_KEYS = (
    'screen_state', 'screen_conf', 'buildings',
    'troop_bar', 'troop_positions', 'buttons', 'buttons_age_s', 'readings',
    'fresh', 'timestamp',
)


def build_world(models=None, max_age_s=2.0, **flags):
    """
    Build the shared world snapshot from the PerceptionThread cache.

    Args:
        models: the loaded models dict (may contain 'perception_thread').
                If None or the thread is stale, returns an empty-but-valid world.
        max_age_s: max age of the perception cache to consider it fresh.
        **flags: extra bookkeeping flags merged into the world (e.g.
                 attacks_since_chat_check=3).

    Returns:
        world: dict — never None; all WORLD_KEYS present.
    """
    world = {
        'screen_state': None,
        'screen_conf': 0.0,
        'buildings': None,
        'troop_bar': None,
        'troop_positions': None,
        # Boutons visibles (CNN UI). Détectés à cadence réduite et CONSERVÉS
        # entre deux passages : `buttons_age_s` dit depuis quand, à charge de
        # l'appelant d'en tenir compte s'il veut du frais.
        'buttons': {},
        'buttons_age_s': None,
        'readings': {},
        'fresh': False,
        'timestamp': 0.0,
    }

    pt = models.get('perception_thread') if models else None
    if pt is not None:
        try:
            if pt.is_fresh(max_age_s=max_age_s):
                state = pt.get_latest()
                bts = state.get('buttons_ts') or 0.0
                world.update(
                    screen_state=state.get('screen_state'),
                    screen_conf=state.get('screen_conf', 0.0),
                    buildings=state.get('buildings'),
                    troop_bar=state.get('troop_bar'),
                    troop_positions=state.get('troop_positions'),
                    buttons=state.get('buttons') or {},
                    buttons_age_s=(time.time() - bts) if bts else None,
                    readings=state.get('readings') or {},
                    fresh=True,
                    timestamp=state.get('timestamp', time.time()),
                )
        except Exception:
            # Perception unavailable → keep the empty-but-valid world.
            pass

    # Derived convenience flags (agents may also read screen_state directly).
    world['on_village_home'] = (world['screen_state'] == 'village_home')

    # Caller-supplied bookkeeping flags win over derived defaults.
    world.update(flags)
    return world
