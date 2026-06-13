# clashai/combat/environment_v4/env.py
# ClashEnvV4 — assembles the domain mixins on top of ClashEnvV3.
#
# Changes vs V3:
# - 37 actions (role × sector + auto-targeted spells)
# - Compacted observation (54 dims instead of 76)
# - TroopManager for role-based deployment
# - Centralized reward shaping
# - Agent freely chooses the order of spells/abilities

from clashai.combat.legacy.environment import ClashEnvV3
from clashai.combat.environment_v4.core import CoreMixin
from clashai.combat.environment_v4.observation import ObservationMixin
from clashai.combat.environment_v4.actions import ActionsMixin
from clashai.combat.environment_v4.reward import RewardMixin
from clashai.combat.environment_v4.observe import ObserveMixin
from clashai.combat.environment_v4.capture import CaptureMixin
from clashai.combat.environment_v4.heuristic import HeuristicMixin


class ClashEnvV4(
    ObservationMixin,
    ActionsMixin,
    RewardMixin,
    ObserveMixin,
    CaptureMixin,
    HeuristicMixin,
    CoreMixin,
    ClashEnvV3,
):
    """
    V4 environment — simplified action space (37 actions).

    Inherits from V3 for ADB navigation; overrides:
    - Observation (54 dims)
    - Action mask (37 actions)
    - Action execution (role × sector)
    - Reward shaping (separate module)

    Implementation split across domain mixins (Phase 3):
      core / observation / actions / reward / observe / capture / heuristic.
    The mixins precede ClashEnvV3 in the MRO so their overrides win.
    """
