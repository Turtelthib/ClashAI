# clashai/perception/reward_reader/
# Read stars (0-3) and destruction % from the attack results screen.
#
# Split into focused modules (Phase 3):
#   constants.py  — thresholds + template dirs
#   green.py      — shared green-channel isolation
#   stars.py      — star counting (HSV silver detection)
#   percentage.py — % via digit template matching
#   results.py    — read_attack_results / calculate_reward (top-level API)
#
# Public API re-exported so callers keep using:
#   from clashai.perception.reward_reader import read_attack_results

from clashai.perception.reward_reader.green import isolate_green
from clashai.perception.reward_reader.stars import count_stars
from clashai.perception.reward_reader.percentage import (
    read_percentage,
    read_percentage_from_stars,
    find_pct_region,
    load_digit_templates,
)
from clashai.perception.reward_reader.results import (
    read_attack_results,
    calculate_reward,
    extract_result_screen,
)

__all__ = [
    'isolate_green',
    'count_stars',
    'read_percentage', 'read_percentage_from_stars',
    'find_pct_region', 'load_digit_templates',
    'read_attack_results', 'calculate_reward', 'extract_result_screen',
]
