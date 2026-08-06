# clashai/perception/reward_reader/constants.py
# Thresholds + template dirs for reward (stars + %) reading.

from clashai.paths import DEBUG_DIR, REWARD_DIGITS_DIR, REWARD_TEMPLATES_DIR

TEMPLATES_DIR = REWARD_TEMPLATES_DIR
DIGITS_DIR = REWARD_DIGITS_DIR

# Digit matching
DIGIT_MATCH_THRESHOLD = 0.60
PCT_MATCH_THRESHOLD = 0.50

# Stars (HSV)
STAR_MIN_AREA = 1000
STAR_MAX_ASPECT = 2.5
STAR_SATURATION_MAX = 60
STAR_VALUE_MIN = 180
