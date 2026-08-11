"""Single source of truth for the audit protocol constants.

Every notebook cell, script and test in this repository imports its thresholds
from here. Before this module existed the protocol was declared once (n >= 30,
1000 bootstrap resamples) and then silently contradicted by hardcoded literals
of 20, 10 and 8 scattered through the notebook, and the bootstrap was never
implemented at all. Anything that needs a support threshold must import
``MIN_SUPPORT``; anything that reports a group statistic must report a
bootstrap interval built with ``BOOTSTRAP_SAMPLES``.
"""

from __future__ import annotations

# Minimum number of held-out observations a group needs before its statistics
# are treated as part of the primary audit finding. Groups below this are still
# reported, but flagged INCONCLUSIVE.
MIN_SUPPORT = 30

# Groups smaller than this are not reported at all - the estimate is not
# meaningfully different from noise and printing it invites over-reading.
MIN_REPORTABLE = 5

BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 169

# Pass/fail thresholds for the scorecard, in millions of euros / unitless ratio.
BIAS_GAP_THRESHOLD = 1.0
MAE_RATIO_THRESHOLD = 1.5

# Seasons used throughout. The temporal split trains on everything before
# HOLDOUT_SEASON and tests on HOLDOUT_SEASON itself.
TRAIN_SEASONS = (2015, 2016, 2017)
HOLDOUT_SEASON = 2018

# The FIFA edition released in year Y ships in September of year Y-1 and is
# built from scouting data gathered before that. To describe a player as they
# were *before* a transfer in season S (which opens in the summer of S), the
# newest safe edition is the one released in the autumn of S-1, i.e. FIFA (S).
# Using FIFA (S+1) - the previous behaviour - snapshots ratings updated months
# after the move, partly in reaction to the move and its fee.
FIFA_EDITION_OFFSET = 0

AUDIT_CONFIG = {
    "min_support": MIN_SUPPORT,
    "min_reportable": MIN_REPORTABLE,
    "thresholds": {
        "bias_gap": BIAS_GAP_THRESHOLD,
        "mae_ratio": MAE_RATIO_THRESHOLD,
    },
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "confidence_level": CONFIDENCE_LEVEL,
    "random_seed": RANDOM_SEED,
    "train_seasons": list(TRAIN_SEASONS),
    "holdout_season": HOLDOUT_SEASON,
}
