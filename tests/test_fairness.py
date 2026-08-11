"""Tests for the bootstrap CIs, minimum-support rule and computed scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fta.config import MIN_SUPPORT
from fta.fairness import (
    CONCLUSIVE,
    Scorecard,
    assert_scorecard_fresh,
    bootstrap_ci,
    build_scorecard,
    conclusive,
    group_residual_table,
    support_verdict,
)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    sample = rng.normal(5.0, 1.0, 500)
    point, low, high = bootstrap_ci(sample)
    assert low < point < high
    assert point == pytest.approx(sample.mean())


def test_bootstrap_ci_is_wider_for_a_smaller_sample():
    rng = np.random.default_rng(0)
    big = rng.normal(0, 10, 400)
    small = big[:20]
    _, lo_b, hi_b = bootstrap_ci(big)
    _, lo_s, hi_s = bootstrap_ci(small)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_bootstrap_ci_is_deterministic_given_a_seed():
    sample = np.arange(50, dtype=float)
    assert bootstrap_ci(sample, seed=7) == bootstrap_ci(sample, seed=7)


def test_bootstrap_ci_handles_degenerate_input():
    assert all(np.isnan(v) for v in bootstrap_ci([]))
    point, low, high = bootstrap_ci([3.0])
    assert point == 3.0 and np.isnan(low) and np.isnan(high)


def test_support_verdict_uses_the_single_declared_threshold():
    assert support_verdict(MIN_SUPPORT) == CONCLUSIVE
    assert support_verdict(MIN_SUPPORT - 1) != CONCLUSIVE
    assert "inconclusive" in support_verdict(20)
    assert "not reported" in support_verdict(3)


def test_a_heavy_tailed_group_of_twenty_is_not_conclusive():
    """The audit's actual headline claim, restated as a test.

    A mean residual of about -1.3M over 20 heavy-tailed transfers is exactly
    the finding the README led with. This asserts the interval is wide enough
    to contain the global mean - i.e. that the claim is not supported at that
    sample size.
    """
    rng = np.random.default_rng(42)
    # 20 draws with transfer-fee-like dispersion (sd ~ 10M).
    residuals = rng.normal(-1.3, 10.0, 20)
    y_true = np.zeros(20)
    table = group_residual_table(y_true, residuals, ["South America"] * 20)
    row = table.iloc[0]
    assert row["n"] == 20
    assert row["verdict"] != CONCLUSIVE
    assert row["residual_ci_low"] < 0 < row["residual_ci_high"]


def test_group_residual_table_reports_every_group_with_a_verdict():
    rng = np.random.default_rng(1)
    groups = ["W. Europe"] * 60 + ["South America"] * 20 + ["Africa"] * 11
    y_true = rng.normal(20, 8, len(groups))
    y_pred = y_true + rng.normal(0, 3, len(groups))
    table = group_residual_table(y_true, y_pred, groups)

    assert set(table["group"]) == {"W. Europe", "South America", "Africa"}
    # Small groups are kept but flagged, never silently dropped.
    small = table[table["n"] < MIN_SUPPORT]
    assert len(small) == 2
    assert (small["verdict"] != CONCLUSIVE).all()
    assert set(conclusive(table)["group"]) == {"W. Europe"}
    assert table["residual_ci_low"].notna().all()


def test_residual_significance_is_measured_against_the_global_mean_not_zero():
    """A model with a uniform offset must not flag every group as biased."""
    rng = np.random.default_rng(3)
    groups = ["A"] * 100 + ["B"] * 100
    y_true = rng.normal(20, 5, 200)
    y_pred = y_true + 5.0  # every prediction 5M high, identically
    table = group_residual_table(y_true, y_pred, groups)
    assert not table["residual_ci_excludes_global"].any()


def test_scorecard_is_computed_not_transcribed():
    rng = np.random.default_rng(5)
    groups = np.array(["W. Europe"] * 120 + ["South America"] * 60)
    y_true = rng.normal(20, 6, len(groups))
    y_pred = y_true + rng.normal(0, 2, len(groups))
    # Push one adequately-powered group clearly off-centre.
    y_pred[groups == "South America"] -= 6.0

    card = build_scorecard("rf_test", y_true, y_pred, groups)
    assert isinstance(card, Scorecard)
    assert card.n_conclusive_groups == 2
    assert card.max_signed_residual_gap > 1.0
    assert card.status == "FAIL"
    assert card.rmse > 0 and card.mae > 0


def test_scorecard_says_undetermined_when_the_gap_is_within_noise():
    rng = np.random.default_rng(11)
    groups = np.array(["W. Europe"] * 40 + ["South America"] * 35)
    y_true = rng.normal(20, 25, len(groups))
    y_pred = y_true + rng.normal(0, 25, len(groups))
    card = build_scorecard("rf_noisy", y_true, y_pred, groups)
    assert card.status in {"UNDETERMINED", "PASS"}
    if card.status == "UNDETERMINED":
        assert "noise" in card.recommendation


def test_scorecard_refuses_to_certify_when_no_group_has_support():
    rng = np.random.default_rng(9)
    groups = np.array(["A"] * 11 + ["B"] * 14 + ["C"] * 20)
    y_true = rng.normal(20, 6, len(groups))
    card = build_scorecard("rf_thin", y_true, y_true + 4.0, groups)
    assert card.status == "INCONCLUSIVE"
    assert card.n_conclusive_groups == 0
    assert np.isnan(card.max_signed_residual_gap)


def test_assert_scorecard_fresh_catches_drift():
    """The guard that would have caught the hardcoded cell-58 scorecard."""
    rng = np.random.default_rng(13)
    groups = np.array(["A"] * 80 + ["B"] * 80)
    y_true = rng.normal(20, 6, len(groups))
    card = build_scorecard("m", y_true, y_true + 1.0, groups)

    assert_scorecard_fresh(card, {"status": card.status, "rmse": card.rmse})

    with pytest.raises(AssertionError, match="drifted"):
        assert_scorecard_fresh(card, {"max_signed_residual_gap": 3.80})
