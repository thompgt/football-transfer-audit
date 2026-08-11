"""Tests for leakage-safe aggregates, the temporal split and inflation handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fta.features import (
    FoldSafeAggregates,
    TemporalSplit,
    build_ten_factors,
    group_position,
    parse_fifa_stat,
    temporal_split,
)


def test_parse_fifa_stat_handles_the_plus_minus_format():
    assert parse_fifa_stat("94+3") == 94.0
    assert parse_fifa_stat("94-1") == 94.0
    assert parse_fifa_stat(72) == 72.0
    assert np.isnan(parse_fifa_stat("n/a"))
    assert np.isnan(parse_fifa_stat(None))


def test_group_position_buckets():
    assert group_position("ST, LW") == 3
    assert group_position("CAM") == 2
    assert group_position("CB, RB") == 1
    assert group_position("GK") == 0
    assert group_position(None) == 0


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    seasons = np.repeat([2015, 2016, 2017, 2018], 25)
    # Deliberate market inflation plus a huge 2018-only outlier league.
    base = {2015: 7.0, 2016: 10.0, 2017: 13.0, 2018: 9.0}
    fee = np.array([base[s] for s in seasons]) + rng.normal(0, 1, len(seasons))
    league = np.where(seasons == 2018, "NewLeague", "Premier League")
    return pd.DataFrame(
        {
            "Season_transferred": seasons,
            "League_to": league,
            "Transfer_fee_in_mln": fee,
        }
    )


def test_temporal_split_is_forward_looking(panel):
    split = temporal_split(panel["Season_transferred"], holdout_season=2018)
    assert isinstance(split, TemporalSplit)
    train_seasons = set(panel.loc[split.train_idx, "Season_transferred"])
    test_seasons = set(panel.loc[split.test_idx, "Season_transferred"])
    assert train_seasons == {2015, 2016, 2017}
    assert test_seasons == {2018}
    assert not split.train_idx.intersection(split.test_idx).size


def test_temporal_split_refuses_an_empty_holdout(panel):
    with pytest.raises(ValueError, match="empty"):
        temporal_split(panel["Season_transferred"], holdout_season=2099)


def test_aggregates_never_see_the_holdout_fees(panel):
    """The leak: league strength used to be built from all rows including test."""
    split = temporal_split(panel["Season_transferred"])
    agg = FoldSafeAggregates().fit(panel.loc[split.train_idx])

    # Nothing fitted may mention the holdout season or the holdout-only league.
    assert 2018 not in set(agg.season_mean_.index)
    assert "NewLeague" not in set(agg.league_median_.index)

    test = agg.transform(panel.loc[split.test_idx])
    # An unseen league falls back to the *training* global median, not to a
    # statistic derived from its own held-out fees.
    assert (test["league_median_fee_to"] == agg.global_median_).all()
    assert test["buying_league_strength"].notna().all()


def test_aggregates_are_identical_regardless_of_what_test_rows_contain(panel):
    """Refitting with a wildly different holdout must not move a train feature."""
    split = temporal_split(panel["Season_transferred"])
    agg = FoldSafeAggregates().fit(panel.loc[split.train_idx])
    train_feature = agg.transform(panel.loc[split.train_idx])["league_median_fee_to"].to_numpy()

    tampered = panel.copy()
    tampered.loc[split.test_idx, "Transfer_fee_in_mln"] *= 1000
    agg2 = FoldSafeAggregates().fit(tampered.loc[split.train_idx])
    train_feature2 = agg2.transform(tampered.loc[split.train_idx])["league_median_fee_to"].to_numpy()

    np.testing.assert_allclose(train_feature, train_feature2)


def test_deflation_removes_the_market_drift(panel):
    """Raw fees trend across seasons; deflated fees should not."""
    split = temporal_split(panel["Season_transferred"])
    agg = FoldSafeAggregates().fit(panel.loc[split.train_idx])
    train = panel.loc[split.train_idx]

    raw_by_season = train.groupby("Season_transferred")["Transfer_fee_in_mln"].median()
    assert raw_by_season.max() / raw_by_season.min() > 1.5  # the drift is real

    deflated = agg.deflate(train["Transfer_fee_in_mln"], train["Season_transferred"])
    deflated_by_season = deflated.groupby(train["Season_transferred"].values).median()
    np.testing.assert_allclose(deflated_by_season.to_numpy(), 1.0, atol=1e-9)


def test_deflate_reflate_round_trips(panel):
    split = temporal_split(panel["Season_transferred"])
    agg = FoldSafeAggregates().fit(panel.loc[split.train_idx])
    train = panel.loc[split.train_idx]
    d = agg.deflate(train["Transfer_fee_in_mln"], train["Season_transferred"])
    r = agg.reflate(d, train["Season_transferred"])
    np.testing.assert_allclose(r.to_numpy(), train["Transfer_fee_in_mln"].to_numpy())


def test_build_ten_factors_refuses_to_compute_the_leaky_aggregate_itself():
    merged = pd.DataFrame(
        {
            "Season_transferred": [2016],
            "age": [24],
            "overall": [82],
            "potential": [88],
            "contract_valid_until": [2020],
            "attacking_finishing": ["80+2"],
            "mentality_positioning": [78],
            "mentality_vision": [75],
            "attacking_crossing": [70],
            "Nationality": ["Brazil"],
            "player_positions": ["ST, LW"],
            "League_to": ["Premier League"],
        }
    )
    with pytest.raises(ValueError, match="buying_league_strength"):
        build_ten_factors(merged)

    merged["buying_league_strength"] = 11.0
    out = build_ten_factors(merged)
    assert out["Contract_Duration"].iloc[0] == 4.0
    assert out["xG_Proxy"].iloc[0] == 158.0
    assert out["xA_Proxy"].iloc[0] == 145.0
    assert out["Passport_Premium"].iloc[0] == 1
    assert out["Position_Feature"].iloc[0] == 3
    assert out["Home_Nation_Transfer"].iloc[0] == 0  # Brazilian -> England
