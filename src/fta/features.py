"""Feature engineering, with the leakage paths closed.

Three separate leaks were live in the original pipeline and all three are
addressed here:

1. **Target-derived aggregates over the whole dataset.** League strength was a
   rolling mean of fee medians computed across every row, then handed to the
   model as a feature; season mean fee did the same. Each test row's own fee
   contributed to a feature it was then predicted from. ``FoldSafeAggregates``
   fits on the training fold only and maps onto test rows.
2. **FIFA ratings from after the transfer.** A 2016 summer move was matched to
   FIFA 17, released that September, whose overall/potential had already
   absorbed the move. ``fta.data.load_fifa`` now labels each edition with the
   season it legitimately precedes.
3. **Random splits across a time series.** ``temporal_split`` splits by season
   and returns explicitly-named objects so a later cell cannot silently rebind
   the same names to a random split, which is exactly what notebook cell 26 did
   to cell 22's temporal split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fta.config import HOLDOUT_SEASON


def parse_fifa_stat(stat):
    """FIFA sub-ratings ship as ``'94+3'`` / ``'94-1'``; take the base value."""
    if isinstance(stat, str):
        import re

        parts = re.split(r"[+-]", stat.strip())
        try:
            return float(parts[0])
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(stat)
    except (TypeError, ValueError):
        return np.nan


@dataclass
class TemporalSplit:
    """Train/test index sets for a strictly forward-looking evaluation."""

    train_idx: pd.Index
    test_idx: pd.Index
    holdout_season: int

    def __post_init__(self):
        if len(self.test_idx) == 0:
            raise ValueError(f"Temporal split produced an empty {self.holdout_season} test set.")
        overlap = self.train_idx.intersection(self.test_idx)
        if len(overlap):
            raise ValueError("Temporal split leaked: train and test share rows.")


def temporal_split(seasons: pd.Series, holdout_season: int = HOLDOUT_SEASON) -> TemporalSplit:
    """Train on every season strictly before ``holdout_season``, test on it.

    Returns a ``TemporalSplit`` object rather than a bare four-tuple. The names
    ``X_train``/``X_test`` were, in the notebook, produced once by a temporal
    split (cell 22) and then rebound by a random 70/30 split (cell 26), so
    every downstream claim of "out-of-sample" depended on execution order. A
    distinct type makes that substitution impossible to do by accident.
    """
    seasons = pd.Series(seasons)
    return TemporalSplit(
        train_idx=seasons.index[seasons < holdout_season],
        test_idx=seasons.index[seasons == holdout_season],
        holdout_season=holdout_season,
    )


class FoldSafeAggregates:
    """Target-derived group aggregates fitted on training rows only.

    ``fit`` sees training rows; ``transform`` maps the fitted statistics onto
    any rows. Unseen categories fall back to the global training statistic, so
    a 2018-only league gets the training-set average rather than a value
    derived from its own held-out fees.
    """

    def __init__(self, target: str = "Transfer_fee_in_mln"):
        self.target = target
        self.league_median_ = None
        self.season_mean_ = None
        self.season_median_ = None
        self.global_median_ = None
        self.global_mean_ = None
        self.league_strength_ = None

    def fit(self, train_df: pd.DataFrame, league_col: str = "League_to", season_col: str = "Season_transferred"):
        y = train_df[self.target]
        self.global_median_ = float(y.median())
        self.global_mean_ = float(y.mean())
        self.league_median_ = train_df.groupby(league_col)[self.target].median()
        self.season_mean_ = train_df.groupby(season_col)[self.target].mean()
        self.season_median_ = train_df.groupby(season_col)[self.target].median()

        # 3-year rolling mean of the buying league's median fee, computed
        # inside the training fold and carried forward. The original computed
        # this across all rows including the test season.
        per_league_season = (
            train_df.groupby([league_col, season_col])[self.target]
            .median()
            .reset_index()
            .sort_values([league_col, season_col])
        )
        per_league_season["strength"] = per_league_season.groupby(league_col)[self.target].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        # The value carried into unseen (later) seasons is the last training
        # value for that league.
        self.league_strength_ = per_league_season.groupby(league_col)["strength"].last()
        self._league_col = league_col
        self._season_col = season_col
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.league_median_ is None:
            raise RuntimeError("FoldSafeAggregates.transform called before fit.")
        out = df.copy()
        out["league_median_fee_to"] = out[self._league_col].map(self.league_median_).fillna(self.global_median_)
        out["buying_league_strength"] = out[self._league_col].map(self.league_strength_).fillna(self.global_median_)
        out["season_mean_fee"] = out[self._season_col].map(self.season_mean_).fillna(self.global_mean_)
        out["season_median_fee"] = out[self._season_col].map(self.season_median_).fillna(self.global_median_)
        return out

    def deflate(self, fees: pd.Series, seasons: pd.Series) -> pd.Series:
        """Express fees in units of that season's training-fold median fee.

        Between 2014 and 2018 the median top-250 fee ran roughly 7 -> 10 -> 12
        -> 13 -> 9 M EUR. Season was never a feature, so that market drift had
        nowhere to go but into the player attributes: a 2017 winger looks
        "worth more" than an identical 2014 winger and the model attributes the
        difference to the player. Modelling the deflated target removes the
        drift instead of laundering it.
        """
        scale = pd.Series(seasons).map(self.season_median_).fillna(self.global_median_)
        return pd.Series(np.asarray(fees, dtype=float) / scale.values, index=pd.Series(fees).index)

    def reflate(self, deflated, seasons: pd.Series) -> pd.Series:
        scale = pd.Series(seasons).map(self.season_median_).fillna(self.global_median_)
        return pd.Series(np.asarray(deflated, dtype=float) * scale.values, index=scale.index)


TOP_NATIONS = [
    "Brazil", "Argentina", "France", "Germany", "Spain",
    "England", "Italy", "Portugal", "Netherlands", "Belgium",
]

LEAGUE_TO_COUNTRY = {
    "Premier League": "England", " England": "England", "League One": "England",
    "Championship": "England",
    "LaLiga": "Spain", "LaLiga2": "Spain", "Primera División": "Spain",
    "Serie A": "Italy", "Serie B": "Italy", "Serie C - B": "Italy",
    "1.Bundesliga": "Germany", "Bundesliga": "Germany", "2.Bundesliga": "Germany",
    "Ligue 1": "France", "Ligue 2": "France",
    "Liga NOS": "Portugal", " Portugal": "Portugal", "Ledman Liga Pro": "Portugal",
    "Eredivisie": "Netherlands",
    "Série A": "Brazil", " Brazil": "Brazil",
    "Süper Lig": "Turkey",
    "Premier Liga": "Russia", " Russia": "Russia",
    "Jupiler Pro League": "Belgium", " Belgium": "Belgium",
    "Super League": "China", " China": "China",
    "MLS": "United States",
    "Argentina": "Argentina", "Torneo Final": "Argentina",
    "Mexico": "Mexico", "Liga MX Clausura": "Mexico", "Liga MX Apertura": "Mexico",
    "Scotland": "Scotland", "Premiership": "Scotland",
}


def group_position(pos) -> int:
    """0 GK / 1 DEF / 2 MID / 3 FWD from a FIFA position string."""
    if not isinstance(pos, str):
        return 0
    first = pos.split(",")[0].strip()
    if first in {"ST", "CF", "LW", "RW", "LS", "RS", "RF", "LF"}:
        return 3
    if first in {"CAM", "CM", "CDM", "LM", "RM", "LAM", "RAM", "LDM", "RDM"}:
        return 2
    if first in {"CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB"}:
        return 1
    return 0


TEN_FACTORS = [
    "Contract_Duration",
    "Age_Feature",
    "Financial_Strength",
    "Ability_Overall",
    "Ability_Potential",
    "xG_Proxy",
    "xA_Proxy",
    "Passport_Premium",
    "Position_Feature",
    "Home_Nation_Transfer",
]


def build_ten_factors(merged: pd.DataFrame, season_col: str = "Season_transferred") -> pd.DataFrame:
    """Materialise the ten audit factors on a joined transfer/FIFA frame.

    ``Financial_Strength`` must already have been supplied by a fitted
    ``FoldSafeAggregates`` (as ``buying_league_strength``) - this function
    refuses to compute it, because computing it here is what made it leak.
    """
    if "buying_league_strength" not in merged.columns:
        raise ValueError(
            "buying_league_strength missing: fit FoldSafeAggregates on the "
            "training fold and transform before building features, so the "
            "league aggregate is not derived from held-out fees."
        )

    out = merged.copy()
    contract_end = pd.to_numeric(out.get("contract_valid_until"), errors="coerce")
    out["Contract_Duration"] = (contract_end - out[season_col]).clip(lower=0).fillna(2.0)
    out["Age_Feature"] = pd.to_numeric(out["age"], errors="coerce")
    out["Financial_Strength"] = out["buying_league_strength"]
    out["Ability_Overall"] = pd.to_numeric(out["overall"], errors="coerce")
    out["Ability_Potential"] = pd.to_numeric(out["potential"], errors="coerce")
    out["xG_Proxy"] = out["attacking_finishing"].map(parse_fifa_stat) + out["mentality_positioning"].map(parse_fifa_stat)
    out["xA_Proxy"] = out["mentality_vision"].map(parse_fifa_stat) + out["attacking_crossing"].map(parse_fifa_stat)
    out["Passport_Premium"] = out["Nationality"].isin(TOP_NATIONS).astype(int)
    out["Position_Feature"] = out["player_positions"].map(group_position)
    dest_country = out["League_to"].map(LEAGUE_TO_COUNTRY)
    out["Home_Nation_Transfer"] = (out["Nationality"] == dest_country).astype(int)
    return out
