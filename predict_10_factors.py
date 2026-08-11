"""Ten-factor transfer-fee model with SHAP and LIME explanations.

Run from the repository root::

    python predict_10_factors.py

All the data plumbing now comes from ``src/fta`` so it is shared with
``scripts/run_audit.py`` and covered by ``tests/``. This script is the
explainability story only.

Five correctness problems in the previous version are fixed here:

* **Inflation.** The median top-250 fee ran roughly 7 -> 10 -> 13 -> 9 M EUR
  across the pooled seasons and season was not a feature, so market drift was
  absorbed into player attributes. The model now trains on the season-deflated
  target by default; ``--raw-fee`` restores the old behaviour for comparison.
* **Random split across a time series.** Replaced with a strict temporal split:
  train on 2015-2017, test on 2018.
* **Target-derived aggregates over the full dataset.** Buying-league strength
  was a rolling mean of fee medians computed across every row and then fed back
  as a feature. It is now fitted on the training fold only.
* **FIFA ratings from after the transfer.** A 2016 summer move was matched to
  FIFA 17, released that September, whose ratings had already reacted to the
  move. Each season is now pinned to the last edition released before its
  window.
* **LIME plots silently falling back to the wrong player.** The old code did
  ``candidates = df_full.head(1)`` when a scenario player was missing while
  keeping the title, and git history ("Fix player names in LIME plots") shows
  wrong figures actually shipped. A missing scenario is now an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fta import config  # noqa: E402
from fta.data import load_fifa, load_transfers  # noqa: E402
from fta.features import (  # noqa: E402
    TEN_FACTORS,
    FoldSafeAggregates,
    build_ten_factors,
    temporal_split,
)
from fta.matching import strict_join  # noqa: E402
from fta.regions import to_region  # noqa: E402

PLOTS = REPO_ROOT / "plots"

FEATURE_NOTES = [
    ("Contract_Duration", "Years left on contract at the move. Long contracts remove the seller's urgency."),
    ("Age_Feature", "Player age at the move."),
    ("Financial_Strength", "3-year rolling median fee of the buying league, fitted on training seasons only."),
    ("Ability_Overall", "FIFA overall rating from the edition released before the window."),
    ("Ability_Potential", "FIFA potential rating, same edition."),
    ("xG_Proxy", "Finishing + Positioning. A proxy, not a measured xG."),
    ("xA_Proxy", "Vision + Crossing. A proxy, not a measured xA."),
    ("Passport_Premium", "1 if the player holds a top-10 footballing nationality."),
    ("Position_Feature", "0 GK / 1 DEF / 2 MID / 3 FWD."),
    ("Home_Nation_Transfer", "1 if moving to a league in the player's own country."),
]

SCENARIOS = [
    {
        "label": "Young Brazilian Talent",
        "player_name": "Richarlison",
        "season": 2017,
        "desc": "Young prospect moving from Brazil to the Premier League (Fluminense to Watford).",
    },
    {
        "label": "English Domestic Move",
        "player_name": "Alex Oxlade-Chamberlain",
        "season": 2017,
        "desc": "English player moving domestically between top clubs (Arsenal to Liverpool).",
    },
    {
        "label": "Superstar Juggernaut",
        "player_name": "Paul Pogba",
        "season": 2016,
        "desc": "Marquee signing with world-record fee context (Juventus to Man Utd).",
    },
    {
        "label": "Veteran Superstar",
        "player_name": "Cristiano Ronaldo",
        "season": 2018,
        "desc": "Elite veteran (33) moving for a high fee to a top league (Real to Juventus).",
    },
    {
        "label": "Mid-tier Competitive",
        "player_name": "Daley Blind",
        "season": 2018,
        "desc": "Prime-age established player moving between competitive leagues (Man Utd to Ajax).",
    },
]


class ScenarioNotFound(LookupError):
    """A named LIME scenario player is not in the joined dataset.

    Raised rather than falling back to an arbitrary row. A plot titled "Paul
    Pogba" that explains whoever happened to sort first is worse than no plot:
    it is a wrong figure that looks right, and this repository shipped several.
    """


def build_dataset():
    """Join, split, and engineer features without leaking the holdout season."""
    transfers = load_transfers()
    fifa = load_fifa()

    seasons = list(config.TRAIN_SEASONS) + [config.HOLDOUT_SEASON]
    transfers = transfers[transfers["Season_transferred"].isin(seasons)].copy()
    transfers = transfers.dropna(subset=["Transfer_fee", "Name"])
    transfers = transfers[transfers["Age"] > 0]
    transfers["Transfer_fee_in_mln"] = transfers["Transfer_fee"] / 1e6
    transfers = transfers.reset_index(drop=True)

    matched = strict_join(transfers, fifa)
    matched["Region"] = matched["Nationality"].map(to_region)
    df = matched[matched["match_tier"] != "unmatched"].reset_index(drop=True)

    split = temporal_split(df["Season_transferred"])
    agg = FoldSafeAggregates().fit(df.loc[split.train_idx])
    df = build_ten_factors(agg.transform(df))

    X = df[TEN_FACTORS].astype(float)
    X = X.fillna(X.loc[split.train_idx].median())
    return df, X, split, agg


def find_scenario_row(df: pd.DataFrame, spec: dict) -> pd.Series:
    mask = (df["Name"] == spec["player_name"]) & (df["Season_transferred"] == spec["season"])
    hits = df[mask]
    if hits.empty:
        # Try a normalized contains-match before giving up, so an accent or a
        # middle name does not lose a legitimate scenario.
        loose = (
            df["Name"].str.contains(spec["player_name"].split()[-1], case=False, na=False)
            & (df["Season_transferred"] == spec["season"])
        )
        hits = df[loose]
    if hits.empty:
        raise ScenarioNotFound(
            f"{spec['player_name']} ({spec['season']}) is not in the joined dataset. "
            "Either the join dropped this transfer or the scenario list is stale. "
            "Fix one of those - do not plot a different player under this title."
        )
    return hits.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-fee",
        action="store_true",
        help="Train on the raw fee instead of the season-deflated fee (the old, inflation-confounded behaviour).",
    )
    parser.add_argument(
        "--skip-missing-scenarios",
        action="store_true",
        help="Warn and continue instead of failing when a named scenario player is absent.",
    )
    args = parser.parse_args()

    PLOTS.mkdir(exist_ok=True)

    print("Loading and joining data...")
    df, X, split, agg = build_dataset()
    print(f"Dataset: {len(df)} matched transfers "
          f"({len(split.train_idx)} train {config.TRAIN_SEASONS} / "
          f"{len(split.test_idx)} test {config.HOLDOUT_SEASON})")

    y_raw = df["Transfer_fee_in_mln"].astype(float)
    seasons = df["Season_transferred"]
    X_train, X_test = X.loc[split.train_idx], X.loc[split.test_idx]
    y_train_raw, y_test_raw = y_raw.loc[split.train_idx], y_raw.loc[split.test_idx]

    deflated = not args.raw_fee
    y_train = y_train_raw if args.raw_fee else agg.deflate(y_train_raw, seasons.loc[split.train_idx])

    model = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    if deflated:
        pred_test = agg.reflate(pred_test, seasons.loc[split.test_idx]).to_numpy()

    target_label = "season-deflated fee" if deflated else "raw fee"
    print(f"\n--- Evaluation on the held-out {config.HOLDOUT_SEASON} season "
          f"(target: {target_label}) ---")
    print(f"MAE:  {mean_absolute_error(y_test_raw, pred_test):.3f} M EUR")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_raw, pred_test)):.3f} M EUR")
    print(f"R2:   {r2_score(y_test_raw, pred_test):.3f}")

    print("\n--- Feature ranges (training fold) ---")
    stats = X_train.describe()
    for feat, note in FEATURE_NOTES:
        print(f"{feat:22}: [{stats.loc['min', feat]:7.1f} - {stats.loc['max', feat]:7.1f}] "
              f"mean {stats.loc['mean', feat]:7.1f}  | {note}")

    # ------------------------------------------------------------------ SHAP
    import shap

    print("\nComputing SHAP values on the held-out season...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=TEN_FACTORS, show=False)
    plt.title(
        f"SHAP: ten-factor RF on held-out {config.HOLDOUT_SEASON} ({target_label})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_summary.png", dpi=140)
    plt.close()
    print(f"Generated plot: shap_summary.png")

    # ------------------------------------------------------------------ LIME
    from lime.lime_tabular import LimeTabularExplainer

    def predict_fn(arr):
        # LIME hands raw ndarrays to the model; rewrapping them keeps sklearn's
        # feature-name check satisfied instead of emitting a warning per call.
        return model.predict(pd.DataFrame(arr, columns=TEN_FACTORS))

    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=TEN_FACTORS,
        class_names=["Transfer_fee_in_mln"],
        mode="regression",
        random_state=42,
    )

    missing = []
    for spec in SCENARIOS:
        try:
            row = find_scenario_row(df, spec)
        except ScenarioNotFound as exc:
            if not args.skip_missing_scenarios:
                raise
            print(f"WARNING: {exc}")
            missing.append(spec["label"])
            continue

        row_values = row[TEN_FACTORS].to_numpy(dtype=float)
        exp = lime_explainer.explain_instance(row_values, predict_fn, num_features=10)

        lime_df = pd.DataFrame(exp.as_list(), columns=["feature", "weight"]).sort_values("weight")

        pred = predict_fn(row_values.reshape(1, -1))[0]
        if deflated:
            pred = float(pred * agg.season_median_.get(row["Season_transferred"], agg.global_median_))
        actual = float(row["Transfer_fee_in_mln"])

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2a9d8f" if w > 0 else "#e76f51" for w in lime_df["weight"]]
        ax.barh(lime_df["feature"], lime_df["weight"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"{spec['label']} | Pred EUR {pred:.2f}M | Actual EUR {actual:.2f}M",
            fontsize=12, fontweight="bold",
        )
        # LIME weights describe a local linear surrogate of *this model*. They
        # are not estimates of what the market pays, and nothing downstream may
        # phrase them causally.
        ax.set_xlabel("LIME local surrogate weight (explains the model, not the market)")
        ax.grid(axis="x", alpha=0.3)

        metadata = (
            f"{spec['desc']}\nPlayer: {row['Name']}\nNation: {row['Nationality']}\n"
            f"Age: {row['Age']}\nSeason: {row['Season_transferred']}\n"
            f"Match tier: {row['match_tier']}"
        )
        ax.text(
            0.98, 0.05, metadata, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
        )

        plt.tight_layout()
        fname = f"lime_{spec['label'].lower().replace(' ', '_')}.png"
        plt.savefig(PLOTS / fname, dpi=140)
        plt.close()
        print(f"Generated plot: {fname} for {row['Name']} ({row['Season_transferred']})")

    if missing:
        print(f"\nScenarios skipped because the join dropped them: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
