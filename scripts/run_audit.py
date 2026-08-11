#!/usr/bin/env python
"""Run the fairness audit end to end and write every published number to disk.

This script is the answer to the two most serious audit findings: the final
scorecard and the README's headline regional figure were both **hardcoded
literals** transcribed by hand, not computed. Everything either of them
claimed now comes out of this file, into ``results/``, and from there into
``viz.py`` and the README. If the pipeline changes, the published numbers
change with it or CI fails.

    python scripts/run_audit.py

Outputs (all JSON/CSV, all regenerated from scratch):

    results/region_stats.json    - per-region residuals with bootstrap CIs
    results/scorecard.json       - the computed audit verdict
    results/attrition.csv        - join match rate by league / region / season
    results/ambiguity.csv        - surname-collision rate by region
    results/run_metadata.json    - inputs, seeds, package versions
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fta import config  # noqa: E402
from fta.data import load_fifa, load_transfers  # noqa: E402
from fta.fairness import (  # noqa: E402
    build_scorecard,
    group_residual_table,
    permutation_proxy_leakage,
)
from fta.features import (  # noqa: E402
    TEN_FACTORS,
    FoldSafeAggregates,
    build_ten_factors,
    temporal_split,
)
from fta.matching import ambiguity_rate_by, attrition_table, strict_join  # noqa: E402
from fta.regions import to_region  # noqa: E402

RESULTS = REPO_ROOT / "results"


def _clean(obj):
    """Recursively make a payload strict-JSON safe.

    NaN must become ``null``, not the literal ``NaN`` that Python's json module
    emits by default - bare NaN is not valid JSON and every consumer outside
    Python chokes on it. An undefined confidence interval on an n=1 group is
    exactly the case that produces one.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if np.isnan(value) or np.isinf(value) else value
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    print("Loading data...")
    transfers = load_transfers()
    fifa = load_fifa(years=(2015, 2016, 2017, 2018, 2019))

    # Keep only the seasons where a FIFA edition released *before* the window
    # exists, so no rating post-dates the move it describes.
    seasons = list(config.TRAIN_SEASONS) + [config.HOLDOUT_SEASON]
    transfers = transfers[transfers["Season_transferred"].isin(seasons)].copy()
    transfers = transfers.dropna(subset=["Transfer_fee", "Name"])
    transfers = transfers[transfers["Age"] > 0]
    transfers["Transfer_fee_in_mln"] = transfers["Transfer_fee"] / 1e6
    transfers = transfers.reset_index(drop=True)
    print(f"  transfers in scope ({min(seasons)}-{max(seasons)}): {len(transfers)}")
    print(f"  FIFA player-season rows: {len(fifa)}")

    # ---------------------------------------------------------------- joins
    print("\nJoining transfers to FIFA player-seasons (strict, identity-evidenced)...")
    matched = strict_join(transfers, fifa)
    matched["Region"] = matched["Nationality"].map(to_region)
    n_matched = int((matched["match_tier"] != "unmatched").sum())
    print(f"  matched {n_matched}/{len(matched)} ({n_matched / len(matched):.1%})")
    print("  by tier:")
    for tier, count in matched["match_tier"].value_counts().items():
        print(f"    {tier:16s} {count:5d}")

    # Attrition accounting - the ~5-in-6 rows that vanish are not a random
    # sample, so the rate is reported by every axis the audit disaggregates on.
    attrition_frames = []
    for axis in ("League_from", "League_to", "Season_transferred"):
        tbl = attrition_table(matched, by=axis)
        tbl.insert(0, "axis", axis)
        tbl = tbl.rename(columns={axis: "value"})
        attrition_frames.append(tbl)
    attrition = pd.concat(attrition_frames, ignore_index=True)
    attrition.to_csv(RESULTS / "attrition.csv", index=False)

    # Surname-collision rate by region: quantifies the join bias rather than
    # asserting it. Region here comes from the matched rows only, so it is a
    # lower bound on the true rate.
    region_lookup = matched.set_index(matched.index)["Region"]
    amb_input = transfers.copy()
    amb_input["Region"] = region_lookup.reindex(amb_input.index).fillna("Unmapped")
    ambiguity = ambiguity_rate_by(amb_input, fifa, by="Region")
    ambiguity.to_csv(RESULTS / "ambiguity.csv", index=False)
    print("\nSurname-collision (ambiguous-match) rate by region:")
    print(ambiguity.to_string(index=False))

    model_df = matched[matched["match_tier"] != "unmatched"].copy().reset_index(drop=True)

    # ------------------------------------------------------ temporal split
    print(f"\nTemporal split: train {config.TRAIN_SEASONS} / test {config.HOLDOUT_SEASON}")
    split = temporal_split(model_df["Season_transferred"])
    print(f"  train {len(split.train_idx)} rows | test {len(split.test_idx)} rows")

    # --------------------------------------------- fold-safe aggregates
    agg = FoldSafeAggregates().fit(model_df.loc[split.train_idx])
    model_df = agg.transform(model_df)
    featured = build_ten_factors(model_df)

    X = featured[TEN_FACTORS].astype(float)
    X = X.fillna(X.loc[split.train_idx].median())
    y_raw = featured["Transfer_fee_in_mln"].astype(float)
    seasons_col = featured["Season_transferred"]

    X_train, X_test = X.loc[split.train_idx], X.loc[split.test_idx]
    y_train_raw, y_test_raw = y_raw.loc[split.train_idx], y_raw.loc[split.test_idx]
    regions_test = featured.loc[split.test_idx, "Region"].to_numpy()

    # ------------------------------------------------ models: raw vs deflated
    # Deflation is the default, not an experimental appendix: the median fee
    # runs 7 -> 10 -> 13 -> 9 M across these seasons and season is not a
    # feature, so without it the drift is absorbed into player attributes.
    print("\nTraining models...")
    results = {}

    rf_raw = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    rf_raw.fit(X_train, y_train_raw)
    pred_raw = rf_raw.predict(X_test)
    results["rf_raw_fee"] = pred_raw

    y_train_def = agg.deflate(y_train_raw, seasons_col.loc[split.train_idx])
    rf_def = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    rf_def.fit(X_train, y_train_def)
    pred_def = agg.reflate(rf_def.predict(X_test), seasons_col.loc[split.test_idx]).to_numpy()
    results["rf_season_deflated"] = pred_def

    # ------------------------------------------------------------ scorecards
    scorecards = [
        build_scorecard(name, y_test_raw.to_numpy(), preds, regions_test)
        for name, preds in results.items()
    ]
    scorecard_df = pd.DataFrame([s.as_dict() for s in scorecards])
    print("\n=== FINAL AUDIT SCORECARD (computed) ===")
    print(scorecard_df.to_string(index=False))

    primary_name = "rf_season_deflated"
    primary_pred = results[primary_name]
    primary_card = next(s for s in scorecards if s.model == primary_name)

    # ------------------------------------------------------- region stats
    region_table = group_residual_table(y_test_raw.to_numpy(), primary_pred, regions_test)
    print("\nPer-region residuals on the held-out season, with 95% bootstrap CIs:")
    print(
        region_table[
            ["group", "n", "mean_residual", "residual_ci_low", "residual_ci_high", "verdict"]
        ].to_string(index=False)
    )

    region_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": primary_name,
        "holdout_season": config.HOLDOUT_SEASON,
        "min_support": config.MIN_SUPPORT,
        "bootstrap_samples": config.BOOTSTRAP_SAMPLES,
        "confidence_level": config.CONFIDENCE_LEVEL,
        "global_mean_residual": region_table.attrs["global_mean_residual"],
        "global_mae": region_table.attrs["global_mae"],
        "units": "millions of euros, residual = predicted - actual",
        "regions": region_table.to_dict(orient="records"),
    }
    _dump(RESULTS / "region_stats.json", region_payload)

    _dump(
        RESULTS / "scorecard.json",
        {
            "generated_utc": region_payload["generated_utc"],
            "audit_config": config.AUDIT_CONFIG,
            "primary_model": primary_name,
            "verdict": primary_card.status,
            "models": [card.as_dict() for card in scorecards],
        },
    )

    # ------------------------------------------- proxy-leakage permutation
    # Replaces the README's claim that a low SHAP rank for Passport_Premium is
    # evidence of merit-based inputs. It is not: nationality can enter through
    # league and club. This measures how much of the residual gap each feature
    # actually carries.
    print("\nProxy-leakage permutation test (how much of the max regional gap each feature carries):")
    # Uses the raw-fee model so predictions and residuals share one scale.
    leak = permutation_proxy_leakage(rf_raw, X_test, y_test_raw, regions_test, n_repeats=5)
    leak.to_csv(RESULTS / "proxy_leakage.csv", index=False)
    print(leak.head(10).to_string(index=False))

    _dump(
        RESULTS / "run_metadata.json",
        {
                "generated_utc": region_payload["generated_utc"],
                "python": platform.python_version(),
                "packages": {
                    m.__name__: getattr(m, "__version__", "?")
                    for m in (np, pd, __import__("sklearn"))
                },
                "n_transfers_in_scope": int(len(transfers)),
                "n_matched": n_matched,
                "match_rate": float(n_matched / len(transfers)),
                "n_train": int(len(split.train_idx)),
                "n_test": int(len(split.test_idx)),
            "audit_config": config.AUDIT_CONFIG,
        },
    )

    print(f"\nWrote results to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
