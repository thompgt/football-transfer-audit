"""Group fairness metrics with bootstrap confidence intervals.

The audit protocol has always *declared* 95% bootstrap intervals and a minimum
support of 30 (see ``docs/fairness-metrics-plan.md`` and the notebook's audit
policy cell). Neither was implemented: group means were reported as bare point
estimates over as few as 10 observations. This module implements what was
promised, and makes it impossible to report a group statistic without also
reporting its interval and its support verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fta.config import (
    BIAS_GAP_THRESHOLD,
    BOOTSTRAP_SAMPLES,
    CONFIDENCE_LEVEL,
    MAE_RATIO_THRESHOLD,
    MIN_REPORTABLE,
    MIN_SUPPORT,
    RANDOM_SEED,
)

CONCLUSIVE = "conclusive"
INCONCLUSIVE = f"inconclusive (n < {MIN_SUPPORT})"
NOT_REPORTED = f"not reported (n < {MIN_REPORTABLE})"


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
):
    """Percentile bootstrap interval for ``statistic`` over ``values``.

    Returns ``(point_estimate, low, high)``. Resampling is with replacement at
    the original sample size, which is the right thing for the heavy-tailed
    transfer-fee residuals here: the interval widens honestly when a group's
    mean is being carried by one or two record fees.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    point = float(statistic(arr))
    if arr.size == 1:
        return (point, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    stats = statistic(arr[idx], axis=1)
    alpha = 1.0 - confidence
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(low), float(high))


def support_verdict(n: int) -> str:
    if n < MIN_REPORTABLE:
        return NOT_REPORTED
    if n < MIN_SUPPORT:
        return INCONCLUSIVE
    return CONCLUSIVE


def group_residual_table(
    y_true,
    y_pred,
    groups,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Per-group signed residual and MAE, each with a 95% bootstrap interval.

    Residual is defined as ``predicted - actual`` throughout the repository, so
    a positive mean residual means the model *overvalues* that group.

    Every row carries ``n`` and a ``verdict``. Groups with n < MIN_SUPPORT are
    kept in the table - hiding them would be its own distortion - but marked
    inconclusive so no downstream consumer can quote them as a finding.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "group": np.asarray(groups),
        }
    )
    df["residual"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["residual"].abs()

    global_mae = float(df["abs_error"].mean())
    global_residual = float(df["residual"].mean())

    rows = []
    for group, sub in df.groupby("group", sort=False):
        n = len(sub)
        res_pt, res_lo, res_hi = bootstrap_ci(sub["residual"], n_samples=n_samples, seed=seed)
        mae_pt, mae_lo, mae_hi = bootstrap_ci(sub["abs_error"], n_samples=n_samples, seed=seed)
        # A group's residual is only "significant" if its interval excludes the
        # global mean residual - not zero. Comparing to zero would flag every
        # group whenever the model has an overall offset.
        significant = bool(
            not np.isnan(res_lo) and (res_lo > global_residual or res_hi < global_residual)
        )
        rows.append(
            {
                "group": group,
                "n": n,
                "mean_residual": res_pt,
                "residual_ci_low": res_lo,
                "residual_ci_high": res_hi,
                "residual_gap": res_pt - global_residual,
                "residual_ci_excludes_global": significant,
                "MAE": mae_pt,
                "mae_ci_low": mae_lo,
                "mae_ci_high": mae_hi,
                "mae_ratio": mae_pt / global_mae if global_mae else np.nan,
                "verdict": support_verdict(n),
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_residual").reset_index(drop=True)
    out.attrs["global_mae"] = global_mae
    out.attrs["global_mean_residual"] = global_residual
    return out


def conclusive(table: pd.DataFrame) -> pd.DataFrame:
    """Rows that clear the minimum-support bar."""
    return table[table["verdict"] == CONCLUSIVE]


@dataclass
class Scorecard:
    """The audit verdict for one model, computed - never transcribed."""

    model: str
    rmse: float
    mae: float
    r2: float
    max_signed_residual_gap: float
    max_mae_ratio: float
    n_conclusive_groups: int
    n_inconclusive_groups: int
    status: str
    recommendation: str
    basis: str

    def as_dict(self):
        return asdict(self)


def build_scorecard(model_name: str, y_true, y_pred, groups, **kwargs) -> Scorecard:
    """Compute a scorecard row from live predictions.

    Cell 58 of ``cleaned_model_pipeline.ipynb`` used to hardcode every number
    here except the baseline RMSE, with the comment "# From previous results",
    and those literals produced the FAIL / Conditional Deployment verdict the
    README reported. This function derives all of them, and refuses to base a
    verdict on groups that do not meet minimum support.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    table = group_residual_table(y_true, y_pred, groups, **kwargs)
    solid = conclusive(table)

    if solid.empty:
        max_gap = float("nan")
        max_ratio = float("nan")
        status = "INCONCLUSIVE"
        recommendation = (
            f"No group reaches the minimum support of n={MIN_SUPPORT} on the "
            "held-out set; the audit cannot certify or reject this model on "
            "regional fairness."
        )
        basis = "0 groups at or above minimum support"
    else:
        max_gap = float(solid["residual_gap"].abs().max())
        max_ratio = float(solid["mae_ratio"].max())
        gap_fail = max_gap > BIAS_GAP_THRESHOLD
        ratio_fail = max_ratio > MAE_RATIO_THRESHOLD
        # A threshold breach only counts as a finding if the group's interval
        # actually excludes the global mean. Otherwise the "breach" is noise.
        breaching = solid[solid["residual_gap"].abs() > BIAS_GAP_THRESHOLD]
        supported = bool(breaching["residual_ci_excludes_global"].any())

        if (gap_fail or ratio_fail) and supported:
            status = "FAIL"
            recommendation = (
                "Do not deploy: at least one adequately-powered group shows a "
                "residual gap beyond threshold whose confidence interval "
                "excludes the global mean."
            )
        elif gap_fail or ratio_fail:
            status = "UNDETERMINED"
            recommendation = (
                "Point estimates breach threshold but every breaching group's "
                "confidence interval contains the global mean - the disparity "
                "is not distinguishable from sampling noise at this sample size."
            )
        else:
            status = "PASS"
            recommendation = (
                "No adequately-powered group breaches the residual-gap or "
                "MAE-ratio threshold."
            )
        basis = f"{len(solid)} groups at or above minimum support (n>={MIN_SUPPORT})"

    return Scorecard(
        model=model_name,
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        max_signed_residual_gap=max_gap,
        max_mae_ratio=max_ratio,
        n_conclusive_groups=int(len(solid)),
        n_inconclusive_groups=int(len(table) - len(solid)),
        status=status,
        recommendation=recommendation,
        basis=basis,
    )


def assert_scorecard_fresh(scorecard: Scorecard, expected: dict, tol: float = 1e-6) -> None:
    """Fail loudly if a committed scorecard no longer matches a live rerun.

    Guards against the failure mode this repository actually shipped: numbers
    frozen into a cell, the pipeline changing underneath them, and the stale
    figures continuing to drive the published verdict.
    """
    live = scorecard.as_dict()
    drift = {
        k: (expected[k], live[k])
        for k in expected
        if k in live
        and not (
            isinstance(live[k], float)
            and isinstance(expected[k], (int, float))
            and (
                abs(live[k] - expected[k]) <= tol
                or (np.isnan(live[k]) and np.isnan(float(expected[k])))
            )
        )
        and live[k] != expected[k]
    }
    if drift:
        raise AssertionError(
            "Scorecard has drifted from the committed figures: "
            + "; ".join(
                f"{key}: committed={committed!r} live={live_value!r}"
                for key, (committed, live_value) in drift.items()
            )
        )


def permutation_proxy_leakage(model, X, y, sensitive_series, n_repeats: int = 20, seed: int = RANDOM_SEED):
    """How much of a group's residual gap survives permuting a candidate proxy.

    The README used to read a low SHAP rank for ``Passport_Premium`` as
    evidence that the model was merit-based. A low importance on an *explicit*
    nationality flag says nothing about nationality entering through league or
    club. This measures the thing the claim needs: for each feature, shuffle it
    and see how much the max signed residual gap across groups moves.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(sensitive_series)

    base = group_residual_table(y, model.predict(X), groups)
    base_gap = float(conclusive(base)["residual_gap"].abs().max()) if not conclusive(base).empty else np.nan

    rows = []
    for col in X.columns:
        gaps = []
        for _ in range(n_repeats):  # noqa: B007
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            tbl = group_residual_table(y, model.predict(X_perm), groups, n_samples=1)
            solid = conclusive(tbl)
            if not solid.empty:
                gaps.append(float(solid["residual_gap"].abs().max()))
        if gaps:
            rows.append(
                {
                    "feature": col,
                    "baseline_max_gap": base_gap,
                    "permuted_max_gap": float(np.mean(gaps)),
                    "gap_removed": base_gap - float(np.mean(gaps)),
                }
            )
    return pd.DataFrame(rows).sort_values("gap_removed", ascending=False).reset_index(drop=True)
