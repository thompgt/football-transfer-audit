"""Render the headline fairness figure from computed results.

This file used to open with a hand-transcribed dict::

    region_stats = {
        'Africa': (-0.2470, 11),
        'Other': (2.5050, 14),
        'South America': (-1.2996, 20),
        'W. Europe': (-0.3660, 60),
    }

Four numbers, copied out of a notebook cell by hand, presented as the audit's
central result and reproduced in the README. Nothing connected them to the
model: rerunning the pipeline could not change the published figure, and did
not, even when the pipeline changed.

The figure now reads ``results/region_stats.json``, written by
``scripts/run_audit.py``. It refuses to render if that file is missing, and it
draws the 95% bootstrap intervals and the minimum-support verdict, so a bar
built on 19 transfers cannot be read as if it were built on 200.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = REPO_ROOT / "results" / "region_stats.json"
OUTPUT_PATH = REPO_ROOT / "fairness_audit.png"


def load_region_stats(path: Path = RESULTS_PATH) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\n"
            "The headline figure is computed, not transcribed. Run:\n"
            "    python scripts/run_audit.py\n"
            "and try again."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    payload = load_region_stats()
    regions = [r for r in payload["regions"] if r["verdict"] != f"not reported (n < 5)"]
    regions = [r for r in regions if r["residual_ci_low"] is not None]
    if not regions:
        raise SystemExit("No region clears the reporting floor; nothing to plot.")
    regions.sort(key=lambda r: r["mean_residual"])

    global_mean = payload["global_mean_residual"]
    labels = [f"{r['group']}\n(n={r['n']})" for r in regions]
    values = [r["mean_residual"] for r in regions]
    lo = [r["mean_residual"] - r["residual_ci_low"] for r in regions]
    hi = [r["residual_ci_high"] - r["mean_residual"] for r in regions]
    conclusive = [r["verdict"] == "conclusive" for r in regions]

    # Adequately-powered groups get solid bars; under-powered ones are hatched
    # and greyed, so the eye cannot mistake a 19-transfer estimate for a finding.
    colors = ["#2a6f97" if ok else "#b8b8b8" for ok in conclusive]
    hatches = ["" if ok else "//" for ok in conclusive]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#2b2b2b", linewidth=0.8, zorder=3)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.errorbar(
        labels, values, yerr=[lo, hi], fmt="none",
        ecolor="#2b2b2b", elinewidth=1.4, capsize=6, capthick=1.4, zorder=4,
    )
    ax.axhline(0, color="#2b2b2b", linewidth=1.0, zorder=2)
    ax.axhline(
        global_mean, color="#c9184a", linewidth=1.4, linestyle="--", zorder=2,
        label=f"Global mean residual ({global_mean:+.2f} M€)",
    )

    ax.set_title(
        "Mean valuation residual by player origin region\n"
        f"held-out {payload['holdout_season']} season, model: {payload['model']}, "
        f"{int(payload['confidence_level'] * 100)}% bootstrap CIs "
        f"({payload['bootstrap_samples']} resamples)",
        fontsize=12.5, fontweight="bold",
    )
    ax.set_ylabel("Mean residual, predicted − actual fee (millions €)")
    ax.set_xlabel(
        "Player origin region — solid = meets minimum support "
        f"(n ≥ {payload['min_support']}); hatched grey = inconclusive"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.text(
        0.5, -0.02,
        "A bar is only evidence of a disparity if its interval excludes the global mean line. "
        f"Generated from results/region_stats.json ({payload['generated_utc']}).",
        ha="center", fontsize=8.5, color="#555555",
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    print(f"Wrote {OUTPUT_PATH}")

    for r in regions:
        excludes = r["residual_ci_excludes_global"]
        print(
            f"  {r['group']:16s} n={r['n']:4d}  {r['mean_residual']:+.3f} M€ "
            f"[{r['residual_ci_low']:+.3f}, {r['residual_ci_high']:+.3f}]  "
            f"{r['verdict']}"
            f"{'  <- interval excludes global mean' if excludes else ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
