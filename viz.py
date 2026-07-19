import matplotlib.pyplot as plt

# Mean prediction residual (predicted - actual fee, in millions of euros) by player
# origin region, taken directly from the fairness diagnostics in
# notebooks/cleaned_model_pipeline.ipynb (Simpson's Paradox Diagnostics cell) for the
# "rf_weighted" model, the best-performing fairness-aware variant selected in the
# in-processing model comparison. Regions with fewer than 10 held-out test
# transfers are dropped to match the minimum-sample-size rule used throughout the
# audit (see docs/fairness-metrics-plan.md).
#
# region: (mean_residual, n_holdout_transfers)
region_stats = {
    'Africa': (-0.2470, 11),
    'Other': (2.5050, 14),
    'South America': (-1.2996, 20),
    'W. Europe': (-0.3660, 60),
}

regions = [f"{r}\n(n={n})" for r, (_, n) in region_stats.items()]
bias_scores = [val for val, _ in region_stats.values()]

plt.figure(figsize=(10, 6))
plt.bar(regions, bias_scores, color=['#4caf50' if s > 0 else '#f44336' for s in bias_scores])
plt.axhline(0, color='black', linewidth=1)
plt.title("Random Forest ADS: Mean Valuation Residual by Player Origin Region")
plt.ylabel('Mean Residual, Predicted − Actual Fee (Millions €)')
plt.xlabel('Player Origin Region (held-out test set)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('fairness_audit.png', bbox_inches='tight')
