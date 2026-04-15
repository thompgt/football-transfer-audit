# Fairness metrics plan: transfer-fee model audit

This document records **which metrics we will use**, **what they mean**, and **why we chose them** when evaluating the **existing Random Forest regressor** that predicts player transfer fees. The goal is not only strong aggregate fit but **whether errors are distributed fairly across demographics**—avoiding a model that is accurate overall yet **systematically over- or under-predicts** certain groups or inflicts **larger mistakes** on some groups than others.

---

## Scope and model under audit

| Choice | Decision |
|--------|----------|
| **Model** | The **original Random Forest** already used in the project notebooks (not the linear baseline). |
| **Target** | Observed transfer fee (continuous). |
| **Predictions** | Out-of-sample only: e.g. the held-out **test set** used after `train_test_split`, or—if we refactor—**out-of-fold** predictions from cross-validation so every row has a single honest \(\hat{y}\). |
| **Subgroups** | **`League_from`**, **`Nationality`**, and optionally **`Region`** (manual mapping from nationality when country-level sample sizes are too small). |

We will **pre-specify** a **minimum sample size** per reported cell (e.g. minimum \(n\) transfers per nationality or per league) so rankings are not driven by noise.

---

## Metrics

### 1. Signed residual (mean and median)

**Definition.** For each transfer \(i\),

\[
\text{residual}_i = \hat{y}_i - y_i
\]

where \(y_i\) is the actual fee and \(\hat{y}_i\) is the RF prediction.

- **Positive** residual → the model **overpredicts** the fee (predicted higher than paid).
- **Negative** residual → the model **underpredicts** the fee.

We summarize by subgroup \(g\) (e.g. each nationality, each `League_from`, each region):

- **Mean residual** \(\mathbb{E}[\text{residual} \mid g]\) — average direction and size of systematic bias.
- **Median residual** — robust alternative when a few mega-transfers skew the mean.

**Optional (heavy-tailed fees).** **Log-scale residual**:

\[
\log(1 + \hat{y}_i) - \log(1 + y_i)
\]

interprets bias in **multiplicative** terms and can be less dominated by record-breaking fees.

**Why we chose it.** This is the direct analogue of “who is overpredicted vs underpredicted” in tools like **COMPAS**, adapted to **regression**: we care whether some demographics are consistently shifted **high** or **low** relative to reality, even when global \(R^2\) or RMSE looks acceptable.

---

### 2. Mean absolute error (MAE) and root mean squared error (RMSE) by subgroup

**Definition.** Within each subgroup \(g\):

- **MAE**\(_g = \mathbb{E}\big[\lvert \hat{y} - y \rvert \mid g\big]\)
- **RMSE**\(_g = \sqrt{\mathbb{E}\big[(\hat{y} - y)^2 \mid g\big]}\)

**Why we chose it.** **Signed** metrics alone can hide a problem: groups with **near-zero mean residual** might still suffer **much larger typical errors** (noisier or less reliable predictions). Reporting **MAE** and **RMSE** by `League_from`, nationality, and region answers: *who pays the price in sheer prediction error magnitude?* That aligns with prioritizing **error disparities** over a single headline accuracy figure.

---

### 3. Calibration-style check (predicted-fee bins)

**Definition.** Partition the evaluation set by **predicted fee** \(\hat{y}\) (e.g. **deciles** or fixed monetary bins). Within each bin \(b\), compare across groups:

- mean **actual** fee \(\mathbb{E}[y \mid b, g]\), and/or  
- mean **residual** \(\mathbb{E}[\hat{y} - y \mid b, g]\).

**Why we chose it.** COMPAS-style critiques often ask: *at comparable predicted risk, do realized outcomes (or errors) differ by group?* Here, “comparable predicted risk” becomes **comparable predicted fee bin**. If, within the same bin of \(\hat{y}\), some nationalities or leagues systematically show **different actual fees** or **different residuals**, the model is **miscalibrated** across those groups at that level of predicted fee—distinct from overall mean residual.

---

### 4. (Optional) Threshold-based “error rates”

**Definition.** If the narrative needs a literal **rate** (proportion), define a **clear operational rule**, for example:

- **Large error:** \(\lvert \hat{y} - y \rvert > \tau\) for a chosen \(\tau\) (e.g. in millions or as a fraction of \(y\)), or  
- **High predicted fee:** \(\hat{y}\) in the top decile, then report subgroup fractions.

Report **false-like** constructs only with explicit \(\tau\) and justification.

**Why it is optional.** Regression does not natively yield a single “positive class”; thresholds introduce **policy choices**. We use this only when we need to mirror **classification-style** reporting or stakeholder language.

---

## What we are not claiming

- These metrics **do not prove** discrimination or intent; they **describe** whether the **RF’s errors** differ across **observed** groupings.
- Differences can reflect **legitimate structure** (e.g. unmodeled factors correlated with league or nationality), **data quality**, or **selection** into transfers—not only model bias. The write-up should discuss **confounding** and **missing covariates** where relevant.
- **Nationality** and **league** are **proxies** for complex social and market structure; **region** aggregates are a **stability** device, not a substitute for careful interpretation at country level when \(n\) allows.

---

## Evaluation workflow (next steps)

1. Reproduce or load **RF predictions** on the chosen **out-of-sample** rows.
2. Join **true fee**, **prediction**, **`League_from`**, **`Nationality`** (and **region** if used).
3. Filter to subgroups meeting **minimum \(n\)**.
4. Produce tables and plots for: **mean/median residual**, **MAE/RMSE**, and **binned calibration** comparisons.
5. Narrate **which groups** are most over- vs underpredicted and where **error magnitude** is largest, tying back to this plan.

---

## Summary table

| Metric | What it answers | Why it was chosen |
|--------|-----------------|-------------------|
| Mean / median signed residual | Systematic over- vs underprediction by group | Core “COMPAS spirit” for regression; detects demographic skew in direction of error |
| MAE / RMSE by group | Who gets larger mistakes | Captures unequal error severity not visible from mean residual alone |
| Residuals within predicted-fee bins | Fairness at comparable predicted fee | Mirrors calibration / “same score, different outcomes” reasoning |
| Log residuals (optional) | Multiplicative bias under heavy tails | Stabilizes interpretation when a few huge fees dominate |
| Threshold “rates” (optional) | Proportions above a defined error bar | Only when we need explicit rates; requires justified \(\tau\) |

This file should stay aligned with the notebooks and any **data dictionary** updates (e.g. engineered nationality fields and merge provenance) as the audit proceeds.
