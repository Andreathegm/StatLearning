# Statistical Learning — Course Projects

**University of Florence (UniFi)**
Department of Statistics, Computer Science, Applications "G. Parenti"

---

## Overview

This repository collects the coursework and independent projects developed during the Statistical Learning course at the University of Florence. The work spans three main areas: confidence interval estimation, high-dimensional variable selection, nonparametric regression with splines, and Bayesian causal inference via Bayesian Causal Forests. Each project is self-contained, with its own data-generating processes, estimation routines, and diagnostic tools.

---

## Repository Structure

```
.
├── Homework1/                  # Confidence interval coverage: Wald vs Wilson
├── Homework2_spline/           # Spline regression on the Boston dataset
├── Contest/                    # High-dimensional variable selection via multi-split
└── BCF_on_treatment_effect/    # Bayesian Causal Forests: BART, ps-BART, XBCF
```

---

## Projects

### Homework 1 — Confidence Interval Coverage

**Directory:** `Homework1/`

A Monte Carlo study of the actual coverage probability of two confidence intervals for a Bernoulli proportion: the Wald interval and the Wilson (score) interval. The study is conducted across a grid of sample sizes and simulation counts to quantify how quickly each method converges to the nominal 95% level, with particular attention to boundary cases (p near 0 or 1).


---

### Homework 2 — Spline Regression on the Boston Dataset

**Directory:** `Homework2_spline/`

Comparison of ordinary linear regression against natural and B-spline additive models for predicting the average number of rooms per dwelling (`rm`) in the Boston housing dataset. Model selection is performed via 5-fold cross-validation over a range of spline degrees of freedom, and the selected model is evaluated on a held-out test set.

---

### Contest — High-Dimensional Variable Selection via Multi-Split

**Directory:** `Contest/`

An implementation and empirical evaluation of the multi-split procedure for controlled variable selection in the high-dimensional regime (p >> n). The project reproduces the FWER and FDR analysis from the selective inference literature, comparing single-split with multi-split aggregation across Monte Carlo replications.

---

### BCF on Treatment Effect — Bayesian Causal Forests

**Directory:** `BCF_on_treatment_effect/`

The main project of the course. A Monte Carlo simulation study comparing three methods for estimating heterogeneous treatment effects (CATE) and the average treatment effect (ATE) under targeted selection and regularisation-induced confounding, as studied in Hahn, Murray and Carvalho (2020).

**Methods compared:**

- **BART naive** — Bayesian Additive Regression Trees with treatment indicator included as a raw covariate.
- **ps-BART** — BART with the estimated propensity score appended as an additional covariate.
- **XBCF** — Accelerated Bayesian Causal Forests, which explicitly separates the prognostic function mu(x) from the treatment effect function tau(x).

**Reference:** Hahn, P.R., Murray, J.S. and Carvalho, C.M. (2020). Bayesian Regression Tree Models for Causal Inference: Regularization, Confounding, and Heterogeneous Treatment Effects. *Bayesian Analysis*, 15(3), 965-1056.

---

## Dependencies

All code is written in R. The following packages are required:

| Package | Used in |
|---|---|
| `dbarts` | BCF project — BART and ps-BART fitting |
| `XBCF` | BCF project — XBCF fitting |
| `glmnet` | Contest — Lasso variable selection |
| `hdi` | Contest — Multi-split via third-party implementation |
| `huge` | Contest — Synthetic covariate generation |
| `MASS` | Homework 2 — Boston dataset |
| `splines` | Homework 2 — `ns()` and `bs()` bases |
| `gam` | Homework 2 — Generalised Additive Models |
| `ggplot2` | All projects — plotting |
| `patchwork` | BCF project — multi-panel figures |
| `dplyr` / `tidyr` | BCF project — data manipulation |
| `scales` | BCF project — axis formatting |
| `progress` | BCF project — progress bar |
| `plotly` | BCF project — interactive 3D surfaces |

Install all packages at once with:

```r
install.packages(c(
  "dbarts", "XBCF", "glmnet", "hdi", "huge",
  "MASS", "splines", "gam", "ggplot2", "patchwork",
  "dplyr", "tidyr", "scales", "progress", "plotly"
))
```

---

## Reproducibility

All stochastic components are controlled via explicit `set.seed()` calls. The BCF simulation additionally saves a checkpoint `.rds` file after each Monte Carlo replication, so a run that is interrupted can be inspected up to the last completed iteration. Final aggregated results and diagnostic images are written to `result_experiment/<dgp_name>/` subdirectories.

---

Statistical Learning course project, University of Florence (UniFi).
Academic year 2024/2025.
