# =============================================================================
#  Inspired by Hahn, Murray & Carvalho (2020)
#  "Bayesian Regression Tree Models for Causal Inference:
#   Regularization, Confounding, and Heterogeneous Treatment Effects"
#
#  FOCUS: Targeted Selection + Regularization-Induced Confounding (RIC)
#
#  Methods compared:
#    1. BART naïve  — no propensity score (dbarts)
#    2. ps-BART     — propensity score as covariate (dbarts)
#    3. XBCF        — mu/tau separation
#
#  Available DGPs (via ACTIVE_DGP):
#    "no_confounding"      — baseline
#    "targeted_selection"  — RIC ps-BART & XBCF >> naive (in theory)
#    "paper_example1"      — RIC with homogeneous tau, pure confounding
#    "ht_nl_clean"         — mu non-linear + heterogeneous tau on x4 (orthogonal to pi)
#    "ht_nl"               — x5 appears in {mu, tau, pi} -> collinearity
# =============================================================================

# ── 0. Dependencies ───────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(dbarts)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(progress)
  library(XBCF)
})

# =============================================================================
#  GLOBAL PARAMETERS
# =============================================================================
ACTIVE_DGP   <- "targeted_selection"   # change this to run one specific DGP
# To run all DGPs in one batch, see the block at the end of the file.

N_MC     <- 100                     # Monte Carlo replications
N        <- 250                    # sample size

# BART / ps-BART
NDPOST       <- 600                    # post burn-in draws
NSKIP        <- 200                    # burn-in
NTREE        <- 200                    # number of trees for outcome model

# XBCF
XBCF_BURNIN  <- 200
XBCF_MCMC    <- 600
XBCF_MU_TREES  <- 200
XBCF_TAU_TREES <- 50

set.seed(1234)

# =============================================================================
#  SECTION 1: DGPs
# =============================================================================
dgp_bcf_winner_fixed <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
  colnames(X) <- paste0("x", 1:5)
  
  mu <- sin(pi * X[,"x1"]) + X[,"x2"]
  
  pi_raw <- pnorm(mu - 0.5 * X[,"x4"])
  pi_true <- pmin(pmax(pi_raw, 0.05), 0.95)
  Z <- rbinom(n, 1, pi_true)
  
  tau_true <- 0.5 + 0.5 * X[,"x4"]
  
  Y <- mu + tau_true * Z + rnorm(n, 0, 1)
  
  list(X = X, Z = Z, Y = Y, mu_true = mu, pi_true = pi_true, 
       tau_true = tau_true, sigma_true = 1, ATE_true = mean(tau_true),
       dgp_label = "BCF Winner Fixed", tau_type  = "heterogeneous")
}

dgp_bcf_winner <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
  colnames(X) <- paste0("x", 1:5)
  
  mu <- 3 * sin(X[,"x1"]) + 2 * X[,"x2"]^2 - 2 * X[,"x3"]
  
  pi_raw <- pnorm(0.6 * mu - 0.2 * X[,"x4"])
  pi_true <- pmin(pmax(pi_raw, 0.05), 0.95)
  
  Z <- rbinom(n, 1, pi_true)
  
  tau_true <- 0.2 + 0.3 * X[,"x4"]
  
  Y <- mu + tau_true * Z + rnorm(n, 0, 1)
  
  list(
    X = X, Z = Z, Y = Y,
    mu_true = mu, pi_true = pi_true, tau_true = tau_true,
    sigma_true = 1, ATE_true = mean(tau_true),
    dgp_label = "BCF Winner (High Confounding, Small Tau)",
    tau_type  = "heterogeneous"
  )
}

dgp_no_confounding <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- data.frame(
    x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n),
    x4 = sample(c(0, 1),    size = n, replace = TRUE),
    x5 = sample(c(1, 2, 3), size = n, replace = TRUE)
  )
  
  g_func   <- ifelse(X$x5 == 1, 2, ifelse(X$x5 == 2, -1, -4))
  mu       <- -6 + g_func + 6 * abs(X$x3 - 1)
  tau_true <- 1 + 2 * X$x2 * X$x4
  
  pi_true  <- pmin(pmax(0.5 + rnorm(n, 0, 0.05), 0.1), 0.9)
  
  Z <- rbinom(n, 1, pi_true)
  Y <- mu + tau_true * Z + rnorm(n)
  
  list(
    X = as.matrix(X), Z = Z, Y = Y,
    mu_true = mu, pi_true = pi_true, tau_true = tau_true,
    sigma_true = 1, ATE_true = mean(tau_true),
    dgp_label = "no_confounding",
    tau_type  = "heterogeneous"
  )
}

dgp_targeted_selection <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- matrix(rnorm(n * 3), nrow = n, ncol = 3)
  colnames(X) <- c("x1", "x2", "x3")
  
  q        <- -1 * (X[,1] > X[,2]) + 1 * (X[,1] < X[,2])
  pi_true  <- pnorm(q)
  Z        <- rbinom(n, 1, pi_true)
  
  tau_true <- 0.5  * (X[,3] > -3/4) +
    0.25 * (X[,3] >  0)   +
    0.25 * (X[,3] >  3/4)
  
  sigma_true <- diff(range(q + tau_true * pi_true)) / 8
  Y          <- q + tau_true * Z + rnorm(n, sd = sigma_true)
  
  list(
    X          = X, Z = Z, Y = Y,
    mu_true    = q,
    pi_true    = pi_true, tau_true   = tau_true,
    sigma_true = sigma_true, ATE_true   = mean(tau_true),
    dgp_label  = "Targeted Selection",
    tau_type   = "heterogeneous"
  )
}

dgp_paper_example1 <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- data.frame(x1 = runif(n, 0, 1), x2 = runif(n, 0, 1))
  mu <- -3 + 6 * pnorm(2 * (X$x1 - X$x2))
  tau_true <- rep(-1, n)
  
  pi_raw <- 0.8 * pnorm(mu / (0.1 * (2 - X$x1 - X$x2) + 0.25)) +
    0.025 * (X$x1 + X$x2) + 0.05
  pi_true <- pmin(pmax(pi_raw, 0.01), 0.99)
  
  Z <- rbinom(n, size = 1, prob = pi_true)
  Y <- mu + tau_true * Z + rnorm(n, mean = 0, sd = 1)
  
  list(
    X          = as.matrix(X), Z = Z, Y = Y,
    mu_true    = mu, pi_true = pi_true, tau_true = tau_true,
    sigma_true = 1, ATE_true = mean(tau_true),
    dgp_label  = "Paper Example 1", tau_type = "homogeneous"
  )
}

dgp_ht_nl_clean <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- data.frame(
    x1 = rnorm(n, 0, 1),
    x2 = rnorm(n, 0, 1),
    x3 = rnorm(n, 0, 1),
    x4 = sample(c(0, 1),    size = n, replace = TRUE),
    x5 = sample(c(1, 2, 3), size = n, replace = TRUE)
  )
  
  g_func <- ifelse(X$x5 == 1, 2, ifelse(X$x5 == 2, -1, -4))
  mu     <- -6 + g_func + 6 * abs(X$x3 - 1)
  
  tau_true <- 1 + 2 * X$x2 * X$x4
  
  s_mu    <- sd(mu)
  u       <- runif(n, 0, 1)
  pi_raw  <- 0.8 * pnorm((3 * mu / s_mu) - 0.5 * X$x1) + 0.05 + u / 10
  pi_true <- pmin(pmax(pi_raw, 0.01), 0.99)
  
  Z <- rbinom(n, size = 1, prob = pi_true)
  Y <- mu + tau_true * Z + rnorm(n, mean = 0, sd = 1)
  
  list(
    X          = as.matrix(X),
    Z          = Z,
    Y          = Y,
    mu_true    = mu,
    pi_true    = pi_true,
    tau_true   = tau_true,
    sigma_true = 1,
    ATE_true   = mean(tau_true),
    dgp_label  = "ht_nl_clean (tau on x4, orthogonal to pi)",
    tau_type   = "heterogeneous"
  )
}

dgp_ht_nl <- function(n, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  X <- data.frame(
    x1 = rnorm(n, 0, 1), x2 = rnorm(n, 0, 1), x3 = rnorm(n, 0, 1),
    x4 = sample(c(0, 1),    size = n, replace = TRUE),
    x5 = sample(c(1, 2, 3), size = n, replace = TRUE)
  )
  
  g_func <- ifelse(X$x5 == 1, 2, ifelse(X$x5 == 2, -1, -4))
  mu     <- -6 + g_func + 6 * abs(X$x3 - 1)
  tau_true <- 1 + 2 * X$x2 * X$x5
  
  s_mu   <- sd(mu)
  u      <- runif(n, 0, 1)
  pi_raw <- 0.8 * pnorm((3 * mu / s_mu) - 0.5 * X$x1) + 0.05 + u / 10
  pi_true <- pmin(pmax(pi_raw, 0.01), 0.99)
  
  Z <- rbinom(n, size = 1, prob = pi_true)
  Y <- mu + tau_true * Z + rnorm(n, mean = 0, sd = 1)
  
  list(
    X          = as.matrix(X), Z = Z, Y = Y,
    mu_true    = mu, pi_true = pi_true, tau_true = tau_true,
    sigma_true = 1, ATE_true = mean(tau_true),
    dgp_label  = "ht_nl_dgp", tau_type = "heterogeneous"
  )
}

generate_data <- function(n, seed, dgp_name) {
  switch(dgp_name,
         "no_confounding"     = dgp_no_confounding(n, seed),
         "targeted_selection" = dgp_targeted_selection(n, seed),
         "paper_example1"     = dgp_paper_example1(n, seed),
         "ht_nl_clean"        = dgp_ht_nl_clean(n, seed),
         "ht_nl"              = dgp_ht_nl(n, seed),
         "dgp_bcf_winner"     = dgp_bcf_winner(n,seed),
         "dgp_winner_fixed"   = dgp_bcf_winner_fixed(n,seed),
         stop("DGP not recognized.")
  )
}

# =============================================================================
#  SECTION 2: FITTING FUNCTIONS — BART, ps-BART and XBCF
# =============================================================================

# ── 2A: BART naïve ───────────────────────────────────────────────────────────
fit_bart_naive <- function(dat, ndpost = NDPOST, nskip = NSKIP, ntree = NTREE) {
  X_train <- cbind(dat$X, Z = dat$Z)
  X_w1    <- X_train; X_w1[, "Z"] <- 1
  X_w0    <- X_train; X_w0[, "Z"] <- 0
  
  fit <- bart(x.train = X_train, y.train = dat$Y, x.test = rbind(X_w1, X_w0),
              ndpost = ndpost, nskip = nskip, ntree = ntree, verbose = FALSE)
  
  n          <- nrow(dat$X)
  ite_post   <- fit$yhat.test[, 1:n] - fit$yhat.test[, (n+1):(2*n)]
  
  list(
    ite_post   = ite_post, ate_post = rowMeans(ite_post),
    cate_mean  = colMeans(ite_post), cate_lower = apply(ite_post, 2, quantile, 0.025),
    cate_upper = apply(ite_post, 2, quantile, 0.975), varcount = fit$varcount
  )
}

# ── 2B: ps-BART ──────────────────────────────────────────────────────────────
# Requires pre-computed pi_hat
fit_ps_bart <- function(dat, pi_hat, ndpost = NDPOST, nskip = NSKIP, ntree = NTREE) {
  X_aug   <- cbind(dat$X, ps = pi_hat, Z = dat$Z)
  X_w1    <- X_aug; X_w1[, "Z"] <- 1
  X_w0    <- X_aug; X_w0[, "Z"] <- 0
  
  fit <- bart(x.train = X_aug, y.train = dat$Y, x.test = rbind(X_w1, X_w0),
              ndpost = ndpost, nskip = nskip, ntree = ntree, verbose = FALSE)
  
  n          <- nrow(dat$X)
  ite_post   <- fit$yhat.test[, 1:n] - fit$yhat.test[, (n+1):(2*n)]
  
  list(
    ite_post   = ite_post, ate_post = rowMeans(ite_post), cate_mean = colMeans(ite_post),
    cate_lower = apply(ite_post, 2, quantile, 0.025), cate_upper = apply(ite_post, 2, quantile, 0.975),
    varcount = fit$varcount
  )
}

# ── 2C: fit_xbcf wrapper ─────────────────────────────────────────────────────
# Requires pre-computed pi_hat
fit_xbcf <- function(dat, pi_hat, num_sweeps = XBCF_MCMC + XBCF_BURNIN, burnin = XBCF_BURNIN) {
  
  # Order variables: continuous first, categorical last
  is_cat <- apply(dat$X, 2, function(x) length(unique(x)) <= 5)
  X_cont <- dat$X[, !is_cat, drop = FALSE]
  X_cat  <- dat$X[, is_cat, drop = FALSE]
  X_ordered <- cbind(X_cont, X_cat)
  pcat <- ncol(X_cat)
  
  # XBCF call
  fit <- XBCF(
    y          = matrix(dat$Y,  ncol = 1),
    z          = matrix(dat$Z,  ncol = 1),
    x_mod      = X_ordered,
    x_con      = X_ordered,
    pihat      = matrix(pi_hat, ncol = 1),
    pcat_con   = pcat,
    pcat_mod   = pcat,
    n_trees_con = 200L,
    n_trees_mod = 50L,
    alpha_con  = 0.95,
    beta_con   = 2.0,
    alpha_mod  = 0.25,
    beta_mod   = 3.0,
    pr_scale   = TRUE,
    trt_scale  = TRUE,
    num_sweeps = num_sweeps,
    burnin     = burnin
  )
  
  # Transpose to match BART output format [ndpost x n]
  ite_post <- t(fit$tauhats.adjusted)
  
  list(
    ite_post   = ite_post,
    ate_post   = rowMeans(ite_post),
    cate_mean  = colMeans(ite_post),
    cate_lower = apply(ite_post, 2, quantile, 0.025),
    cate_upper = apply(ite_post, 2, quantile, 0.975)
  )
}

# ── 2D: Metrics for a single replication ─────────────────────────────────────
compute_metrics <- function(fit_out, dat) {
  ate_true  <- dat$ATE_true
  tau_true  <- dat$tau_true
  ate_mean  <- mean(fit_out$ate_post)
  ate_lower <- quantile(fit_out$ate_post, 0.025)
  ate_upper <- quantile(fit_out$ate_post, 0.975)
  
  data.frame(
    ate_bias     = ate_mean - ate_true,
    ate_sqerr    = (ate_mean - ate_true)^2,   # for RMSE aggregation
    ate_covered  = as.numeric(ate_lower <= ate_true & ate_true <= ate_upper),
    ate_len      = as.numeric(ate_upper - ate_lower),
    cate_rmse    = sqrt(mean((fit_out$cate_mean - tau_true)^2)),
    cate_covered = mean(fit_out$cate_lower <= tau_true & tau_true <= fit_out$cate_upper),
    cate_len     = mean(fit_out$cate_upper - fit_out$cate_lower)
  )
}

# =============================================================================
#  SECTION 3: MAIN EXPERIMENT FUNCTION (one DGP)
# =============================================================================
run_experiment <- function(dgp_name, n_mc = N_MC, n_obs = N, seed_base = 1234,use_pi_true = TRUE) {
  
  cat(strrep("=", 65), "\n")
  cat(sprintf("  SIMULATION — DGP: '%s'\n", dgp_name))
  cat(sprintf("  %d replications, n = %d\n", n_mc, n_obs))
  cat(strrep("=", 65), "\n\n")
  
  pb <- progress_bar$new(
    format = "  [:bar] :current/:total  ETA: :eta",
    total = n_mc, clear = FALSE, width = 55
  )
  pb$tick(0)
  
  # Output directories
  base_dir <- file.path("result_experiment", dgp_name)
  num_dir  <- file.path(base_dir, "numeric")
  img_dir  <- file.path(base_dir, "images")
  dir.create(num_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)
  
  results_list <- vector("list", n_mc)
  
  for (r in 1:n_mc) {
    pb$tick()
    dat <- generate_data(n = n_obs, seed = seed_base + r, dgp_name = dgp_name)
    
    # Shared propensity score estimation via BART
    if (!use_pi_true){
      print("estimating pi")
      ps_fit <- bart(x.train = dat$X, y.train = dat$Z,
                     ndpost = 400, nskip = 100, ntree = 50, verbose = FALSE)
      pi_hat <- colMeans(pnorm(ps_fit$yhat.train))
      
    }
    else{
      print("Using pi true...")
      pi_hat <- dat$pi_true
    }
    
    bart_out   <- tryCatch(fit_bart_naive(dat), error = function(e) NULL)
    psbart_out <- tryCatch(fit_ps_bart(dat, pi_hat), error = function(e) NULL)
    xbcf_out   <- tryCatch(fit_xbcf(dat, pi_hat), error = function(e) NULL)
    
    if (is.null(bart_out) || is.null(psbart_out) || is.null(xbcf_out)) {
      warning(sprintf("Replication %d: fitting error, skipped.", r))
      next
    }
    
    results_list[[r]] <- bind_rows(
      compute_metrics(bart_out,   dat) %>% mutate(method = "BART naïve"),
      compute_metrics(psbart_out, dat) %>% mutate(method = "ps-BART"),
      compute_metrics(xbcf_out,   dat) %>% mutate(method = "XBCF")
    ) %>% mutate(replication = r, dgp = dgp_name)
  }
  
  results_mc <- bind_rows(results_list)
  
  # ── Summary table ─────────────────────────────────────────────────────────
  cat("\n\n")
  cat(strrep("-", 65), "\n")
  cat(sprintf("  RESULTS — DGP: '%s'\n", dgp_name))
  cat(strrep("-", 65), "\n")
  
  summary_tbl <- results_mc %>%
    group_by(method) %>%
    summarise(
      ATE_bias   = round(mean(ate_bias),     4),
      ATE_RMSE   = round(sqrt(mean(ate_sqerr)), 4),
      ATE_cover  = round(mean(ate_covered),  3),
      ATE_len    = round(mean(ate_len),      4),
      CATE_RMSE  = round(mean(cate_rmse),    4),
      CATE_cover = round(mean(cate_covered), 3),
      CATE_len   = round(mean(cate_len),     4),
      .groups    = "drop"
    )
  
  print(as.data.frame(summary_tbl))
  cat(strrep("-", 65), "\n")
  

  # Save numeric results
  write.csv(results_mc,  file.path(num_dir, "mc_results.csv"), row.names = TRUE)
  write.csv(summary_tbl, file.path(num_dir, "summary_table.csv"), row.names = TRUE)
  cat(sprintf("  Numeric results saved in %s\n\n\n", num_dir))
  
  # ── Demo dataset for diagnostics (n=500) ────────────────────────────────
  cat(strrep("=", 65), "\n")
  cat("  DIAGNOSTICS — single demo replication (n = 500)\n")
  cat(strrep("=", 65), "\n\n")
  
  dat_demo <- generate_data(n = 500, seed = 1234, dgp_name = dgp_name)
  
  ps_fit_demo <- bart(x.train = dat_demo$X, y.train = dat_demo$Z,
                      ndpost = 400, nskip = 100, ntree = 50, verbose = FALSE)
  pi_hat_demo <- colMeans(pnorm(ps_fit_demo$yhat.train))
  
  psbart_demo <- fit_ps_bart(dat_demo, pi_hat_demo, ndpost = 800, nskip = 300, ntree = 200)
  bart_demo   <- fit_bart_naive(dat_demo, ndpost = 800, nskip = 300, ntree = 200)
  xbcf_demo   <- fit_xbcf(dat_demo, pi_hat_demo, num_sweeps = 1100, burnin = 300)
  
  # ── PS diagnostics ──────────────────────────────────────────────────────
  brier_score  <- mean((pi_hat_demo - dat_demo$Z)^2)
  brier_oracle <- mean((dat_demo$pi_true - dat_demo$Z)^2)
  cor_ps       <- cor(dat_demo$pi_true, pi_hat_demo)
  
  cal_df <- data.frame(ps_hat = pi_hat_demo, pi_true = dat_demo$pi_true, Z = dat_demo$Z) %>%
    mutate(decile = ntile(ps_hat, 10)) %>%
    group_by(decile) %>%
    summarise(mean_ps_hat = mean(ps_hat), obs_rate = mean(Z),
              mean_pi_true = mean(pi_true), .groups = "drop")
  
  # =========================================================================
  #  SECTION 6: PLOTS — THEME AND PALETTE
  # =========================================================================
  pal <- c("BART naïve" = "#E05C5C", "ps-BART" = "#3A86FF", "XBCF" = "#00C698")
  
  theme_paper <- theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(size = 10, color = "gray40"),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      strip.text       = element_text(face = "bold")
    )
  
  dgp_subtitle <- sprintf("DGP: '%s'  |  %d MC replications, n = %d",
                          dgp_name, n_mc, n_obs)
  
  # ── PANEL A — MC results ────────────────────────────────────────────────
  p1 <- results_mc %>%
    ggplot(aes(x = cate_covered, fill = method, color = method)) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    geom_vline(xintercept = 0.95, linetype = "dashed", color = "black", linewidth = 0.8) +
    scale_fill_manual(values = pal) + scale_color_manual(values = pal) +
    labs(title = "Coverage 95% CI — CATE", subtitle = "Dashed line = nominal 95%",
         x = "Coverage", y = "Density", fill = NULL, color = NULL) + theme_paper
  
  p2 <- results_mc %>%
    ggplot(aes(x = method, y = cate_rmse, fill = method)) +
    geom_boxplot(alpha = 0.7, width = 0.4, outlier.shape = 21, outlier.size = 1.5) +
    scale_fill_manual(values = pal) +
    labs(title = "RMSE(CATE)", subtitle = "Lower is better",
         x = NULL, y = "RMSE", fill = NULL) +
    theme_paper + theme(legend.position = "none")
  
  ate_cov_avg <- results_mc %>%
    group_by(method) %>%
    summarise(cov = mean(ate_covered), .groups = "drop")
  
  p3 <- results_mc %>%
    arrange(replication) %>%
    group_by(method) %>%
    mutate(cum_cov = cummean(ate_covered)) %>%
    ggplot(aes(x = replication, y = cum_cov, color = method)) +
    geom_line(linewidth = 0.9) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "black") +
    scale_color_manual(values = pal) +
    scale_y_continuous(limits = c(0, 1.02)) +
    labs(title = "cumulative ATE",
         x = "Replication", y = "Cumulative Cover", color = NULL) +
    theme_paper
  
  p4 <- results_mc %>%
    ggplot(aes(x = ate_bias, fill = method, color = method)) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
    scale_fill_manual(values = pal) + scale_color_manual(values = pal) +
    labs(title = "ATE bias distribution",
         subtitle = "Dashed line = zero bias (ideal)",
         x = "Bias (estimate − true)", y = "Density", fill = NULL, color = NULL) + theme_paper
  
  panel_A <- (p1 | p2) / (p3 | p4) +
    plot_annotation(title = "Panel A — Monte Carlo Simulation Results",
                    subtitle = dgp_subtitle)
  
  # ── PANEL B — Propensity score diagnostics ──────────────────────────────
  ps_diag_df <- data.frame(
    pi_true  = dat_demo$pi_true,
    ps_hat   = pi_hat_demo,
    treated  = factor(dat_demo$Z, labels = c("Z=0 (Control)", "Z=1 (Treated)"))
  )
  
  p5 <- ps_diag_df %>%
    ggplot(aes(x = pi_true, y = ps_hat, color = treated)) +
    geom_point(alpha = 0.35, size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.9) +
    scale_color_manual(values = c("gray50", "#3A86FF")) +
    labs(title = "Propensity Score: true vs estimated",
         x = "π_true", y = "π̂", color = NULL) + theme_paper
  
  p6 <- ps_diag_df %>%
    ggplot(aes(x = ps_hat, fill = treated, color = treated)) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    scale_fill_manual(values  = c("gray60", "#3A86FF")) +
    scale_color_manual(values = c("gray60", "#3A86FF")) +
    labs(title = "Overlap: PS distribution by group",
         x = "Estimated PS", y = "Density", fill = NULL, color = NULL) + theme_paper
  
  p7 <- cal_df %>%
    pivot_longer(cols = c(obs_rate, mean_pi_true),
                 names_to  = "series",
                 values_to = "prob") %>%
    mutate(series = recode(series,
                           "obs_rate"     = "observed π",
                           "mean_pi_true" = "true π"
    )) %>%
    ggplot(aes(x = mean_ps_hat, y = prob, color = series)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 3) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("observed π" = "#E05C5C",
                                  "true π"          = "#3A86FF")) +
    labs(title = "PS calibration plot",
         x = "PS stimato medio (decile)", y = "Probabilità", color = NULL) +
    theme_paper
  
  ts_df <- data.frame(
    mu = dat_demo$mu_true, pi_true = dat_demo$pi_true,
    Z  = factor(dat_demo$Z, labels = c("Z=0", "Z=1"))
  )
  
  p8 <- ts_df %>%
    ggplot(aes(x = mu, y = pi_true, color = Z)) +
    geom_point(alpha = 0.4, size = 1.5) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1) +
    scale_color_manual(values = c("gray50", "#E05C5C")) +
    labs(title = sprintf("μ(x) vs π(x) — %s", dgp_name),
         x = "μ(x)", y = "π(x)", color = NULL) + theme_paper
  
  panel_B <- (p5 | p6) / (p7 | p8) +
    plot_annotation(title = "Panel B — Propensity Score Diagnostics")
  
  # ── PANEL C — MCMC diagnostics ──────────────────────────────────────────
  ate_trace_psbart <- rowMeans(psbart_demo$ite_post)
  ate_trace_xbcf   <- rowMeans(xbcf_demo$ite_post)
  ate_true_d       <- dat_demo$ATE_true
  
  trace_df <- bind_rows(
    data.frame(iter = seq_along(ate_trace_psbart), ate = ate_trace_psbart,
               method = "ps-BART"),
    data.frame(iter = seq_along(ate_trace_xbcf), ate = ate_trace_xbcf,
               method = "XBCF")
  )
  
  p_trace <- ggplot(trace_df, aes(x = iter, y = ate, color = method)) +
    geom_line(alpha = 0.6, linewidth = 0.4) +
    geom_hline(yintercept = ate_true_d, color = "red", linetype = "dashed",
               linewidth = 1) +
    scale_color_manual(values = pal) +
    labs(title = "MCMC trace — ATE", subtitle = "Red line = true ATE",
         x = "Iteration", y = "ATE") + theme_paper
  
  p_post <- ggplot(trace_df, aes(x = ate, fill = method)) +
    geom_density(alpha = 0.4, linewidth = 0.8) +
    geom_vline(xintercept = ate_true_d, color = "red", linetype = "dashed",
               linewidth = 1.2) +
    scale_fill_manual(values = pal) +
    labs(title = "Posterior distribution of ATE", x = "ATE", y = "Density") + theme_paper
  
  
  varimp_bart   <- colMeans(bart_demo$varcount)
  varimp_psbart <- colMeans(psbart_demo$varcount)
  
  varimp_df <- bind_rows(
    data.frame(
      variable = names(varimp_bart),
      count    = varimp_bart / sum(varimp_bart),   
      method   = "BART naïve"
    ),
    data.frame(
      variable = names(varimp_psbart),
      count    = varimp_psbart / sum(varimp_psbart),
      method   = "ps-BART"
    )
  )
  
  p_varimp <- varimp_df %>%
    ggplot(aes(x = reorder(variable, count), y = count, fill = method)) +
    geom_col(position = "dodge", alpha = 0.8, width = 0.6) +
    coord_flip() +
    scale_fill_manual(values = pal) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(title    = "Variable importance",
         x = NULL, y = "% split", fill = NULL) +
    theme_paper
  
  panel_C <- (p_trace | p_post) / p_varimp +
    plot_annotation(title = "Panel C — MCMC Diagnostics and Variable Importance")
  
  # ── Save plots ────────────────────────────────────────────────────────────
  ggsave(file.path(img_dir, "panel_A_mc_results.png"),  panel_A,
         width = 14, height = 10, dpi = 150)
  ggsave(file.path(img_dir, "panel_B_ps_diag.png"),     panel_B,
         width = 14, height = 10, dpi = 150)
  ggsave(file.path(img_dir, "panel_C_mcmc_diag.png"),   panel_C,
         width = 14, height = 10, dpi = 150)
  
  cat(sprintf("\n  ✓ Images saved in %s\n", img_dir))
  
  invisible(list(mc_results = results_mc, summary = summary_tbl))
}

# =============================================================================
#  EXECUTION: single DGP (as set by ACTIVE_DGP)
# =============================================================================
dgp_list <- c("no_confounding","targeted_selection","paper_example1",
  "ht_nl_clean","ht_nl","dgp_bcf_winner,dgp_winner_fixed")
for (dgp in dgp_list) {
   cat("\n\n>>> Starting DGP:", dgp, "\n")
   run_experiment(dgp)
}