# =============================================================================
#  SCRIPT 2: PLOTTING AND DEMO RUN
# =============================================================================
source("bcf_paper_simulation.R")
suppressPackageStartupMessages({
  library(dbarts)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(XBCF)
})

plot_experiment_results <- function(dgp_name, n_mc = 50, n_obs = 250) {
  
  base_dir <- file.path("result_experiment", dgp_name)
  num_dir  <- file.path(base_dir, "numeric")
  img_dir  <- file.path(base_dir, "images")
  
  csv_file <- file.path(num_dir, "mc_results.csv")
  
  if (!file.exists(csv_file)) {
    stop(sprintf("File not found: %s.", csv_file))
  }
  
  results_mc <- read.csv(csv_file, row.names = 1)
  
  pal <- c("BART naïve" = "#E05C5C", "ps-BART" = "#3A86FF", "XBCF" = "#00C698")
  
  theme_paper <- theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(size = 10, color = "gray40"),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      strip.text       = element_text(face = "bold")
    )
  
  dgp_subtitle <- sprintf("DGP: '%s'  |  %d MC replications, n = %d", dgp_name, n_mc, n_obs)
  
  # =========================================================================
  #  PANEL A 
  # =========================================================================
  cat("Panel A generation from csv file...\n")
  
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
         x = NULL, y = "RMSE", fill = NULL) + theme_paper + theme(legend.position = "none")
  
  p3 <- results_mc %>%
    arrange(replication) %>%
    group_by(method) %>%
    mutate(cum_cov = cummean(ate_covered)) %>%
    ggplot(aes(x = replication, y = cum_cov, color = method)) +
    geom_line(linewidth = 0.9) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "black") +
    scale_color_manual(values = pal) +
    scale_y_continuous(limits = c(0, 1.02)) +
    labs(title = "Cumulative ATE coverage",
         x = "Replication", y = "Cumulative Cover", color = NULL) + theme_paper
  
  p4 <- results_mc %>%
    ggplot(aes(x = ate_bias, fill = method, color = method)) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
    scale_fill_manual(values = pal) + scale_color_manual(values = pal) +
    labs(title = "ATE bias distribution",
         subtitle = "Dashed line = zero bias (ideal)",
         x = "Bias (estimate - true)", y = "Density", fill = NULL, color = NULL) + theme_paper
  
  panel_A <- (p1 | p2) / (p3 | p4) +
    plot_annotation(title = "Panel A — Monte Carlo Simulation Results",
                    subtitle = dgp_subtitle)
  
  # =========================================================================
  #  PANEL B & C: one single run (n=500, seed=1234)
  # =========================================================================
  cat("Diagnostic Run (Panel B e C)...\n")
  
  dat_demo <- generate_data(n = 500, seed = 1234, dgp_name = dgp_name)
  
  ps_fit_demo <- bart(x.train = dat_demo$X, y.train = dat_demo$Z,
                      ndpost = 400, nskip = 100, ntree = 50, verbose = FALSE)
  pi_hat_demo <- colMeans(pnorm(ps_fit_demo$yhat.train))
  
  psbart_demo <- fit_ps_bart(dat_demo, pi_hat_demo, ndpost = 800, nskip = 300, ntree = 200)
  bart_demo   <- fit_bart_naive(dat_demo, ndpost = 800, nskip = 300, ntree = 200)
  xbcf_demo   <- fit_xbcf(dat_demo, pi_hat_demo, num_sweeps = 1100, burnin = 300)
  
  # Dati per Panel B
  ps_diag_df <- data.frame(
    pi_true  = dat_demo$pi_true,
    ps_hat   = pi_hat_demo,
    treated  = factor(dat_demo$Z, labels = c("Z=0 (Control)", "Z=1 (Treated)"))
  )
  
  cal_df <- data.frame(ps_hat = pi_hat_demo, pi_true = dat_demo$pi_true, Z = dat_demo$Z) %>%
    mutate(decile = ntile(ps_hat, 10)) %>%
    group_by(decile) %>%
    summarise(mean_ps_hat = mean(ps_hat), obs_rate = mean(Z),
              mean_pi_true = mean(pi_true), .groups = "drop")
  
  p5 <- ps_diag_df %>%
    ggplot(aes(x = pi_true, y = ps_hat, color = treated)) +
    geom_point(alpha = 0.35, size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.9) +
    scale_color_manual(values = c("gray50", "#3A86FF")) +
    labs(title = "Propensity Score: true vs estimated",
         x = expression(pi[true]), y = expression(hat(pi)), color = NULL) + theme_paper
  
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
                           "mean_pi_true" = "true π")) %>%
    ggplot(aes(x = mean_ps_hat, y = prob, color = series)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 3) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("observed π" = "#E05C5C", "true π" = "#3A86FF")) +
    labs(title = "PS calibration plot",
         x = "Mean estimated PS (decile)", y = "Probability", color = NULL) + theme_paper
  
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
  
  ate_trace_psbart <- rowMeans(psbart_demo$ite_post)
  ate_trace_xbcf   <- rowMeans(xbcf_demo$ite_post)
  ate_true_d       <- dat_demo$ATE_true
  
  trace_df <- bind_rows(
    data.frame(iter = seq_along(ate_trace_psbart), ate = ate_trace_psbart, method = "ps-BART"),
    data.frame(iter = seq_along(ate_trace_xbcf), ate = ate_trace_xbcf, method = "XBCF")
  )
  
  p_trace <- ggplot(trace_df, aes(x = iter, y = ate, color = method)) +
    geom_line(alpha = 0.6, linewidth = 0.4) +
    geom_hline(yintercept = ate_true_d, color = "red", linetype = "dashed", linewidth = 1) +
    scale_color_manual(values = pal) +
    labs(title = "MCMC trace — ATE", subtitle = "Red line = true ATE",
         x = "Iteration", y = "ATE") + theme_paper
  
  p_post <- ggplot(trace_df, aes(x = ate, fill = method)) +
    geom_density(alpha = 0.4, linewidth = 0.8) +
    geom_vline(xintercept = ate_true_d, color = "red", linetype = "dashed", linewidth = 1.2) +
    scale_fill_manual(values = pal) +
    labs(title = "Posterior distribution of ATE", x = "ATE", y = "Density") + theme_paper
  
  varimp_bart   <- colMeans(bart_demo$varcount)
  varimp_psbart <- colMeans(psbart_demo$varcount)
  
  varimp_df <- bind_rows(
    data.frame(variable = names(varimp_bart), count = varimp_bart / sum(varimp_bart), method = "BART naïve"),
    data.frame(variable = names(varimp_psbart), count = varimp_psbart / sum(varimp_psbart), method = "ps-BART")
  )
  
  p_varimp <- varimp_df %>%
    ggplot(aes(x = reorder(variable, count), y = count, fill = method)) +
    geom_col(position = "dodge", alpha = 0.8, width = 0.6) +
    coord_flip() +
    scale_fill_manual(values = pal) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(title = "Variable importance", x = NULL, y = "% split", fill = NULL) + theme_paper
  
  panel_C <- (p_trace | p_post) / p_varimp +
    plot_annotation(title = "Panel C — MCMC Diagnostics and Variable Importance")
  

  dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)
  
  ggsave(file.path(img_dir, "panel_A_mc_results_replot.png"), panel_A, width = 14, height = 10, dpi = 150)
  ggsave(file.path(img_dir, "panel_B_ps_diag_replot.png"),    panel_B, width = 14, height = 10, dpi = 150)
  ggsave(file.path(img_dir, "panel_C_mcmc_diag_replot.png"),  panel_C, width = 14, height = 10, dpi = 150)
  
  cat(sprintf("New plot saved in %s\n\n", img_dir))
}

dgp_list <- c("no_confounding", "targeted_selection", "paper_example1", "ht_nl_clean", "ht_nl")

for (dgp in dgp_list) {
  cat("=== Starting plotting DGP (and demo run):", dgp, "===\n")
  tryCatch({
    plot_experiment_results(dgp)
  }, error = function(e) {
    cat("Error for", dgp, ":", e$message, "\n")
  })
}