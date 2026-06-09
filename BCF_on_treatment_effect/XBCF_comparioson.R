# ==========================================
# FILE: main_simulation_xbcf.R
# Main script to run experiments and test models (BART, PS-BART, XBCF)
# ==========================================

# --- Package loading ---
#.libPaths(~/Rlibs)
library(dbarts)
library(XBCF)

source("DGP_library.R")

# ==========================================
# HELPER: Prepare X matrix for XBCF
# ==========================================
# XBCF requires:
#   1. A numeric matrix (no data.frame, no factors)
#   2. Categorical columns placed at the END
#   3. pcat = number of those trailing categorical columns
#
# For DGPs with 2 continuous variables (dgp_paper_example1, dgp_simple):
#   x1, x2 -> all continuous -> pcat = 0
#
# For DGPs with 5 variables (easier_dgp, dgp, dgp_enriched, ht_l_dgp, ht_nl_dgp):
#   Continuous : x1, x2, x3
#   Binary     : x4 (0/1)          -> 1 categorical column
#   3-level cat: x5 (1,2,3)        -> expanded to x5.2, x5.3 via dummy encoding
#   Final order: x1, x2, x3 | x4, x5.2, x5.3
#   pcat = 3

prepare_xbcf_matrix <- function(X) {
  # Detect whether we are in the simple (2-var) or complex (5-var) DGP
  if (ncol(X) == 2) {
    # dgp_paper_example1 / dgp_simple: both variables are continuous
    X_mat  <- as.matrix(X)
    pcat   <- 0L
  } else {
    # Complex DGPs: x1, x2, x3 continuous; x4 binary; x5 3-level categorical
    X_tmp       <- X
    X_tmp$x5    <- as.factor(X_tmp$x5)                        # tell R x5 is categorical
    X_expanded  <- model.matrix(~ . - 1, data = X_tmp)        # expands x5 into x5.2, x5.3
    
    # Identify column groups
    cont_cols   <- c("x1", "x2", "x3")
    cat_cols    <- setdiff(colnames(X_expanded), cont_cols)    # x4, x5.2, x5.3
    
    # Reorder: continuous first, categorical last (required by XBCF)
    X_mat       <- as.matrix(X_expanded[, c(cont_cols, cat_cols)])
    pcat        <- length(cat_cols)                            # = 3
  }
  
  return(list(X_mat = X_mat, pcat = as.integer(pcat)))
}

# ==========================================
# HELPER: Estimate propensity score via BART
# ==========================================
estimate_pi <- function(X, Z) {
  fit <- bart(x.train = as.matrix(X), y.train = Z,ndpost = 1000, nskip = 500,verbose = FALSE)
  pmin(pmax(colMeans(pnorm(fit$yhat.train)), 0.01), 0.99)
  # NOTE: dbarts returns yhat on probit scale for binary outcomes,
  #       so we apply pnorm() to convert to probabilities.
}

# ==========================================
# Experiment Settings
# ==========================================
n_simulations <- 100

# XBCF hyperparameters
# num_sweeps = total MCMC iterations; burnin = warm-up to discard
# Effective posterior draws = num_sweeps - burnin
xbcf_num_sweeps <- 2000
xbcf_burnin     <- 1000   # must be strictly < num_sweeps

# BART hyperparameters (kept for vanilla BART and PS-BART)
nskip_grid   <- c(1000)
ndpost_fixed <- 2000

# ==========================================
# DGP SELECTION
# ==========================================
# We focus on scenarios where BCF/XBCF has a structural advantage over vanilla BART:
#
# dgp_paper_example1 : classic BCF benchmark; strong confounding via mu entering pi.
#                      BCF designed exactly for this setup -> expected best performer.
#
# dgp_enriched       : like dgp but propensity score also depends on x4 (extra confounding).
#                      The richer pi -> pi_hat overlap helps BCF more than BART.
#
# ht_l_dgp           : heterogeneous treatment effect, linear functional form.
#                      BCF's dedicated moderation tree should capture tau(x) well.
#
# ht_nl_dgp          : heterogeneous treatment effect, non-linear functional form.
#                      Hardest case; tests whether XBCF's speed allows more sweeps
#                      to compensate for the extra complexity.
#
# Excluded DGPs and reasons:
#   dgp_simple   : tau is constant (-1) and mu is very simple; BART/BCF gap is minimal.
#   easier_dgp   : homogeneous tau, easier mu; no meaningful heterogeneity to exploit.
#   dgp          : redundant with dgp_enriched for our purposes.

experiments_to_run <- c(
  "dgp_paper_example1",
  "dgp_enriched",
  "ht_l_dgp",
  "ht_nl_dgp"
)

# ==========================================
# MAIN LOOP
# ==========================================
start_time <- Sys.time()
cat(paste0("Experiment started at ", start_time, "\n"))

for (dgp_name in experiments_to_run) {
  
  cat("\n================================================\n")
  cat("=== Running experiment on DGP:", dgp_name, "===\n")
  cat("================================================\n")
  
  data_generator <- match.fun(dgp_name)
  
  # Pre-allocate metric matrices (rows = simulations, cols = nskip settings)
  make_metrics <- function() {
    list(
      bias     = matrix(NA, n_simulations, length(nskip_grid)),
      rmse     = matrix(NA, n_simulations, length(nskip_grid)),
      coverage = matrix(NA, n_simulations, length(nskip_grid)),
      length   = matrix(NA, n_simulations, length(nskip_grid)),
      time     = matrix(NA, n_simulations, length(nskip_grid)),
      tau_est  = vector("list", n_simulations),  # ŌåÉ aggiunto: lista di vettori
      tau_true = vector("list", n_simulations)   # ŌåÉ aggiunto: tau vero per ogni sim
    )
  }
  
  metrics_bart   <- make_metrics()
  metrics_ps_bart <- make_metrics()
  metrics_xbcf   <- make_metrics()
  
  for (i in 1:n_simulations) {
    cat(sprintf("\n==> Simulation %d / %d [%s]\n", i, n_simulations,
                format(Sys.time(), "%H:%M:%S")))
    
    # Generate data
    data    <- data_generator(n = 250, seed = i)
    pi_hat  <- estimate_pi(data$X, data$Z)
    n_obs   <- nrow(data$X)
    
    # Prepare matrices for BART
    X_train_bart  <- cbind(data$X, Z = data$Z)
    X_test1_bart  <- cbind(data$X, Z = 1)
    X_test0_bart  <- cbind(data$X, Z = 0)
    
    # Prepare matrices for PS-BART (propensity score added as covariate)
    X_train_psbart <- cbind(data$X, pihat = pi_hat, Z = data$Z)
    X_test1_psbart <- cbind(data$X, pihat = pi_hat, Z = 1)
    X_test0_psbart <- cbind(data$X, pihat = pi_hat, Z = 0)
    
    # Prepare matrices for XBCF (dummy-encoded, categoricals last)
    xbcf_prep <- prepare_xbcf_matrix(data$X)
    X_xbcf    <- xbcf_prep$X_mat
    pcat      <- xbcf_prep$pcat
    
    for (j in 1:length(nskip_grid)) {
      current_nskip <- nskip_grid[j]
      
      # -------------------------------------------------------
      # 1. BART VANILLA
      # -------------------------------------------------------
      start_bart <- Sys.time()
      invisible(capture.output({
        fit_bart <- bart(
          x.train = X_train_bart,
          y.train = data$Y,
          x.test  = rbind(X_test1_bart, X_test0_bart),
          nskip   = current_nskip,
          ndpost  = ndpost_fixed,
          verbose = FALSE
        )
      }))
      time_bart <- as.numeric(difftime(Sys.time(), start_bart, units = "secs"))
      
      cate_draws_bart <- fit_bart$yhat.test[, 1:n_obs] - fit_bart$yhat.test[, (n_obs + 1):(2 * n_obs)]
      cate_est_bart   <- colMeans(cate_draws_bart)
      intervals_bart  <- apply(cate_draws_bart, 2, quantile, probs = c(0.025, 0.975))
      
      # BART salva tau_est e tau_true
      metrics_bart$tau_est[[i]]  <- cate_est_bart
      metrics_bart$tau_true[[i]] <- data$tau_true
      
      metrics_bart$bias[i, j]     <- mean(abs(cate_est_bart - data$tau_true))
      metrics_bart$rmse[i, j]     <- sqrt(mean((cate_est_bart - data$tau_true)^2))
      metrics_bart$coverage[i, j] <- mean(intervals_bart[1, ] <= data$tau_true &
                                            intervals_bart[2, ] >= data$tau_true)
      metrics_bart$length[i, j]   <- mean(intervals_bart[2, ] - intervals_bart[1, ])
      metrics_bart$time[i, j]     <- time_bart
      
      cat(sprintf("  [%s] [BART]    Fitted in %6.1fs | Bias: %.4f | RMSE: %.4f | Cov: %.3f\n",
                  format(Sys.time(), "%H:%M:%S"), time_bart,
                  metrics_bart$bias[i, j], metrics_bart$rmse[i, j], metrics_bart$coverage[i, j]))
      
      # -------------------------------------------------------
      # 2. PS-BART
      # -------------------------------------------------------
      start_psbart <- Sys.time()
      invisible(capture.output({
        fit_psbart <- bart(
          x.train = X_train_psbart,
          y.train = data$Y,
          x.test  = rbind(X_test1_psbart, X_test0_psbart),
          nskip   = current_nskip,
          ndpost  = ndpost_fixed,
          verbose = FALSE
        )
      }))
      time_psbart <- as.numeric(difftime(Sys.time(), start_psbart, units = "secs"))
      
      cate_draws_psbart <- fit_psbart$yhat.test[, 1:n_obs] - fit_psbart$yhat.test[, (n_obs + 1):(2 * n_obs)]
      cate_est_psbart   <- colMeans(cate_draws_psbart)
      intervals_psbart  <- apply(cate_draws_psbart, 2, quantile, probs = c(0.025, 0.975))
      
      # PS-BART salva tau_est e tau_true
      metrics_ps_bart$tau_est[[i]]  <- cate_est_psbart
      metrics_ps_bart$tau_true[[i]] <- data$tau_true
      
      metrics_ps_bart$bias[i, j]     <- mean(abs(cate_est_psbart - data$tau_true))
      metrics_ps_bart$rmse[i, j]     <- sqrt(mean((cate_est_psbart - data$tau_true)^2))
      metrics_ps_bart$coverage[i, j] <- mean(intervals_psbart[1, ] <= data$tau_true &
                                               intervals_psbart[2, ] >= data$tau_true)
      metrics_ps_bart$length[i, j]   <- mean(intervals_psbart[2, ] - intervals_psbart[1, ])
      metrics_ps_bart$time[i, j]     <- time_psbart
      
      cat(sprintf("  [%s] [PS-BART] Fitted in %6.1fs | Bias: %.4f | RMSE: %.4f | Cov: %.3f\n",
                  format(Sys.time(), "%H:%M:%S"), time_psbart,
                  metrics_ps_bart$bias[i, j], metrics_ps_bart$rmse[i, j], metrics_ps_bart$coverage[i, j]))
      
      # -------------------------------------------------------
      # 3. XBCF
      # -------------------------------------------------------
      # XBCF separates the outcome into:
      #   y = mu(x_con) + tau(x_mod) * z + error
      # x_con = matrix for the prognostic tree (includes pihat as first column)
      # x_mod = matrix for the moderation tree (treatment effect heterogeneity)
      # pihat is passed both as a standalone argument AND prepended to x_con,
      # following the original BCF paper convention.
      
      # IMPORTANT: XBCF already prepends pihat to x_con, so we must NOT do it manually.
      X_con_xbcf <- X_xbcf                      # pihat will be added automatically
      X_mod_xbcf <- X_xbcf                      # moderation tree uses raw covariates
      
      # pcat_con = pcat (pihat is continuous, so it doesn't count as categorical)
      # pcat_mod = pcat (same categorical structure as X_xbcf)
      
      start_xbcf <- Sys.time()
      fit_xbcf <- tryCatch({
        XBCF(
          y          = matrix(data$Y,  ncol = 1),
          z          = matrix(data$Z,  ncol = 1),
          x_mod      = X_mod_xbcf,
          x_con      = X_con_xbcf,
          pihat      = matrix(pi_hat,  ncol = 1),
          pcat_con   = pcat,
          pcat_mod   = pcat,
          
          n_trees_con = 200L,    # Prognostic: default BART
          n_trees_mod = 50L,     # Treatment effect: 
          
          alpha_con = 0.95,      
          beta_con  = 2.0,       
          alpha_mod = 0.25,      
          beta_mod  = 3.0,      
          
          pr_scale   = TRUE,     
          trt_scale  = TRUE,  
          
          num_sweeps = xbcf_num_sweeps,
          burnin     = xbcf_burnin
        )
      }, error = function(e) {
        cat(sprintf("  [XBCF ERROR] Simulation %d: %s\n", i, conditionMessage(e)))
        NULL
      })
      time_xbcf <- as.numeric(difftime(Sys.time(), start_xbcf, units = "secs"))
      
      if (!is.null(fit_xbcf)) {
        cate_draws_xbcf <- t(fit_xbcf$tauhats.adjusted)  # n_draws ├Ś n_obs
        cate_est_xbcf   <- colMeans(cate_draws_xbcf)
        intervals_xbcf  <- apply(cate_draws_xbcf, 2, quantile, probs = c(0.025, 0.975))
        
        # XBCF salva tau_est e tau_true
        metrics_xbcf$tau_est[[i]]  <- cate_est_xbcf
        metrics_xbcf$tau_true[[i]] <- data$tau_true
        
        metrics_xbcf$bias[i, j]     <- mean(abs(cate_est_xbcf - data$tau_true))
        metrics_xbcf$rmse[i, j]     <- sqrt(mean((cate_est_xbcf - data$tau_true)^2))
        metrics_xbcf$coverage[i, j] <- mean(intervals_xbcf[1, ] <= data$tau_true &
                                              intervals_xbcf[2, ] >= data$tau_true)
        metrics_xbcf$length[i, j]   <- mean(intervals_xbcf[2, ] - intervals_xbcf[1, ])
        metrics_xbcf$time[i, j]     <- time_xbcf
        
        cat(sprintf("  [%s] [XBCF]    Fitted in %6.1fs | Bias: %.4f | RMSE: %.4f | Cov: %.3f\n",
                    format(Sys.time(), "%H:%M:%S"), time_xbcf,
                    metrics_xbcf$bias[i, j], metrics_xbcf$rmse[i, j], metrics_xbcf$coverage[i, j]))
      } else {
        metrics_xbcf$time[i, j] <- time_xbcf
        cat(sprintf("  [%s] [XBCF]    FAILED after %6.1fs ŌĆö metrics set to NA\n",
                    format(Sys.time(), "%H:%M:%S"), time_xbcf))
      }
    } # end j loop (nskip_grid)
    
    # -------------------------------------------------------
    # CHECKPOINT: save partial results after every simulation
    # -------------------------------------------------------
    dir_name <- file.path("table_results", dgp_name)
    if (!dir.exists(dir_name)) dir.create(dir_name, recursive = TRUE)
    
    checkpoint_file <- file.path(dir_name, paste0("checkpoint_raw_", dgp_name, ".rds"))
    saveRDS(
      list(
        last_completed_simulation = i,
        settings = list(
          dgp_name         = dgp_name,
          n_simulations    = n_simulations,
          nskip_grid       = nskip_grid,
          ndpost_fixed     = ndpost_fixed,
          xbcf_num_sweeps  = xbcf_num_sweeps,
          xbcf_burnin      = xbcf_burnin
        ),
        metrics_bart    = metrics_bart,
        metrics_ps_bart = metrics_ps_bart,
        metrics_xbcf    = metrics_xbcf
      ),
      file = checkpoint_file
    )
  } # end i loop (simulations)
  
  # -------------------------------------------------------
  # FINAL AGGREGATION
  # -------------------------------------------------------
  summarise_metrics <- function(m, nskip_grid) {
    data.frame(
      nskip        = nskip_grid,
      Bias         = colMeans(m$bias,     na.rm = TRUE),
      RMSE         = colMeans(m$rmse,     na.rm = TRUE),
      Coverage     = colMeans(m$coverage, na.rm = TRUE),
      Length       = colMeans(m$length,   na.rm = TRUE),
      Avg_Time_Secs = colMeans(m$time,    na.rm = TRUE)
    )
  }
  
  final_bart    <- summarise_metrics(metrics_bart,    nskip_grid)
  final_psbart  <- summarise_metrics(metrics_ps_bart, nskip_grid)
  final_xbcf    <- summarise_metrics(metrics_xbcf,    nskip_grid)
  
  cat("\n=== AVERAGE RESULTS BART VANILLA ===\n"); print(final_bart)
  cat("\n=== AVERAGE RESULTS PS-BART ===\n");      print(final_psbart)
  cat("\n=== AVERAGE RESULTS XBCF ===\n");         print(final_xbcf)
  
  # -------------------------------------------------------
  # SAVE RESULTS
  # -------------------------------------------------------
  cat("\nSaving final results...\n")
  timestamp <- format(Sys.time(), "%d%m%Y_%H%M%S")
  
  write.csv(final_bart,   file.path(dir_name, paste0("final_bart_",   dgp_name, "_", timestamp, ".csv")), row.names = TRUE)
  write.csv(final_psbart, file.path(dir_name, paste0("final_psbart_", dgp_name, "_", timestamp, ".csv")), row.names = TRUE)
  write.csv(final_xbcf,   file.path(dir_name, paste0("final_xbcf_",   dgp_name, "_", timestamp, ".csv")), row.names = TRUE)
  
  saveRDS(
    list(
      settings        = list(dgp_name = dgp_name, n_simulations = n_simulations,
                             nskip_grid = nskip_grid, ndpost_fixed = ndpost_fixed,
                             xbcf_num_sweeps = xbcf_num_sweeps, xbcf_burnin = xbcf_burnin),
      metrics_bart    = metrics_bart,
      metrics_ps_bart = metrics_ps_bart,
      metrics_xbcf    = metrics_xbcf
    ),
    file = file.path(dir_name, paste0("raw_metrics_full_", dgp_name, "_", timestamp, ".rds"))
  )
  
  # Remove checkpoint now that final results are saved
  if (file.exists(checkpoint_file)) file.remove(checkpoint_file)
  
  # -------------------------------------------------------
  # DIAGONAL PLOT: tau_true vs tau_estimated
  # -------------------------------------------------------
  # Combina tutte le simulazioni in un unico dataframe
  build_tau_df <- function(metrics, model_name) {
    data.frame(
      tau_true = unlist(metrics$tau_true),
      tau_est  = unlist(metrics$tau_est),
      model    = model_name
    )
  }
  
  df_plot <- rbind(
    build_tau_df(metrics_bart,    "BART"),
    build_tau_df(metrics_ps_bart, "PS-BART"),
    build_tau_df(metrics_xbcf,    "XBCF")
  )
  
  # Limiti comuni per entrambi gli assi
  axis_lim <- range(c(df_plot$tau_true, df_plot$tau_est), na.rm = TRUE)
  
  # Colori per i tre modelli
  colors <- c("BART" = "#E74C3C", "PS-BART" = "#3498DB", "XBCF" = "#2ECC71")
  
  plot_file <- file.path(dir_name, paste0("diagonal_plot_", dgp_name, "_", timestamp, ".png"))
  png(plot_file, width = 800, height = 700, res = 120)
  
  plot(NULL, xlim = axis_lim, ylim = axis_lim,
       xlab = expression(tau ~ "true"),
       ylab = expression(hat(tau) ~ "estimated"),
       main = paste0("True vs Estimated CATE ", dgp_name),
       las = 1)
  
  # Linea diagonale perfetta
  abline(0, 1, col = "black", lwd = 2, lty = 2)
  
  # Punti per ogni modello
  for (mod in c("BART", "PS-BART", "XBCF")) {
    sub <- df_plot[df_plot$model == mod, ]
    points(sub$tau_true, sub$tau_est,
           col = adjustcolor(colors[mod], alpha.f = 0.25),
           pch = 16, cex = 0.6)
  }
  
  # Legenda
  legend("topleft", legend = names(colors), col = colors,
         pch = 16, pt.cex = 1.2, bty = "n")
  
  dev.off()
  cat("Diagonal plot saved:", plot_file, "\n")
  
  cat("Successfully saved in folder:", dir_name, "\n")
  
} # end dgp loop

cat("\nAll experiments completed.\n")
cat(paste0("Total elapsed time: ", round(difftime(Sys.time(), start_time, units = "mins"), 1), " minutes\n"))