# ==========================================
# FILE: main_simulation.R
# Main script to run experiments and test models (BART, PS-BART, BCF)
# ==========================================
is_server <- TRUE 

if (is_server) {
  custom_lib <- "~/Rlibs"
  
  if (!dir.exists(custom_lib)) {
    dir.create(custom_lib, recursive = TRUE)
  }
  
  .libPaths(c(custom_lib, .libPaths()))
  cat("Server enviroment: using ", custom_lib," to find packages \n")
} else {
  cat("Local env: using standard lib.\n")
}

if (!require("dbarts", quietly = TRUE)) {
  install.packages("dbarts")
  library(dbarts)
}

if (!require("bcf", quietly = TRUE)) {
  install.packages("bcf")
  library(bcf)
}
source("DGP_library.R")

## Calculate P(Z_i=1 | X_i = x_i) for every i 
estimate_pi = function(X, Z) {
  fit = bart(x.train = as.matrix(X), y.train = Z, verbose = FALSE)
  pmin(pmax(colMeans(fit$yhat.train), 0.01), 0.99)
}

# --- Experiment Settings ---
n_simulations = 50
nskip_grid = c(1000)
ndpost_fixed = 2000

# Define the list of experiments (DGPs) to run sequentially
experiments_to_run = c(
  "dgp_paper_example1", 
  "dgp_simple", 
  "easier_dgp", 
  "dgp", 
  "dgp_enriched",
  "ht_nl_dgp",
  "ht_l_dgp"
)

start_time <- Sys.time()
print(paste0("Experiment started at ", start_time))

for (dgp_name in experiments_to_run) {
  cat("\n================================================\n")
  cat("=== Running experiment on DGP:", dgp_name, "===\n")
  cat("================================================\n")
  
  data_generator = match.fun(dgp_name)
  
  metrics_bart = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                      rmse = matrix(NA, n_simulations, length(nskip_grid)),
                      coverage = matrix(NA, n_simulations, length(nskip_grid)),
                      length = matrix(NA, n_simulations, length(nskip_grid)),
                      time = matrix(NA, n_simulations, length(nskip_grid)))
  
  metrics_ps_bart = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                         rmse = matrix(NA, n_simulations, length(nskip_grid)),
                         coverage = matrix(NA, n_simulations, length(nskip_grid)),
                         length = matrix(NA, n_simulations, length(nskip_grid)),
                         time = matrix(NA, n_simulations, length(nskip_grid)))
  
  metrics_bcf = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                     rmse = matrix(NA, n_simulations, length(nskip_grid)),
                     coverage = matrix(NA, n_simulations, length(nskip_grid)),
                     length = matrix(NA, n_simulations, length(nskip_grid)),
                     time = matrix(NA, n_simulations, length(nskip_grid)))
  
  for(i in 1:n_simulations) {
    cat(sprintf("\n==> Simulation %d / %d [%s]\n", i, n_simulations, format(Sys.time(), "%H:%M:%S")))
    
    data = data_generator(n = 250, seed = i)
    pi_hat = estimate_pi(data$X, data$Z)
    
    X_train_bart = cbind(data$X, Z = data$Z)
    X_test1_bart = cbind(data$X, Z = 1)
    X_test0_bart = cbind(data$X, Z = 0)
    
    X_train_psbart = cbind(data$X, pihat = pi_hat, Z = data$Z)
    X_test1_psbart = cbind(data$X, pihat = pi_hat, Z = 1)
    X_test0_psbart = cbind(data$X, pihat = pi_hat, Z = 0)
    
    X_mat = as.matrix(data$X)
    
    for(j in 1:length(nskip_grid)) {
      current_nskip = nskip_grid[j]
      
      # ---------------------------------------------------------
      # 1. RUN BART VANILLA
      # ---------------------------------------------------------
      start_bart <- Sys.time()
      invisible(capture.output({
        fit_bart = bart(x.train = X_train_bart, y.train = data$Y, 
                        x.test = rbind(X_test1_bart, X_test0_bart), 
                        nskip = current_nskip, ndpost = ndpost_fixed, verbose = FALSE)
      }))
      end_bart <- Sys.time()
      time_bart <- as.numeric(difftime(end_bart, start_bart, units = "secs"))
      
      cate_draws_bart = fit_bart$yhat.test[, 1:250] - fit_bart$yhat.test[, 251:500]
      cate_est_bart = colMeans(cate_draws_bart)
      
      intervals_bart = apply(cate_draws_bart, 2, quantile, probs = c(0.025, 0.975))
      cov_bart = mean(intervals_bart[1, ] <= data$tau_true & intervals_bart[2, ] >= data$tau_true)
      len_bart = mean(intervals_bart[2, ] - intervals_bart[1, ])
      
      metrics_bart$bias[i, j] = mean(abs(cate_est_bart - data$tau_true))
      metrics_bart$rmse[i, j] = sqrt(mean((cate_est_bart - data$tau_true)^2))
      metrics_bart$coverage[i, j] = cov_bart
      metrics_bart$length[i, j] = len_bart
      metrics_bart$time[i, j] = time_bart
      
      # Logging con Timestamp
      timestamp_log <- format(Sys.time(), "%H:%M:%S")
      cat(sprintf("  [%s] [BART]    Fitted in %6.1fs | Bias: %.4f | RMSE: %.4f | Cov: %.3f\n", 
                  timestamp_log, time_bart, metrics_bart$bias[i, j], metrics_bart$rmse[i, j], metrics_bart$coverage[i, j]))
      
      # ---------------------------------------------------------
      # 2. RUN PS-BART
      # ---------------------------------------------------------
      start_psbart <- Sys.time()
      invisible(capture.output({
        fit_psbart = bart(x.train = X_train_psbart, y.train = data$Y, 
                          x.test = rbind(X_test1_psbart, X_test0_psbart), 
                          nskip = current_nskip, ndpost = ndpost_fixed, verbose = FALSE)
      }))
      end_psbart <- Sys.time()
      time_psbart <- as.numeric(difftime(end_psbart, start_psbart, units = "secs"))
      
      cate_draws_psbart = fit_psbart$yhat.test[, 1:250] - fit_psbart$yhat.test[, 251:500]
      cate_est_psbart = colMeans(cate_draws_psbart)
      
      intervals_psbart = apply(cate_draws_psbart, 2, quantile, probs = c(0.025, 0.975))
      cov_psbart = mean(intervals_psbart[1, ] <= data$tau_true & intervals_psbart[2, ] >= data$tau_true)
      len_psbart = mean(intervals_psbart[2, ] - intervals_psbart[1, ])
      
      metrics_ps_bart$bias[i, j] = mean(abs(cate_est_psbart - data$tau_true))
      metrics_ps_bart$rmse[i, j] = sqrt(mean((cate_est_psbart - data$tau_true)^2))
      metrics_ps_bart$coverage[i, j] = cov_psbart
      metrics_ps_bart$length[i, j] = len_psbart
      metrics_ps_bart$time[i, j] = time_psbart
      
      # Logging con Timestamp
      timestamp_log <- format(Sys.time(), "%H:%M:%S")
      cat(sprintf("  [%s] [PS-BART] Fitted in %6.1fs | Bias: %.4f | RMSE: %.4f | Cov: %.3f\n", 
                  timestamp_log, time_psbart, metrics_ps_bart$bias[i, j], metrics_ps_bart$rmse[i, j], metrics_ps_bart$coverage[i, j]))
      
      # ---------------------------------------------------------
      # 3. RUN BCF
      # ---------------------------------------------------------
      start_bcf <- Sys.time()
      invisible(capture.output({
        fit_bcf = bcf(y = data$Y, z = data$Z, x_control = X_mat, x_moderate = X_mat, 
                      pihat = pi_hat, 
                      nburn = current_nskip, nsim = ndpost_fixed, 
                      no_output = TRUE)
      }))
      end_bcf <- Sys.time()
      time_bcf <- as.numeric(difftime(end_bcf, start_bcf, units = "secs"))
      
      cate_draws_bcf = fit_bcf$tau 
      cate_est_bcf = colMeans(cate_draws_bcf)
      
      intervals_bcf = apply(cate_draws_bcf, 2, quantile, probs = c(0.025, 0.975))
      cov_bcf = mean(intervals_bcf[1, ] <= data$tau_true & intervals_bcf[2, ] >= data$tau_true)
      len_bcf = mean(intervals_bcf[2, ] - intervals_bcf[1, ])
      
      metrics_bcf$bias[i, j] = mean(abs(cate_est_bcf - data$tau_true))
      metrics_bcf$rmse[i, j] = sqrt(mean((cate_est_bcf - data$tau_true)^2))
      metrics_bcf$coverage[i, j] = cov_bcf
      metrics_bcf$length[i, j] = len_bcf
      metrics_bcf$time[i, j] = time_bcf
      
      # Logging con Timestamp
      timestamp_log <- format(Sys.time(), "%H:%M:%S")
      cat(sprintf("  [%s] [BCF]     Fitted in %6.1fs | Bias: %.4f | RMSE: %.4f | Cov: %.3f\n", 
                  timestamp_log, time_bcf, metrics_bcf$bias[i, j], metrics_bcf$rmse[i, j], metrics_bcf$coverage[i, j]))
    }
    
    # --- CHECKPOINT ---
    dir_name = paste0("table_results/", dgp_name)
    if (!dir.exists(dir_name)) {
      dir.create(dir_name, recursive = TRUE)
    }
    checkpoint_file = file.path(dir_name, paste0("checkpoint_raw_", dgp_name, ".rds"))
    
    checkpoint_results = list(
      last_completed_simulation = i,
      settings = list(dgp_name = dgp_name, n_simulations = n_simulations, nskip_grid = nskip_grid, ndpost_fixed = ndpost_fixed),
      metrics_bart = metrics_bart,
      metrics_ps_bart = metrics_ps_bart,
      metrics_bcf = metrics_bcf
    )
    saveRDS(checkpoint_results, file = checkpoint_file)
  }
  
  # --- FINAL AGGREGATION ---
  final_bart = data.frame(nskip = nskip_grid, 
                          Bias = colMeans(metrics_bart$bias), 
                          RMSE = colMeans(metrics_bart$rmse), 
                          Coverage = colMeans(metrics_bart$coverage),
                          Length = colMeans(metrics_bart$length),
                          Avg_Time_Secs = colMeans(metrics_bart$time)) 
  
  final_psbart = data.frame(nskip = nskip_grid, 
                            Bias = colMeans(metrics_ps_bart$bias), 
                            RMSE = colMeans(metrics_ps_bart$rmse), 
                            Coverage = colMeans(metrics_ps_bart$coverage),
                            Length = colMeans(metrics_ps_bart$length),
                            Avg_Time_Secs = colMeans(metrics_ps_bart$time)) 
  
  final_bcf = data.frame(nskip = nskip_grid, 
                         Bias = colMeans(metrics_bcf$bias), 
                         RMSE = colMeans(metrics_bcf$rmse), 
                         Coverage = colMeans(metrics_bcf$coverage),
                         Length = colMeans(metrics_bcf$length),
                         Avg_Time_Secs = colMeans(metrics_bcf$time))
  
  cat("\n=== AVERAGE RESULTS BART VANILLA ===\n")
  print(final_bart)
  cat("\n=== AVERAGE RESULTS PS-BART ===\n")
  print(final_psbart)
  cat("\n=== AVERAGE RESULTS BCF ===\n")
  print(final_bcf)
  
  # --- SAVE RESULTS ---
  cat("\nSaving final results...\n")
  
  timestamp = format(Sys.time(), "%d%m%Y_%H%M%S")
  
  file_csv_bart = file.path(dir_name, paste0("final_bart_", dgp_name, "_", timestamp, ".csv"))
  file_csv_psbart = file.path(dir_name, paste0("final_psbart_", dgp_name, "_", timestamp, ".csv"))
  file_csv_bcf = file.path(dir_name, paste0("final_bcf_", dgp_name, "_", timestamp, ".csv"))
  
  write.csv(final_bart, file = file_csv_bart, row.names = TRUE)
  write.csv(final_psbart, file = file_csv_psbart, row.names = TRUE)
  write.csv(final_bcf, file = file_csv_bcf, row.names = TRUE)
  
  file_rds_raw = file.path(dir_name, paste0("raw_metrics_full_", dgp_name, "_", timestamp, ".rds"))
  
  full_results = list(
    settings = list(dgp_name = dgp_name, n_simulations = n_simulations, nskip_grid = nskip_grid, ndpost_fixed = ndpost_fixed),
    metrics_bart = metrics_bart,
    metrics_ps_bart = metrics_ps_bart,
    metrics_bcf = metrics_bcf
  )
  saveRDS(full_results, file = file_rds_raw)
  
  if (file.exists(checkpoint_file)) file.remove(checkpoint_file)
  
  cat("Successfully saved in folder:", dir_name, "\n")
}

print("Experiments completed")
end_time <- Sys.time()
print(paste0("Elapsed time : ", end_time - start_time))