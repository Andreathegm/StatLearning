# ==========================================
# FILE: main_simulation.R
# Main script to run experiments and test models (BART, PS-BART, BCF)
# ==========================================

source("DGP_library.R")

if (!require("dbarts", quietly = TRUE)) install.packages("dbarts")
if (!require("bcf", quietly = TRUE)) install.packages("bcf")

library(dbarts)
library(bcf)

## Calculate P(Z_i=1 | X_i = x_i) for every i 
estimate_pi = function(X, Z) {
  fit = bart(x.train = as.matrix(X), y.train = Z, verbose = FALSE)
  pmin(pmax(colMeans(fit$yhat.train), 0.01), 0.99)
}

# --- Experiment Settings ---
# n_simulations = 50
# nskip_grid = c(250,500,1000)
n_simulations = 50
nskip_grid = c(250,500,1000)
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
print(paste0("Experiment started at ",start_time))
for (dgp_name in experiments_to_run) {
  cat("\n================================================\n")
  cat("=== Running experiment on DGP:", dgp_name, "===\n")
  cat("================================================\n")
  
  # Convert the string name to the actual generating function
  data_generator = match.fun(dgp_name)
  
  # Initialize result holders for the current DGP (Added "length")
  metrics_bart = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                      rmse = matrix(NA, n_simulations, length(nskip_grid)),
                      coverage = matrix(NA, n_simulations, length(nskip_grid)),
                      length = matrix(NA, n_simulations, length(nskip_grid)))
  
  metrics_ps_bart = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                         rmse = matrix(NA, n_simulations, length(nskip_grid)),
                         coverage = matrix(NA, n_simulations, length(nskip_grid)),
                         length = matrix(NA, n_simulations, length(nskip_grid)))
  
  metrics_bcf = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                     rmse = matrix(NA, n_simulations, length(nskip_grid)),
                     coverage = matrix(NA, n_simulations, length(nskip_grid)),
                     length = matrix(NA, n_simulations, length(nskip_grid)))
  
  for(i in 1:n_simulations) {
    cat("==> Simulation", i, "/", n_simulations, "\n")
    
    # Generate data using the wrapper function
    data = data_generator(n = 250, seed = i)
    pi_hat = estimate_pi(data$X, data$Z)
    
    # MATRICES FOR BART VANILLA
    X_train_bart = cbind(data$X, Z = data$Z)
    X_test1_bart = cbind(data$X, Z = 1)
    X_test0_bart = cbind(data$X, Z = 0)
    
    # MATRICES FOR PS-BART (Adding pi_hat as a covariate)
    X_train_psbart = cbind(data$X, pihat = pi_hat, Z = data$Z)
    X_test1_psbart = cbind(data$X, pihat = pi_hat, Z = 1)
    X_test0_psbart = cbind(data$X, pihat = pi_hat, Z = 0)
    
    # MATRIX FOR BCF
    X_mat = as.matrix(data$X)
    
    for(j in 1:length(nskip_grid)) {
      current_nskip = nskip_grid[j]
      
      # ---------------------------------------------------------
      # 1. RUN BART VANILLA
      # ---------------------------------------------------------
      invisible(capture.output({
        fit_bart = bart(x.train = X_train_bart, y.train = data$Y, 
                        x.test = rbind(X_test1_bart, X_test0_bart), 
                        nskip = current_nskip, ndpost = ndpost_fixed, verbose = FALSE)
        
      }))
      
      print(paste0("Bart fitted (simulation ",i,"/",n_simulations,")"))
      
      cate_draws_bart = fit_bart$yhat.test[, 1:250] - fit_bart$yhat.test[, 251:500]
      cate_est_bart = colMeans(cate_draws_bart)
      
      intervals_bart = apply(cate_draws_bart, 2, quantile, probs = c(0.025, 0.975))
      cov_bart = mean(intervals_bart[1, ] <= data$tau_true & intervals_bart[2, ] >= data$tau_true)
      len_bart = mean(intervals_bart[2, ] - intervals_bart[1, ])
      
      metrics_bart$bias[i, j] = mean(abs(cate_est_bart - data$tau_true))
      metrics_bart$rmse[i, j] = sqrt(mean((cate_est_bart - data$tau_true)^2))
      metrics_bart$coverage[i, j] = cov_bart
      metrics_bart$length[i, j] = len_bart
      
      # ---------------------------------------------------------
      # 2. RUN PS-BART (Propensity Score BART)
      # ---------------------------------------------------------
      invisible(capture.output({
        fit_psbart = bart(x.train = X_train_psbart, y.train = data$Y, 
                          x.test = rbind(X_test1_psbart, X_test0_psbart), 
                          nskip = current_nskip, ndpost = ndpost_fixed, verbose = FALSE)
        
      }))
      print(paste0("ps-Bart fitted (simulation ",i,"/",n_simulations,")"))
      
      
      cate_draws_psbart = fit_psbart$yhat.test[, 1:250] - fit_psbart$yhat.test[, 251:500]
      cate_est_psbart = colMeans(cate_draws_psbart)
      
      intervals_psbart = apply(cate_draws_psbart, 2, quantile, probs = c(0.025, 0.975))
      cov_psbart = mean(intervals_psbart[1, ] <= data$tau_true & intervals_psbart[2, ] >= data$tau_true)
      len_psbart = mean(intervals_psbart[2, ] - intervals_psbart[1, ])
      
      metrics_ps_bart$bias[i, j] = mean(abs(cate_est_psbart - data$tau_true))
      metrics_ps_bart$rmse[i, j] = sqrt(mean((cate_est_psbart - data$tau_true)^2))
      metrics_ps_bart$coverage[i, j] = cov_psbart
      metrics_ps_bart$length[i, j] = len_psbart
      
      # ---------------------------------------------------------
      # 3. RUN BCF
      # ---------------------------------------------------------
      invisible(capture.output({
        fit_bcf = bcf(y = data$Y, z = data$Z, x_control = X_mat, x_moderate = X_mat, 
                      pihat = pi_hat, 
                      nburn = current_nskip, nsim = ndpost_fixed, 
                      no_output = TRUE)
        
      }))
      print(paste0("BCF fitted (simulation ",i,"/",n_simulations,")"))
      
      
      cate_draws_bcf = fit_bcf$tau  # [mcmc iterations kept, num. observation]
      cate_est_bcf = colMeans(cate_draws_bcf)
      
      intervals_bcf = apply(cate_draws_bcf, 2, quantile, probs = c(0.025, 0.975))
      cov_bcf = mean(intervals_bcf[1, ] <= data$tau_true & intervals_bcf[2, ] >= data$tau_true)
      len_bcf = mean(intervals_bcf[2, ] - intervals_bcf[1, ])
      
      metrics_bcf$bias[i, j] = mean(abs(cate_est_bcf - data$tau_true))
      metrics_bcf$rmse[i, j] = sqrt(mean((cate_est_bcf - data$tau_true)^2))
      metrics_bcf$coverage[i, j] = cov_bcf
      metrics_bcf$length[i, j] = len_bcf
    }
  }
  
  # --- FINAL AGGREGATION ---
  final_bart = data.frame(nskip = nskip_grid, 
                          Bias = colMeans(metrics_bart$bias), 
                          RMSE = colMeans(metrics_bart$rmse), 
                          Coverage = colMeans(metrics_bart$coverage),
                          Length = colMeans(metrics_bart$length))
  
  final_psbart = data.frame(nskip = nskip_grid, 
                            Bias = colMeans(metrics_ps_bart$bias), 
                            RMSE = colMeans(metrics_ps_bart$rmse), 
                            Coverage = colMeans(metrics_ps_bart$coverage),
                            Length = colMeans(metrics_ps_bart$length))
  
  final_bcf = data.frame(nskip = nskip_grid, 
                         Bias = colMeans(metrics_bcf$bias), 
                         RMSE = colMeans(metrics_bcf$rmse), 
                         Coverage = colMeans(metrics_bcf$coverage),
                         Length = colMeans(metrics_bcf$length))
  
  cat("\n=== AVERAGE RESULTS BART VANILLA ===\n")
  print(final_bart)
  cat("\n=== AVERAGE RESULTS PS-BART ===\n")
  print(final_psbart)
  cat("\n=== AVERAGE RESULTS BCF ===\n")
  print(final_bcf)
  
  # --- SAVE RESULTS ---
  cat("\nSaving results...\n")
  
  # Create the folder if it does not exist
  dir_name = paste0("table_results/",dgp_name)
  if (!dir.exists(dir_name)) {
    dir.create(dir_name,recursive = TRUE)
  }
  
  # Generate a unique timestamp for the file names
  timestamp = format(Sys.time(), "%d%m%Y_%H%M%S")
  
  # Specific paths for CSV files including the DGP name
  file_csv_bart = file.path(dir_name, paste0("final_bart_", dgp_name, "_", timestamp, ".csv"))
  file_csv_psbart = file.path(dir_name, paste0("final_psbart_", dgp_name, "_", timestamp, ".csv"))
  file_csv_bcf = file.path(dir_name, paste0("final_bcf_", dgp_name, "_", timestamp, ".csv"))
  
  write.csv(final_bart, file = file_csv_bart, row.names = TRUE)
  write.csv(final_psbart, file = file_csv_psbart, row.names = TRUE)
  write.csv(final_bcf, file = file_csv_bcf, row.names = TRUE)
  
  # Path for the raw RDS file
  file_rds_raw = file.path(dir_name, paste0("raw_metrics_full_", dgp_name, "_", timestamp, ".rds"))
  
  # Create a full list containing all settings and data, then save
  full_results = list(
    settings = list(dgp_name = dgp_name, n_simulations = n_simulations, nskip_grid = nskip_grid, ndpost_fixed = ndpost_fixed),
    metrics_bart = metrics_bart,
    metrics_ps_bart = metrics_ps_bart,
    metrics_bcf = metrics_bcf
  )
  saveRDS(full_results, file = file_rds_raw)
  
  cat("Successfully saved in folder:", dir_name, "\n")
}

print("Experiments completed")
end_time <- Sys.time()
print(paste0("Elapsed time : ",end_time - start_time))