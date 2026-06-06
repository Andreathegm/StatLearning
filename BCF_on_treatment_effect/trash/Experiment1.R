if (!require("dbarts", quietly = TRUE)) install.packages("dbarts")
if (!require("bcf", quietly = TRUE)) install.packages("bcf")
library(dbarts)
library(bcf)

generate_data = function(n, seed) {
  set.seed(seed)
  x1 = runif(n, 0, 1)
  x2 = runif(n, 0, 1)
  mu = -3 + 6 * pnorm(2 * (x1 - x2))
  pi_x = 0.8 * pnorm(mu / (0.1 * (2 - x1 - x2) + 0.25)) + 0.025 * (x1 + x2) + 0.05
  pi_x = pmin(pmax(pi_x, 0.01), 0.99)
  z = rbinom(n, 1, pi_x)
  y = mu + (-1) * z + rnorm(n, sd = 1)
  return(list(X = data.frame(x1, x2), Y = y, Z = z, pi_true = pi_x, mu_true = mu, tau_true = rep(-1, n)))
}

## we calculate P(Z_i=1 | X_i = x_i) for every i 
estimate_pi = function(X, Z) {
  fit = bart(x.train = as.matrix(X), y.train = Z, verbose = FALSE)
  ## fit$yhat.train gives us a  matrix mcmc_samples(from ndraw) * n (num. of observation)
  pmin(pmax(colMeans(fit$yhat.train), 0.01), 0.99)
}

# --- Experiment setting ---
n_simulations = 20
#nskip_grid = c(250, 500, 1000, 2000)
nskip_grid = c(1000)
ndpost_fixed = 2000


### results holder
metrics_bart = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                    rmse = matrix(NA, n_simulations, length(nskip_grid)),
                    coverage = matrix(NA, n_simulations, length(nskip_grid)))

# metrics_ps.bart = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
#                    rmse = matrix(NA, n_simulations, length(nskip_grid)),
#                    coverage = matrix(NA, n_simulations, length(nskip_grid)))


metrics_bcf = list(bias = matrix(NA, n_simulations, length(nskip_grid)),
                   rmse = matrix(NA, n_simulations, length(nskip_grid)),
                   coverage = matrix(NA, n_simulations, length(nskip_grid)))



cat("Start simulation ...\n\n")

for(i in 1:n_simulations) {
  cat("==> Simulation", i, "/", n_simulations, "\n")
  data = generate_data(n = 250, seed = i)
  pi_hat = estimate_pi(data$X, data$Z)
  
  X_train_bart = cbind(data$X, Z = data$Z)
  X_test1_bart = cbind(data$X, Z = 1)
  X_test0_bart = cbind(data$X, Z = 0)
  X_mat = as.matrix(data$X)
  
  for(j in 1:length(nskip_grid)) {
    current_nskip = nskip_grid[j]
    
    # 1. RUN BART
    fit_bart = bart(x.train = X_train_bart, y.train = data$Y, 
                    x.test = rbind(X_test1_bart, X_test0_bart), 
                    nskip = current_nskip, ndpost = ndpost_fixed, verbose = FALSE)
    
    cate_draws_bart = fit_bart$yhat.test[, 1:250] - fit_bart$yhat.test[, 251:500]
    cate_est_bart = colMeans(cate_draws_bart)
    
    # Credible intervals @ 95% for BART and coverage 
    intervals_bart = apply(cate_draws_bart, 2, quantile, probs = c(0.025, 0.975))
    cov_bart = mean(intervals_bart[1, ] <= -1 & intervals_bart[2, ] >= -1)
    
    metrics_bart$bias[i, j] = mean(abs(cate_est_bart - (-1)))
    metrics_bart$rmse[i, j] = sqrt(mean((cate_est_bart - (-1))^2))
    metrics_bart$coverage[i, j] = cov_bart
    
    # 2. RUN BCF
    fit_bcf = bcf(y = data$Y, z = data$Z, x_control = X_mat, x_moderate = X_mat, 
                  pihat = pi_hat, 
                  nburn = current_nskip, nsim = ndpost_fixed, 
                  use_tauscale = TRUE, use_muscale = TRUE, 
                  verbose = FALSE, no_output = TRUE, n_chains = 4)
    
    cate_draws_bcf = fit_bcf$tau  # [mcmc iteration that are kept, num.observation]
    cate_est_bcf = colMeans(cate_draws_bcf)
    
    # Credible intervals @ 95% per BCF
    intervals_bcf = apply(cate_draws_bcf, 2, quantile, probs = c(0.025, 0.975))
    cov_bcf = mean(intervals_bcf[1, ] <= -1 & intervals_bcf[2, ] >= -1)
    
    metrics_bcf$bias[i, j] = mean(abs(cate_est_bcf - (-1)))
    metrics_bcf$rmse[i, j] = sqrt(mean((cate_est_bcf - (-1))^2))
    metrics_bcf$coverage[i, j] = cov_bcf
  }
}

final_bart = data.frame(nskip = nskip_grid, Bias = colMeans(metrics_bart$bias), RMSE = colMeans(metrics_bart$rmse), Coverage = colMeans(metrics_bart$coverage))
final_bcf = data.frame(nskip = nskip_grid, Bias = colMeans(metrics_bcf$bias), RMSE = colMeans(metrics_bcf$rmse), Coverage = colMeans(metrics_bcf$coverage))

cat("\n=== MEAN RESULTS BART VANILLA ===\n")
print(final_bart)
cat("\n=== MEAN RESULTS BCF ===\n")
print(final_bcf)

# --- Saving the results ---
cat("\n Saving results ...\n")

dir_name = "table_results"
if (!dir.exists(dir_name)) {
  dir.create(dir_name)
}

timestamp = format(Sys.time(), "%d%m%Y_%H%M%S")

file_csv_bart = file.path(dir_name, paste0("final_bart_", timestamp, ".csv"))
file_csv_bcf = file.path(dir_name, paste0("final_bcf_", timestamp, ".csv"))

write.csv(final_bart, file = file_csv_bart, row.names = TRUE)
write.csv(final_bcf, file = file_csv_bcf, row.names = TRUE)

file_rds_raw = file.path(dir_name, paste0("raw_metrics_full_", timestamp, ".rds"))

full_results = list(
  settings = list(n_simulations = n_simulations, nskip_grid = nskip_grid, ndpost_fixed = ndpost_fixed),
  metrics_bart = metrics_bart,
  metrics_bcf = metrics_bcf
)
saveRDS(full_results, file = file_rds_raw)

cat("All file saved in :", dir_name, "\n")