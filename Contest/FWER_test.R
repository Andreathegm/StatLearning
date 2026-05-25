library(hdi)
source("utils.R")
###    Using third-party multi-split to get E[FP] (expectation of false positive)
###    E[TP] , E[FP/(TP+FP) , E[Indicator of at least 1 false positive]]

FWER_test <- function(n_obs,p,B,s0,alpha,N_mc, snr){

# Initialize data frames for storing results
res_single <- data.frame(TP = numeric(N_mc),FP = numeric(N_mc), 
                         FWER_ind = numeric(N_mc),FDR = numeric(N_mc))
res_multi  <- data.frame(TP = numeric(N_mc), FP = numeric(N_mc),
                         FWER_ind = numeric(N_mc),FDR = numeric(N_mc))


# --- START MONTE CARLO SIMULATION ---
for (i in seq_len(N_mc)) {
  cat(sprintf("Running iteration %d of %d...\n", i, N_mc))
  
  X <- generate_data(n_obs = n_obs,n_var = p)
  active <- sample(1:p, s0)
  active <- sort(active)
  
  beta <- rep(0, p)
  beta[active] <- runif(s0, 1, s0)
  signal <- X %*% beta
  var_signal <- var(signal)          # Varianza empirica del segnale Var(X*beta)
  sigma <- sqrt(var_signal / snr)
  Y <- linear_dgp(X,beta,n_obs,sigma)

  
  # 2. Single Split model (B = 1) with gamma = c(1)
  fit.single <- hdi(x = X, 
                    y = Y, 
                    method = "multi.split", 
                    B = 1, 
                    gamma = c(1), 
                    verbose = FALSE)
  
  signif_single <- which(fit.single$pval.corr < alpha)
  res_single[i, ] <- calculate_metrics(signif_single, active)
  
  # 3. Multi Split model (B > 1)
  fit.multi <- hdi(x = X, 
                   y = Y, 
                   method = "multi.split", 
                   B = B, 
                   verbose = FALSE)
  
  signif_multi <- which(fit.multi$pval.corr < alpha)
  res_multi[i, ] <- calculate_metrics(signif_multi, active)
}
# --- END SIMULATION ---

# Calculate Expectations (Means over N_mc iterations)
expected_single <- colMeans(res_single)
expected_multi  <- colMeans(res_multi)

paste0("\n--- EXPECTED METRICS (Single Split, B = ",B,") ---\n")
print(expected_single)

paste0("\n--- EXPECTED METRICS (Multi Split, B = ",B,") ---\n")
print(expected_multi)
return (list(res_single=res_single, res_multi=res_multi))
}