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

estimate_pi = function(X, Z) {
  fit = bart(x.train = as.matrix(X), y.train = Z, verbose = FALSE)
  pmin(pmax(colMeans(fit$yhat.train), 0.01), 0.99)
}

# --- IMPOSTAZIONI ESPERIMENTO ---
# Usiamo 5 simulazioni per tenere i tempi umani. Se vuoi curve più lisce, aumenta n_simulations
n_simulations = 5 
#nskip_grid = c(250, 500, 1000, 2000)
nskip_grid = c(250, 500, 1000, 2000)# La griglia di burn-in
ndpost_fixed = 2000                  # Campioni tenuti FISSI

# Matrici per salvare i risultati (Bias)
bias_bart_matrix = matrix(NA, nrow = n_simulations, ncol = length(nskip_grid))
bias_bcf_matrix = matrix(NA, nrow = n_simulations, ncol = length(nskip_grid))
colnames(bias_bart_matrix) = paste0("nskip_", nskip_grid)
colnames(bias_bcf_matrix) = paste0("nskip_", nskip_grid)

cat("Inizio simulazione curva di convergenza su", n_simulations, "replicazioni...\n")
cat("Valori di nskip testati:", paste(nskip_grid, collapse=", "), "\n")
cat("Valore fisso di ndpost:", ndpost_fixed, "\n\n")

for(i in 1:n_simulations) {
  cat("==> Simulazione", i, "/", n_simulations, "\n")
  data = generate_data(n = 250, seed = i)
  pi_hat = estimate_pi(data$X, data$Z)
  
  X_train_bart = cbind(data$X, Z = data$Z)
  X_test1_bart = cbind(data$X, Z = 1)
  X_test0_bart = cbind(data$X, Z = 0)
  X_mat = as.matrix(data$X)
  
  for(j in 1:length(nskip_grid)) {
    current_nskip = nskip_grid[j]
    cat("    Testando nskip =", current_nskip, "...\n")
    
    # BART Vanilla 
    fit_bart = bart(x.train = X_train_bart, y.train = data$Y, 
                    x.test = rbind(X_test1_bart, X_test0_bart), 
                    nskip = current_nskip, ndpost = ndpost_fixed, verbose = FALSE)
    yhat1_bart = fit_bart$yhat.test[, 1:250]
    yhat0_bart = fit_bart$yhat.test[, 251:500]
    ate_bart = mean(colMeans(yhat1_bart - yhat0_bart))
    
    # BCF 
    fit_bcf = bcf(y = data$Y, z = data$Z, x_control = X_mat, x_moderate = X_mat, 
                  pihat = pi_hat, 
                  nburn = current_nskip, nsim = ndpost_fixed, 
                  use_tauscale = TRUE, use_muscale = TRUE, 
                  verbose = FALSE, no_output = TRUE, n_chains = 1)
    ate_bcf = mean(colMeans(fit_bcf$tau))
    
    # Calcolo del Bias (True ATE = -1)
    bias_bart_matrix[i, j] = ate_bart - (-1)
    bias_bcf_matrix[i, j] = ate_bcf - (-1)
  }
}

# --- AGGREGAZIONE DATI E PRINT ---
mean_bias_bart = colMeans(bias_bart_matrix)
mean_bias_bcf = colMeans(bias_bcf_matrix)

cat("\n=== RISULTATI FINALI MEDI ===\n")
print(data.frame(nskip = nskip_grid, Bias_BART = mean_bias_bart, Bias_BCF = mean_bias_bcf))

# --- PLOT DELLA CURVA DI CONVERGENZA ---
# Genera un grafico base R con entrambe le curve e la linea dello zero (bias nullo)
plot(nskip_grid, mean_bias_bcf, type="b", col="forestgreen", pch=16, lwd=2,
     ylim=range(c(mean_bias_bcf, mean_bias_bart, 0)),
     xlab="Burn-in (nskip)", ylab="Bias Medio (Distanza dal vero ATE)",
     main="Curva di Convergenza: BCF vs BART")
lines(nskip_grid, mean_bias_bart, type="b", col="firebrick", pch=15, lwd=2, lty=2)
abline(h=0, col="black", lty=3, lwd=1.5) 
legend("topright", legend=c("BCF", "BART Vanilla"), col=c("forestgreen", "firebrick"), 
       pch=c(16, 15), lty=c(1, 2), lwd=2)