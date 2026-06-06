# ============================================
# STEP 2: Test BART vanilla su singolo dataset
# ============================================
library(BART)

source("dgps.R")  # per il DGP originale (lo useremo per confronto)

# Usa il DGP CORRETTO
dgp_simple_fixed <- function(n = 250, tau = -1) {
  x1 <- runif(n, 0, 1)
  x2 <- runif(n, 0, 1)
  mu <- 3 * (x1 - x2)
  pi_x <- 0.8 * pnorm(mu / (0.1*(2 - x1 - x2) + 0.25)) + 
    0.025*(x1 + x2) + 0.05
  z <- rbinom(n, 1, pi_x)
  y <- mu + tau * z + rnorm(n)
  return(list(X = data.frame(x1, x2), Y = y, Z = z,
              pi_true = pi_x, mu_true = mu, tau_true = rep(tau, n)))
}

set.seed(123)
d <- dgp_simple_fixed(n = 250, tau = -1)

# BART vanilla: Z è "just another covariate"
X_train <- as.matrix(cbind(d$X, Z = d$Z))
X_test1 <- as.matrix(cbind(d$X, Z = 1))
X_test0 <- as.matrix(cbind(d$X, Z = 0))

cat("Fitting BART vanilla (può richiedere 10-20 sec)...\n")
fit_bart <- wbart(x.train = X_train, y.train = d$Y,
                  x.test = rbind(X_test1, X_test0),
                  nskip = 250, ndpost = 1000, printevery = 500)

n <- length(d$Y)
yhat_1 <- fit_bart$yhat.test[, 1:n]
yhat_0 <- fit_bart$yhat.test[, (n+1):(2*n)]
cate_samples <- yhat_1 - yhat_0

ate_est <- mean(rowMeans(cate_samples))
ate_low <- quantile(rowMeans(cate_samples), 0.025)
ate_up  <- quantile(rowMeans(cate_samples), 0.975)

cat("\n=== RISULTATI BART VANILLA ===\n")
cat("ATE stimata:", round(ate_est, 4), "\n")
cat("Vero ATE:", d$tau_true[1], "\n")
cat("Bias:", round(ate_est - d$tau_true[1], 4), "\n")
cat("IC 95%:", round(ate_low, 4), "-", round(ate_up, 4), "\n")
cat("IC contiene vero?", (ate_low <= d$tau_true[1] & d$tau_true[1] <= ate_up), "\n")
cat("SD CATE:", round(sd(colMeans(cate_samples)), 4), "\n")

# Confronto con naive empirical
empirical_diff <- mean(d$Y[d$Z == 1]) - mean(d$Y[d$Z == 0])
cat("\nConfronto naive:", round(empirical_diff, 4), "(grezzo, confounded)\n")