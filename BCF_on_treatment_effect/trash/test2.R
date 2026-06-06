# ============================================
# STEP 5: Simulazione Monte Carlo con DGP ORIGINALE del paper
# Usando bart() di dbarts, come nel codice originale
# ============================================

# Verifica/installa dbarts
if (!require("dbarts", quietly = TRUE)) {
  install.packages("dbarts")
}
library(dbarts)
library(bcf)

# Copia esatta del DGP del paper
g = function(x){
  ifelse(x == 1, 2, ifelse(x == 2, -1, -4))
}

generate_data_paper = function(n, p=5, tau_type="homogeneous", mu_type="linear", seed){
  set.seed(seed)
  
  x = matrix(rnorm(n*p), nrow=n, ncol=p)
  x[,1] = rnorm(n)
  x[,2] = rnorm(n)
  x[,3] = rnorm(n)
  x[,4] = rbinom(n, 1, 0.5)
  x[,5] = sample(c(1,2,3), n, replace = TRUE)
  u = runif(n, 0, 1)
  sigma = 5  # CRUCIALE: rumore grande
  
  # Tau
  if(tau_type == "homogeneous"){tau_x = rep(3, n)}
  if(tau_type == "heterogeneous"){tau_x = 1 + 2*x[,2]*x[,4]}
  
  # Mu
  if(mu_type == "linear"){mu_x = 1 + g(x[,5]) + x[,1]*x[,3]}
  if(mu_type == "nonlinear"){mu_x = -6 + g(x[,5]) + 6*abs(x[,3] - 1)}
  
  # Pi e Pihat (come nel paper: bart() di dbarts!)
  pi.x = 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*x[,1]) + 0.05 + u/10
  z = rbinom(n, 1, pi.x)
  
  # Stima pi con bart() di dbarts (come nel paper originale)
  pihat_bart = apply(bart(x, z, verbose=FALSE)$yhat.train, 2, mean)
  
  # Y
  y = mu_x + tau_x * z + rnorm(n, sd=sigma)
  
  return(list(
    X = data.frame(x1=x[,1], x2=x[,2], x3=x[,3], x4=x[,4], x5=x[,5]),
    Y = y,
    Z = z,
    pi_true = pi.x,
    mu_true = mu_x,
    tau_true = tau_x,
    pi_hat = pihat_bart
  ))
}

# METODI con bart() di dbarts (come nel paper)
run_bart_method_paper <- function(data) {
  # BART vanilla: Z è "just another covariate"
  X_train <- cbind(data$X, Z = data$Z)
  X_test1 <- cbind(data$X, Z = 1)
  X_test0 <- cbind(data$X, Z = 0)
  
  fit <- bart(x.train = X_train, y.train = data$Y,
              x.test = rbind(X_test1, X_test0),
              nskip = 250, ndpost = 1000, verbose = FALSE)
  
  n <- length(data$Y)
  yhat_1 <- fit$yhat.test[, 1:n]
  yhat_0 <- fit$yhat.test[, (n+1):(2*n)]
  cate_samples <- yhat_1 - yhat_0
  
  list(
    ATE = list(est = mean(rowMeans(cate_samples)),
               lower = quantile(rowMeans(cate_samples), 0.025),
               upper = quantile(rowMeans(cate_samples), 0.975)),
    CATE = list(est = colMeans(cate_samples),
                lower = apply(cate_samples, 2, quantile, 0.025),
                upper = apply(cate_samples, 2, quantile, 0.975))
  )
}

run_ps_bart_method_paper <- function(data) {
  # ps-BART: include pi_hat come covariata
  X_train <- cbind(data$X, pi_hat = data$pi_hat, Z = data$Z)
  X_test1 <- cbind(data$X, pi_hat = data$pi_hat, Z = 1)
  X_test0 <- cbind(data$X, pi_hat = data$pi_hat, Z = 0)
  
  fit <- bart(x.train = X_train, y.train = data$Y,
              x.test = rbind(X_test1, X_test0),
              nskip = 250, ndpost = 1000, verbose = FALSE)
  
  n <- length(data$Y)
  yhat_1 <- fit$yhat.test[, 1:n]
  yhat_0 <- fit$yhat.test[, (n+1):(2*n)]
  cate_samples <- yhat_1 - yhat_0
  
  list(
    ATE = list(est = mean(rowMeans(cate_samples)),
               lower = quantile(rowMeans(cate_samples), 0.025),
               upper = quantile(rowMeans(cate_samples), 0.975)),
    CATE = list(est = colMeans(cate_samples),
                lower = apply(cate_samples, 2, quantile, 0.025),
                upper = apply(cate_samples, 2, quantile, 0.975))
  )
}

run_bcf_method_paper <- function(data) {
  X_mat <- as.matrix(data$X)
  
  fit <- bcf(y = data$Y, z = data$Z,
             x_control = X_mat, x_moderate = X_mat,
             pihat = data$pi_hat,
             nburn = 250, nsim = 1000,
             verbose = FALSE, no_output = TRUE, n_chains = 1)
  
  cate_samples <- fit$tau
  
  list(
    ATE = list(est = mean(rowMeans(cate_samples)),
               lower = quantile(rowMeans(cate_samples), 0.025),
               upper = quantile(rowMeans(cate_samples), 0.975)),
    CATE = list(est = colMeans(cate_samples),
                lower = apply(cate_samples, 2, quantile, 0.025),
                upper = apply(cate_samples, 2, quantile, 0.975))
  )
}

# ============================================
# SIMULAZIONE MONTE CARLO
# ============================================
n_sims <- 50
n_obs <- 250

methods <- list(
  "BART"    = run_bart_method_paper,
  "ps-BART" = run_ps_bart_method_paper,
  "BCF"     = run_bcf_method_paper
)

ate_est <- matrix(NA, nrow = n_sims, ncol = length(methods))
colnames(ate_est) <- names(methods)

cat("Inizio Monte Carlo:", n_sims, "simulazioni...\n")
cat("Configurazione: homogeneous, linear, n=250, sigma=5\n")
cat("Usando bart() di dbarts (come nel paper originale)\n\n")

for(i in 1:n_sims) {
  if(i %% 10 == 0) cat("Simulazione", i, "/", n_sims, "\n")
  
  data <- generate_data_paper(n = n_obs, tau_type = "homogeneous", 
                              mu_type = "linear", seed = i)
  
  for(m in names(methods)) {
    tryCatch({
      res <- methods[[m]](data)
      ate_est[i, m] <- res$ATE$est
    }, error = function(e) {
      cat("Errore in", m, "sim", i, ":", conditionMessage(e), "\n")
    })
  }
}

# ANALISI
true_ate <- 3

cat("\n=== RISULTATI MONTE CARLO (", n_sims, " simulazioni) ===\n")
cat("Vero ATE:", true_ate, "\n\n")

bias <- colMeans(ate_est, na.rm = TRUE) - true_ate
rmse <- apply(ate_est, 2, function(x) sqrt(mean((x - true_ate)^2, na.rm = TRUE)))
sd_est <- apply(ate_est, 2, sd, na.rm = TRUE)

cat("Bias medio:\n")
print(round(bias, 4))
cat("\nRMSE:\n")
print(round(rmse, 4))
cat("\nSD delle stime:\n")
print(round(sd_est, 4))

# Boxplot
png("ric_monte_carlo_dbarts.png", width = 900, height = 700, res = 120)
boxplot(ate_est,
        main = "RIC: DGP originale con bart() di dbarts (50 sim)",
        ylab = "Stima ATE",
        col = c("coral", "skyblue", "palegreen"),
        ylim = c(min(ate_est, na.rm = TRUE) - 1, max(ate_est, na.rm = TRUE) + 1))
abline(h = true_ate, col = "darkred", lwd = 2, lty = 2)
legend("topright", legend = paste("Vero ATE =", true_ate), 
       col = "darkred", lty = 2, lwd = 2, bty = "n")
dev.off()

cat("\nPlot salvato: ric_monte_carlo_dbarts.png\n")