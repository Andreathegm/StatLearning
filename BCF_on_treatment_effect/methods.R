#install.packages(c("bcf", "BART"))
library(bcf)
library(BART)

# ==============================================================================
# 1. METODO: BART (Vanilla / Standard)
# Z here is treated as any other covariate
# ==============================================================================
run_bart_method <- function(data) {
  X_train <- as.matrix(cbind(data$X, Z = data$Z))
  
  # Creiamo i due scenari di test per la scomposizione causale
  X_test1 <- as.matrix(cbind(data$X, Z = 1))
  X_test0 <- as.matrix(cbind(data$X, Z = 0))
  
  # Adattiamo il modello BART per variabili continue (usando le impostazioni di default)
  # nskip e ndpost controllano il burn-in e il numero di istanti MCMC salvati
  fit_bart <- wbart(x.train = X_train, y.train = data$Y, 
                    x.test = rbind(X_test1, X_test0),
                    nskip = 250, ndpost = 1000, pr = FALSE)
  
  # Separiamo le predizioni dei due mondi potenziali dal blocco unico di test
  # fit_bart$yhat.test contiene 1000 righe (i passaggi MCMC) e 2*n colonne
  n <- length(data$Y)
  yhat_1 <- fit_bart$yhat.test[, 1:n]
  yhat_0 <- fit_bart$yhat.test[, (n + 1):(2 * n)]
  
  # Calcoliamo la distribuzione a posteriori del CATE per ogni individuo
  cate_samples <- yhat_1 - yhat_0
  
  # Estraiamo le metriche di sintesi (Media e quantili per l'intervallo al 95%)
  cate_est <- colMeans(cate_samples)
  cate_low <- apply(cate_samples, 2, quantile, probs = 0.025)
  cate_up  <- apply(cate_samples, 2, quantile, probs = 0.975)
  
  # L'ATE è la media delle stime dei CATE ad ogni iterazione MCMC
  ate_samples <- rowMeans(cate_samples)
  ate_est <- mean(ate_samples)
  ate_low <- quantile(ate_samples, probs = 0.025)
  ate_up  <- quantile(ate_samples, probs = 0.975)
  
  return(list(
    ATE = list(est = ate_est, lower = ate_low, upper = ate_up),
    CATE = list(est = cate_est, lower = cate_low, upper = cate_up)
  ))
}

# ==============================================================================
# 2. METODO: ps-BART (Propensity Score BART)
# Estende BART includendo il Propensity Score stimato come covariata [cite: 281, 436]
# ==============================================================================
run_ps_bart_method <- function(data) {
  prop_model <- glm(Z ~ ., data = data$X, family = binomial)
  pi_hat <- predict(prop_model, type = "response")
  
  X_train <- as.matrix(cbind(data$X, pi_hat = pi_hat, Z = data$Z))
  X_test1 <- as.matrix(cbind(data$X, pi_hat = pi_hat, Z = 1))
  X_test0 <- as.matrix(cbind(data$X, pi_hat = pi_hat, Z = 0))
  
  fit_ps_bart <- wbart(x.train = X_train, y.train = data$Y, 
                       x.test = rbind(X_test1, X_test0),
                       nskip = 250, ndpost = 1000, pr = FALSE)
  
  n <- length(data$Y)
  yhat_1 <- fit_ps_bart$yhat.test[, 1:n]
  yhat_0 <- fit_ps_bart$yhat.test[, (n + 1):(2 * n)]
  
  cate_samples <- yhat_1 - yhat_0
  
  cate_est <- colMeans(cate_samples)
  cate_low <- apply(cate_samples, 2, quantile, probs = 0.025)
  cate_up  <- apply(cate_samples, 2, quantile, probs = 0.975)
  
  ate_samples <- rowMeans(cate_samples)
  ate_est <- mean(ate_samples)
  ate_low <- quantile(ate_samples, probs = 0.025)
  ate_up  <- quantile(ate_samples, probs = 0.975)
  
  return(list(
    ATE = list(est = ate_est, lower = ate_low, upper = ate_up),
    CATE = list(est = cate_est, lower = cate_low, upper = cate_up)
  ))
}

# ==============================================================================
# 3. METODO: BCF (Bayesian Causal Forests)
# Separa nativamente la regolarizzazione di mu(x) e tau(x) [cite: 15, 388]
# ==============================================================================
run_bcf_method <- function(data) {
  # 1. Stima preliminare obbligatoria del propensity score [cite: 120, 319]
  prop_model <- glm(Z ~ ., data = data$X, family = binomial)
  pi_hat <- predict(prop_model, type = "response")
  
  # Trasformiamo il dataframe delle sole covariate X in matrice numerica per il pacchetto bcf
  X_matrix <- as.matrix(data$X)
  
  # 2. Eseguiamo il fit del modello BCF.
  # Il pacchetto implementa internamente l'architettura dettagliata nel paper:
  # assegna priori diversi e più penalizzanti su tau rispetto a mu[cite: 392, 393].
  fit_bcf <- bcf(y = data$Y, z = data$Z, x_control = X_matrix, x_moderate = X_matrix, 
                 pihat = pi_hat, nburn = 250, nsim = 1000)
  
  # Nel modello bcf, l'effetto condizionato tau è un parametro esplicito [cite: 357]
  # fit_bcf$tau estrae direttamente i campioni MCMC dell'effetto del trattamento
  cate_samples <- fit_bcf$tau
  
  cate_est <- colMeans(cate_samples)
  cate_low <- apply(cate_samples, 2, quantile, probs = 0.025)
  cate_up  <- apply(cate_samples, 2, quantile, probs = 0.975)
  
  # Calcoliamo l'ATE facendo la media di riga dei CATE stimati ad ogni iterazione
  ate_samples <- rowMeans(cate_samples)
  ate_est <- mean(ate_samples)
  ate_low <- quantile(ate_samples, probs = 0.025)
  ate_up  <- quantile(ate_samples, probs = 0.975)
  
  return(list(
    ATE = list(est = ate_est, lower = ate_low, upper = ate_up),
    CATE = list(est = cate_est, lower = cate_low, upper = cate_up)
  ))
}