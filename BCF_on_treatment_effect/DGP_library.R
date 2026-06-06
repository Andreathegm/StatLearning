# ==========================================
# FILE: DGP_library.R
# Contains the universal simulation engine and all Data Generating Processes (DGPs)
# ==========================================

# ==========================================
# 1. THE UNIVERSAL ENGINE (Boilerplate)
# ==========================================
generate_data_universal <- function(n, seed, generate_X_func, calc_mu_func, calc_tau_func, calc_pi_func, sd_error = 1) {
  # Set seed for reproducibility
  set.seed(seed)
  
  # 1. Generate the covariate matrix (Dynamic)
  X <- generate_X_func(n)
  
  # 2. Calculate the Prognostic Score (mu) and Treatment Effect (tau)
  mu <- calc_mu_func(X)
  tau <- calc_tau_func(X)
  
  # 3. Calculate the Propensity Score (pi) and apply safety clipping (overlap assumption)
  pi_x <- calc_pi_func(X, mu)
  pi_x <- pmin(pmax(pi_x, 0.01), 0.99)
  
  # 4. Treatment assignment and error term
  z <- rbinom(n, size = 1, prob = pi_x)
  epsilon <- rnorm(n, mean = 0, sd = sd_error)
  
  # 5. Generate the final Outcome
  y <- mu + tau * z + epsilon
  
  return(list(
    X = X, 
    Y = y, 
    Z = z, 
    pi_true = pi_x, 
    mu_true = mu, 
    tau_true = tau
  ))
}

# ==========================================
# 2. WRAPPERS (Original Scenarios)
# ==========================================

dgp_paper_example1 <- function(n, seed) {
  gen_X    <- function(n) data.frame(x1 = runif(n, 0, 1), x2 = runif(n, 0, 1))
  calc_mu  <- function(X) -3 + 6 * pnorm(2 * (X$x1 - X$x2))
  calc_tau <- function(X) rep(-1, nrow(X))
  calc_pi  <- function(X, mu) 0.8 * pnorm(mu / (0.1 * (2 - X$x1 - X$x2) + 0.25)) + 0.025 * (X$x1 + X$x2) + 0.05
  
  return(generate_data_universal(n, seed, gen_X, calc_mu, calc_tau, calc_pi, sd_error = 1))
}

dgp_simple <- function(n, seed, tau_param = -1) {
  gen_X    <- function(n) data.frame(x1 = runif(n, 0, 1), x2 = runif(n, 0, 1))
  calc_mu  <- function(X) 3 * (X$x1 - X$x2)
  calc_tau <- function(X) rep(tau_param, nrow(X))
  calc_pi  <- function(X, mu) 0.8 * pnorm(mu / (0.1 * (2 - X$x1 - X$x2) + 0.25)) + 0.025 * (X$x1 + X$x2) + 0.05
  
  return(generate_data_universal(n, seed, gen_X, calc_mu, calc_tau, calc_pi, sd_error = 1))
}

# ==========================================
# 3. WRAPPERS (New Scenarios with 5 Variables)
# ==========================================

# Base Generator for complex DGPs
gen_X_complex <- function(n) {
  data.frame(
    x1 = rnorm(n, mean = 0, sd = 1),
    x2 = rnorm(n, mean = 0, sd = 1),
    x3 = rnorm(n, mean = 0, sd = 1),
    x4 = sample(c(0, 1), size = n, replace = TRUE),
    x5 = sample(c(1, 2, 3), size = n, replace = TRUE)
  )
}

easier_dgp <- function(n, seed, effect_type = "homogeneous", functional_form = "linear") {
  
  calc_mu <- function(X) {
    g_func <- ifelse(X$x5 == 1, 2, ifelse(X$x5 == 2, -1, -4))
    if (functional_form == "linear") {
      return(1 + g_func + X$x1 * X$x3)
    } else {
      return(-6 + g_func + 6 * abs(X$x3 - 1))
    }
  }
  
  calc_tau <- function(X) {
    if (effect_type == "homogeneous") {
      return(rep(3, nrow(X)))
    } else {
      return(1 + 2 * X$x2 * X$x4)
    }
  }
  
  calc_pi <- function(X, mu) {
    s <- sd(mu)
    u <- runif(nrow(X), min = 0, max = 1)
    return(0.8 * pnorm((3 * mu / s) - 0.5 * X$x1) + 0.05 + u / 10)
  }
  
  return(generate_data_universal(n, seed, gen_X_complex, calc_mu, calc_tau, calc_pi, sd_error = 1))
}

dgp <- function(n, seed, effect_type = "homogeneous", functional_form = "linear") {
  # Almost identical to easier_dgp, only the interaction for heterogeneous tau changes (uses x5 instead of x4)
  calc_mu <- function(X) {
    g_func <- ifelse(X$x5 == 1, 2, ifelse(X$x5 == 2, -1, -4))
    if (functional_form == "linear") return(1 + g_func + X$x1 * X$x3) else return(-6 + g_func + 6 * abs(X$x3 - 1))
  }
  
  calc_tau <- function(X) {
    if (effect_type == "homogeneous") return(rep(3, nrow(X))) else return(1 + 2 * X$x2 * X$x5)
  }
  
  calc_pi <- function(X, mu) {
    s <- sd(mu)
    u <- runif(nrow(X), min = 0, max = 1)
    return(0.8 * pnorm((3 * mu / s) - 0.5 * X$x1) + 0.05 + u / 10)
  }
  
  return(generate_data_universal(n, seed, gen_X_complex, calc_mu, calc_tau, calc_pi, sd_error = 1))
}

dgp_enriched <- function(n, seed, effect_type = "homogeneous", functional_form = "linear") {
  calc_mu <- function(X) {
    g_func <- ifelse(X$x5 == 1, 2, ifelse(X$x5 == 2, -1, -4))
    if (functional_form == "linear") return(1 + g_func + X$x1 * X$x3) else return(-6 + g_func + 6 * abs(X$x3 - 1))
  }
  
  calc_tau <- function(X) {
    if (effect_type == "homogeneous") return(rep(3, nrow(X))) else return(1 + 2 * X$x2 * X$x5)
  }
  
  # Here the propensity score contains the enriched dynamics (+ 0.3 * x4)
  calc_pi <- function(X, mu) {
    s <- sd(mu)
    u <- runif(nrow(X), min = 0, max = 1)
    return(0.8 * pnorm((3 * mu / s) - 0.5 * X$x1 + 0.3 * X$x4) + 0.05 + u / 10)
  }
  
  return(generate_data_universal(n, seed, gen_X_complex, calc_mu, calc_tau, calc_pi, sd_error = 1))
}

ht_nl_dgp <- function(n,seed){
  return(dgp(n, seed, effect_type = "ht", functional_form = "linear"))
}

ht_l_dgp <- function(n,seed){
  return(dgp(n, seed, effect_type = "ht", functional_form = "nl"))
}