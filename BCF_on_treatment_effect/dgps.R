easier_dgp<- function(n = 250, effect_type = "homogeneous", functional_form = "linear") {
  x1 <- rnorm(n, mean = 0, sd = 1)
  x2 <- rnorm(n, mean = 0, sd = 1)
  x3 <- rnorm(n, mean = 0, sd = 1)
  x4 <- sample(c(0, 1), size = n, replace = TRUE)
  x5 <- sample(c(1, 2, 3), size = n, replace = TRUE)
  
  g_func <- function(x) {
    ifelse(x == 1, 2,
           ifelse(x == 2, -1, -4))
  }
  
  if (effect_type == "homogeneous") {
    tau <- rep(3, n)
  } else if (effect_type == "heterogeneous") {
    tau <- 1 + 2 * x2 * x4
  }
  
  if (functional_form == "linear") {
    mu <- 1 + g_func(x5) + x1 * x3
  } else if (functional_form == "nonlinear") {
    mu <- -6 + g_func(x5) + 6 * abs(x3 - 1)
  }
  
  s <- sd(mu)
  u <- runif(n, min = 0, max = 1)
  pi_x <- 0.8 * pnorm((3 * mu / s) - 0.5 * x1) + 0.05 + u / 10
  
  z <- rbinom(n, size = 1, prob = pi_x)
  epsilon <- rnorm(n, mean = 0, sd = 1)
  
  y <- mu + tau * z + epsilon
  
  X_covariates <- data.frame(x1, x2, x3, x4, x5)
  
  return(list(
    X = X_covariates,    # predictors
    Y = y,               # response variable
    Z = z,               # Treatment variable (0 or 1)
    pi_true = pi_x,      # real propensity score
    mu_true = mu,        # real prognostic score
    tau_true = tau       # treatment effect
  ))
}

dgp<- function(n = 250, effect_type = "homogeneous", functional_form = "linear") {
  x1 <- rnorm(n, mean = 0, sd = 1)
  x2 <- rnorm(n, mean = 0, sd = 1)
  x3 <- rnorm(n, mean = 0, sd = 1)
  x4 <- sample(c(0, 1), size = n, replace = TRUE)
  x5 <- sample(c(1, 2, 3), size = n, replace = TRUE)
  
  g_func <- function(x) {
    ifelse(x == 1, 2,
           ifelse(x == 2, -1, -4))
  }
  
  if (effect_type == "homogeneous") {
    tau <- rep(3, n)
  } else if (effect_type == "heterogeneous") {
    tau <- 1 + 2 * x2 * x5
  }
  
  if (functional_form == "linear") {
    mu <- 1 + g_func(x5) + x1 * x3
  } else if (functional_form == "nonlinear") {
    mu <- -6 + g_func(x5) + 6 * abs(x3 - 1)
  }
  
  s <- sd(mu)
  u <- runif(n, min = 0, max = 1)
  pi_x <- 0.8 * pnorm((3 * mu / s) - 0.5 * x1) + 0.05 + u / 10
  
  z <- rbinom(n, size = 1, prob = pi_x)
  epsilon <- rnorm(n, mean = 0, sd = 1)
  
  y <- mu + tau * z + epsilon
  
  X_covariates <- data.frame(x1, x2, x3, x4, x5)
  
  return(list(
    X = X_covariates,    # predictors
    Y = y,               # response variable
    Z = z,               # Treatment variable (0 or 1)
    pi_true = pi_x,      # real propensity score
    mu_true = mu,        # real prognostic score
    tau_true = tau       # treatment effect
  ))
}

dgp_enriched<- function(n = 250, effect_type = "homogeneous", functional_form = "linear") {
  x1 <- rnorm(n, mean = 0, sd = 1)
  x2 <- rnorm(n, mean = 0, sd = 1)
  x3 <- rnorm(n, mean = 0, sd = 1)
  x4 <- sample(c(0, 1), size = n, replace = TRUE)
  x5 <- sample(c(1, 2, 3), size = n, replace = TRUE)
  
  g_func <- function(x) {
    ifelse(x == 1, 2,
           ifelse(x == 2, -1, -4))
  }
  
  if (effect_type == "homogeneous") {
    tau <- rep(3, n)
  } else if (effect_type == "heterogeneous") {
    tau <- 1 + 2 * x2 * x5
  }
  
  if (functional_form == "linear") {
    mu <- 1 + g_func(x5) + x1 * x3
  } else if (functional_form == "nonlinear") {
    mu <- -6 + g_func(x5) + 6 * abs(x3 - 1)
  }
  
  s <- sd(mu)
  u <- runif(n, min = 0, max = 1)
  pi_x <- 0.8 * pnorm((3 * mu / s) - 0.5 * x1 + 0.3 * x4) + 0.05 + u / 10
  pi_x <- pmin(pmax(pi_x, 0.01), 0.99)
  
  z <- rbinom(n, size = 1, prob = pi_x)
  epsilon <- rnorm(n, mean = 0, sd = 1)
  
  y <- mu + tau * z + epsilon
  
  X_covariates <- data.frame(x1, x2, x3, x4, x5)
  
  return(list(
    X = X_covariates,    # predictors
    Y = y,               # response variable
    Z = z,               # Treatment variable (0 or 1)
    pi_true = pi_x,      # real propensity score
    mu_true = mu,        # real prognostic score
    tau_true = tau       # treatment effect
  ))
}