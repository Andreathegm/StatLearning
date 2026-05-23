library(huge)
generate_data <- function(n_obs,n_var){
  data <- huge.generator(
    n = n_obs,
    d = n_var,
    graph = "hub",     
    v = 0.3,
    u = 0.1
  )
  return (data$data)
}

linear_dgp <- function(X,beta,n,sigma){
  Y <- X %*% beta + rnorm(n, sd = sigma)
  return(Y)
}

calculate_metrics <- function(selected_vars, active_vars) {
  TP <- length(intersect(selected_vars, active_vars))
  FP <- length(setdiff(selected_vars, active_vars))
  FDR <- if ((TP + FP) > 0) FP / (TP + FP) else 0
  # FWER indicator: 1 if there is AT LEAST ONE False Positive, 0 otherwise
  FWER_ind <- ifelse(FP > 0, 1, 0)
  
  return(c(TP = TP, FP = FP,FDR = FDR, FWER_ind = FWER_ind))
}