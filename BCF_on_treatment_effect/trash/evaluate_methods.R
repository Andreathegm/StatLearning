source("utils.R")

evaluate_method <- function(true_data, method_results) {
  
  true_cate <- true_data$tau_true
  true_ate  <- mean(true_cate)
  
  # ATE metrics
  ate_rmse  <- rmse(true_ate, method_results$ATE$est)
  ate_cover <- coverage(true_ate, method_results$ATE$lower, method_results$ATE$upper)
  ate_len   <- avarage_coverage_length(method_results$ATE$lower, method_results$ATE$upper)
  
  # CATE metrics
  cate_rmse  <- rmse(true_cate, method_results$CATE$est)
  cate_cover <- coverage(true_cate, method_results$CATE$lower, method_results$CATE$upper)
  cate_len   <- avarage_coverage_length(method_results$CATE$lower, method_results$CATE$upper)
  
  return(c(
    ATE_rmse = ate_rmse, ATE_cover = ate_cover, ATE_len = ate_len,
    CATE_rmse = cate_rmse, CATE_cover = cate_cover, CATE_len = cate_len
  ))
}