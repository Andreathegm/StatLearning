source("evaluate_methods.R")

run_simulation_benchmark <- function(n_sims = 200, n_obs = 250, effect = "heterogeneous", form = "linear", methods_list,dgp) {
  
  all_results <- list()
  
  for (i in 1:n_sims) {
    print(paste0("Simulation",i,"/",n_sims))
 
    
    sim_data <- dgp(n = n_obs, effect_type = effect, functional_form = form)
    
    iter_results <- matrix(NA, nrow = length(methods_list), ncol = 6)
    rownames(iter_results) <- names(methods_list)
    colnames(iter_results) <- c("ATE_rmse", "ATE_cover", "ATE_len", "CATE_rmse", "CATE_cover", "CATE_len")
    
    for (method_name in names(methods_list)) {
      method_func <- methods_list[[method_name]]
      
      model_output <- method_func(sim_data)
      
      iter_results[method_name, ] <- evaluate_method(sim_data, model_output)
    }
    
    all_results[[i]] <- iter_results
  }
  
  final_table <- Reduce("+", all_results) / n_sims
  
  return(round(final_table, 3))
}