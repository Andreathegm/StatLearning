source("methods.R")
source("run_benchmark.R")
source("simulation_study.R")

my_real_methods <- list(
  "BCF"      = run_bcf_method,
  "BART"     = run_bart_method,
  "ps-BART"  = run_ps_bart_method
)

set.seed(1234)

## heter./linear
table_heter_linear <- run_simulation_benchmark(
  n_sims = 200, 
  n_obs = 250, 
  effect = "heterogeneous", 
  form = "linear", 
  methods_list = my_real_methods,
  dgp = dgp
)
saveRDS(table_heter_linear, "results_heter_linear.rds")
write.csv(table_heter_linear, "results_heter_linear.csv", row.names = TRUE)

## heter./nonlinear
table_heter_nonlinear <- run_simulation_benchmark(
  n_sims = 200, 
  n_obs = 250, 
  effect = "heterogeneous", 
  form = "nonlinear", 
  methods_list = my_real_methods,
  dgp = dgp
)
saveRDS(table_heter_nonlinear, "results_heter_nonlinear.rds")
write.csv(table_heter_nonlinear, "results_heter_nonlinear.csv", row.names = TRUE)

## homo./linear
table_homo_linear <- run_simulation_benchmark(
  n_sims = 200, 
  n_obs = 250, 
  effect = "homogeneous", 
  form = "linear", 
  methods_list = my_real_methods,
  dgp = dgp
)
saveRDS(table_homo_linear, "results_homo_linear.rds")
write.csv(table_homo_linear, "results_homo_linear.csv", row.names = TRUE)

## homo./nonlinear
table_homo_nonlinear <- run_simulation_benchmark(
  n_sims = 200, 
  n_obs = 250, 
  effect = "homogeneous", 
  form = "nonlinear", 
  methods_list = my_real_methods,
  dgp = dgp
)
saveRDS(table_homo_nonlinear, "results_homo_nonlinear.rds")
write.csv(table_homo_nonlinear, "results_homo_nonlinear.csv", row.names = TRUE)

print(table_heter_linear)
print(table_heter_nonlinear)
print(table_homo_linear)
print(table_homo_nonlinear)