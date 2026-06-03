source("methods.R")
source("run_benchmark.R")
source("simulation_study.R")

my_real_methods <- list(
  "BART"     = run_bart_method,
  "ps-BART"  = run_ps_bart_method,
  "BCF"      = run_bcf_method
)

set.seed(1234)
comparion_table <- run_simulation_benchmark(
  n_sims = 50, 
  n_obs = 250, 
  effect = "heterogeneous", 
  form = "linear", 
  methods_list = my_real_methods,
  dgp = dgp
)

print(comparison_table)