source("methods.R")
source("run_benchmark.R")
source("dgps.R")

### MAIN

my_real_methods <- list(
  "BCF"      = run_bcf_method,
  "BART"     = run_bart_method,
  "ps-BART"  = run_ps_bart_method
)

set.seed(1234)

run_all(
  methods = my_real_methods,
  dgp = dgp,
  configs = c("heter_linear", "heter_nonlinear", "homo_linear", "homo_nonlinear"),
  prefix = "paper_"
)

run_all(
  methods = my_real_methods,
  dgp = dgp_enriched,
  configs = c("heter_linear", "heter_nonlinear"),
  prefix = "enriched_"
)