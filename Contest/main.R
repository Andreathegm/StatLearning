source("ECDF_test.R")
source("FWER_test.R")

set.seed(123)

n_obs  <- 100     # observations
p  <- 1000          # variables -> p >> n
s0 <- 10
B <- 10000
alpha <- 0.05
snr <- 16
print_sel_var <- FALSE

ECDF_test(n_obs,p,B,s0,alpha,snr,print_only_active_var = print_sel_var)

N_mc = 500# num. of Montecarlo simulation
B=50
res = FWER_test(n_obs,p,B,s0,alpha,N_mc,snr)
res$res_single
res$res_multi
