source("ECDF_test.R")
source("FWER_test.R")

set.seed(123)

n_obs  <- 100     # observations
p  <- 1000          # variables -> p >> n
s0 <- 10
B <- 50
alpha <- 0.05

ECDF_test(n_obs,p,B,s0,alpha)

N_mc = 50         # num. of Montecarlo simulation
res = FWER_test(n_obs,p,B,s0,alpha,N_mc)
res$res_single
res$res_multi
