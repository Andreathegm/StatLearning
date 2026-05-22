source("ECDF_test.R")
source("FWER_test.R")

set.seed(123)

n_obs  <- 50     # observations
p  <- 100          # variables -> p >> n
s0 <- 5
B <- 10
alpha <- 0.05

ECDF_test(n_obs,p,B,s0,alpha)

N_mc = 10         # num. of Montecarlo simulation
FWER_test(n_obs,p,B,s0,alpha,N_mc)