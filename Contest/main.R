library(huge)
library(hdi)
source("multisplit.R")
source("single_split.R")

set.seed(123)

n_obs  <- 100      # osservazioni
p  <- 20     # variabili -> p >> n
s0 <- 5      # numero variabili realmente attive
active <- sample(1:p, s0)

### we define the betas . They'll stay constant during montecarlo simulation
beta <- rep(0, p)
beta[active] <- runif(s0, 1, 5)

generate_data <- function(n_obs,n_var){
    data <- huge.generator(
    n = n_obs,
    d = n_var,
    graph = "random",     # struttura con correlazioni
    v = 0.3,
    u = 0.1
  )
  return (data$data)
}

X <- data$data
active <- sort(active)

linear_dgp <- function(X,beta,n,sigma){
  Y <- X %*% beta + rnorm(n, sd = sigma)
  return(Y)
}

#### Now we process the data
#### 1) MULTI-SPLIT done by 'our hands' to calculate p_value
####    distribution for each variable j for j in [1,p]

#### DEFINITION OF CONSTANTS
B = 10
N = 1

for (i in seq_len(N)){
  X <- generate_data(n_obs = n_obs,n_var = p)
  Y = linear_dgp(X,beta,n_obs,1)
  p_values_dataframe <- multisplit(
    x = X,
    y = Y,
    alpha = 0.05,
    B = B
  )
  ### 2) plot ecdf for each variable
  for (i in seq_len(nrow(p_values_dataframe)))
    plot_pvalue_ecdf(
      p_values_dataframe[i,],
      main_title = paste0("ecdf of p-values of variable",i)
    )
 }

### 3) Using third-party multi-split to get #of E[FP] (expectation of false positive)
###    E[TP] , E[FP/(TP+FP)]
N = 10

## single split
FPs <- c()
for(i in seq_len(N))
   
  fit.multi <- hdi(
    x = X,
    y = Y,
    B = 1,
    gamma = c(1),
    verbose = TRUE
  )
  
  summary(fit.multi)
  fit.multi$selected
  fit.multi$pval
  
  significat_var <- which(fit.multi$pval.corr < 0.05)
  active
  
  FP <- length(significat_var - active)
  TP <- lenght(significat_var) - FP
  
  FP_fraction <- FP/(FP+TP)




