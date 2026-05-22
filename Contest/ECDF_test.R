source("plot.R")
source("multisplit.R")
source("utils.R")

ECDF_test <- function(n_obs,p,B,s0,alpha,print_only_active_var = TRUE){
  
  
active <- sample(1:p, s0) # num of variables that are active
active <- sort(active)


beta <- rep(0, p)
beta[active] <- runif(s0, 1, 5)

####    MULTI-SPLIT done by 'our hands' to calculate p_value
####    distribution for each variable j for j in [1,p]

X <- generate_data(n_obs = n_obs,n_var = p)
Y <- linear_dgp(X,beta,n_obs,1)
p_values_dataframe <- multisplit(
    x = X,
    y = Y,
    alpha = alpha,
    B = B
  )
  ### plot ecdf for each variable
if (print_only_active_var){
    indexes <- active
    print(active)
} else {
  indexes <- seq_len(nrow(p_values_dataframe))
}
  
for (i in indexes)
  plot_pvalue_ecdf(
        as.numeric(p_values_dataframe[i,]),
        main_title = paste0("ecdf of variable",i)
      )
  
}


