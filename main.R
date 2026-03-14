source("utils.R")
alpha = 0.05

## First Plot
p_vec <- c(0.01,0.5,0.9)
p_vec <- c(0.001)
function_vec <- list(wald.conf.int,wilson.conf.int)
n <- 1000
names <- c(paste0("wald, n = ",n),paste0("wilson, n = ",n))
for (i in 1:length(p_vec)){
  
  for (j in 1:length(function_vec)){
    
    cov1 <- montecarlo.CI.mean(p_vec[i],1000,alpha,50,function_vec[[j]])
    plot.montecarlo.CIs(p_vec[i],n.sim = 50,CIs = cov1$CIs,p.insides = cov1$p.insides,names[j])
  }
}









## Second Plot
n_vec <- c(50, 100, 400, 1600, 6400)
nsim_vec <- c(50, 100, 400, 1600, 6400)

p_vec <- c(0.01,0.9)
p_vec <- c(0.001)
function_vec <- list(wald.conf.int,wilson.conf.int)
names <- c("Wald","Wilson")
for (i in 1:length(p_vec)){
  
  for (j in 1:length(function_vec)){
    
    n.nsimMatrix <- montecarlo.CI.mean.p.nsim.matrix(nsim_vec = nsim_vec,n_vec = n_vec,alpha = alpha,p_vec[i],function_vec[[j]])
    plot.heatmap.results(n.nsimMatrix,n_vec,nsim_vec,p_vec[i],names[j])
    paste0(i,"(",j,")"," plot completed")
    
    }
  }





### Third Plot
n <- 1000
n.sim <- 500
ps <- seq(from=0.000,to=1,by=0.005)
function_vec <- list(wald.conf.int,wilson.conf.int)
ylim <- c(0.85,1)
names <- c("Wald","Wilson")
for (i in 1:length(function_vec)){
  ys <- montercarlo.CI.meanfunction(ps,n,alpha,n.sim,function_vec[[i]])
  plot.actual.coverage(ps,ys,ylim,paste0("Actual Cover Probability - ",names[i]),1-alpha)
  }



  