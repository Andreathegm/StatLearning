montecarlo.CI.mean <- function(p, n, alpha, n.sim, conf.int.function) {
  sign.level <- 1 - alpha/2 
  z.sign.level <- qnorm(sign.level)
  p.hats <- vector(mode="numeric", length=n.sim)
  CIs <- data.frame(CI.lower=numeric(n.sim), CI.upper=numeric(n.sim))
  p.insides <- vector(mode="numeric", length=n.sim)
  
  for (i in 1:n.sim) {
    set.seed(i)
    x <- rbinom(n, 1, p)
    p.hat <- mean(x)
    p.hats[i] <- p.hat
    CIs[i,] <- conf.int.function(p.hat,z.sign.level,n)
    p.insides[i] <- as.numeric(p >= CIs[i,1] & p <= CIs[i,2])
  }
  
  conf.level.estimate <- mean(p.insides)
  
  return(list(estimate = conf.level.estimate, CIs = CIs, p.insides = p.insides, n.sim = n.sim))
}

plot.montecarlo.CIs <- function(p,n.sim,CIs,p.insides,method) {
  
  plot(0, 0, xlim = c(min(CIs$CI.lower),max(CIs$CI.upper)), ylim = c(1, n.sim), type = "n",
       xlab = "proportion", ylab = "n.sim",
       main = paste0("CI actual coverage 95% p = ",p,"\n",method))
  
  abline(v = p, lwd = 2, lty = 2)
  colori <- ifelse(p.insides == 1, "darkgray", "red")
  
  segments(x0 = CIs$CI.lower, y0 = 1:n.sim,
           x1 = CIs$CI.upper, y1 = 1:n.sim, 
           col = colori, lwd = 2)
}

plot.heatmap.results <- function(matrice, n_vec, nsim_vec,p,method) {
  par(mar = c(5, 5, 4, 2)) 
  
  target <- 0.95
  ampiezza <- 0.05
  breaks <- seq(target - ampiezza, target + ampiezza, length.out = 101)
  col_func <- colorRampPalette(c("red", "yellow", "darkgreen", "yellow", "red"))
  col_palette <- col_func(100)
  
  image(1:nrow(matrice), 1:ncol(matrice), matrice, 
        col = col_palette, 
        breaks = breaks,
        axes = FALSE, 
        xlab = "nsim", 
        ylab = "n",
        main = paste0("CI actual covarage (Target 0.95) for p = ",p,"\n",method))
  
  axis(1, at = 1:nrow(matrice), labels = nsim_vec, las = 1, cex.axis = 0.9)
  axis(2, at = 1:ncol(matrice), labels = n_vec, las = 1, cex.axis = 0.9)
  

  for (i in 1:nrow(matrice)) {
    for (j in 1:ncol(matrice)) {
      valore <- round(matrice[i, j], 3)
      text(i, j, labels = valore, font = 2, cex = 0.8)
    }
  }
  
  box(lwd = 1.5) 
}


montecarlo.CI.mean.p.nsim.matrix <- function(nsim_vec,n_vec,alpha,p,conf.int.function){
  matrice_risultati <- matrix(NA, 
                              nrow = length(nsim_vec), 
                              ncol = length(n_vec))
  
  rownames(matrice_risultati) <- paste0("nsim_", nsim_vec)
  colnames(matrice_risultati) <- paste0("n_", n_vec)
  
  
  for (i in 1:length(nsim_vec)) {
    for (j in 1:length(n_vec)) {
      res <- montecarlo.CI.mean(p, n_vec[j], alpha, nsim_vec[i],conf.int.function)
      matrice_risultati[i, j] <- res$estimate
    }
  }
  return (matrice_risultati)
}

montercarlo.CI.meanfunction <- function(ps,n,alpha,n.sim,conf.int.function){
  ys <- c()
  for(i in 1:length(ps))
    ys[i] <- montecarlo.CI.mean(ps[i],n,alpha,n.sim,conf.int.function)$estimate
  return (ys)
}

wilson.conf.int <- function(p.hat, quantile, n){
  adj.p.hat <- p.hat + (quantile^2) / (2 * n)
  denom <- 1 + (quantile^2) / n
  
  adj.std <- sqrt( (p.hat * (1 - p.hat) / n) + (quantile^2 / (4 * n^2)) )
  
  CI.lower <- (adj.p.hat - quantile * adj.std) / denom
  if (CI.lower < 1e-12)
      CI.lower <- 0
  CI.upper <- (adj.p.hat + quantile * adj.std) / denom
  
  return(c(CI.lower, CI.upper))
}

wald.conf.int <- function(p.hat,quantile,n){
  var.p.hat <- ((p.hat)*(1-p.hat))/n
  CI.lower <- p.hat - (quantile*(sqrt(var.p.hat)))
  CI.upper <- p.hat + (quantile*(sqrt(var.p.hat)))
  return (c(CI.lower,CI.upper))
}
  

