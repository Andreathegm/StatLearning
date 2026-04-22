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

plot.heatmap.results <- function(matrice, n_vec, nsim_vec, p, method) {
  # 1. Configurazione Layout
  layout(matrix(c(1, 2), nrow = 1), widths = c(0.85, 0.15))
  
  # Margini per il grafico principale
  par(mar = c(5, 5, 4, 1)) 
  
  breaks <- seq(0, 1, length.out = 101)
  
  # Palette personalizzata (70% rosso/arancio, poi transizione al verde)
  c1 <- colorRampPalette(c("red", "red", "darkorange"))(70)
  c2 <- colorRampPalette(c("darkorange", "greenyellow"))(20)
  c3 <- colorRampPalette(c("greenyellow", "darkgreen"))(5)
  c4 <- colorRampPalette(c("darkgreen", "lawngreen"))(5)
  col_palette <- c(c1, c2, c3, c4)
  
  # 2. Disegno della Heatmap principale
  image(1:nrow(matrice), 1:ncol(matrice), matrice, 
        col = col_palette, 
        breaks = breaks,
        axes = FALSE, 
        xlab = "nsim", 
        ylab = "n",
        # MODIFICA: cex.main riduce la dimensione del titolo (0.8 è l'80% del default)
        main = paste0("CI actual coverage (Target 0.95) for p = ", p, "\n", method),
        cex.main = 0.9) 
  
  axis(1, at = 1:nrow(matrice), labels = nsim_vec, las = 1, cex.axis = 0.9)
  axis(2, at = 1:ncol(matrice), labels = n_vec, las = 1, cex.axis = 0.9)
  
  for (i in 1:nrow(matrice)) {
    for (j in 1:ncol(matrice)) {
      valore <- round(matrice[i, j], 3)
      text_col <- "black"
      text(i, j, labels = valore, font = 2, cex = 0.8, col = text_col)
    }
  }
  box(lwd = 1.5)
  
  # 3. Disegno della Legenda (Color Bar)
  par(mar = c(5, 1, 4, 3))
  legend_image <- as.matrix(seq(0, 1, length.out = 100))
  
  image(x = 1, y = breaks, z = t(legend_image), 
        col = col_palette, 
        breaks = breaks,
        axes = FALSE, xlab = "", ylab = "")
  
  axis(4, at = seq(0, 1, by = 0.1), las = 1, cex.axis = 0.8)
  abline(h = 0.95, col = "white", lwd = 2, lty = 2)
  
  layout(1) 
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

plot.actual.coverage <- function(ps, ys, ylim, title, nominal.level) {
  par(mar = c(5, 5, 4, 2))
  plot(ps, ys,
       type = "l",
       col = "blue",
       lwd = 1.5,
       xlab = "p",
       ylab = "Actual cover probability",
       main = title,
       ylim = ylim,
       xaxt = "n", 
       yaxt = "n") 
  
  # Asse X
  axis(1, at = seq(0, 1, by = 0.05), labels = FALSE)
  etichette_x <- seq(0, 0.9, by = 0.15)
  axis(1, at = etichette_x, 
       labels = sprintf("%.2f", etichette_x), 
       tick = FALSE) 
  
  # Asse Y 
  axis(2, at = c(0.9, 1.0), 
       labels = c("0.9", "1.0"), 
       las = 1)
  
  # Linea tratteggiata
  abline(h = nominal.level,
         col = "red",
         lwd = 1.5,
         lty = 2)
}

  

