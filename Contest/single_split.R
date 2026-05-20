one_split <- function(x, y, alpha= 0.05){
  
  x <- as.matrix(x)
  y <- as.numeric(y)
  
  n <- nrow(x)
  p <- ncol(x)

  idx.sel <- sample(seq_len(n), floor(n/2))
  
  idx.inf <- setdiff(seq_len(n), idx.sel)
  
  X.sel <- x[idx.sel, , drop = FALSE]
  Y.sel <- y[idx.sel]
  
  X.inf <- x[idx.inf, , drop = FALSE]
  Y.inf <- y[idx.inf]

  library(glmnet)
  
  cvfit <- cv.glmnet(
    X.sel,
    Y.sel,
    alpha = 1
  )
  
  beta.hat <- coef(cvfit, s = "lambda.min")
  
  # variabili selezionate
  selected <- which(beta.hat[-1] != 0)
  
  pvals <- rep(1, p)
  
  # se il lasso non seleziona nulla
  if(length(selected) == 0){
    return(list(
      pval = pvals,
      selected = selected
    ))
  }

  
  dat.inf <- data.frame(
    y = Y.inf,
    X.inf[, selected, drop = FALSE]
  )
  
  fit.ols <- lm(y ~ ., data = dat.inf)
  
  # p-value OLS
  ols.pvals <- summary(fit.ols)$coefficients[-1, 4]
  
  # caso singola variabile selezionata
  if(length(selected) == 1){
    ols.pvals <- as.numeric(ols.pvals*length(selected))
  }
  
  # assegna i p-value
  pvals[selected] <- ols.pvals * length(selected)
  pvals[pvals > 1] = 1
    
  
  return(list(
    pval = pvals,
    selected = selected[which(pvals[selected] < alpha)]
  ))
}

library(huge)

set.seed(123)

n <- 100
p <- 20
s0 <- 5

dat <- huge.generator(
  n = n,
  d = p,
  graph = "random",
  v = 0.3,
  u = 0.1
)

X <- scale(dat$data)

beta <- rep(0, p)

active <- sample(1:p, s0)

beta[active] <- runif(s0, 1, 5)

Y <- X %*% beta + rnorm(n, sd=1)

Y <- as.numeric(Y)

res <- one_split(X, Y)

res$pval

res$selected
active


# p-value esempio
pvalues <- c(0.01, 0.20, 0.03, 0.50, 0.15, 0.80, 0.04)

# Costruzione istogramma SENZA plot
h <- hist(pvalues,
          breaks = seq(0, 1, by = 0.1),
          plot = FALSE)

# Frequenze relative
freq_rel <- h$counts / sum(h$counts)

# Plot delle percentuali
barplot(freq_rel,
        names.arg = paste(head(h$breaks, -1),
                          tail(h$breaks, -1),
                          sep = "-"),
        col = "lightgray",
        ylab = "Percentuale",
        xlab = "Bin p-value",
        main = "Percentuale di p-value per bin")
par(new = TRUE)

plot(ecdf(pvalues),
     verticals = TRUE,
     do.points = FALSE,
     col = "red",
     lwd = 2,
     axes = FALSE,
     xlab = "",
     ylab = "",
     ylim = c(0, 1))

# Asse destro ECDF
axis(4)

# Label asse destro
mtext("ECDF", side = 4, line = 3)
# Dati iniziali
pvalues <- c(0.01, 0.20, 0.03, 0.50, 0.15, 0.80, 0.04)

# 1. Costruzione istogramma SENZA plot
h <- hist(pvalues,
          breaks = seq(0, 1, by = 0.1),
          plot = FALSE)

# 2. Modifica in frequenze relative
# Inseriamo i nuovi valori direttamente dentro l'oggetto 'h'
h$counts <- h$counts / sum(h$counts)

# 3. Plot dell'istogramma (che ha un asse X continuo compatibile con la ECDF)
plot(h,
     freq = TRUE,                  # Usa i nostri count modificati
     col = "lightgray",
     ylim = c(0, 1),               # UNICA scala Y (condivisa tra barre ed ECDF)
     xlim = c(0, 1),
     axes = FALSE,                 # Disattiviamo gli assi di default per renderli più puliti
     xlab = "p-value",
     ylab = "Frequenza Relativa ed ECDF",
     main = "Distribuzione p-value ed ECDF")

# 4. Assi personalizzati (più "bellini" ed equidistanti)
# Asse X: mettiamo i tick esattamente in corrispondenza dei bordi dei bin (0, 0.1, 0.2...)
axis(1, at = seq(0, 1, by = 0.1))

# Asse Y: da 0 a 1
axis(2, at = seq(0, 1, by = 0.2))

# 5. Sovrapposizione della ECDF
# Usando add = TRUE la ECDF userà lo STESSO sistema di coordinate (X e Y) dell'istogramma.
plot(ecdf(pvalues),
     verticals = TRUE,
     do.points = FALSE,
     col = "red",
     lwd = 2,
     add = TRUE)

plot_pvalue_ecdf <- function(data, 
                             alpha = 0.05, # Nuovo parametro richiesto
                             breaks_seq = seq(0, 1, by = 0.1), 
                             main_title = "Distribuzione p-value, ECDF e f(p)",
                             x_label = "p-value") {
  
  data <- na.omit(data)
  
  # 1. Costruzione istogramma
  h <- hist(data, breaks = breaks_seq, plot = FALSE)
  h$counts <- h$counts / sum(h$counts)
  
  # 2. Plot dell'istogramma modificato
  plot(h,
       freq = TRUE,                  
       col = "lightgray",
       ylim = c(0, 1),               
       xlim = range(breaks_seq),     
       axes = FALSE,                 
       xlab = x_label,
       ylab = "Frequenza Relativa / ECDF / f(p)",
       main = main_title)
  
  # 3. Assi personalizzati
  axis(1, at = breaks_seq)           
  axis(2, at = seq(0, 1, by = 0.2))  
  
  # 4. Sovrapposizione della ECDF (Linea Rossa Continua)
  plot(ecdf(data),
       verticals = TRUE,
       do.points = FALSE,
       col = "red",
       lwd = 2,
       add = TRUE)
  
  # 5. Sovrapposizione della funzione custom f(p) (Linea Blu Tratteggiata)
  # Nota: R richiede espressamente la variabile 'x' dentro la funzione curve()
  curve(pmax(0.05, (3.996 / alpha) * x), 
        from = 0, 
        to = 1, 
        add = TRUE, 
        lty = 2,          # lty = 2 imposta la linea come tratteggiata (dashed)
        col = "blue",     
        lwd = 2)          
}

miei_pvalues <- c(0.01, 0.20, 0.03, 0.50, 0.15, 0.80, 0.04)

# Esecuzione con alpha = 0.05 (Default)
plot_pvalue_ecdf(miei_pvalues, alpha = 0.05)

# Esecuzione con un alpha più grande per vedere meglio la pendenza della retta blu
plot_pvalue_ecdf(miei_pvalues, alpha = 5)
