one_split <- function(x, y, alpha= 0.05){
  
  x <- as.matrix(x)
  y <- as.numeric(y)
  
  n <- nrow(x)
  p <- ncol(x)

  idx.sel <- sample(seq_len(n), floor((n-1)/2))
  
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