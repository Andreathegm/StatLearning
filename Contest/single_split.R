one_split <- function(x, y){
  
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
    ols.pvals <- as.numeric(ols.pvals)
  }
  
  # assegna i p-value
  pvals[selected] <- ols.pvals
  
  return(list(
    pval = pvals,
    selected = selected
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

beta[active] <- 2

Y <- X %*% beta + rnorm(n)

Y <- as.numeric(Y)

res <- one_split(X, Y)

res$pval

res$selected
active