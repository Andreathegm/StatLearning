library(huge)
library(hdi)

set.seed(123)

n  <- 100      # osservazioni
p  <- 20     # variabili -> p >> n
s0 <- 5      # numero variabili realmente attive

data <- huge.generator(
  n = n,
  d = p,
  graph = "random",     # struttura con correlazioni
  v = 0.3,
  u = 0.1
)

data$theta

X <- data$data
dim(X)
class(data)

beta <- rep(0, p) #crea un vettore di p zeri

#active contiene le variabili segnale
active <- sample(1:p, s0)

#coefficienti delle variabili di segnale
beta[active] <- runif(s0, 1, 5)

active
beta[active]
active <- sort(active)

sigma <- 1

Y <- X %*% beta + rnorm(n, sd = sigma)

set.seed(17)
fit.multi <- hdi(
  x= X,
  y= Y,
  B=1,
  verbose = TRUE
)

summary(fit.multi)

fit.multi$selected
fit.multi$pval
which(fit.multi$pval.corr < 0.05)
active


