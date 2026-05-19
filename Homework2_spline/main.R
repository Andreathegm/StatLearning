library(ggplot2)
require(MASS)
library(gam)
library(splines)



data(Boston,package = "MASS")
summary(Boston)
data_pulito <- na.omit(Boston)
summary(data_pulito)

set.seed(1234)  # per riproducibilità

n <- nrow(data_pulito)
split = 0.8
train_index <- sample(1:n, size = split * n)

train <- data_pulito[train_index, ]
test  <- data_pulito[-train_index, ]


modello_regressione_full <- lm(rm ~ . , data =  train)


step_aic <- stepAIC(modello_regressione_full, direction = "backward")

summary(step_aic)

qqnorm(residuals(step_aic))
qqline(residuals(step_aic), col = "red")




fit_gam <- gam(rm ~ ns(zn, 1) + ns(indus, 3) + ns(age, 3) + 
                 ns(rad, 3) + ns(black, 3) + 
                 ns(lstat, 3) + ns(medv, 3), 
               data = train)
class(fit_gam)
summary(train$zn)

termplot(step_aic,se = TRUE, partial.resid = TRUE, col.res = "#905E9F", pch = 16, col.term = "black", col.se = "black")

qqnorm(residuals(fit_gam))
qqline(residuals(fit_gam), col = "red")

plot(fit_gam,residuals = TRUE,pch = 16,col = "#905E9F",se = TRUE)
summary(fit_gam)

mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

pred_lm <- predict(step_aic, newdata = test)

mse(test$medv, pred_lm)

pred_gam <- predict(fit_gam, newdata = test)

mse(test$medv, pred_gam)

