library(ggplot2)
require(MASS)
library(gam)
library(splines)



data(Boston,package = "MASS")
summary(Boston)
data_pulito <- na.omit(Boston)
summary(data_pulito)


modello_regressione_full <- lm(rm ~ . , data =  data_pulito)


step_aic <- stepAIC(modello_regressione_full, direction = "backward")

summary(step_aic)
AIC(step_aic)
deviance(step_aic)
qqnorm(residuals(step_aic))
qqline(residuals(step_aic), col = "red")




fit_gam <- gam(rm ~ s(zn, 3) + ns(indus, 3) + ns(age, 3) + 
                 ns(rad, 3) + ns(black, 3) + 
                 ns(lstat, 3) + ns(medv, 3), 
               data = data_pulito)
class(fit_gam)

termplot(step_aic,se = TRUE, partial.resid = TRUE, col.res = "#905E9F", pch = 16, col.se = "black", col.term = "black" )

qqnorm(residuals(fit_gam))
qqline(residuals(fit_gam), col = "red")

plot(fit_gam,residuals = TRUE,pch = 16,col = "#905E9F",se = TRUE)
summary(fit_gam)


