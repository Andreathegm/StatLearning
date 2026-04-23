install.packages("palmerpenguins")
install.packages("ggeffects")
require(palmerpenguins)
require(ggeffects)
library(ggplot2)


data <- penguins
data_pulito <- na.omit(data)
summary(data_pulito)

data("penguins")


modello_logistico_wo_year <- glm(sex ~ . - year,
                         data =  data_pulito,
                         family = binomial)
summary(modello_logistico)


modello_logistico_wo_y_i <- glm(sex ~ . - year -island,
                                 data =  data_pulito,
                                 family = binomial)

effetti <- ggpredict(modello_logistico_wo_y_i)

plot(effetti)

summary(modello_logistico_wo_y_i)




# 6. Calcolo degli Odds Ratio (per un'interpretazione più semplice)
# Un OR > 1 indica che all'aumentare della variabile aumentano le probabilità
print("Odds Ratio:")
exp(coef(modello_logistico_wo_y_i))

penguins_clean$probabilita_predetta <- predict(modello_logistico, type = "response")

# 8. Visualizzazione (Opzionale)
# Creiamo un grafico per vedere come il modello separa le specie
ggplot(penguins_clean, aes(x = body_mass_g, y = probabilita_predetta, color = species)) +
  geom_point(alpha = 0.5) +
  labs(title = "Probabilità di essere Gentoo in base alla massa corporea",
       y = "Probabilità Predetta",
       x = "Massa Corporea (g)") +
  theme_minimal()