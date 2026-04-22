install.packages("palmerpenguins")

data <- penguins
data_pulito <- na.omit(data)
summary(data_pulito)