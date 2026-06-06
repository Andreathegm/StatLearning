if (!require("plotly", quietly = TRUE)) install.packages("plotly")
library(plotly)

grid_size <- 50
x1_seq <- seq(0, 1, length.out = grid_size)
x2_seq <- seq(0, 1, length.out = grid_size)

mu_matrix <- matrix(NA, nrow = grid_size, ncol = grid_size)
pi_matrix <- matrix(NA, nrow = grid_size, ncol = grid_size)

for(i in 1:grid_size) {
  for(j in 1:grid_size) {
    x1 <- x1_seq[i]
    x2 <- x2_seq[j]
    
    mu_matrix[i, j] <- -3 + 6 * pnorm(2 * (x1 - x2))
    
    pi_val <- 0.8 * pnorm(mu_matrix[i, j] / (0.1 * (2 - x1 - x2) + 0.25)) + 0.025 * (x1 + x2) + 0.05
    pi_matrix[i, j] <- pmin(pmax(pi_val, 0.01), 0.99)
  }
}

camera_view <- list(
  eye = list(x = -1.5, y = -1.5, z = 0.8)
)

plot_mu <- plot_ly(x = ~x1_seq, y = ~x2_seq, z = ~mu_matrix) %>% 
  add_surface(colorscale = "Viridis") %>% 
  layout(
    title = "Prognostic Function μ(x)",
    scene = list(
      camera = camera_view,
      xaxis = list(title = "x₁"),
      yaxis = list(title = "x₂"),
      zaxis = list(title = "μ")
    )
  )

plot_pi <- plot_ly(x = ~x1_seq, y = ~x2_seq, z = ~pi_matrix) %>% 
  add_surface(colorscale = "Cividis") %>% 
  layout(
    title = "Propensity Function π(x)",
    scene = list(
      camera = camera_view,
      xaxis = list(title = "x₁"),
      yaxis = list(title = "x₂"),
      zaxis = list(title = "π")
    )
  )


plot_mu
plot_pi