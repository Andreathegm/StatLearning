rmse <- function(true_values, estimated_values) {
  sqrt(mean((true_values - estimated_values)^2))
}

coverage <- function(true_values, lower_bounds, upper_bounds) {
  mean(true_values >= lower_bounds & true_values <= upper_bounds)
}

avarage_coverage_length <- function(lower_bounds, upper_bounds) {
  mean(upper_bounds - lower_bounds)
}