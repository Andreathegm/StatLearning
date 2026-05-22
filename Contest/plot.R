plot_pvalue_ecdf <- function(data, 
                             alpha = 0.05, 
                             breaks_seq = NULL, 
                             n_bins = 10,
                             main_title = "Distribuzione p-value, ECDF e f(p)",
                             x_label = "p-value") {
  
  data <- na.omit(data)
  
  max_val <- max(data)
  
  if (is.null(breaks_seq)) {
    breaks_seq <- seq(0, max_val, length.out = n_bins + 1)
  }
  
  h <- hist(data, breaks = breaks_seq, plot = FALSE)
  h$counts <- h$counts / sum(h$counts)
  
  plot(h,
       freq = TRUE,                  
       col = "lightgray",
       ylim = c(0, 1),                
       xlim = c(0, max_val),      
       axes = FALSE,                  
       xlab = x_label,
       ylab = "Frequencies / ECDF / f(p)",
       main = main_title)
  
  axis(1, at = signif(breaks_seq, 3))            
  axis(2, at = seq(0, 1, by = 0.2))  
  
  plot(ecdf(data),
       verticals = TRUE,
       do.points = FALSE,
       col = "red",
       lwd = 2,
       add = TRUE)
  
  curve(pmax(0.05, (3.996 / alpha) * x), 
        from = 0, 
        to = max_val, 
        add = TRUE, 
        lty = 2,          
        col = "blue",     
        lwd = 2)          
}