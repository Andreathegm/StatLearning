source("single_split.R")

multisplit <- function(x, y, alpha, B) {
  p <- ncol(x)
  pvals_mat <- matrix(NA, nrow = p, ncol = B)
  
  for (i in seq_len(B)) {
    cat(sprintf("Split iteration %d of %d...\n", i, B))
    res <- one_split(x, y, alpha = alpha)
    pvals_mat[, i] <- res$pval
  }
  
  df_pvals <- as.data.frame(pvals_mat)
  colnames(df_pvals) <- paste0("B_", seq_len(B))
  
  return(df_pvals)
}