dir <- "table_results"
dgps <- c("dgp_enriched", "dgp_paper_example1", "ht_l_dgp", "ht_nl_dgp")
data <- data.frame()

for (dgp in dgps) {
  
  percorso_dgp <- file.path(dir, dgp)
  file_csv <- list.files(percorso_dgp, pattern = "\\.csv$", full.names = TRUE)
  
  for (file in file_csv) {
    df <- read.csv(file, stringsAsFactors = FALSE)
    df$dgp <- dgp
    df$modello <- ifelse(grepl("psbart", basename(file), ignore.case = TRUE), "BART",
                         ifelse(grepl("bart", basename(file), ignore.case = TRUE), "PS-BART",
                                ifelse(grepl("xbcf", basename(file), ignore.case = TRUE), "XBCF", "other")))
    data <- rbind(data, df)
  }
}

head(data)

# (Opzionale) Salva il risultato combinato
write.csv(data, file.path(dir, "all_data.csv"), row.names = FALSE)