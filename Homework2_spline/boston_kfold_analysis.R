# =============================================================================
# ANALISI BOSTON DATASET - K-FOLD CROSS VALIDATION
# Modelli: Regressione Lineare vs Spline (ns + bs)
# =============================================================================
# PACCHETTI NECESSARI:
# - MASS       : contiene il dataset Boston
# - splines    : per costruire le basi delle spline (ns() e bs())
# - ggplot2    : per i grafici (boxplot finale)
#
# Se non li hai installati, esegui:
# install.packages(c("MASS", "splines", "ggplot2"))
# =============================================================================

library(MASS)      # Per il dataset Boston
library(splines)   # Per ns() e bs()
library(ggplot2)   # Per il boxplot finale


# =============================================================================
# SEZIONE 1: CARICAMENTO E SPLIT DEL DATASET
# =============================================================================

#' Carica il dataset Boston e lo divide in TRAIN e TEST
#'
#' @param seed      Seed per la riproducibilità (default: 42)
#' @param test_frac Proporzione da usare come TEST (default: 0.2 = 20%)
#'
#' @return Lista con $train e $test
carica_e_splitta <- function(seed = 42, test_frac = 0.2) {
  
  data("Boston", package = "MASS")
  cat("Dataset Boston caricato:", nrow(Boston), "righe,", ncol(Boston), "colonne\n")
  
  set.seed(seed)
  
  n_test      <- floor(nrow(Boston) * test_frac)
  indici_test <- sample(1:nrow(Boston), size = n_test, replace = FALSE)
  
  train <- Boston[-indici_test, ]
  test  <- Boston[indici_test, ]
  
  cat("Dimensioni TRAIN:", nrow(train), "righe\n")
  cat("Dimensioni TEST: ", nrow(test),  "righe\n\n")
  
  return(list(train = train, test = test))
}


# =============================================================================
# SEZIONE 2: CREAZIONE DELLE FOLD PER K-FOLD CV
# =============================================================================

#' Divide il dataset di TRAIN in K fold
#'
#' @param dati  Dataframe di training
#' @param K     Numero di fold (default: 5)
#' @param seed  Seed per la riproducibilità
#'
#' @return Vettore di interi (da 1 a K): l'i-esimo elemento è la fold
#'         a cui appartiene la riga i del dataframe
crea_fold <- function(dati, K = 5, seed = 42) {
  
  set.seed(seed)
  n <- nrow(dati)
  
  # Mescoliamo gli indici casualmente per evitare pattern nelle fold
  indici_mescolati <- sample(1:n)
  
  # Assegniamo ogni riga a una fold usando il modulo (%%)
  # La sequenza prodotta è: 1,2,3,...,K,1,2,3,...,K,...
  fold_assegnazioni <- rep(NA, n)
  fold_assegnazioni[indici_mescolati] <- (seq_along(indici_mescolati) - 1) %% K + 1
  
  cat("Fold create:", K, "fold\n")
  cat("Dimensione approssimativa di ogni fold:", floor(n / K), "osservazioni\n\n")
  
  return(fold_assegnazioni)
}


# =============================================================================
# SEZIONE 3: FUNZIONE DI CALCOLO RMSE
# =============================================================================

#' Calcola il Mean Squared Error (Errore Quadratico Medio)
#'
#' RMSE = media di (y_reale - y_predetto)^2
#' Più è basso, meglio il modello ha predetto.
#'
#' @param y_reale    Vettore dei valori reali
#' @param y_predetto Vettore dei valori predetti
#'
#' @return Singolo valore numerico: l'RMSE
calcola_rmse <- function(y_reale, y_predetto) {
  errori          <- y_reale - y_predetto
  errori_quadrati <- errori^2
  mse             <- mean(errori_quadrati)
  rmse            <- sqrt(mse)
  return(rmse)
}


# =============================================================================
# SEZIONE 4: K-FOLD CV CON REGRESSIONE LINEARE
# =============================================================================

#' Esegue la K-Fold Cross Validation con Regressione Lineare
#'
#' @param dati            Dataframe di training
#' @param fold_vettore    Output di crea_fold()
#' @param formula_modello Formula R (es: rm ~ lstat + crim + ...)
#' @param K               Numero di fold
#'
#' @return Vettore di K valori RMSE (uno per fold)
kfold_regressione_lineare <- function(dati, fold_vettore, formula_modello, K = 5) {
  
  rmse_per_fold <- numeric(K)
  
  cat("--- Regressione Lineare: K-Fold CV ---\n")
  cat("  Formula:", deparse(formula_modello), "\n")
  
  for (k in 1:K) {
    
    # Tutte le fold tranne la k-esima → addestramento
    train_fold <- dati[fold_vettore != k, ]
    # La fold k → validazione
    val_fold   <- dati[fold_vettore == k, ]
    
    modello    <- lm(formula_modello, data = train_fold)
    previsioni <- predict(modello, newdata = val_fold)
    
    # Estraiamo il nome della risposta dalla formula (parte a sinistra del ~)
    variabile_risposta <- as.character(formula_modello)[2]
    y_reale <- val_fold[[variabile_risposta]]
    
    rmse_per_fold[k] <- calcola_rmse(y_reale, previsioni)
    cat(sprintf("  Fold %d: RMSE = %.4f\n", k, rmse_per_fold[k]))
  }
  
  cat(sprintf("  MEDIA RMSE (Reg. Lineare): %.4f\n\n", mean(rmse_per_fold)))
  
  return(rmse_per_fold)
}


# =============================================================================
# SEZIONE 5: COSTRUTTORE DI FORMULA SPLINE
# =============================================================================

#' Costruisce una formula lm() con termini ns(), bs() e/o lineari
#'
#' Perché due tipi di spline?
#'   - ns() (Natural Spline): si comporta linearmente FUORI dai nodi di
#'     confine → stabile in K-Fold. Usarla per la maggior parte delle variabili.
#'   - bs() (B-Spline): con df=df=3 non piazza nodi interni → non crasha
#'     su variabili con pochi valori distinti (es. zn che è quasi tutta zeri).
#'
#' Esempio con vars_ns=c("lstat","nox"), vars_bs=c("zn"), vars_lineari=c("crim"):
#'   rm ~ ns(lstat, df=3) + ns(nox, df=3) + bs(zn, df=3) + crim
#'
#' @param var_y        Nome della variabile risposta (stringa)
#' @param vars_ns      Vettore di variabili da avvolgere con ns()
#' @param vars_bs      Vettore di variabili da avvolgere con bs() (default: NULL)
#' @param vars_lineari Vettore di variabili da tenere lineari (default: NULL)
#' @param df_spline    Gradi di libertà per TUTTE le spline (default: 4)
#'
#' @return Oggetto formula R pronto per lm()
costruisci_formula_spline <- function(var_y,
                                      vars_ns,
                                      vars_bs      = NULL,
                                      vars_lineari = NULL,
                                      df_spline    = 4) {
  
  # Costruiamo i termini ns() per ogni variabile in vars_ns
  # sapply() applica la funzione a ogni elemento e restituisce un vettore
  termini_ns    <- sapply(vars_ns, function(v) paste0("ns(", v, ", df = ", df_spline, ")"))
  tutti_termini <- termini_ns
  
  # Aggiungiamo i termini bs() se presenti
  if (!is.null(vars_bs) && length(vars_bs) > 0) {
    termini_bs    <- sapply(vars_bs, function(v) paste0("bs(", v, ", df = ", df_spline, ")"))
    tutti_termini <- c(tutti_termini, termini_bs)
  }
  
  # Aggiungiamo i termini lineari se presenti (entrano senza wrapper)
  if (!is.null(vars_lineari) && length(vars_lineari) > 0) {
    tutti_termini <- c(tutti_termini, vars_lineari)
  }
  
  lato_destro     <- paste(tutti_termini, collapse = " + ")
  formula_stringa <- paste(var_y, "~", lato_destro)
  
  return(as.formula(formula_stringa))
}


# =============================================================================
# SEZIONE 6: K-FOLD CV CON SPLINE
# =============================================================================

#' Esegue la K-Fold Cross Validation con Spline (ns + bs)
#'
#' @param dati         Dataframe di training
#' @param fold_vettore Output di crea_fold()
#' @param var_y        Nome della variabile risposta (stringa)
#' @param vars_ns      Variabili da avvolgere con ns()
#' @param vars_bs      Variabili da avvolgere con bs() (default: NULL)
#' @param vars_lineari Variabili da tenere lineari (default: NULL)
#' @param df_spline Grado della spline (default: 4)
#' @param K            Numero di fold
#'
#' @return Vettore di K valori RMSE (uno per fold)
kfold_spline <- function(dati, fold_vettore, var_y,
                         vars_ns, vars_bs = NULL, vars_lineari = NULL,
                         df_spline = 4, K = 5) {
  
  rmse_per_fold <- numeric(K)
  
  # Costruiamo la formula UNA SOLA VOLTA fuori dal loop:
  # è identica per tutte le fold, non ha senso ricostruirla K volte
  formula_spline <- costruisci_formula_spline(
    var_y        = var_y,
    vars_ns      = vars_ns,
    vars_bs      = vars_bs,
    vars_lineari = vars_lineari,
    df_spline    = df_spline
  )
  
  cat(sprintf("--- Spline (df=%d): K-Fold CV ---\n", df_spline))
  cat("  Formula:", deparse(formula_spline), "\n")
  
  for (k in 1:K) {
    
    train_fold <- dati[fold_vettore != k, ]
    val_fold   <- dati[fold_vettore == k, ]
    
    modello    <- lm(formula_spline, data = train_fold)
    previsioni <- predict(modello, newdata = val_fold)
    
    y_reale <- val_fold[[var_y]]
    
    rmse_per_fold[k] <- calcola_rmse(y_reale, previsioni)
    cat(sprintf("  Fold %d: RMSE = %.4f\n", k, rmse_per_fold[k]))
  }
  
  cat(sprintf("  MEDIA RMSE (Spline df=%d): %.4f\n\n", df_spline, mean(rmse_per_fold)))
  
  return(rmse_per_fold)
}


# =============================================================================
# SEZIONE 7: PREDICT FINALE SUL TEST SET
# =============================================================================

#' Addestra il modello finale su TUTTO il train e lo valuta sul TEST
#'
#' La firma è coerente con kfold_spline: stessi parametri, stessa
#' costruisci_formula_spline → garanzia che il modello valutato sul test
#' è identico a quello valutato in cross-validation.
#'
#' @param train        Dataframe di training completo
#' @param test         Dataframe di test
#' @param tipo_modello "lineare" o "spline"
#' @param var_y        Nome della variabile risposta (stringa)
#' @param formula_lm   Formula per la regressione lineare (solo se tipo="lineare")
#' @param vars_ns      Variabili con ns() (solo se tipo="spline")
#' @param vars_bs      Variabili con bs() (solo se tipo="spline", default: NULL)
#' @param vars_lineari Variabili lineari (solo se tipo="spline", default: NULL)
#' @param df_spline    Gradi della spline (solo se tipo="spline")
#'
#' @return Lista con $modello, $previsioni, $rmse_test
valuta_su_test <- function(train, test, tipo_modello, var_y,
                           formula_lm   = NULL,
                           vars_ns      = NULL,
                           vars_bs      = NULL,
                           vars_lineari = NULL,
                           df_spline    = 4) {
  
  if (tipo_modello == "lineare") {
    
    modello    <- lm(formula_lm, data = train)
    previsioni <- predict(modello, newdata = test)
    
  } else if (tipo_modello == "spline") {
    
    # Usiamo la stessa funzione usata in kfold_spline → coerenza totale
    formula_spline <- costruisci_formula_spline(
      var_y        = var_y,
      vars_ns      = vars_ns,
      vars_bs      = vars_bs,
      vars_lineari = vars_lineari,
      df_spline    = df_spline
    )
    cat("  Formula spline finale:", deparse(formula_spline), "\n")
    modello    <- lm(formula_spline, data = train)
    previsioni <- predict(modello, newdata = test)
    
  } else {
    stop("tipo_modello deve essere 'lineare' o 'spline'")
  }
  
  y_test_reale <- test[[var_y]]
  rmse_test     <- calcola_rmse(y_test_reale, previsioni)
  
  cat(sprintf("RMSE sul TEST set (%s): %.4f\n", tipo_modello, rmse_test))
  
  return(list(modello = modello, previsioni = previsioni, mse_test = rmse_test))
}


# =============================================================================
# SEZIONE 8: BOXPLOT DI CONFRONTO
# =============================================================================

#' Crea un boxplot per confrontare gli RMSE delle K fold tra i due modelli
#'
#' Il boxplot mostra non solo la media ma anche la VARIABILITÀ dell'errore:
#'   - linea centrale = mediana degli RMSE
#'   - rettangolo     = 50% centrale dei valori (IQR)
#'   - baffi          = range (esclusi outlier)
#'   - punti          = i K valori RMSE reali (uno per fold)
#'
#' @param lista_rmse Lista nominata di vettori RMSE
#'                  (es: list("Lineare"=vettore, "Spline"=vettore))
#' @param titolo    Titolo del grafico
#'
#' @return Oggetto ggplot
crea_boxplot_confronto <- function(lista_rmse, titolo = "Confronto RMSE - K-Fold CV") {
  
  # Costruiamo un dataframe in formato "lungo":
  # una riga per ogni coppia (RMSE, nome_modello)
  df_plot <- data.frame(
    RMSE     = unlist(lista_rmse),
    Modello = rep(names(lista_rmse), sapply(lista_rmse, length))
    # rep() ripete ogni nome tante volte quante sono le fold di quel modello
    # sapply(lista_rmse, length) restituisce il numero di fold per ogni modello
  )
  
  grafico <- ggplot(df_plot, aes(x = Modello, y = RMSE, fill = Modello)) +
    geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 2) +
    # geom_jitter sovrappone i punti reali, spostati leggermente in orizzontale
    # per evitare sovrapposizioni (width = ampiezza dello spostamento)
    geom_jitter(width = 0.15, size = 2.5, alpha = 0.8) +
    labs(
      title   = titolo,
      x       = "Model",
      y       = "RMSE (Mean Squared Error)",
      #caption = "punto = RMSE di una fold. Modello migliore = RMSE più basso."
    ) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "none",
      plot.title      = element_text(hjust = 0.5, face = "bold")
    )
  
  return(grafico)
}

# Testa tutti i df da 3 a max_df e restituisce la media RMSE per ognuno
evaluate_df <- function(max_df, dati, fold_vettore, response, vars_ns, vars_bs, K) {
  
  rmse_per_df <- c()   # vettore vuoto, lo riempiamo ad ogni iterazione
  
  for (i in 3:max_df) {   # le {} racchiudono TUTTO il corpo del loop
    
    rmse_spline <- kfold_spline(
      dati         = dati,        # usa il parametro, non la variabile globale
      fold_vettore = fold_vettore,
      var_y        = response,
      vars_ns      = vars_ns,
      vars_bs      = vars_bs,
      vars_lineari = NULL,
      df_spline    = i,
      K            = K
    )
    
    rmse_per_df <- c(rmse_per_df, mean(rmse_spline))  # append la media di questo df
  }
  
  return(rmse_per_df)  # restituisce un vettore lungo (max_df - 2)
}


# Plotta RMSE medio (y) al variare del df della spline (x)
plot_rmse_vs_df <- function(rmse_per_df, max_df) {
  
  # Costruiamo il dataframe per ggplot:
  # df_values = 3, 4, 5, ..., max_df  (i gradi testati)
  # rmse_values = la media RMSE corrispondente
  df_plot <- data.frame(
    df_values  = 3:max_df,
    rmse_values = rmse_per_df
  )
  
  grafico <- ggplot(df_plot, aes(x = df_values, y = rmse_values)) +
    geom_line(color = "steelblue", linewidth = 1) +   # linea che collega i punti
    geom_point(color = "steelblue", size = 3) +        # un punto per ogni df testato
    # geom_vline traccia una linea verticale sul df con RMSE minimo
    geom_vline(xintercept = df_plot$df_values[ which.min(df_plot$rmse_values) ],
               linetype = "dashed", color = "red") +
    # which.min() restituisce l'indice del valore minimo nel vettore
    labs(
      title   = "Mean RMSE as df increases",
      x       = "Degrees of Freedom (df)",
      y       = "Mean RMSE (K-Fold CV)",
      #caption = "Red dotted line = optimal df (min RMSE)"
    ) +
    scale_x_continuous(breaks = 3:max_df) +  # un tick per ogni df testato
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  return(grafico)
}

# --- Chiamata nel main ---


# =============================================================================
# SCRIPT PRINCIPALE
# =============================================================================

cat("========================================================\n")
cat("   ANALISI BOSTON - REGRESSIONE LINEARE VS SPLINE\n")
cat("========================================================\n\n")

# --- STEP 1: Carica e splitta ---
dati_split <- carica_e_splitta(seed = 42, test_frac = 0.2)
train <- dati_split$train
test  <- dati_split$test

# --- STEP 2: Definisci risposta e covariate ---
response <- "rm"

# Prendiamo tutti i nomi delle colonne di Boston ed escludiamo:
#   - la variabile risposta (rm)
#   - chas (variabile dummy 0/1, non ha senso metterci una spline sopra)
f_names    <- names(Boston)
covariates <- f_names[ !f_names %in% c(response, "chas") ]

# zn è quasi tutta zeri → ns() crasha (nodi interni coincidono col boundary)
# la separiamo e la tratteremo con bs() che con df=3 non piazza nodi interni
vars_ns <- covariates[ covariates != "zn" ]   # tutte le altre → ns()
vars_bs <- "zn"                                # solo zn        → bs()

cat("Variabile risposta:", response, "\n")
cat("Covariate con ns():", paste(vars_ns, collapse = ", "), "\n")
cat("Covariate con bs():", paste(vars_bs, collapse = ", "), "\n\n")

# --- STEP 3: Crea le fold ---
K            <- 5
fold_vettore <- crea_fold(train, K = K, seed = 42)

# --- STEP 4: Formula della regressione lineare ---
formula_lineare <- as.formula(
  paste(response, "~", paste(covariates, collapse = " + "))
)

# --- STEP 5: K-Fold CV per entrambi i modelli ---
rmse_lineare <- kfold_regressione_lineare(
  dati            = train,
  fold_vettore    = fold_vettore,
  formula_modello = formula_lineare,
  K               = K
)

rmse_spline <- kfold_spline(
  dati         = train,
  fold_vettore = fold_vettore,
  var_y        = response,
  vars_ns      = vars_ns,
  vars_bs      = vars_bs,
  vars_lineari = NULL,
  df_spline    = 3,
  K            = K
)

# --- STEP 6: Riepilogo CV ---
cat("========================================================\n")
cat("RIEPILOGO MEDIA RMSE DALLA K-FOLD CV:\n")
cat(sprintf("  Regressione Lineare : %.4f\n", mean(rmse_lineare)))
cat(sprintf("  Spline (df=3)       : %.4f\n", mean(rmse_spline)))
cat("Il modello con RMSE più basso è il migliore in cross-validation.\n\n")

# --- STEP 7: Valutazione finale sul test set ---
cat("========================================================\n")
cat("VALUTAZIONE FINALE SUL TEST SET:\n\n")

risultato_lineare <- valuta_su_test(
  train        = train,
  test         = test,
  tipo_modello = "lineare",
  var_y        = response,
  formula_lm   = formula_lineare
)

risultato_spline <- valuta_su_test(
  train        = train,
  test         = test,
  tipo_modello = "spline",
  var_y        = response,
  vars_ns      = vars_ns,
  vars_bs      = vars_bs,
  vars_lineari = NULL,
  df_spline    = 3
)

# --- STEP 8: Boxplot ---
cat("\n========================================================\n")
cat("CREAZIONE BOXPLOT DI CONFRONTO...\n")

grafico <- crea_boxplot_confronto(
  lista_rmse = list(
    "Linear Regression" = rmse_lineare,
    "Spline (df=3)"       = rmse_spline
  ),
  titolo = "RMSE - K-Fold CV (K=5)"
)

print(grafico)

max_df <- 9

rmse_per_df <- evaluate_df(
  max_df       = max_df,
  dati         = train,
  fold_vettore = fold_vettore,
  response     = response,
  vars_ns      = vars_ns,
  vars_bs      = vars_bs,
  K            = K
)

cat("df ottimale:", which.min(rmse_per_df) + 2, "\n")

grafico_df <- plot_rmse_vs_df(rmse_per_df, max_df)
print(grafico_df)

cat("\n✓ Analisi completata!\n")