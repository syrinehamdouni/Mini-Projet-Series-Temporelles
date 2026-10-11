# =====================================================================================
# classique.R — Socle classique en R (section 3) : ARIMA/SARIMA, VAR, cointégration/VECM, GARCH
# À lancer depuis la RACINE du projet :  Rscript R/classique.R
# Données : les CSV data/KO.csv et data/PEP.csv créés par python/01_donnees_eda.py (mêmes données qu'en Python,
#           donc résultats comparables). Les modèles avancés (LSTM/GRU) sont en Python uniquement (autorisé).
# Packages : install.packages(c("forecast","tseries","urca","vars","rugarch"))
# =====================================================================================
set.seed(42)                                   # graine fixe -> reproductibilité
suppressPackageStartupMessages({library(forecast); library(tseries); library(urca); library(vars); library(rugarch)})
dir.create("figures_R", showWarnings = FALSE); dir.create("results_R", showWarnings = FALSE)

# ---- Fonction maison : test ARCH-LM (régression de e² sur ses retards) -> évite une dépendance à un package archivé
arch_lm <- function(e, lags = 10) {
  m <- embed(e^2, lags + 1)                                    # colonnes : e²_t, e²_{t-1}, ...
  r2 <- summary(lm(m[, 1] ~ m[, -1]))$r.squared
  stat <- nrow(m) * r2
  c(stat = stat, p = 1 - pchisq(stat, lags))                   # H0 : pas d'effet ARCH
}
interp <- function(p, h0) cat(sprintf("   p=%.4f -> %s H0 (%s)\n", p, ifelse(p < 0.05, "on REJETTE", "on ne rejette PAS"), h0))

# ---- 1. Données : série temporelle, log-prix, split chronologique 90/10 -------------------------------------
ko  <- read.csv("data/KO.csv",  stringsAsFactors = FALSE)
pep <- read.csv("data/PEP.csv", stringsAsFactors = FALSE)
d <- merge(data.frame(Date = as.Date(ko$Date),  KO  = ko$Adj.Close),
           data.frame(Date = as.Date(pep$Date), PEP = pep$Adj.Close), by = "Date")   # dates communes aux deux actifs
d <- d[order(d$Date), ]
lko <- log(d$KO); lpep <- log(d$PEP)                           # log : stabilise la variance, rendement = différence
n <- length(lko); n_train <- floor(0.9 * n)                    # split chronologique, jamais aléatoire
y_tr <- ts(lko[1:n_train], frequency = 5)                      # fréquence 5 = semaine de bourse (pour tester SARIMA)
cat("Observations :", n, "| train :", n_train, "| test :", n - n_train, "\n")

stationarity_rows <- list()
for (ticker in c("KO", "PEP")) {
  level <- if (ticker == "KO") lko[1:n_train] else lpep[1:n_train]
  for (d_order in 0:1) {
    x <- if (d_order == 0) level else diff(level)
    adf_p <- adf.test(x)$p.value
    kpss_p <- kpss.test(x, null = "Level")$p.value
    conclusion <- if (adf_p < 0.05 && kpss_p >= 0.05) {
      "stationnaire"
    } else if (adf_p >= 0.05 && kpss_p < 0.05) {
      "non stationnaire"
    } else {
      "tests divergents"
    }
    stationarity_rows[[length(stationarity_rows) + 1]] <- data.frame(
      serie = ticker, ordre_difference = d_order, ADF_p = adf_p,
      KPSS_p = kpss_p, conclusion = conclusion
    )
    cat("\n", ticker, if (d_order == 0) " niveau" else " première différence",
        "\nADF : "); interp(adf_p, "racine unitaire")
    cat("KPSS: "); interp(kpss_p, "série stationnaire")
    cat("Conclusion conjointe :", conclusion, "\n")
  }
}
write.csv(do.call(rbind, stationarity_rows), "results_R/stationnarite.csv", row.names = FALSE)
png("figures_R/R_acf_pacf_diff.png", 1000, 400); par(mfrow = c(1, 2))
acf(diff(as.numeric(y_tr)), lag.max = 40, main = "ACF de diff log(KO)"); pacf(diff(as.numeric(y_tr)), lag.max = 40, main = "PACF de diff log(KO)")
dev.off()

# ---- 3.1 ARIMA / SARIMA --------------------------------------------------------------------------------------
dy <- diff(as.numeric(y_tr)); seuil <- 1.96 / sqrt(length(dy))              # seuil de significativité de l'ACF/PACF
p_man <- max(c(0, which(abs(pacf(dy, lag.max = 5, plot = FALSE)$acf) > seuil)))   # p = dernier retard PACF significatif
q_man <- max(c(0, which(abs(acf(dy,  lag.max = 5, plot = FALSE)$acf[-1]) > seuil)))  # q = dernier retard ACF significatif
fit_man <- Arima(y_tr, order = c(p_man, 1, q_man))                           # approche manuelle (à valider sur le graphique)
fit_auto <- auto.arima(y_tr, d = 1, seasonal = FALSE, ic = "aic", stepwise = FALSE, approximation = FALSE)  # recherche exhaustive
fit_sar <- auto.arima(y_tr, d = 1, D = 0, seasonal = TRUE, ic = "aic")       # saisonnalité hebdomadaire (m = 5)
cat("\nOrdre manuel :", p_man, 1, q_man, "| auto :", arimaorder(fit_auto)[1:3], "| SARIMA :", arimaorder(fit_sar), "\n")

diag_res <- function(fit, nm) {                                              # diagnostic complet des résidus
  e <- as.numeric(residuals(fit)); k <- sum(fit$arma[1:4])                   # k = nb de paramètres ARMA (degrés de liberté LB)
  lb10_lag <- max(10, k + 1); lb20_lag <- max(20, k + 2)
  lb10 <- Box.test(e, lb10_lag, "Ljung-Box", fitdf = k)
  lb20 <- Box.test(e, lb20_lag, "Ljung-Box", fitdf = k)
  jb <- jarque.bera.test(e); sw <- shapiro.test(e); ar <- arch_lm(e, 10)
  cat("\n--- Diagnostic :", nm, "\n Ljung-Box(", lb10_lag, "):", sep = ""); interp(lb10$p.value, "résidus indépendants")
  cat(" Ljung-Box(", lb20_lag, "):", sep = ""); interp(lb20$p.value, "résidus indépendants")
  cat(" Jarque-Bera  :"); interp(jb$p.value, "résidus normaux")
  cat(" Shapiro-Wilk :"); interp(sw$p.value, "résidus normaux")
  cat(" ARCH-LM(10)  :"); interp(ar["p"], "variance conditionnelle constante")
  png(paste0("figures_R/R_resid_", nm, ".png"), 900, 700); checkresiduals(fit, plot = TRUE); dev.off()
  data.frame(modele = nm, AIC = AIC(fit), BIC = BIC(fit), LB10_lag = lb10_lag,
             LB10_p = lb10$p.value, LB20_lag = lb20_lag, LB20_p = lb20$p.value,
             JB_p = jb$p.value, Shapiro_p = sw$p.value, ARCH10_p = ar["p"])
}
tab_arima <- rbind(diag_res(fit_man, "ARIMA_manuel"), diag_res(fit_auto, "ARIMA_auto"), diag_res(fit_sar, "SARIMA"))
write.csv(tab_arima, "results_R/diag_arima.csv", row.names = FALSE); print(round(tab_arima[, -1], 4))
png("figures_R/R_prevision_arima.png", 900, 400)
plot(forecast(fit_auto, h = 10, level = 95), include = 60, main = "ARIMA auto — prévision à 10 jours (log KO), IC 95 %"); dev.off()

# ---- 3.2 VAR + Granger (sur les rendements), cointégration + VECM (sur les log-prix) ----------------------
ret_tr <- diff(cbind(KO = lko, PEP = lpep))[1:(n_train - 1), ]               # rendements : stationnaires (d = 1)
p_var <- VARselect(ret_tr, lag.max = 10, type = "const")$selection["SC(n)"]  # BIC (SC) : plus parcimonieux que l'AIC
var_fit <- VAR(ret_tr, p = p_var, type = "const")
cat("\nVAR(", p_var, ") estimé\n", sep = "")
cat("Granger PEP -> KO :"); interp(causality(var_fit, cause = "PEP")$Granger$p.value, "PEP ne cause pas KO")
cat("Granger KO -> PEP :"); interp(causality(var_fit, cause = "KO")$Granger$p.value, "KO ne cause pas PEP")
granger_pep_ko <- causality(var_fit, cause = "PEP")$Granger$p.value
granger_ko_pep <- causality(var_fit, cause = "KO")$Granger$p.value
white_p <- serial.test(var_fit, lags.pt = 20, type = "PT.asymptotic")$serial$p.value
normal_p <- normality.test(var_fit)$jb.mul$JB$p.value
arch_p <- arch.test(var_fit, lags.multi = 5)$arch.mul$p.value
cat("Portemanteau      :"); interp(white_p, "résidus = bruit blanc")
cat("Normalité (JB)    :"); interp(normal_p, "résidus normaux")
cat("ARCH multivarié   :"); interp(arch_p, "variance conditionnelle constante")
write.csv(data.frame(VAR_order = as.integer(p_var), stable = all(Mod(roots(var_fit)) > 1),
                     Granger_PEP_to_KO_p = granger_pep_ko, Granger_KO_to_PEP_p = granger_ko_pep,
                     Portmanteau_p = white_p, Jarque_Bera_p = normal_p, ARCH_multivariate_p = arch_p),
          "results_R/diag_var.csv", row.names = FALSE)

lv <- cbind(KO = lko[1:n_train], PEP = lpep[1:n_train])
cat("\nCointégration — Phillips-Ouliaris (H0 : pas de cointégration) :"); interp(po.test(lv)$p.value, "pas de cointégration")
K <- max(2, VARselect(lv, lag.max = 10, type = "const")$selection["SC(n)"])  # K = retards en niveaux (= k_ar_diff + 1)
jo <- ca.jo(lv, type = "trace", ecdet = "const", K = K)                      # Johansen (trace)
print(summary(jo))
trace_critical <- jo@cval[, "5pct"]
johansen_rank <- if (jo@teststat[1] > trace_critical[1]) {
  2L
} else if (jo@teststat[2] > trace_critical[2]) {
  1L
} else {
  0L
}
rang1 <- johansen_rank == 1L
cat("Johansen exact rank (trace, 5 %) :", johansen_rank, "\n")
if (rang1) {
  vecm <- cajorls(jo, r = 1)                                                 # VECM de rang 1
  cat("beta (relation de long terme) :\n"); print(vecm$beta)
  alpha <- vecm$rlm$coefficients["ect1", ]
  cat("alpha (vitesse d'ajustement) :", round(alpha, 4), "\n")
  cat("Un alpha négatif et significatif = la variable revient vers l'équilibre ; demi-vie = ln(0.5)/ln(1+alpha).\n")
  print(summary(vecm$rlm))
}
write.csv(data.frame(Phillips_Ouliaris_p = po.test(lv)$p.value, Johansen_rank = johansen_rank,
                     Johansen_teststat_r_le_1 = jo@teststat[1], Johansen_critical_r_le_1 = trace_critical[1],
                     Johansen_teststat_r_0 = jo@teststat[2], Johansen_critical_r_0 = trace_critical[2],
                     VECM_rank1_fitted = rang1),
          "results_R/cointegration.csv", row.names = FALSE)

# ---- 3.3 ARCH-LM puis GARCH sur les rendements de KO (en %) ---------------------------------------------------
r <- 100 * ret_tr[, "KO"]
cat("\nARCH-LM sur rendements KO :"); interp(arch_lm(r - mean(r), 10)["p"], "variance conditionnelle constante")
spec <- function(dist) ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                                  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE), distribution.model = dist)
fit_n <- ugarchfit(spec("norm"), data = r); fit_t <- ugarchfit(spec("std"), data = r)   # normale vs Student-t
cat("BIC GARCH(1,1) normal :", infocriteria(fit_n)["Bayes", ], "| Student-t :", infocriteria(fit_t)["Bayes", ], "\n")
fit_g <- if (infocriteria(fit_t)["Bayes", ] < infocriteria(fit_n)["Bayes", ]) fit_t else fit_n   # on garde le meilleur BIC
z <- as.numeric(residuals(fit_g, standardize = TRUE))                        # résidus standardisés : doivent être homoscédastiques
cat("\n--- Diagnostic des résidus standardisés du GARCH\n Ljung-Box z :"); interp(Box.test(z, 10, "Ljung-Box")$p.value, "pas d'autocorrélation")
cat(" Ljung-Box z²:"); interp(Box.test(z^2, 10, "Ljung-Box")$p.value, "pas d'autocorrélation de la variance")
cat(" ARCH-LM z   :"); interp(arch_lm(z, 10)["p"], "homoscédasticité retrouvée")
cat(" Jarque-Bera :"); interp(jarque.bera.test(z)$p.value, "résidus normaux")
cf <- coef(fit_g); pers <- sum(cf[grepl("^alpha|^beta", names(cf))])
cat(sprintf("Persistance = %.4f ; demi-vie d'un choc de volatilité = %.1f jours\n", pers, log(0.5) / log(pers)))
png("figures_R/R_garch_volatilite.png", 1000, 450)
plot(as.numeric(sigma(fit_g)), type = "l", main = "Volatilité conditionnelle GARCH(1,1) — KO (% journalier)", xlab = "jours (train)", ylab = "sigma"); dev.off()
cat("Volatilité prévue (10 jours) :", round(as.numeric(sigma(ugarchforecast(fit_g, n.ahead = 10))), 3), "\n")
cat("\nFIN. Figures dans figures_R/, tableaux dans results_R/.\n")
sessionInfo()                                                                # versions des packages (reproductibilité)
