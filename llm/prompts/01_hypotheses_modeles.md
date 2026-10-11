Contexte : séries journalières Coca-Cola (KO) et PepsiCo (PEP), 2019-2024 (Yahoo Finance, prix ajustés), split chronologique 90/10.

Tu es un expert en séries temporelles. Voici les statistiques descriptives des rendements log (échantillon d'apprentissage) :
                   ret_KO   ret_PEP
moyenne           0.00033   0.00047
variance          0.00017   0.00018
ecart_type        0.01318   0.01349
skewness         -0.95020  -0.64697
kurtosis_excess  10.06613  22.49589
min              -0.10173  -0.12136
max               0.06278   0.12166
JB_p              0.00000   0.00000

Résultats des tests de stationnarité (ADF : H0 = racine unitaire ; KPSS : H0 = stationnaire) :
  serie  ADF_p  KPSS_p                         conclusion
 log_KO 0.4291    0.01 NON stationnaire (racine unitaire)
log_PEP 0.2782    0.01 NON stationnaire (racine unitaire)
 ret_KO 0.0000    0.10                       stationnaire
ret_PEP 0.0000    0.10                       stationnaire

TÂCHE : propose une liste ordonnée d'hypothèses de modèles à tester (ARIMA, SARIMA, VAR, cointégration/VECM, GARCH, réseaux
récurrents...). Pour chaque modèle, justifie par les chiffres ci-dessus et indique quel diagnostic le validerait ou l'invaliderait.
N'invente aucun chiffre : si une information manque, dis-le.
