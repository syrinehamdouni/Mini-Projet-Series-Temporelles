Contexte : séries journalières Coca-Cola (KO) et PepsiCo (PEP), 2019-2024 (Yahoo Finance, prix ajustés), split chronologique 90/10.

À partir des performances ci-dessous, formule des recommandations d'investissement SIMULÉES (exercice pédagogique) sur KO et PEP :
      modele    MSE   RMSE    MAE   MAPE  temps_fit_s complexite      interpretabilite
         GRU 1.9517 1.3970 1.0776 1.7262        13.97     Élevée  Faible (boîte noire)
         VAR 1.9565 1.3988 1.0856 1.7399         0.02    Moyenne      Élevée (Granger)
      SARIMA 1.9660 1.4021 1.0716 1.7129         0.38    Moyenne                Élevée
  ARIMA_auto 1.9660 1.4021 1.0716 1.7129         0.33     Faible                Élevée
       naive 1.9713 1.4040 1.0798 1.7264          NaN     Aucune                Totale
ARIMA_manuel 1.9765 1.4059 1.0805 1.7293         2.55     Faible                Élevée
        LSTM 1.9829 1.4081 1.0690 1.7119        12.13     Élevée  Faible (boîte noire)
        VECM 2.2899 1.5133 1.1508 1.8345         0.01    Moyenne Élevée (équilibre LT)
Volatilité prévue par le GARCH sur les 10 prochains jours (écart-type journalier en %) :
 horizon  vol_journaliere_%
       1             0.8596
       2             0.8758
       3             0.8911
       4             0.9056
       5             0.9194
       6             0.9326
       7             0.9451
       8             0.9570
       9             0.9684
      10             0.9793

TÂCHE : (1) donne tes recommandations ; (2) TOI-MÊME, liste ensuite explicitement les limites, risques et hypothèses fragiles de ces
recommandations (efficience des marchés, coûts de transaction, instabilité des paramètres, risque de modèle, surapprentissage...).
