Contexte : séries journalières Coca-Cola (KO) et PepsiCo (PEP), 2019-2024 (Yahoo Finance, prix ajustés), split chronologique 90/10.

Tu t'adresses à un décideur NON statisticien (directeur financier). Explique simplement, sans jargon, les résultats suivants.

Tableau de comparaison des modèles de prévision à 10 jours (prix de KO en USD ; RMSE/MAE en dollars, MAPE en %) :
      modele    MSE   RMSE    MAE   MAPE  temps_fit_s complexite      interpretabilite
         GRU 1.9517 1.3970 1.0776 1.7262        13.97     Élevée  Faible (boîte noire)
         VAR 1.9565 1.3988 1.0856 1.7399         0.02    Moyenne      Élevée (Granger)
      SARIMA 1.9660 1.4021 1.0716 1.7129         0.38    Moyenne                Élevée
  ARIMA_auto 1.9660 1.4021 1.0716 1.7129         0.33     Faible                Élevée
       naive 1.9713 1.4040 1.0798 1.7264          NaN     Aucune                Totale
ARIMA_manuel 1.9765 1.4059 1.0805 1.7293         2.55     Faible                Élevée
        LSTM 1.9829 1.4081 1.0690 1.7119        12.13     Élevée  Faible (boîte noire)
        VECM 2.2899 1.5133 1.1508 1.8345         0.01    Moyenne Élevée (équilibre LT)

Test de Diebold-Mariano contre la baseline naïve :
      modele  DM_stat_vs_naive  p_value                                   conclusion
ARIMA_manuel           -0.1889   0.8504 moins bon que la baseline — NON significatif
  ARIMA_auto            0.1883   0.8509  meilleur que la baseline — NON significatif
      SARIMA            0.1883   0.8509  meilleur que la baseline — NON significatif
         VAR            0.1620   0.8715  meilleur que la baseline — NON significatif
        VECM           -1.7936   0.0749 moins bon que la baseline — NON significatif
        LSTM           -0.0827   0.9342 moins bon que la baseline — NON significatif
         GRU            0.2296   0.8187  meilleur que la baseline — NON significatif

Résultats clés : {"granger_PEP_to_KO_p": 0.014069790663758953, "granger_KO_to_PEP_p": 0.056221807577860945, "engle_granger_p": 0.031100603956177145, "johansen_rank": 2, "garch_best": "GARCH(1,1) t", "garch_persistence": 0.9656424497753342, "garch_half_life_days": 19.825929792704823, "best_arima": "ARIMA_auto", "best_rmse_model": "GRU"}

TÂCHE : (1) ce que les résultats disent de KO et PEP, (2) ce que signifient cointégration, Granger et volatilité conditionnelle
en langage courant, (3) le modèle à retenir et pourquoi. N'invente aucune p-value ni aucun chiffre absent du texte ci-dessus.
