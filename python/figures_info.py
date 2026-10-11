"""Catalogue des graphiques : où les trouver, ce qu'ils montrent, comment les interpréter.
Utilisé par l'application Streamlit et par le générateur de présentation (une seule source de vérité).
Les 'regles' sont des règles de lecture (si… alors…) : le résultat réel dépend de vos données, à vérifier sur le graphique."""

FIGURES = [
    dict(f="fig01_prix.png", script="01_donnees_eda.py", section="1-2 Données / exploration", titre="Prix de KO et PEP",
         montre="Haut : prix ajustés en USD (échelles différentes). Bas : prix rebasés à 100 pour comparer les dynamiques. Trait pointillé = début du test.",
         regles="Tendance durable à la hausse/baisse => série non stationnaire en niveau. Si les deux courbes bougent ensemble => actifs corrélés, "
                "candidats à la cointégration. Un krach visible (mars 2020) = rupture à signaler."),
    dict(f="fig02_indicateurs.png", script="01_donnees_eda.py", section="1-2 Données / exploration", titre="Indicateurs dérivés : MM20, rendements, volatilité réalisée",
         montre="Moyenne mobile 20 j de KO, rendements logarithmiques des deux actifs, volatilité réalisée 21 j annualisée.",
         regles="Rendements autour de 0 sans tendance => stationnaires. Périodes agitées suivies de périodes calmes (clusters) => variance non constante => "
                "justifie ARCH/GARCH."),
    dict(f="fig03_acf_pacf_niveau.png", script="01_donnees_eda.py", section="2 Exploration", titre="ACF / PACF de log(KO) (série brute)",
         montre="Autocorrélations et autocorrélations partielles du log-prix, 40 retards, bandes de confiance à 95 %.",
         regles="ACF qui décroît très lentement et PACF forte au retard 1 => racine unitaire (non stationnaire) => il faut différencier."),
    dict(f="fig04_acf_pacf_diff.png", script="01_donnees_eda.py", section="2 / 3.1 Identification ARIMA", titre="ACF / PACF de Δlog(KO) (série différenciée)",
         montre="Mêmes graphiques sur le rendement. C'est LE graphique qui sert à choisir p et q.",
         regles="PACF coupe après p retards et ACF décroît => AR(p). ACF coupe après q retards et PACF décroît => MA(q). Aucun retard significatif => "
                "rendements ≈ bruit blanc (marche aléatoire : ARIMA(0,1,0)). Quelques pics isolés à 5 % sont normaux (1 retard sur 20)."),
    dict(f="fig05_stl_KO.png", script="01_donnees_eda.py", section="2 Exploration", titre="Décomposition STL de log(KO)",
         montre="4 panneaux : observé, tendance, saisonnalité (période 252 j), reste.",
         regles="Tendance = mouvement de fond. Saisonnalité d'amplitude minuscule devant la tendance (voir stl_forces.csv : force proche de 0) => pas de "
                "vraie saisonnalité annuelle. Le reste doit ressembler à du bruit ; ses pics = chocs (COVID)."),
    dict(f="fig06_outliers.png", script="01_donnees_eda.py", section="2 Exploration", titre="Valeurs aberrantes (IQR)",
         montre="Boxplot des rendements et rendements de KO avec les points hors de [Q1-1.5·IQR ; Q3+1.5·IQR] en rouge.",
         regles="Les points rouges se regroupent-ils sur des dates d'événements réels (mars 2020) ? Les chocs sont conservés, pas supprimés ; les indicateurs IQR "
                "sont archivés pour audit et ne sont pas des variables de modèle."),
    dict(f="fig07_resid_ARIMA_manuel.png", script="02_arima_sarima.py", section="3.1 ARIMA", titre="Diagnostic des résidus — ARIMA manuel",
         montre="Résidus dans le temps, ACF des résidus, histogramme vs loi normale, QQ-plot.",
         regles="ACF des résidus dans la bande => indépendance (Ljung-Box non rejeté). QQ-plot qui s'écarte aux extrémités / histogramme pointu => queues "
                "épaisses (normalité rejetée, très courant en finance). Résidus à amplitude variable => effet ARCH."),
    dict(f="fig08_resid_ARIMA_auto.png", script="02_arima_sarima.py", section="3.1 ARIMA", titre="Diagnostic des résidus — ARIMA auto",
         montre="Idem pour l'ordre trouvé par la recherche automatique.", regles="Même lecture ; comparer avec le modèle manuel (mêmes défauts ?)."),
    dict(f="fig09_resid_SARIMA.png", script="02_arima_sarima.py", section="3.1 SARIMA", titre="Diagnostic des résidus — SARIMA (m=5)",
         montre="Idem avec une composante saisonnière hebdomadaire.",
         regles="Le SARIMA améliore-t-il l'AIC/BIC ET les résidus ? Sinon la saisonnalité ajoute de la complexité sans valeur."),
    dict(f="fig10_prevision_arima.png", script="02_arima_sarima.py", section="3.1 / 6 Prévision", titre="Prévision à 10 jours avec IC 95 %",
         montre="60 derniers jours d'apprentissage, 10 jours observés du test, prévision et intervalle de prédiction.",
         regles="L'intervalle s'élargit avec l'horizon (incertitude cumulative). Si l'observé reste dans l'IC => intervalle bien calibré."),
    dict(f="fig11_resid_VAR_KO.png", script="03_var_cointegration.py", section="3.2 VAR", titre="Diagnostic des résidus — équation KO du VAR",
         montre="4 graphiques de diagnostic des résidus du VAR (équation KO).", regles="Même lecture que pour ARIMA ; les tests multivariés sont dans diag_var.csv."),
    dict(f="fig12_spread_cointegration.png", script="03_var_cointegration.py", section="3.2 Cointégration", titre="Écart de long terme KO − b·PEP",
         montre="Combinaison linéaire des log-prix issue de la régression de long terme.",
         regles="Écart qui revient sans cesse vers sa moyenne (mean-reverting) => cointégration plausible. Écart qui dérive comme une marche aléatoire => pas "
                "de cointégration. À croiser avec Engle-Granger/Johansen (key_results.json)."),
    dict(f="fig13_garch_volatilite.png", script="04_garch.py", section="3.3 GARCH", titre="Volatilité conditionnelle estimée",
         montre="Rendements de KO (haut) et écart-type conditionnel du GARCH (bas), avec des événements connus.",
         regles="Pics de volatilité alignés avec les événements (krach COVID, remontée des taux) => le modèle capte les clusters. Retour progressif au calme "
                "= persistance (Σα+Σβ proche de 1)."),
    dict(f="fig14_garch_residus.png", script="04_garch.py", section="3.3 GARCH", titre="Résidus standardisés du GARCH",
         montre="Diagnostic des résidus standardisés du GARCH retenu.",
         regles="Doivent redevenir sans autocorrélation et homoscédastiques (ARCH-LM et Ljung-Box sur z² non rejetés, voir diag_garch.csv). Un QQ-plot encore "
                "épais justifie la loi de Student."),
    dict(f="fig15_dl_pertes.png", script="05_deep_learning.py", section="4.1 LSTM/GRU", titre="Courbes de perte train / validation",
         montre="MSE à chaque epoch sur l'apprentissage et la validation (chronologique) pour LSTM et GRU.",
         regles="Perte de validation qui remonte pendant que celle du train baisse => surapprentissage (l'arrêt précoce le limite). Deux courbes proches et "
                "plates => le réseau apprend peu (signal faible, normal sur des rendements)."),
    dict(f="fig16_dl_cv_walkforward.png", script="05_deep_learning.py", section="4.1 Validation croisée temporelle", titre="Validation croisée walk-forward",
         montre="RMSE du rendement cumulé à 10 jours par pli (passé -> futur) pour LSTM et GRU, avec la baseline naïve.",
         regles="Barres systématiquement sous la ligne naïve => gain réel et robuste. Barres au-dessus ou instables d'un pli à l'autre => pas de gain robuste."),
    dict(f="fig17_previsions_test.png", script="06_evaluation.py", section="6 Prévision", titre="Prévisions de tous les modèles sur le test",
         montre="Blocs de 10 jours consécutifs de chaque modèle superposés au prix observé.",
         regles="Courbes quasi confondues => les modèles font la même chose (≈ dernier prix connu). Un modèle qui s'écarte beaucoup sans mieux suivre l'observé "
                "ne fait que du bruit."),
    dict(f="fig18_metriques.png", script="06_evaluation.py", section="6 Évaluation", titre="RMSE par modèle",
         montre="Comparaison du RMSE (USD) de tous les modèles ; le tableau complet MSE/RMSE/MAE/MAPE est dans metrics.csv.",
         regles="Plus petit = meilleur, mais lire avec le test de Diebold-Mariano : des RMSE proches ne sont pas forcément différents."),
    dict(f="fig19_dm_pvalues.png", script="06_evaluation.py", section="6 Évaluation", titre="Test de Diebold-Mariano (p-values par paires)",
         montre="Carte des p-values du test DM pour chaque paire de modèles.",
         regles="Cellule rouge (p < 0.05) => différence de précision significative. Cellules vertes/jaunes => on ne peut pas dire que les modèles diffèrent, "
                "même si les RMSE sont différents."),
]
