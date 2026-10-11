
# Proposition ordonnée de modèles pour KO et PEP (2019–2024)

## 1. Interprétation préalable des résultats

Les statistiques descriptives et les tests de stationnarité orientent vers une stratégie en deux étapes : modéliser les rendements pour la dynamique de la moyenne, puis leur volatilité conditionnelle.

| Résultat observé                                    | Interprétation statistique                                                                   | Conséquence pour la modélisation                                  |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Moyennes : KO = 0,00033 ; PEP = 0,00047             | Rendements journaliers moyens positifs dans l'échantillon d'apprentissage                    | Inclure une constante et vérifier sa significativité              |
| Écarts-types : 0,01318 et 0,01349                   | Volatilité marginale légèrement supérieure pour PEP                                          | Comparer les modèles de volatilité des deux titres                |
| Skewness négative                                   | Asymétrie à gauche, plus marquée pour KO (-0,95020)                                          | Examiner les distributions asymétriques et les chocs négatifs     |
| Kurtosis excess : 10,06613 et 22,49589              | Queues épaisses, particulièrement pour PEP                                                   | Tester des innovations Student-t plutôt que supposer la normalité |
| JB p = 0,00000 pour les deux titres                 | Rejet de la normalité selon le test de Jarque–Bera, sous réserve de la précision d'affichage | Éviter de retenir la normalité sans justification                 |
| ADF p = 0,4291 pour log(KO), 0,2782 pour log(PEP)   | L'ADF ne rejette pas l'hypothèse de racine unitaire                                          | Modéliser les logarithmes des prix après différenciation          |
| KPSS p = 0,01 pour les deux logarithmes             | Rejet de l'hypothèse de stationnarité aux seuils usuels                                      | Renforce l'hypothèse de non-stationnarité des prix                |
| ADF p = 0,0000 et KPSS p = 0,10 pour les rendements | Résultats compatibles avec la stationnarité des rendements                                   | Tester des modèles sur les rendements plutôt que sur les niveaux  |

Précaution : une p-value affichée à 0,0000 est une valeur arrondie, pas nécessairement une probabilité exactement nulle. De même, KPSS p = 0,10 peut correspondre à une p-value plafonnée à 0,10 par l'implémentation du test.

## 2. Liste ordonnée des hypothèses de modèles à tester

### 1 — ARIMA sur les logarithmes des prix

Priorité élevée

Hypothèse : les logarithmes des prix sont intégrés d'ordre 1, I(1)I(1)I(1), et leurs premières différences, correspondant aux rendements logarithmiques, sont stationnaires.

Les tests ADF et KPSS sur les niveaux soutiennent cette hypothèse, tandis que les résultats sur les rendements sont compatibles avec la stationnarité.

Modèles candidats :

* ARIMA(0,1,0), comme référence de marche aléatoire ;

* ARIMA(1,1,0) ;

* ARIMA(0,1,1) ;

* ARIMA(p,1,q), avec des ordres limités sélectionnés sur l'échantillon d'apprentissage.

Diagnostics de validation :

* Résidus sans autocorrélation significative : ACF des résidus et test de Ljung–Box.

* Paramètres estimés cohérents et modèle stable.

* Performances hors échantillon meilleures que celles de la référence, sur les mêmes dates et avec la même métrique.

Critères d'invalidation : autocorrélation résiduelle persistante ou absence d'amélioration hors échantillon.

Les statistiques fournies ne permettent pas de déterminer les ordres ppp et qqq. Il faut examiner l'ACF/PACF des rendements et comparer plusieurs spécifications.

### 2 — GARCH sur les rendements

Priorité très élevée

Hypothèse : les rendements ont une volatilité conditionnelle variable dans le temps, même si leur moyenne est stationnaire.

Cette hypothèse est motivée par les kurtosis excessives de KO (10,06613) et surtout de PEP (22,49589), ainsi que par le rejet de la normalité. Ces résultats sont compatibles avec des épisodes de volatilité élevée, mais ne prouvent pas à eux seuls l'existence d'un effet ARCH/GARCH.

Modèles candidats :

* GARCH(1,1) avec innovations normales ;

* GARCH(1,1) avec innovations Student-t ;

* éventuellement EGARCH ou GJR-GARCH pour étudier une asymétrie de la volatilité.

Diagnostics de validation :

1. Tester les effets ARCH dans les rendements ou dans les résidus d'un modèle de moyenne, à l'aide du test ARCH-LM.

2. Vérifier l'absence d'autocorrélation significative des résidus standardisés.

3. Vérifier l'absence d'effets ARCH résiduels après estimation.

4. Comparer les AIC/BIC et les performances de prévision de volatilité hors échantillon.

Critères d'invalidation : absence d'effets ARCH détectables, paramètres non admissibles ou persistance des effets ARCH dans les résidus standardisés au carré.

Une loi Student-t est une hypothèse particulièrement pertinente à tester compte tenu des queues épaisses. Elle ne doit toutefois être retenue que si les diagnostics et les critères de comparaison la favorisent.

### 3 — ARMA sur les rendements, éventuellement combiné à GARCH

Hypothèse : les rendements sont stationnaires et peuvent présenter une dépendance linéaire de courte durée dans leur moyenne.

Les rendements de KO et PEP sont stationnaires selon les tests fournis. Leurs moyennes sont positives, mais rien dans les statistiques descriptives ne démontre une autocorrélation.

Modèles candidats :

* ARMA(0,0), avec constante, comme référence ;

* ARMA(1,0) ;

* ARMA(0,1) ;

* quelques ARMA(p,q) supplémentaires si l'ACF/PACF le justifie ;

* ARMA(p,q)-GARCH(1,1) si la moyenne et la variance conditionnelle présentent chacune une dynamique.

Diagnostics de validation :

* ACF/PACF des rendements et test de Ljung–Box pour identifier une dépendance de la moyenne ;

* test ARCH-LM sur les résidus du modèle de moyenne ;

* AIC/BIC et performances hors échantillon.

Critère d'invalidation : si les rendements ne présentent pas d'autocorrélation significative et que les modèles ARMA n'améliorent pas les prévisions, conserver une moyenne constante est préférable.

### 4 — VAR sur les rendements de KO et PEP

Hypothèse : les rendements passés d'un titre apportent de l'information sur les rendements futurs de l'autre, au-delà de sa propre dynamique.

Cette approche est adaptée à deux séries stationnaires, mais les statistiques fournies ne renseignent pas sur leur corrélation ni sur les éventuelles relations croisées.

Modèle candidat :

(rKO,trPEP,t)=c+∑i=1pAi(rKO,t−irPEP,t−i)+εt\begin{pmatrix} r_{KO,t}\\ r_{PEP,t} \end{pmatrix} = c+\sum_{i=1}^{p}A_i \begin{pmatrix} r_{KO,t-i}\\ r_{PEP,t-i} \end{pmatrix} +\varepsilon_t(rKO,trPEP,t)=c+i=1∑pAi(rKO,t−irPEP,t−i)+εt

Commencer par un VAR(1), puis sélectionner le nombre de retards à l'aide de critères comme AIC et BIC, en tenant compte de la taille de l'échantillon.

Diagnostics de validation :

* Test de causalité au sens de Granger pour les deux directions.

* Test de Ljung–Box multivarié ou tests d'autocorrélation des résidus.

* Stabilité du VAR et amélioration des prévisions par rapport à deux modèles univariés.

Critères d'invalidation : absence de dépendance croisée utile, résidus mal spécifiés ou absence d'amélioration prédictive.

Attention : la causalité de Granger signifie une capacité prédictive conditionnelle, et non une causalité économique au sens strict.

### 5 — Cointégration de Johansen et VECM

Hypothèse : les logarithmes des prix de KO et PEP partagent une relation d'équilibre de long terme, malgré leur non-stationnarité individuelle.

Les tests ADF et KPSS sont compatibles avec des séries de prix I(1)I(1)I(1), condition nécessaire à l'approche classique de cointégration, mais insuffisante pour conclure qu'une relation de cointégration existe.

Étapes :

1. Vérifier l'ordre d'intégration des deux séries.

2. Tester la cointégration avec Johansen, en spécifiant correctement les constantes, les tendances éventuelles et le nombre de retards.

3. Si le rang de cointégration est égal à 1, estimer un VECM.

4. Si le rang est nul, privilégier un VAR sur les rendements ou les différences.

Diagnostics de validation :

* Test de Johansen indiquant un rang de cointégration positif.

* Terme de correction d'erreur statistiquement pertinent et cohérent avec le mécanisme d'ajustement.

* Résidus correctement spécifiés et amélioration éventuelle des prévisions.

Critère d'invalidation : absence de cointégration détectable. Dans ce cas, un VECM classique n'est pas justifié par les résultats.

La proximité sectorielle de Coca-Cola et PepsiCo ne suffit pas à établir une relation de cointégration.

### 6 — SARIMA sur les prix ou rendements

Hypothèse : il existe une structure saisonnière récurrente dans les séries.

Les données sont journalières et couvrent 2019–2024, mais les statistiques descriptives et les tests de stationnarité ne fournissent aucune preuve de saisonnalité.

Pour les données boursières, il faut distinguer une saisonnalité calendaire liée aux séances de cotation d'une périodicité artificielle due aux jours non négociés.

Modèles candidats : un ARIMA de référence, puis un SARIMA avec une période saisonnière justifiée par l'analyse des données.

Diagnostics de validation :

* ACF révélant des pics périodiques reproductibles.

* Analyse des effets calendaires, avec prise en compte des jours fériés et des séances de marché.

* Amélioration des résidus et des performances hors échantillon.

Critère d'invalidation : absence de saisonnalité stable ou absence de gain prédictif par rapport à ARIMA.

Il est prématuré de fixer une période saisonnière sss sans les graphiques et les diagnostics correspondants.

### 7 — Réseaux récurrents : LSTM ou GRU

Hypothèse : une architecture récurrente peut apprendre des dépendances temporelles non linéaires qui échappent aux modèles statistiques précédents.

Les queues épaisses et l'asymétrie des rendements rendent une exploration de modèles non linéaires envisageable, mais ne démontrent pas qu'un réseau récurrent sera supérieur.

Protocole de test :

* Utiliser les rendements, éventuellement complétés par des variables de volatilité et des indicateurs techniques disponibles à la date de prévision.

* Comparer LSTM et GRU à des références simples : moyenne constante, marche aléatoire pour les prix et ARIMA/GARCH selon la cible.

* Effectuer une validation chronologique, sans mélange aléatoire des observations.

* Ajuster les hyperparamètres uniquement sur les données d'apprentissage et de validation.

Diagnostics de validation :

* Gain hors échantillon sur une métrique définie à l'avance.

* Résultats robustes sur plusieurs fenêtres temporelles.

* Absence de fuite de données et de surapprentissage manifeste.

Critères d'invalidation : absence de gain face aux références simples, résultats instables ou forte dégradation sur la période de test.

Avec un split chronologique 90/10, le test représente une période finale limitée. Il est important de préserver cette période jusqu'à la comparaison finale et d'effectuer le réglage des modèles par validation chronologique à l'intérieur des 90 % d'apprentissage.

### 8 — Modèles hybrides : ARIMA-GARCH ou LSTM-GARCH

Hypothèse : la dynamique de la moyenne et celle de la volatilité sont différentes et nécessitent des composants distincts.

ARIMA-GARCH est une extension naturelle si les diagnostics indiquent une dépendance dans la moyenne et des effets ARCH dans les résidus. LSTM-GARCH peut être exploré ensuite, mais il est plus complexe et son intérêt empirique doit être démontré.

Diagnostics de validation :

* Résidus standardisés sans autocorrélation ni effets ARCH résiduels.

* Comparaison avec ARIMA seul, GARCH seul et ARMA-GARCH.

* Amélioration de la prévision de volatilité, de la densité prédictive ou de la perte de prévision choisie.

Critère d'invalidation : complexité supplémentaire sans amélioration mesurable hors échantillon.

## 3. Synthèse des priorités

| Rang | Famille de modèles    | Objectif principal                                           | Information encore nécessaire           |
| ---- | --------------------- | ------------------------------------------------------------ | --------------------------------------- |
| 1    | ARIMA                 | Prévoir les prix ou rendements par différenciation           | ACF/PACF, sélection des ordres          |
| 2    | GARCH(1,1), Student-t | Modéliser la volatilité conditionnelle                       | Test ARCH-LM, diagnostics résiduels     |
| 3    | ARMA / ARMA-GARCH     | Modéliser la moyenne et, si nécessaire, la volatilité        | Autocorrélation des rendements          |
| 4    | VAR                   | Détecter les dépendances croisées entre KO et PEP            | Corrélations, retards, tests de Granger |
| 5    | Johansen / VECM       | Tester une relation de long terme entre les prix             | Rang de cointégration                   |
| 6    | SARIMA                | Capturer une saisonnalité récurrente                         | ACF saisonnière, effets calendaires     |
| 7    | LSTM / GRU            | Détecter des dépendances non linéaires                       | Validation chronologique rigoureuse     |
| 8    | Modèles hybrides      | Combiner moyenne, volatilité et éventuellement non-linéarité | Gains mesurables face aux références    |

## 4. Conclusion scientifique

Les résultats disponibles justifient en priorité l'étude des modèles ARIMA sur les logarithmes des prix et des modèles de volatilité GARCH sur les rendements. Les queues épaisses et les asymétries invitent à comparer les innovations normales et Student-t, tandis que la stationnarité des rendements permet d'envisager des modèles ARMA et VAR.

En revanche, les données fournies ne permettent pas encore de conclure à l'existence d'effets ARCH, d'une saisonnalité, d'une causalité prédictive entre KO et PEP ou d'une relation de cointégration. Les modèles LSTM et hybrides doivent rester des candidats expérimentaux, à retenir uniquement si leurs performances hors échantillon dépassent celles de modèles de référence correctement spécifiés.

Enfin, le choix définitif doit reposer sur les diagnostics résiduels, la parcimonie et les performances de prévision sur les 10 % de données finales, sans utiliser cette période pour sélectionner les modèles.
