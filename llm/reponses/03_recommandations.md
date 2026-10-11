# Recommandations d’investissement simulées — Coca-Cola (KO) et PepsiCo (PEP)

Exercice pédagogique fondé sur les données 2019–2024 et les prévisions de volatilité à 10 jours. Il ne s’agit pas de recommandations financières personnalisées.

## 1. Recommandations proposées

Ma recommandation principale : adopter une stratégie prudente, conserver une référence simple pour les prévisions de prix et utiliser le GARCH pour encadrer le risque. Les résultats ne démontrent pas qu'un modèle complexe permet de gagner davantage sur le marché.

## KO — Coca-Cola

Position simulée prudente

Conserver KO dans un portefeuille pédagogique diversifié, ou envisager une petite position simulée si le scénario d'investissement le justifie. Ne pas augmenter l'exposition uniquement parce qu'un modèle prévoit un prix favorable : les résultats fournis ne donnent pas de signal fiable d'achat.

## PEP — PepsiCo

Position simulée conditionnelle

Maintenir PEP dans la liste des actifs à comparer avec KO. Les résultats de prévision fournis concernent principalement le prix de KO : ils ne suffisent donc pas à justifier une position plus importante ou plus faible sur PEP.

Ces choix sont des scénarios d'étude, pas des signaux d'achat ou de vente calculés à partir d'une rentabilité attendue démontrée.

### Quel modèle utiliser pour prendre les décisions ?

| Modèle               | Résultat observé                          | Utilisation proposée                                  |
| -------------------- | ----------------------------------------- | ----------------------------------------------------- |
| GRU                  | RMSE = 1,3970 $                           | Comparaison expérimentale des prévisions              |
| ARIMA automatique    | RMSE = 1,4021 $                           | Référence interprétable pour prévoir KO               |
| Naïf                 | RMSE = 1,4040 $                           | Référence minimale à battre                           |
| GARCH(1,1)-Student-t | Volatilité prévue en hausse sur l'horizon | Surveillance du risque, pas direction du prix         |
| VECM                 | RMSE = 1,5133 $                           | Ne pas privilégier pour cette prévision à court terme |

Le GRU affiche le meilleur RMSE, mais son avantage par rapport au modèle naïf reste très faible et n'est pas statistiquement significatif d'après les tests fournis précédemment. Je ne justifierais donc pas son utilisation opérationnelle uniquement sur cette base.

## 2. Comment interpréter le risque sur les dix prochains jours ?

Les prévisions GARCH indiquent que l'écart-type journalier attendu passe de 0,8596 % au premier jour à 0,9793 % au dixième jour.

Jour 1

# 0,8596 %

Jour 10

# 0,9793 %

La volatilité prévue augmente progressivement. Cela indique une hausse du risque estimé, et non une prévision de baisse du cours.

### Conséquences pour une stratégie simulée

* Limiter la taille des positions : ne pas engager une part trop importante du portefeuille sur un seul titre.

* Prévoir des scénarios de stress : examiner les conséquences d'une baisse importante, en particulier si la volatilité augmente.

* Éviter les décisions fondées sur une seule prévision : actualiser les estimations lorsque de nouvelles données sont disponibles.

* Ne pas confondre risque et rendement : une volatilité plus élevée ne signifie pas nécessairement une meilleure performance attendue.

Les chiffres de volatilité fournis ne permettent pas de calculer à eux seuls une perte maximale, une probabilité de perte ou une valeur à risque fiable : il faudrait préciser la distribution prédictive, l'horizon cumulé et les hypothèses retenues.

## 3. Exemple de portefeuille simulé

Pour illustrer une approche prudente, on pourrait utiliser une allocation fictive comme point de départ, sans prétendre qu'elle est optimale.

### Allocation pédagogique illustrative

Pour un portefeuille fictif de 10 000 $, uniquement à titre d'exemple.

KO

### 20 %

2 000 $

PEP

### 20 %

2 000 $

Liquidités

### 60 %

6 000 $

Cette allocation est un exemple arbitraire de prudence, pas un résultat issu de l'optimisation statistique. La part de liquidités réduit l'exposition aux actions dans cet exemple, mais ne garantit pas l'absence de perte.

La répartition 20 % KO, 20 % PEP et 60 % liquidités n'est pas déduite des RMSE. Pour proposer une allocation réellement optimisée, il faudrait notamment disposer des rendements, des corrélations, des dividendes, des contraintes de risque et des objectifs du portefeuille.

## 4. Limites, risques et hypothèses fragiles

Il est essentiel de reconnaître les limites de ces recommandations avant toute interprétation financière.

1. Efficience des marchés

Les cours peuvent intégrer rapidement les informations publiques. Une bonne prévision historique ne signifie pas qu'une stratégie peut systématiquement obtenir un rendement excédentaire exploitable.

2. Coûts de transaction et dividendes

Les performances fournies ne précisent pas si elles intègrent les commissions, les écarts achat-vente, la fiscalité et les dividendes. Ces éléments peuvent modifier la rentabilité réelle d'une stratégie.

3. Instabilité des paramètres

Les relations observées entre 2019 et 2024 peuvent évoluer avec les taux d'intérêt, l'inflation, la concurrence ou les conditions macroéconomiques. Les modèles doivent être réévalués régulièrement.

4. Surapprentissage des réseaux neuronaux

Le GRU et le LSTM peuvent apprendre des particularités historiques plutôt qu'une relation durable. Leurs performances doivent être confirmées sur plusieurs périodes indépendantes.

5. Risque de modèle et choix des métriques

Le GRU est meilleur selon le RMSE, tandis que le LSTM est meilleur selon le MAE et le MAPE. Le classement dépend donc de la mesure choisie. Une faible erreur de prix ne garantit pas une bonne décision d'achat ou de vente.

6. Limitation du test chronologique

Le test porte sur les 10 % de données finales. Cette période peut ne pas représenter tous les régimes de marché. Il faut compléter l'évaluation par une validation chronologique glissante et conserver un véritable jeu de test indépendant.

7. Hausse de volatilité : hypothèse fragile

Les valeurs GARCH sont des estimations conditionnelles issues d'un modèle. Elles ne sont ni certaines ni garanties. Un événement exceptionnel peut produire une volatilité bien supérieure aux prévisions.

8. Comparabilité de KO et PEP

Les erreurs de prévision présentées concernent le prix de KO. Sans les performances prédictives équivalentes de PEP et une analyse des rendements des deux titres, il est impossible de conclure que KO constitue un meilleur investissement que PEP.

## Conclusion

Pour cet exercice, je privilégierais une approche en trois volets : prévision simple avec le modèle naïf ou ARIMA automatique, surveillance du risque avec GARCH et allocation simulée prudente.

Je ne choisirais pas KO plutôt que PEP, ni le GRU plutôt qu'ARIMA, sur la seule base de ces résultats. Les prévisions de prix sont proches, les gains prédictifs ne sont pas statistiquement établis et la volatilité croissante appelle à la prudence plutôt qu'à une décision directionnelle automatique.


Évolution de la volatilité journalière prévue

Écart-type journalier en pourcentage, selon les prévisions GARCH fournies.

horizon	vol
J1	0,8596
J2	0,8758
J3	0,8911
J4	0,9056
J5	0,9194
J6	0,9326
J7	0,9451
J8	0,957
J9	0,9684
J10	0,9793
