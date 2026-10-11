# Note de synthèse pour le directeur financier — Coca-Cola (KO) et PepsiCo (PEP), 2019–2024

## 1. Que nous apprennent les résultats sur KO et PEP ?

Les deux actions présentent des comportements suffisamment proches pour que plusieurs modèles produisent des prévisions similaires à 10 jours. Toutefois, aucun modèle testé ne démontre une amélioration statistiquement significative par rapport à la méthode la plus simple.

### Comparaison des prévisions du prix de KO

Meilleur RMSE observé

# 1,3970 $

GRU

Le GRU présente la plus faible erreur quadratique moyenne en racine, mais son avantage sur la méthode naïve est minime et n'est pas statistiquement significatif.

Meilleur MAE

## 1,0690 $

LSTM

Meilleur MAPE

## 1,7119 %

LSTM

Voici comment interpréter ces indicateurs :

* RMSE : mesure l'importance des erreurs de prévision, en pénalisant davantage les erreurs importantes. Le GRU obtient le meilleur résultat : 1,3970 $.

* MAE : mesure l'erreur absolue moyenne. Le LSTM obtient le meilleur résultat : 1,0690 $.

* MAPE : exprime l'erreur moyenne en pourcentage du prix réel. Le LSTM obtient le meilleur résultat : 1,7119 %.

Le fait que différents modèles arrivent en tête selon l'indicateur montre qu'il n'existe pas de vainqueur incontestable.

### Que vaut l'avantage du GRU ?

| Modèle            | RMSE     | MAPE     |
| ----------------- | -------- | -------- |
| GRU               | 1,3970 $ | 1,7262 % |
| VAR               | 1,3988 $ | 1,7399 % |
| SARIMA            | 1,4021 $ | 1,7129 % |
| ARIMA automatique | 1,4021 $ | 1,7129 % |
| Naïf              | 1,4040 $ | 1,7264 % |
| LSTM              | 1,4081 $ | 1,7119 % |
| VECM              | 1,5133 $ | 1,8345 % |

Le GRU réduit le RMSE d'environ 0,007 $ par rapport au modèle naïf, soit moins d'un centime par action sur cet indicateur.

Cette différence est très faible. De plus, le test de Diebold-Mariano donne une p-value de 0,8187 pour le GRU face au modèle naïf : il ne permet donc pas de conclure que le GRU prévoit mieux.

Il faut également noter que les modèles LSTM et GRU sont plus complexes et moins faciles à expliquer qu'ARIMA ou le modèle naïf. Leur complexité n'est pas justifiée par un avantage prédictif démontré ici.

## 2. Trois notions importantes expliquées simplement

### A. La cointégration : un possible lien à long terme

Imaginez que les cours de Coca-Cola et de PepsiCo évoluent chacun de leur côté au jour le jour, mais qu'un écart entre eux ait tendance à revenir vers une certaine relation à long terme.

C'est l'idée de la cointégration : deux séries de prix peuvent être individuellement instables, tout en présentant une relation statistique durable entre elles.

Dans les résultats fournis :

* Le test d'Engle–Granger donne une p-value de 0,0311. Selon les hypothèses du test et au seuil de 5 %, cela constitue un indice en faveur d'une relation de cointégration.

* Le rang de Johansen est indiqué à 2.

Point de vigilance : pour deux séries de prix supposées intégrées d'ordre 1, un rang de cointégration de 2 serait inhabituel et indiquerait que les spécifications et les hypothèses des tests doivent être réexaminées. Les deux résultats ne doivent donc pas être considérés comme une confirmation définitive sans vérification.

En pratique, cette relation éventuelle peut être intéressante pour étudier les liens de long terme entre les deux entreprises. Elle ne garantit cependant pas un bénéfice de trading ni de meilleures prévisions à 10 jours.

### B. La causalité de Granger : une information utile pour prévoir

La causalité de Granger répond à une question pratique : connaître les évolutions passées d'une action aide-t-il à prévoir l'autre action ?

Les résultats sont les suivants :

PEP → KO

## p = 0,0141

Relation prédictive détectée

Au seuil de 5 %, les résultats indiquent que les valeurs passées de PEP apportent une information prédictive supplémentaire pour KO, selon le modèle testé.

KO → PEP

## p = 0,0562

Résultat non significatif à 5 %

Les résultats ne permettent pas de rejeter l'hypothèse d'absence de causalité de Granger dans cette direction au seuil de 5 %.

Pour le directeur financier, cela signifie que les données passées de PepsiCo pourraient contribuer à la prévision de Coca-Cola.

Mais attention : Granger ne démontre pas que PepsiCo cause économiquement les variations de Coca-Cola. Il s'agit d'un lien de prévision, dépendant des variables et des retards retenus dans le modèle.

### C. La volatilité conditionnelle : le risque change avec le temps

La volatilité représente l'ampleur des variations du cours. Elle n'est pas nécessairement constante : une période calme peut être suivie d'une période de fortes variations.

Le meilleur modèle de volatilité indiqué est un GARCH(1,1) avec innovations Student-t.

Les résultats indiquent :

* Persistance de la volatilité : 0,9656.

* Demi-vie estimée : environ 19,83 jours de cotation.

La persistance élevée suggère que les épisodes de forte volatilité ont tendance à durer. Après un choc, le risque ne revient donc pas immédiatement à son niveau habituel.

La demi-vie signifie, selon la dynamique estimée, qu'il faudrait environ 19,83 jours de cotation pour que l'excès de variance conditionnelle associé à un choc soit réduit de moitié.

La loi Student-t permet par ailleurs de tenir compte de variations extrêmes plus fréquentes que ne le prévoit une distribution normale.

Conséquence financière : ce modèle peut aider à surveiller le risque et à adapter les limites de risque ou les scénarios de stress. Il ne prédit pas à lui seul si le cours va monter ou baisser.

## 3. Quel modèle retenir ?

Le choix dépend de l'objectif du directeur financier.

## Pour prévoir le prix de KO à 10 jours : modèle naïf comme référence opérationnelle

Il consiste à utiliser le dernier prix observé comme référence pour la prévision. Il est très simple, immédiatement interprétable et ne demande pratiquement aucun coût de calcul.

Les résultats ne démontrent pas que les modèles plus complexes le surpassent significativement.

## Pour analyser le risque : GARCH(1,1)-Student-t

C'est le candidat à retenir pour approfondir l'estimation de la volatilité conditionnelle, compte tenu de la forte persistance observée et des variations extrêmes. Il faut néanmoins vérifier ses diagnostics résiduels avant de l'utiliser en production.

## Pour tester une amélioration : ARIMA automatique

Il offre une bonne combinaison de simplicité et d'interprétabilité. Son RMSE est de 1,4021 $ et son MAPE de 1,7129 %. Il constitue un candidat raisonnable, mais son avantage face au modèle naïf n'est pas statistiquement démontré.

Le GRU reste le meilleur modèle selon le RMSE observé, tandis que le LSTM présente les meilleurs MAE et MAPE. Il serait prématuré de retenir définitivement l'un de ces deux réseaux uniquement sur cette base.

Le modèle VECM affiche les erreurs les plus élevées parmi les modèles comparés : RMSE de 1,5133 $ et MAPE de 1,8345 %. Il ne constitue donc pas le meilleur choix pour cet objectif de prévision à court terme.

## 4. Recommandation finale

Pour une décision financière prudente, je recommande de conserver le modèle naïf comme référence de prévision, de tester ARIMA automatique comme solution interprétable et d'utiliser GARCH(1,1)-Student-t pour l'analyse du risque.

Avant tout déploiement, il reste nécessaire de vérifier les diagnostics des modèles, la cohérence des résultats de cointégration et la stabilité des performances sur d'autres fenêtres chronologiques. Une bonne prévision historique ne garantit pas les mêmes performances à l'avenir.

Le message essentiel pour le directeur financier est simple : les modèles complexes ne fournissent pas encore de gain de prévision démontré, mais l'analyse de la volatilité apporte une information utile pour mieux comprendre et gérer le risque.

