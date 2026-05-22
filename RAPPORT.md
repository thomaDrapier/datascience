# Projet Science des Données — Classification supervisée (assurance automobile)

**Auteurs :** [À COMPLÉTER : vos noms]
**Formation :** FISA — IMT Nord Europe
**Date :** mai 2026

> **Note.** Les passages **« Idées clés — vidéo / pptx »** (fin de section 8, puis
> sections 9, 10, 12 et 13) ne sont PAS à rédiger : ce sont les idées-force à
> présenter à l'oral dans la vidéo / le pptx, exposées librement le jour de la
> présentation. Supprimer cette note dans la version finale.

---

## 1. Introduction et contexte

Ce projet se place dans le cadre d'un projet de science des données de type
classification supervisée, réalisé pour la compagnie d'assurance automobile
« On the Road ». L'objectif métier est de **prédire si un client effectuera une
demande d'indemnisation** au cours de sa période d'assurance. Il s'agit donc d'un
problème de **classification binaire** : la variable de sortie `outcome` vaut 1
(réclamation) ou 0 (pas de réclamation).

Le travail suit les étapes caractéristiques d'un projet de science des données :
compréhension et préparation des données, entraînement et comparaison de modèles,
puis sauvegarde du meilleur modèle en vue d'une mise en production.

## 2. Présentation et importation des données (Partie 3)

Le jeu de données `car_insurance.csv` a été importé avec la fonction
`pd.read_csv()` de Pandas. Sa structure a été vérifiée avec `df.head()` et
`df.info()` : il comporte **10 000 observations** et **18 variables**.

Les variables d'entrée décrivent le profil du client et de son véhicule (âge,
expérience de conduite, score de crédit, kilométrage annuel, antécédents, etc.).
La **variable de sortie** est `outcome`, l'étiquette de classe à prédire.

## 3. Examen des données (Partie 4)

La qualité des données a été diagnostiquée avec `isna()` et `describe()`, et des
histogrammes ont été générés pour les variables numériques (hors `id`). Deux
**valeurs aberrantes** ont été identifiées : `speeding_violations` (une valeur
d'environ 40 000 infractions) et `children` (une valeur d'environ 100 enfants).

**Observations à partir des histogrammes.** Les graphiques de `credit_score` et
`annual_mileage` forment une cloche bien régulière : la majorité des clients se
situent autour d'une moyenne. Les variables `children` et `speeding_violations`
présentent des échelles incohérentes (un client avec 100 enfants ou 40 000
infractions, impossible en réalité). Pour les accidents passés et la conduite en
état d'ivresse, la barre du « 0 » écrase tout le reste, ce qui est normal : la
majorité des conducteurs n'a aucun antécédent. Enfin, la cible `outcome` présente
un gros bloc de « 0 » et un petit bloc de « 1 » : la majorité des clients paie
son assurance sans jamais avoir de sinistre.

*Figure : « Etape 4 - histogrammes.png ».*

## 4. Préparation des données (Partie 5)

Plusieurs traitements ont été appliqués pour obtenir des données numériques et de
qualité :

- **Suppression de la colonne `id`**, inutile pour la prédiction ;
- **Correction des valeurs aberrantes** (`speeding_violations`, `children`) en
  remplaçant les valeurs extrêmes par la médiane de la variable ;
- **Imputation des valeurs manquantes** (`credit_score`, `annual_mileage` — moins
  de 10 % de valeurs manquantes) par la médiane via `fillna()` ;
- **Encodage** de 5 variables qualitatives (`driving_experience`, `education`,
  `income`, `vehicle_year`, `vehicle_type`) avec `LabelEncoder` ;
- **Normalisation** des variables d'entrée avec `StandardScaler` (centrage-réduction).

Le jeu de données nettoyé a été exporté dans `car_insurance_clean.csv`.

## 5. Recherche de corrélations (Partie 6)

Les coefficients de corrélation linéaire de Pearson ont été calculés avec
`corr()`, puis visualisés sous forme de heatmap (Seaborn).

**Commentaire de la visualisation.** Sur la ligne dédiée à la cible `outcome`,
l'expérience de conduite (`driving_experience`, −0,50) et l'âge (`age`, −0,45)
présentent les corrélations négatives les plus fortes : plus un conducteur est
âgé et expérimenté, moins il a de risques de faire une demande d'indemnisation. À
l'inverse, l'année d'immatriculation du véhicule a une influence positive sur le
risque de réclamation. On observe aussi une forte corrélation entre deux variables
d'entrée, l'âge et l'expérience de conduite, ce qui est logique (on accumule des
années de permis en vieillissant).

*Figure : « Etape 6 - Matrice de corrélation.png ».*

## 6. Extraction des jeux d'apprentissage et de test (Partie 7)

Le jeu de données a été découpé avec `train_test_split()` selon une répartition
**80 % / 20 %** : 8 000 échantillons pour l'apprentissage et 2 000 pour le test.
Le paramètre `random_state=42` garantit la reproductibilité du découpage.

## 7. Entraînement du modèle (Partie 8)

Un premier classifieur binaire a été entraîné avec `LogisticRegression()` de
Scikit-Learn. L'entraînement calcule un biais (intercept, b0 ≈ −1,69) et 16
coefficients associés aux variables d'entrée. Un diagramme en barres horizontales
représente l'importance et l'impact de chaque variable (facteurs de risque
positifs, facteurs protecteurs négatifs).

**Réponses aux questions sur la régression logistique :**

- *Hypothèse :* l'algorithme suppose qu'en combinant linéairement toutes les
  informations du client (somme pondérée par des coefficients), on obtient une
  valeur qui se traduit directement en probabilité de réclamation (entre 0 et 1)
  via une courbe en « S », la fonction sigmoïde.
- *Minimisation de la fonction de coût :* pour trouver les meilleurs réglages et
  faire le moins d'erreurs possible, l'algorithme utilise la descente de gradient.
- *Apprentissage :* les paramètres calculés sont le biais et les coefficients.

*Figure : « Etape 8 - Coefficients du modèle entrainé.png ».*

## 8. Évaluation du modèle (Partie 9)

Le modèle entraîné a servi à prédire les classes du jeu de test, puis ses
performances ont été mesurées :

- **Accuracy : 84,00 %** — proportion globale de bonnes réponses sur le jeu de test ;
- **Précision : 77,60 %** — lorsqu'il prédit une réclamation, le modèle a raison
  dans 77,60 % des cas ;
- **Rappel : 69,51 %** — le modèle intercepte environ 70 % des sinistres réels.

Matrice de confusion : 1 240 vrais négatifs (clients sans sinistre bien
identifiés), 440 vrais positifs (réclamations bien interceptées), 127 faux
positifs (conducteurs prudents classés à tort comme à risque).

**Idées clés — vidéo / pptx :**

- le **F1-score** = moyenne harmonique de la précision et du rappel ;
- il n'est élevé que si la précision **et** le rappel sont bons → il résume le
  compromis entre les deux (valeur chiffrée affichée par `9_evaluation.py`) ;
- bilan : modèle globalement correct (84 %), mais point faible = le rappel (~70 %).

*Figure : « Etape 9 - Evaluation du modele.png ».*

## 9. Amélioration de l'évaluation : validation croisée (Partie 10)

Une validation croisée 5-Fold (`KFold`, `cross_val_score`) a été réalisée. Les
cinq passes donnent les accuracies : 0,8400 / 0,8340 / 0,8530 / 0,8445 / 0,8445,
soit une **accuracy moyenne de 0,8432** et un **écart-type de 0,0062**.
L'évaluation simple de la Partie 9 (un seul découpage 80/20) donnait 0,8400.

**Idées clés — vidéo / pptx :**

- 5 passes très resserrées + écart-type minuscule (0,62 %) → performance stable,
  qui ne dépend pas du découpage choisi ;
- le 84 % de la Partie 9 est donc confirmé : ce n'était pas un tirage chanceux ;
- la validation croisée donne une estimation plus fiable qu'un seul découpage,
  car elle moyenne le résultat sur 5 essais au lieu d'un seul, potentiellement biaisé.

## 10. Comparaison des algorithmes (Partie 11)

Trois classifieurs ont été comparés par validation croisée 5-Fold (mêmes passes,
pour une comparaison équitable) :

- **Régression logistique : 0,8432** (écart-type 0,0062) ;
- **K plus proches voisins : 0,8116** (écart-type 0,0057) ;
- **Perceptron : 0,7837** (écart-type 0,0284).

**Idées clés — vidéo / pptx :**

- classement : régression logistique (84,3 %) > KNN (81,2 %) > perceptron (78,4 %) ;
- le perceptron est le plus faible **et** le plus instable : il cherche juste *un*
  hyperplan séparateur, sans optimiser de vraie fonction de coût ;
- la régression logistique optimise le log-loss → solution stable et plus précise ;
- le KNN est correct mais en dessous → la frontière entre les classes est bien
  décrite par un modèle linéaire ;
- conclusion : on retient la régression logistique.

## 11. Sauvegarde du modèle (Partie 12)

Le modèle final retenu — la régression logistique, encapsulée avec son
`StandardScaler` dans un pipeline — a été entraîné sur l'ensemble des données puis
**sérialisé avec la bibliothèque Pickle** (`pickle.dump`). Son chargement a été
vérifié (`pickle.load`), et le modèle rechargé a produit des prédictions sur de
nouvelles données, validant sa réutilisation en production.

## 12. Conclusion

**Idées clés — vidéo / pptx :**

- objectif : prédire si un client effectuera une demande d'indemnisation
  (classification binaire) ;
- résultat : modèle retenu = régression logistique, ~84 % d'accuracy, confirmé
  par validation croisée ;
- le projet a couvert toute la chaîne d'un projet de classification :
  compréhension → préparation → modélisation → évaluation → comparaison → sauvegarde.

## 13. Limites et perspectives

**Idées clés — vidéo / pptx :**

- cible **déséquilibrée** (beaucoup plus de « 0 » que de « 1 ») → l'accuracy de
  84 % est un peu flatteuse ;
- rappel à ~70 % → le modèle laisse passer ~30 % des vraies réclamations :
  gênant pour une assurance dont l'enjeu est justement de les repérer ;
- la validation croisée `KFold` utilisée n'est pas stratifiée ;
- pistes d'amélioration : rééquilibrage des classes, ajustement du seuil de
  décision, essai d'autres modèles.

---

## Annexe technique

- **IDE :** [PyCharm / VS Code — à préciser selon ce que vous avez utilisé]
- **Langage :** Python 3.12.10
- **Système d'exploitation :** Windows 11
- **Bibliothèques :** voir `requirements.txt` (pandas, numpy, matplotlib, seaborn,
  scikit-learn)
- **Reproductibilité :** `random_state = 42` fixé pour tous les découpages et modèles.
