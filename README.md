PROJET SCIENCE DES DONNÉES - CLASSIFICATION SUPERVISÉE (ASSURANCE AUTOMOBILE)

Ce dépôt GitHub contient les scripts et analyses réalisés dans le cadre du projet de
classification supervisée pour la compagnie d'assurance automobile "On the Road".

L'objectif principal est de prédire si un client effectuera une demande d'indemnisation
au cours de sa période d'assurance.

-------------------------------------------------------------------------------
1. ARCHITECTURE DU RÉPERTOIRE
-------------------------------------------------------------------------------

- car_insurance.csv ->        *Jeu de données d'origine (10 000 lignes, 18 colonnes)*
- car_insurance_clean.csv ->  *Jeu de données nettoyé et transformé (généré à l'étape 5)*
- partie_4_examen.py ->       *Analyse descriptive et visualisation des distributions*
- partie_5_preparation.py ->  *Nettoyage, imputation, encodage et normalisation*
- partie_6_correlations.py -> *Analyse des corrélations de Pearson et génération de la Heatmap*
- partie_7_separation.py ->   *Découpage du dataset en sous-ensembles Train (80%) / Test (20%)*
- partie_8_apprentissage.py -> *Entraînement de la Régression Logistique et analyse des poids*
- README.txt ->               *Documentation générale du projet (ce fichier)*

-------------------------------------------------------------------------------
2. SYNTHÈSE DES ÉTAPES RÉALISÉES
-------------------------------------------------------------------------------

PARTIE 3 : IMPORTATION DES DONNÉES
- Utilisation de la bibliothèque Pandas (`pd.read_csv`) pour charger le dataset.
- Vérification de la structure initiale à l'aide des méthodes `df.head()` et `df.info()`.
- Identification du volume de départ : 10 000 observations et 18 variables initiales.

PARTIE 4 : EXAMEN DES DONNÉES
- Diagnostic de la qualité des données à l'aide des méthodes `isna()` et `describe()`.
- Génération d'histogrammes pour toutes les variables numériques (hors variable 'id').
- Découverte et identification de deux données aberrantes introduites dans le dataset :
  * `speeding_violations` : Présence d'une valeur extrême aberrante (~40 000 infractions).
  * `children` : Présence d'une valeur extrême aberrante (~100 enfants).

Observations à partir des histogrammes: 

Les graphiques credit_score et annual_mileage forment cloche bien régulière. Cela montre que la majorité des clients se situent autour d'une moyenne.  

Il y a des valeurs abérrantes : Les graphiques de children et speeding_violations ont des échelles incohérentes :des clients avec 100 enfants ou 40 000 infractions, ce qui est impossible dans la réalité.

Pour les accidents passés et la conduite en état d'ivresse, la barre du "0" écrase tout. C'est normal car la majorité des conducteurs n'a aucun antécédent.

La cible à prédire (outcome) : Il y a un énorme bloc de "0" (pas de réclamation) et un petit bloc de "1" (réclamation). C'est le quotidien des compagnies d'assurance : la majeure partie des clients paye son assurance sans jamais avoir de sinistre.

PARTIE 5 : PRÉPARATION DES DONNÉES
- Suppression de la colonne 'id' car elle est ne sert à rien dans la prédiction.
- Correction des données aberrantes (`speeding_violations` et `children`) en remplaçant les valeurs extrêmes par la médiane de la colonne correspondante.
- Traitement des données manquantes (moins de 10% pour `credit_score` et `annual_mileage`) en remplaçant par la valeur médiane via la méthode `fillna()`.
- Encodage des 5 variables (`driving_experience`, `education`, `income`, `vehicle_year`, `vehicle_type`) en valeurs numériques à l'aide de la classe `LabelEncoder` de Scikit-Learn.
- Exportation d'un fichier intermédiaire propre nommé `car_insurance_clean.csv`.
- Normalisation (centrage et réduction) des variables d'entrée à l'aide de la classe `StandardScaler` de Scikit-Learn pour amener toutes les variables à la même échelle (moyenne = 0, écart-type = 1).

PARTIE 6 : RECHERCHE DE CORRÉLATIONS
- Calcul des coefficients de corrélation linéaire de Pearson à l'aide de la méthode `corr()`.
- Génération d'une Heatmap (carte de chaleur) hautement visuelle à l'aide de Seaborn (`sns.heatmap`) pour détecter instantanément les variables les plus prometteuses et les potentielles redondances (colinéarités).

Commentaire de la visualisation:
Si l'on regarde la ligne tout en bas dédiée à notre cible (outcome), on constate que l'expérience de conduite (driving_experience à -0,50) et l'âge (age à -0,45) possèdent les plus fortes teintes bleues.

Donc plus un conducteur est âgé et possède d'années d'expérience, moins il a de risques de faire une demande d'indemnisation auprès de l'assurance.

À l'inverse, l'année d'immatriculation du véhicule a une influence positive sur l'augmentation du risque de réclamation.

Enfin, on remarque une forte corrélation logique entre deux variables d'entrée : l'âge et l'expérience de conduite sont fortement liés, ce qui confirme de manière évidente qu'on accumule des années de permis en vieillissant.

PARTIE 7 : EXTRACTION DES JEUX D'APPRENTISSAGE ET DE TEST
- Utilisation de la fonction `train_test_split()` de Scikit-Learn.
- Choix d'une répartition standardisée de 80% pour le jeu d'apprentissage (8 000 échantillons) et 20% pour le jeu de test (2 000 échantillons).
- Utilisation d'un état aléatoire fixé (`random_state=42`) pour garantir la reproductibilité parfaite du découpage d'un script à l'autre.

PARTIE 8 : ENTRAÎNEMENT DU MODÈLE
- Implémentation et entraînement d'un premier classifieur binaire à l'aide de la classe `LogisticRegression()` de Scikit-Learn.
- Assimilation des fondements théoriques de l'algorithme :
  * Hypothèse : Le logarithme du rapport des vraisemblances (fonction logit) est supposé linéaire par rapport aux variables explicatives.
  * Optimisation : Minimisation de la fonction de coût (Log-Loss / Entropie croisée) par le biais du critère du maximum de vraisemblance via le solveur L-BFGS.
  * Paramètres : Calcul d'un biais/intercept global (b0 = -1.6900) et de 16 coefficients/poids associés aux caractéristiques.
- Représentation graphique de l'importance et de l'impact des variables sous forme d'un diagramme en barres horizontales triées, mettant en évidence les facteurs de risques positifs (accidents passés) et les facteurs protecteurs négatifs (excellent score de crédit).

Réponse aux questions :
- Hypothèse: l'algorithme fait l'hypothèse qu'en combinant toutes les informations du client de manière linéaire (en les additionnant avec des coefficients), on obtient une valeur qui se traduit directement en probabilité de faire une réclamation (entre 0 et 1) grâce à une courbe en "S" (la fonction sigmoïde).
- Minimisation de la focntion de coût: Pour trouver les meilleurs réglages possibles et faire le moins d'erreurs, l'algorithme utilise une technique mathématique appelée la descente de gradient.
- Apprentissage: Le biais et les coefficients

PARTIE 9 : EVALUATION DU MODELE

- Accuracy (84,00 %) : Le modèle obtient une note globale de 84 % de bonnes réponses sur l'ensemble du jeu de test.  
- Précision (77,60 %) : Lorsqu'il prédit qu'un client va faire une réclamation, le modèle a vu juste dans 77,60 % des cas.  
- Rappel / Recall (69,51 %) : L'algorithme réussit à intercepter et capturer près de 70 % des sinistres réels survenus.  
- Succès sur les bons conducteurs : Identification sans faute de 1 240 clients sans sinistre (Vrais Négatifs).  
- Succès sur les profils à risques : Interception réussie de 440 clients qui allaient faire une réclamation (Vrais Positifs).  
- Fausses alertes limitées : Seulement 127 conducteurs prudents ont été classés à tort comme dangereux par l'algorithme (Faux Positifs).

-------------------------------------------------------------------------------
3. INSTRUCTIONS D'EXÉCUTION
-------------------------------------------------------------------------------
Pour exécuter les scripts dans VS Code, assurez-vous d'avoir installé les dépendances nécessaires dans votre environnement virtuel :
$ pip install pandas numpy matplotlib seaborn scikit-learn

Les scripts doivent être lancés séquentiellement ou de manière autonome, en veillant à ce que les fichiers `.csv` soient présents dans le même dossier de travail.
