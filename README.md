===============================================================================
PROJET SCIENCE DES DONNÉES - CLASSIFICATION SUPERVISÉE (ASSURANCE AUTOMOBILE)
===============================================================================

Ce dépôt GitHub contient les scripts et analyses réalisés dans le cadre du projet de
classification supervisée pour la compagnie d'assurance automobile "On the Road".

L'objectif principal est de prédire si un client effectuera une demande d'indemnisation
(variable cible : 'outcome', 0 = pas de réclamation, 1 = réclamation effectuée) au cours
de sa période d'assurance.

-------------------------------------------------------------------------------
1. ARCHITECTURE DU RÉPERTOIRE
-------------------------------------------------------------------------------
Le projet est structuré de manière modulaire, conformément aux bonnes pratiques de la
science des données. Chaque étape clé possède son propre script Python indépendant :

├── car_insurance.csv         # Jeu de données d'origine (10 000 lignes, 18 colonnes)
├── car_insurance_clean.csv   # Jeu de données nettoyé et transformé (généré à l'étape 5)
├── partie_4_examen.py        # Analyse descriptive et visualisation des distributions
├── partie_5_preparation.py   # Nettoyage, imputation, encodage et normalisation
├── partie_6_correlations.py  # Analyse des corrélations de Pearson et génération de la Heatmap
├── partie_7_separation.py    # Découpage du dataset en sous-ensembles Train (80%) / Test (20%)
├── partie_8_apprentissage.py # Entraînement de la Régression Logistique et analyse des poids
└── README.txt                # Documentation générale du projet (ce fichier)

-------------------------------------------------------------------------------
2. SYNTHÈSE DES ÉTAPES RÉALISÉES
-------------------------------------------------------------------------------

PARTIE 3 : IMPORTATION DES DONNÉES
- Utilisation de la bibliothèque Pandas (`pd.read_csv`) pour charger le dataset.
- Vérification de la structure initiale à l'aide des méthodes `df.head()` et `df.info()`.
- Identification du volume de départ : 10 000 observations et 18 variables initiales.

PARTIE 4 : EXAMEN DES DONNÉES
- Diagnostic complet de la qualité des données à l'aide des méthodes `isna()` et `describe()`.
- Génération d'histogrammes pour toutes les variables numériques pertinentes (hors variable 'id').
- Découverte et identification de deux données aberrantes majeures introduites dans le dataset :
  * `speeding_violations` : Présence d'une valeur extrême aberrante (~40 000 infractions).
  * `children` : Présence d'une valeur extrême aberrante (~100 enfants).

PARTIE 5 : PRÉPARATION DES DONNÉES
- Suppression de la colonne 'id' car elle est dénuée de pouvoir prédictif.
- Correction des données aberrantes (`speeding_violations` et `children`) en remplaçant les valeurs extrêmes par la médiane de la colonne correspondante.
- Traitement des données manquantes (moins de 10% pour `credit_score` et `annual_mileage`) par imputation avec la valeur médiane via la méthode `fillna()`.
- Encodage des 5 variables qualitatives/catégorielles textuelles (`driving_experience`, `education`, `income`, `vehicle_year`, `vehicle_type`) en valeurs numériques à l'aide de la classe `LabelEncoder` de Scikit-Learn.
- Exportation d'un fichier intermédiaire propre nommé `car_insurance_clean.csv`.
- Normalisation (centrage et réduction) des variables d'entrée à l'aide de la classe `StandardScaler` de Scikit-Learn pour amener toutes les variables à la même échelle (moyenne = 0, écart-type = 1).

PARTIE 6 : RECHERCHE DE CORRÉLATIONS
- Calcul des coefficients de corrélation linéaire de Pearson à l'aide de la méthode `corr()`.
- Génération d'une Heatmap (carte de chaleur) hautement visuelle à l'aide de Seaborn (`sns.heatmap`) pour détecter instantanément les variables les plus prometteuses et les potentielles redondances (colinéarités).

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

-------------------------------------------------------------------------------
3. INSTRUCTIONS D'EXÉCUTION
-------------------------------------------------------------------------------
Pour exécuter les scripts dans VS Code, assurez-vous d'avoir installé les dépendances nécessaires dans votre environnement virtuel :

Commande d'installation :
$ pip install pandas numpy matplotlib seaborn scikit-learn

Les scripts doivent être lancés séquentiellement ou de manière autonome, en veillant à ce que les fichiers `.csv` soient présents dans le même dossier de travail.
===============================================================================
