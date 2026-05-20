import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================================
# PARTIE 7 : EXTRACTION DES JEUX D'APPRENTISSAGE ET DE TEST
# =====================================================================

# 1. Chargement des données nettoyées
df = pd.read_csv("car_insurance_clean.csv")
print("=== EXTRACTION DES JEUX EN COURS ===")

# 2. Séparation des features (X) et de la cible (y) sous forme de tableaux Numpy
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

# 3. Division en jeu d'apprentissage et jeu de test
# Nous choisissons une répartition classique : 80% train / 20% test
# Le paramètre random_state garantit que le découpage aléatoire reste identique à chaque exécution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Affichage et calcul des proportions pour répondre aux questions du projet
total_echantillons = len(df)
proportion_train = (len(X_train) / total_echantillons) * 100
proportion_test = (len(X_test) / total_echantillons) * 100

print(f"\nProportion du jeu d'apprentissage : {proportion_train:.0f}%")
print(f"Proportion du jeu de test          : {proportion_test:.0f}%")

print("\n--- Dimensions des tableaux Numpy générés ---")
print(f"X_train (Données d'apprentissage) : {X_train.shape}")
print(f"y_train (Labels d'apprentissage)  : {y_train.shape}")
print(f"X_test  (Données de test)          : {X_test.shape}")
print(f"y_test  (Labels de test)           : {y_test.shape}")