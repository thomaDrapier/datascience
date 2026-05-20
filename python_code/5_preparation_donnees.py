import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =====================================================================
# PARTIE 5 : PRÉPARATION DES DONNÉES
# =====================================================================

# 1. Chargement des données
df = pd.read_csv("car_insurance.csv")
print("=== PRÉPARATION DES DONNÉES EN COURS ===")

# 2. Suppression de la colonne 'id' (non utile pour la classification)
df = df.drop(columns=['id'])
print("[INFO] Colonne 'id' supprimée.")

# 3. Traitement des données aberrantes (identifiées à la Partie 4)
# Remplacement des valeurs aberrantes par la médiane de la variable
median_speeding = df['speeding_violations'].median()
median_children = df['children'].median()

# On considère aberrant 'speeding_violations' > 100 et 'children' > 10
df.loc[df['speeding_violations'] > 100, 'speeding_violations'] = median_speeding
df.loc[df['children'] > 10, 'children'] = median_children
print("[INFO] Valeurs aberrantes corrigées (speeding_violations et children).")

# 4. Traitement des données manquantes (Imputation par la médiane)
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
df['annual_mileage'] = df['annual_mileage'].fillna(df['annual_mileage'].median())
print("[INFO] Valeurs manquantes imputées par la médiane.")

# 5. Transformation des variables qualitatives en variables numériques
le = LabelEncoder()
cat_cols = ['driving_experience', 'education', 'income', 'vehicle_year', 'vehicle_type']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print("[INFO] Variables qualitatives encodées avec LabelEncoder.")

# Enregistrement du DataFrame nettoyé pour les étapes suivantes (Partie 6)
df.to_csv("car_insurance_clean.csv", index=False)
print("[INFO] Fichier intermédiaire 'car_insurance_clean.csv' sauvegardé.")

# 6. Séparation des features (X) et de la cible (y)
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

# 7. Normalisation des données d'entrée
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] Données d'entrée normalisées avec StandardScaler.")

# Vérification finale des dimensions
print(f"\nDimensions finales de X (features normalisées) : {X_scaled.shape}")
print(f"Dimensions finales de y (cible) : {y.shape}")
print("\nExemple de la première ligne de données préparées :")
print(X_scaled[0])