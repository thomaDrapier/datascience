import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# PARTIE 4 : EXAMEN DES DONNÉES
# =====================================================================

# Chargement autonome des données
df = pd.read_csv("car_insurance.csv")

print("=== EXAMEN DES DONNÉES EN COURS ===")

# 1. Analyse de la taille du jeu de données 
print(f"\nTaille du jeu de données : {df.shape[0]} lignes et {df.shape[1]} colonnes")

# 2. Détection des données manquantes 
print("\n--- Décompte des données manquantes (méthode isna) ---")
valeurs_manquantes = df.isna().sum()
print(valeurs_manquantes[valeurs_manquantes > 0])

# 3. Statistiques descriptives de base pour traquer les anomalies [cite: 38]
print("\n--- Statistiques descriptives des variables numériques ---")
print(df.describe())

# 4. Visualisation des distributions par histogrammes (Exclusion de l'ID) [cite: 44, 46]
print("\n[INFO] Génération des histogrammes (hors colonne ID)...")
print("[INFO] Veuillez fermer la fenêtre du graphique pour terminer l'exécution du script.")

# On retire la colonne 'id' temporairement pour l'affichage des graphiques
df_visualisation = df.drop(columns=['id'])

# Création des histogrammes pour les variables numériques utiles 
df_visualisation.hist(figsize=(14, 10), bins=30, edgecolor='black', color='skyblue')
plt.tight_layout()
plt.show()