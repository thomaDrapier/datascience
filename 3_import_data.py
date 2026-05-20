import pandas as pd

# Importation du jeu de données
try:
    df = pd.read_csv("car_insurance.csv")
    print("Fichier 'car_insurance.csv' importé avec succès !")
except FileNotFoundError:
    print("Erreur : Le fichier 'car_insurance.csv' est introuvable au même niveau que 'code.py'.")

# Affichage des dimensions (lignes, colonnes)
print("\n--- Dimensions du jeu de données ---")
print(f"Nombre de lignes : {df.shape[0]} | Nombre de colonnes : {df.shape[1]}")

# Affichage des 5 premières lignes
print("\n--- Aperçu des premières lignes ---")
print(df.head())

# Informations générales sur les colonnes et types détectés
print("\n--- Informations sur les variables ---")
print(df.info())