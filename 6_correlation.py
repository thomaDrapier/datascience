import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# PARTIE 6 : RECHERCHE DE CORRÉLATIONS (VERSION HEATMAP)
# =====================================================================

# 1. Chargement des données nettoyées
df = pd.read_csv("car_insurance_clean.csv")
print("=== RECHERCHE DE CORRÉLATIONS EN COURS ===")

# 2. Calcul de la matrice de corrélation globale (méthode corr)
matrice_corr = df.corr()

# 3. Extraction textuelle des corrélations avec la cible pour le rapport
print("\n--- Corrélation de chaque variable avec la cible 'outcome' ---")
corr_avec_cible = matrice_corr['outcome'].sort_values(ascending=False)
print(corr_avec_cible)

# 4. Génération de la Heatmap
print("\n[INFO] Génération de la Heatmap...")
print("[INFO] Veuillez fermer la fenêtre du graphique pour terminer le script.")

# Configuration de la taille de la figure
plt.figure(figsize=(14, 11))

# Création de la carte de chaleur
sns.heatmap(
    matrice_corr, 
    annot=True,          # Affiche les valeurs numériques dans les cases
    cmap="coolwarm",     # Palette de couleurs (bleu = négatif, rouge = positif)
    fmt=".2f",           # Arrondi à 2 décimales
    linewidths=0.5,      # Espacement entre les cases
    vmin=-1, vmax=1      # Bornes de l'échelle des corrélations
)

plt.title("Matrice de corrélation entre les variables (Heatmap)", fontsize=16, pad=20)
plt.tight_layout()
plt.show()