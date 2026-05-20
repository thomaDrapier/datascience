import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =====================================================================
# PARTIE 8 : ENTRAÎNEMENT D'UN MODÈLE & REPRÉSENTATION GRAPHIQUE
# =====================================================================

print("=== APPRENTISSAGE DU MODÈLE EN COURS ===")

# 1. Rechargement des données propres
df = pd.read_csv("car_insurance_clean.csv")

# 2. Séparation des variables d'entrée (X) et de la cible (y)
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

# 3. Recréation à l'identique du découpage (Train / Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Normalisation des données d'entraînement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Entraînement du modèle de Régression Logistique
modele_logit = LogisticRegression(random_state=42)
modele_logit.fit(X_train_scaled, y_train)

print("[SUCCÈS] Le modèle de Régression Logistique a été entraîné avec succès !")
print(f"-> Constante calculée (Intercept / Biais b0) : {modele_logit.intercept_[0]:.4f}")
print(f"-> Nombre de coefficients calculés (Poids b1 à bn) : {len(modele_logit.coef_[0])}\n")

# =====================================================================
# 6. GÉNÉRATION DU GRAPHIQUE DES COEFFICIENTS
# =====================================================================
print("[INFO] Génération du graphique des coefficients...")

# Récupération des noms des variables d'entrée
noms_variables = df.drop(columns=['outcome']).columns
coefficients = modele_logit.coef_[0]

# Création d'un DataFrame temporaire pour trier les coefficients
df_coefs = pd.DataFrame({
    'Variable': noms_variables,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=True)

# Configuration de la figure
plt.figure(figsize=(12, 8))
couleurs = ['tomato' if c > 0 else 'skyblue' for c in df_coefs['Coefficient']]

plt.barh(df_coefs['Variable'], df_coefs['Coefficient'], color=couleurs, edgecolor='black')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.title("Importance et impact des variables (Coefficients de la Régression Logistique)", fontsize=14, pad=15)
plt.xlabel("Valeur du Coefficient (Poids)", fontsize=12)
plt.ylabel("Variables d'entrée", fontsize=12)
plt.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
print("[INFO] Veuillez fermer la fenêtre du graphique pour terminer l'exécution.")
plt.show()