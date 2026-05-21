import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# =====================================================================
# PARTIE 12 : SAUVEGARDE DU MODÈLE ENTRAÎNÉ
# =====================================================================

print("=== SAUVEGARDE DU MODELE EN COURS ===")

# 1. Chargement des données nettoyées (générées à la Partie 5)
df = pd.read_csv("car_insurance_clean.csv")
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

# 2. Construction et entraînement du modèle final retenu
# La Régression Logistique a obtenu le meilleur score à la Partie 11.
# Le modèle destiné à la "production" est entraîné sur l'ENSEMBLE des données
# disponibles. On sauvegarde tout le pipeline (StandardScaler + LogisticRegression) :
# le scaler est ainsi embarqué avec le modèle, et la mise en production peut prédire
# directement à partir de données brutes, sans étape de normalisation séparée.
modele_final = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
modele_final.fit(X, y)
print("[INFO] Modele final (pipeline StandardScaler + LogisticRegression) entraine.")

# 3. Sauvegarde du modèle sur disque avec Pickle (fonction dump)
fichier_modele = "modele_final.pkl"
with open(fichier_modele, "wb") as f:
    pickle.dump(modele_final, f)
print(f"[INFO] Modele sauvegarde dans le fichier '{fichier_modele}'.")

# 4. Vérification du chargement avec Pickle (fonction load)
with open(fichier_modele, "rb") as f:
    modele_charge = pickle.load(f)
print("[INFO] Modele recharge depuis le fichier avec succes.")

# 5. Test de prédiction sur de "nouvelles données"
# On simule l'arrivee de 5 nouveaux clients en reutilisant 5 lignes du jeu de donnees.
nouvelles_donnees = X[:5]
predictions = modele_charge.predict(nouvelles_donnees)

print("\n--- Test de prediction avec le modele recharge ---")
for i, pred in enumerate(predictions):
    etiquette = "Reclamation (1)" if pred == 1 else "Pas de reclamation (0)"
    print(f"Nouveau client n°{i+1} : classe predite = {etiquette}")
