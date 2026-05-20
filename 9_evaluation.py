import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Importation des métriques d'évaluation requises
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# =====================================================================
# PARTIE 9 : ÉVALUATION DU MODÈLE
# =====================================================================

print("=== ÉVALUATION DU MODÈLE EN COURS ===")

# 1. Rechargement et découpage déterministe des données (comme en Partie 8)
df = pd.read_csv("car_insurance_clean.csv")
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 2. Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Application de la même échelle sur le test

# 3. Ré-entraînement du modèle
modele_logit = LogisticRegression(random_state=42)
modele_logit.fit(X_train_scaled, y_train)

# 4. Simulation du test : Prédictions sur les données d'entrée du jeu de test
y_pred = modele_logit.predict(X_test_scaled)

# 5. Boucle de comparaison (Affichage des 15 premiers échantillons)
print("\n--- Comparaison : Classe Prédite vs Classe Réelle (15 premiers clients) ---")
print(f"{'Client':<10} | {'Classe Prédite':<15} | {'Classe Réelle':<15}")
print("-" * 48)
for i in range(15):
    print(f"N° {i+1:<7} | {int(y_pred[i]):<15} | {int(y_test[i]):<15}")

# 6. Calcul des métriques de performance quantitatives
print("\n--- Métriques d'évaluation quantitatives ---")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score  : {acc:.4f} (Soit {acc*100:.2f}% de bonnes réponses globales)")
print(f"Precision Score : {prec:.4f}")
print(f"Recall Score    : {rec:.4f}")
print(f"F1-Score        : {f1:.4f}")

print("\nMatrice de Confusion :")
print(cm)