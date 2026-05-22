import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# =====================================================================
# PARTIE 10 : AMÉLIORATION DE L'ÉVALUATION (VALIDATION CROISÉE)
# =====================================================================

print("=== VALIDATION CROISÉE EN COURS ===")

# 1. Chargement des données nettoyées (générées à la Partie 5)
df = pd.read_csv("car_insurance_clean.csv")

# 2. Séparation des variables d'entrée (X) et de la cible (y)
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

# 3. Construction du modèle sous forme de pipeline (normalisation + régression logistique)
# Le pipeline garantit que le StandardScaler est ré-ajusté UNIQUEMENT sur le bloc
# d'entraînement de chaque passe : on évite ainsi toute fuite d'information
# provenant du bloc de test (ce qui arriverait en normalisant tout X une seule fois).
modele = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

# 4. Définition de la validation croisée en K passes (K-Fold)
# shuffle=True mélange les données avant le découpage ; random_state=42 -> reproductibilité.
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# 5. Exécution de la validation croisée (métrique : accuracy)
scores = cross_val_score(modele, X, y, cv=kf, scoring='accuracy')

# 6. Affichage du détail des résultats
print(f"\n--- Résultats de la validation croisée ({K} passes) ---")
for i, score in enumerate(scores):
    print(f"Passe n°{i+1} : accuracy = {score:.4f}")

print(f"\nAccuracy moyenne : {scores.mean():.4f}")
print(f"Écart-type       : {scores.std():.4f}")

# 7. Comparaison avec l'évaluation simple de la Partie 9 (un seul découpage train/test)
accuracy_partie9 = 0.8400  # valeur obtenue à la Partie 9
print("\n--- Comparaison avec la Partie 9 ---")
print(f"Accuracy (1 seul decoupage 80/20, Partie 9)    : {accuracy_partie9:.4f}")
print(f"Accuracy moyenne (validation croisee {K}-Fold)  : {scores.mean():.4f}")
