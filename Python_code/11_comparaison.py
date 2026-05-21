import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# =====================================================================
# PARTIE 11 : COMPARAISON DE PLUSIEURS ALGORITHMES DE CLASSIFICATION
# =====================================================================

print("=== COMPARAISON DES CLASSIFIEURS EN COURS ===")

# 1. Chargement des données nettoyées (générées à la Partie 5)
df = pd.read_csv("car_insurance_clean.csv")
X = df.drop(columns=['outcome']).to_numpy()
y = df['outcome'].to_numpy()

# 2. Validation croisée commune aux 3 modèles (mêmes passes -> comparaison équitable)
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# 3. Définition des 3 classifieurs à comparer
# Chaque modèle est encapsulé dans un pipeline avec StandardScaler : la normalisation
# est ré-ajustée sur chaque bloc d'entraînement de la validation croisée (cf. Partie 10).
modeles = {
    "Regression Logistique":  make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
    "Perceptron":             make_pipeline(StandardScaler(), Perceptron(random_state=42)),
    "K Plus Proches Voisins": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

# 4. Pour chaque modèle : apprentissage + évaluation par validation croisée + score
print(f"\n--- Score moyen (accuracy) en validation croisee {K}-Fold ---\n")
resultats = {}
for nom, modele in modeles.items():
    scores = cross_val_score(modele, X, y, cv=kf, scoring='accuracy')
    resultats[nom] = scores.mean()
    print(f"{nom:<24} : accuracy moyenne = {scores.mean():.4f}  (ecart-type = {scores.std():.4f})")

# 5. Identification du meilleur modèle sur ce critère
meilleur = max(resultats, key=resultats.get)
print(f"\n>> Meilleur modele (accuracy) : {meilleur} -> {resultats[meilleur]:.4f}")
