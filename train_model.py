
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from joblib import dump
# 1. Charger les données
dossier = Path(__file__).resolve().parent

df = pd.read_csv(
    dossier / "data" / "UNSW_NB15_training-set.csv"
)

# 2. Choisir les caractéristiques
features = [
    "dur",      # Durée du flux
    "sbytes",   # Octets envoyés
    "dbytes",   # Octets reçus
    "spkts",    # Paquets envoyés
    "dpkts"     # Paquets reçus
]

X = df[features]
y = df["label"]

# 3. Séparer les données
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Créer le modèle
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=30,
    random_state=42
)

# 5. Entraîner le modèle
model.fit(X_train, y_train)

# 6. Effectuer des prédictions
predictions = model.predict(X_val)

# 7. Évaluer les performances
print("\n=== ACCURACY ===")
print(accuracy_score(y_val, predictions))

print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(
    y_val,
    predictions,
    target_names=["Normal", "Attaque"]
))

print("\n=== MATRICE DE CONFUSION ===")
print(confusion_matrix(y_val, predictions))

# 8. Importance des caractéristiques
print("\n=== IMPORTANCE DES VARIABLES ===")
for nom, importance in zip(
    features, model.feature_importances_
):
    print(f"{nom}: {importance:.4f}")
# 1. Modèle de référence
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

print("\n=== COMPARAISON DES MODELES ===")

print("Baseline accuracy :",
      accuracy_score(y_val, baseline_predictions))

print("Decision Tree accuracy :",
      accuracy_score(y_val, predictions))

# 2. Recherche de surapprentissage
print("\n=== OVERFITTING ===")

print("Accuracy entraînement :",
      model.score(X_train, y_train))

print("Accuracy validation :",
      model.score(X_val, y_val))

# 3. Créer le dossier pour les graphiques
graphs = dossier / "graphs"
graphs.mkdir(exist_ok=True)

# 4. Matrice de confusion
fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_val,
    predictions,
    display_labels=["Normal", "Attaque"],
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title("Matrice de confusion - Decision Tree")

fig.tight_layout()
fig.savefig(
    graphs / "confusion_matrix.png",
    dpi=150
)

# 5. Importance des variables
importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

fig2, ax2 = plt.subplots(figsize=(7, 4))

importances.plot.barh(ax=ax2)

ax2.set_title("Importance des caractéristiques")
ax2.set_xlabel("Importance relative")

fig2.tight_layout()
fig2.savefig(
    graphs / "feature_importance.png",
    dpi=150
)

plt.show()
# 1. Créer notre Random Forest
forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

# 2. Entraîner notre modèle
print("\nEntraînement du Random Forest...")

forest.fit(X_train, y_train)

# 3. Effectuer des prédictions
pred_forest = forest.predict(X_val)

# 4. Comparer les modèles
print("\n=== COMPARAISON DES MODELES ===")

modeles = {
    "Decision Tree": predictions,
    "Random Forest": pred_forest
}

for nom, pred in modeles.items():

    accuracy = accuracy_score(y_val, pred)
    precision = precision_score(y_val, pred)
    recall = recall_score(y_val, pred)

    tn, fp, fn, tp = confusion_matrix(
        y_val, pred
    ).ravel()

    print(f"\n--- {nom} ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Faux positifs : {fp}")
    print(f"Faux négatifs : {fn}")

# 5. Vérifier le surapprentissage
print("\n=== RANDOM FOREST OVERFITTING ===")

print("Accuracy entraînement :",
      forest.score(X_train, y_train))

print("Accuracy validation :",
      forest.score(X_val, y_val))

# 1. Créer le dossier models
models_dir = dossier / "models"
models_dir.mkdir(exist_ok=True)

# 2. Définir le chemin du modèle
model_path = models_dir / "ai_nids_random_forest.joblib"

# 3. Sauvegarder le modèle et ses caractéristiques
artifact = {
    "model": forest,
    "features": features
}

dump(artifact, model_path)

print("\n=== MODELE SAUVEGARDE ===")
print("Fichier :", model_path)
print("Caractéristiques :", features)
