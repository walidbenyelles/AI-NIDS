
import pandas as pd
from pathlib import Path
from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Définir les chemins
dossier = Path(__file__).resolve().parent

model_path = dossier / "models" / "ai_nids_random_forest.joblib"

test_path = dossier / "data" / "UNSW_NB15_testing-set.csv"

# 2. Charger le modèle déjà entraîné
artifact = load(model_path)

model = artifact["model"]
features = artifact["features"]

print("Modèle chargé avec succès !")

# 3. Charger le dataset de test
df_test = pd.read_csv(test_path)

print("\nDimensions :", df_test.shape)

# 4. Préparer les données
X_test = df_test[features]
y_test = df_test["label"]

# 5. Faire des prédictions
# Aucun nouvel entraînement ici !
predictions = model.predict(X_test)

# 6. Calculer les performances
print("\n=== RESULTATS DU TEST OFFICIEL ===")

print("Accuracy :",
      accuracy_score(y_test, predictions))

print("Precision :",
      precision_score(y_test, predictions))

print("Recall :",
      recall_score(y_test, predictions))

print("F1-score :",
      f1_score(y_test, predictions))

# 7. Rapport complet
print("\n=== RAPPORT DE CLASSIFICATION ===")

print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Attaque"]
))

# 8. Matrice de confusion
print("\n=== MATRICE DE CONFUSION ===")

print(confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
))
