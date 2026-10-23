
import pandas as pd
from pathlib import Path
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1. Charger le modèle
dossier = Path(__file__).resolve().parent

artifact = load(
    dossier / "models" / "ai_nids_random_forest.joblib"
)

model = artifact["model"]
features = artifact["features"]

# 2. Reconstituer notre ensemble de validation
df = pd.read_csv(
    dossier / "data" / "UNSW_NB15_training-set.csv"
)

X = df[features]
y = df["label"]

_, X_val, _, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Obtenir les scores d'attaque
scores = model.predict_proba(X_val)[:, 1]

# 4. Tester différents seuils
seuils = [0.5, 0.6, 0.7, 0.8, 0.9]

print("\n=== COMPARAISON DES SEUILS ===")

for seuil in seuils:

    predictions = (scores >= seuil).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print(f"\nSeuil : {seuil:.1f}")
    print(f"Recall : {recall:.2%}")
    print(f"Taux de faux positifs : {fpr:.2%}")
    print(f"Faux positifs : {fp}")
    print(f"Faux négatifs : {fn}")
