
import pandas as pd
from pathlib import Path

# Dossier principal du projet
dossier = Path(__file__).resolve().parent

# Charger le dataset officiel
chemin = dossier / "data" / "UNSW_NB15_testing-set.csv"

df = pd.read_csv(chemin)

# Colonnes nécessaires pour la démonstration
colonnes = [
    "proto",
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts"
]

# Sélectionner 250 exemples reproductibles
demo = df[colonnes].sample(
    n=250,
    random_state=42
).reset_index(drop=True)

# Créer le dossier demo
dossier_demo = dossier / "demo"
dossier_demo.mkdir(exist_ok=True)

# Enregistrer les données
demo.to_csv(
    dossier_demo / "sample_traffic.csv",
    index=False
)

print("Dataset de démonstration créé !")
print("Dimensions :", demo.shape)
