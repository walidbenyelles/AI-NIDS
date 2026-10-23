import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
from pathlib import Path
#chemin de notre dataset
chemin = Path(__file__).resolve().parent / "data" / "UNSW_NB15_training-set.csv"
#chargement du dataset
df = pd.read_csv(chemin)
#nombre de lignes et de colonnes
print("Dimensions du dataset :", df.shape)
#Afficher les cinq premières lignes
colonnes = [
    "dur", "proto", "sbytes",
    "dbytes", "label", "attack_cat"
]
print("\nPremières connexions :")
print(df[colonnes].head())

# Afficher toutes les colonnes
print("\n=== COLONNES DU DATASET ===")
print(df.columns.tolist())

# Nombre de connexions normales et malveillantes
print("\n=== REPARTITION NORMAL / ATTAQUE ===")
print(df["label"].value_counts().sort_index())

# Pourcentage de chaque classe
print("\n=== POURCENTAGES ===")
print(df["label"].value_counts(normalize=True) * 100)

# Types d'attaques, si disponibles
colonne_attaque = (
    "attack_cat" if "attack_cat" in df.columns
    else "attack" if "attack" in df.columns
    else None
)

if colonne_attaque:
    print("\n=== CATEGORIES D'ATTAQUES ===")
    print(df[colonne_attaque].value_counts(dropna=False))
