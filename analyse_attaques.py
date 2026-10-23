
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Charger les données
dossier = Path(__file__).resolve().parent
fichier = dossier / "data" / "UNSW_NB15_training-set.csv"

df = pd.read_csv(fichier)

# 2. Vérifier les étiquettes
print("=== NORMAL / ATTAQUE ===")
print(df["label"].value_counts().sort_index())

# 3. Trouver la colonne des catégories
if "attack_cat" in df.columns:
    colonne = "attack_cat"
elif "attack" in df.columns:
    colonne = "attack"
else:
    colonne = None

# 4. Afficher les catégories
if colonne is not None:
    print("\n=== CATEGORIES ET ETIQUETTES ===")
    print(pd.crosstab(
        df[colonne].fillna("Non renseigné"),
        df["label"]
    ))

# 5. Examiner la durée des connexions
print("\n=== DUREE MEDIANE DES FLUX ===")
print(df.groupby("label")["dur"].median())

# 6. Préparer le dossier de graphiques
graphs = dossier / "graphs"
graphs.mkdir(exist_ok=True)

# 7. Repérer les catégories connues
categories_connues = {
    "analysis", "backdoor", "backdoors",
    "dos", "exploits", "fuzzers",
    "generic", "reconnaissance",
    "shellcode", "worms"
}

attaques = df[df["label"] == 1]

categories_valides = False

if colonne is not None:
    noms = attaques[colonne].astype("string").str.strip()
    categories_valides = (
        noms.str.lower().isin(categories_connues).any()
    )

# 8. Créer le graphique
if categories_valides:
    nombres = noms.value_counts().sort_values()

    nombres.plot(kind="barh", color="steelblue")
    plt.title("Répartition des types d'attaques")
    plt.xlabel("Nombre de flux")
    plt.ylabel("Catégorie d'attaque")
    nom_image = "categories_attaques.png"

else:
    print("\nCatégories détaillées non identifiées.")
    print("Analyse des protocoles à la place.")

    top = df["proto"].value_counts().head(10).index

    tableau = pd.crosstab(df["proto"], df["label"])
    tableau = tableau.reindex(top, fill_value=0)

    tableau = tableau.rename(
        columns={0: "Normal", 1: "Attaque"}
    )

    tableau.plot(kind="bar", stacked=True)
    plt.title("Trafic par protocole")
    plt.xlabel("Protocole")
    plt.ylabel("Nombre de flux")
    plt.xticks(rotation=45)
    nom_image = "protocoles.png"

plt.tight_layout()
plt.savefig(graphs / nom_image, dpi=150)
plt.show()
