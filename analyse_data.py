
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Charger le dataset
dossier = Path(__file__).resolve().parent
chemin = dossier / "data" / "UNSW_NB15_training-set.csv"

df = pd.read_csv(chemin)

# 2. Compter les connexions
nb_normal = (df["label"] == 0).sum()
nb_attaques = (df["label"] == 1).sum()

print("Connexions normales :", nb_normal)
print("Connexions malveillantes :", nb_attaques)

# 3. Calculer les pourcentages
total = len(df)

pourcentage_normal = nb_normal / total * 100
pourcentage_attaques = nb_attaques / total * 100

print(f"Trafic normal : {pourcentage_normal:.2f}%")
print(f"Trafic malveillant : {pourcentage_attaques:.2f}%")

print("\nColonnes du dataset :")
print(df.columns.tolist())

print("\nValeurs dans la colonne attack :")
if "attack" in df.columns:
    print(df["attack"].value_counts())
elif "attack_cat" in df.columns:
    print(df["attack_cat"].value_counts())
# 4. Créer un graphique
categories = ["Normal", "Attaque"]
valeurs = [nb_normal, nb_attaques]

plt.bar(categories, valeurs, color=["green", "red"])

plt.title("Répartition du trafic réseau")
plt.xlabel("Type de trafic")
plt.ylabel("Nombre de connexions")

# 5. Enregistrer le graphique
dossier_graphs = dossier / "graphs"
dossier_graphs.mkdir(exist_ok=True)

plt.savefig(dossier_graphs / "repartition.png")

# 6. Afficher le graphique
plt.show()
