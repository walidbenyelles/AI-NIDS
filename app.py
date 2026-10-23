
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# Configuration
st.set_page_config(
    page_title="AI-NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

DOSSIER = Path(__file__).resolve().parent
SEUIL = 0.6


# Charger notre modèle une seule fois
@st.cache_resource
def charger_modele():
    artifact = load(
        DOSSIER / "models" / "ai_nids_random_forest.joblib"
    )
    return artifact["model"], artifact["features"]


# Effacer l'ancienne analyse si le fichier change
def reinitialiser_resultats():
    st.session_state.pop("resultats", None)


model, features = charger_modele()

# Interface
st.title("🛡️ AI-NIDS Security Dashboard")

st.caption(
    "Prototype expérimental de détection "
    "d'intrusions réseau par Machine Learning."
)

st.success("Modèle Random Forest chargé !")
st.info(f"Seuil de détection : {SEUIL}")

# Barre latérale
with st.sidebar:

    st.header("🛡️ AI-NIDS")

    st.divider()

    st.subheader("Informations du modèle")

    st.write("**Algorithme :** Random Forest")
    st.write("**Dataset :** UNSW-NB15")
    st.write(f"**Seuil :** {SEUIL:.0%}")
    st.write(f"**Caractéristiques :** {len(features)}")

    st.divider()

    st.subheader("Comment utiliser l'application ?")

    st.write("1. Importer un fichier CSV")
    st.write("2. Lancer l'analyse")
    st.write("3. Consulter les alertes")
    st.write("4. Exporter les résultats")

    st.divider()

    st.warning(
        "Prototype expérimental. Les alertes "
        "sont des prédictions et ne constituent "
        "pas une preuve d'intrusion."
    )

# Importer un fichier
fichier = st.file_uploader(
    "Importer un dataset réseau au format CSV",
    type=["csv"],
    key="fichier_reseau",
    on_change=reinitialiser_resultats
)

if fichier is not None:

    # Charger les données
    try:
        df = pd.read_csv(
            io.BytesIO(fichier.getvalue())
        )
    except Exception as e:
        st.error(f"Impossible de lire le CSV : {e}")
        st.stop()

    st.subheader("Aperçu des données")
    st.dataframe(df.head(10))

    # Vérifier les colonnes attendues
    manquantes = [
        col for col in features
        if col not in df.columns
    ]

    if manquantes:
        st.error(f"Colonnes manquantes : {manquantes}")
        st.stop()

    if df.empty:
        st.warning("Le fichier CSV est vide.")
        st.stop()

    # Préparer les caractéristiques
    X = df[features].apply(
        pd.to_numeric, errors="coerce"
    )

    if not np.isfinite(X.to_numpy()).all():
        st.error("Valeurs manquantes ou invalides.")
        st.stop()

    # Analyser le trafic
    if st.button("Analyser le trafic", type="primary"):

        index_attaque = list(model.classes_).index(1)

        scores = model.predict_proba(X)[:, index_attaque]

        predictions = (scores >= SEUIL).astype(int)

        resultats = df.copy()
        resultats["prediction"] = predictions
        resultats["score_attaque"] = scores
        resultats["statut"] = np.where(
            predictions == 1,
            "Suspect",
            "Normal"
        )

        # Sauvegarder les résultats en session
        st.session_state["resultats"] = resultats

    # Affichage indépendant du bouton
    if "resultats" in st.session_state:

        resultats = st.session_state["resultats"]

        nb_total = len(resultats)
        nb_normal = int(
            (resultats["prediction"] == 0).sum()
        )
        nb_attaques = int(
            (resultats["prediction"] == 1).sum()
        )

        st.divider()
        st.header("Security Dashboard")

        # Indicateurs
        c1, c2, c3 = st.columns(3)

        c1.metric("Flux analysés", nb_total)
        c2.metric("Prédits normaux", nb_normal)
        c3.metric("Alertes générées", nb_attaques)

        # Onglets
        tab1, tab2, tab3 = st.tabs([
            "Vue d'ensemble",
            "Alertes",
            "Toutes les données"
        ])

        with tab1:

            st.subheader("Répartition du trafic")

            repartition = pd.DataFrame({
                "Type": ["Normal", "Suspect"],
                "Nombre": [nb_normal, nb_attaques]
            })

            st.bar_chart(
                repartition, x="Type", y="Nombre"
            )

            st.subheader("Distribution des scores")

            valeurs = resultats["score_attaque"]

            frequences, limites = np.histogram(
                valeurs,
                bins=np.linspace(0, 1, 11)
            )

            etiquettes = [
                f"{limites[i]:.1f} - {limites[i+1]:.1f}"
                for i in range(len(frequences))
            ]

            distribution = pd.DataFrame({
                "Intervalle": etiquettes,
                "Nombre": frequences
            })

            st.bar_chart(
                distribution,
                x="Intervalle",
                y="Nombre"
            )
            
            # Graphique par protocole
            if "proto" in resultats.columns:

                st.subheader(
                    "Analyse du trafic par protocole"
                )

                # Compter les prédictions par protocole
                protocoles = pd.crosstab(
                    resultats["proto"].fillna("Inconnu"),
                    resultats["statut"]
                )

                # Garantir la présence des deux colonnes
                protocoles = protocoles.reindex(
                    columns=["Normal", "Suspect"],
                    fill_value=0
                )

                # Sélectionner les 10 protocoles
                # avec le plus de flux
                top10 = (
                    protocoles.sum(axis=1)
                    .nlargest(10)
                    .index
                )

                protocoles = (
                    protocoles.loc[top10]
                    .rename_axis("Protocole")
                    .reset_index()
                )

                # Afficher le graphique
                st.bar_chart(
                    protocoles,
                    x="Protocole",
                    y=["Normal", "Suspect"],
                    stack=True
                )


        with tab2:

            st.subheader("Alertes de sécurité")

            alertes = resultats[
                resultats["prediction"] == 1
            ].sort_values(
                "score_attaque",
                ascending=False
            )

            st.write(f"{len(alertes)} alertes générées")

            
            # Colonnes importantes à afficher
            colonnes_affichage = [
                col for col in [
                    "proto",
                    "dur",
                    "sbytes",
                    "dbytes",
                    "statut",
                    "score_attaque"
                ]
                if col in alertes.columns
            ]

            st.dataframe(
                alertes[colonnes_affichage].head(100),
                width="stretch",
                column_config={
                    "score_attaque":
                        st.column_config.ProgressColumn(
                            "Score d'attaque",
                            min_value=0,
                            max_value=1,
                            format="%.2f"
                        )
                }
            )


            st.download_button(
                "Télécharger les alertes",
                data=alertes.to_csv(index=False),
                file_name="alertes.csv",
                mime="text/csv"
            )

        with tab3:

            st.subheader("Toutes les connexions")

            filtre = st.selectbox(
                "Statut",
                ["Tous", "Normal", "Suspect"]
            )

            if filtre == "Tous":
                donnees = resultats
            else:
                donnees = resultats[
                    resultats["statut"] == filtre
                ]

            st.dataframe(
                donnees.head(1000),
                width="stretch"
            )

            st.download_button(
                "Télécharger les résultats",
                data=donnees.to_csv(index=False),
                file_name="resultats.csv",
                mime="text/csv"
            )
