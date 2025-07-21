import os
import math
import tempfile
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import openai
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


st.set_page_config(
    page_title="🚲 Vélib' Dashboard — Temps réel",
    layout="wide",
    initial_sidebar_state="expanded",
)

openai.api_key = os.getenv("OPENAI_API_KEY")

zoom_level = 0.70
zoom_s = 0.5
st.markdown(
    f"""
    <style>
    .main {{zoom:{zoom_level};}}
    .custom-sidebar {{zoom:{zoom_s};}}
    section.main > div:first-child {{padding-top:0.3rem;}}
    div[data-testid="stMetric"] div {{justify-content:flex-start;}}
    .dynamic-shadow{{transition:box-shadow 0.3s,transform 0.3s;
                    box-shadow:0 4px 8px rgba(0,0,0,0.1);border-radius:10px;}}
    .dynamic-shadow:hover{{transform:translateY(-5px);
                         box-shadow:0 8px 16px rgba(0,0,255,0.3);}}
    .report-card{{background:linear-gradient(135deg,#6e8efb,#a777e3);
                  padding:20px;border-radius:10px;color:white;
                  box-shadow:0 4px 8px rgba(0,0,0,0.1);}}
    .report-card:hover{{box-shadow:0 8px 16px rgba(0,0,255,0.3);}}
    </style>
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """,
    unsafe_allow_html=True,
)


def haversine(lat1, lon1, lat2, lon2) -> float:
        """Distance en km entre 2 points lat/lon."""
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

def find_rebalance_pairs(df: pd.DataFrame, max_pairs: int = 5) -> list[tuple]:
        """
        Pour chaque station vide, trouve la station la plus proche
        avec ≥ 5 vélos disponibles. Renvoie max_pairs suggestions.
        """
        need = df[df["bikes"] == 0].copy()
        supply = df[df["bikes"] >= 5].copy()
        pairs = []
        for _, row in need.sort_values("capacity", ascending=False).head(max_pairs).iterrows():
            dists = haversine(row["lat"], row["lon"], supply["lat"], supply["lon"])
            idx_min = dists.idxmin()
            donor = supply.loc[idx_min]
            pairs.append(
                {
                    "dest_name": row["name"],
                    "dest_arr": row["arr"],
                    "donor_name": donor["name"],
                    "donor_bikes": int(donor["bikes"]),
                    "distance_km": round(float(dists[idx_min]), 2),
                }
            )
            supply = supply.drop(idx_min)
        return pairs

def generate_report(summary_dict: dict, pairs: list[dict]) -> str:
        """
        Appel OpenAI : produit une synthèse opérationnelle
        + conseils (≤ 120 mots).
        """
        now = datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")
        pairs_txt = "\n".join(
            [
                f"- {p['donor_name']} ({p['donor_bikes']} vélos, {p['distance_km']} km) → {p['dest_name']}"
                for p in pairs
            ]
        ) or "Aucune suggestion de ré‑équilibrage (réseau stable)."
        prompt = f"""
        Contexte : réseau Vélib' temps réel le {now}.
        Indicateurs :
        • stations totales : {summary_dict['n_total']}
        • stations vides : {summary_dict['k_empty']}
        • stations pleines : {summary_dict['k_full']}
        • stations HS : {summary_dict['k_hs']}
        • taux opérationnel : {summary_dict['p_op']:.1f} %

        Voici des propositions de ré‑équilibrage :
        {pairs_txt}

        Rédige un bref rapport (≤ 120 mots) en français :
        ➊ Résume la situation générale.
        ➋ Identifie la station la plus urgente.
        ➌ Donne 2 conseils opérationnels immédiats basés sur les paires ci‑dessus.
        ➍ Termine par une phrase proactive.
        """
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Tu es un expert logistique Vélib'."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=180,
                temperature=0.7,
                top_p=0.9,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            return f"Erreur OpenAI : {e}"

    # ─────────────────────────────────────
    # CHARGEMENT DES DONNÉES
    # ─────────────────────────────────────
DATA_URL = (
        "https://opendata.paris.fr/api/records/1.0/search/"
        "?dataset=velib-disponibilite-en-temps-reel&rows=5000&sort=stationcode"
    )

@st.cache_data(ttl=120, show_spinner="📡 Chargement des données…")
def load_data() -> pd.DataFrame:
        r = requests.get(DATA_URL, timeout=20)
        r.raise_for_status()
        recs = []
        for rec in r.json()["records"]:
            f = rec["fields"]
            lat, lon = f.get("coordonnees_geo", [None, None])
            recs.append(
                {
                    "code": f.get("stationcode"),
                    "name": f.get("name"),
                    "arr": f.get("nom_arrondissement_communes"),
                    "capacity": f.get("capacity", 0),
                    "bikes": f.get("numbikesavailable", 0),
                    "docks": f.get("numdocksavailable", 0),
                    "lat": lat,
                    "lon": lon,
                    "installed": str(f.get("is_installed")).lower() in {"1", "true", "yes", "oui"},
                    "renting": str(f.get("is_renting")).lower() in {"1", "true", "yes", "oui"},
                    "returning": str(f.get("is_returning")).lower() in {"1", "true", "yes", "oui"},
                }
            )
        df = pd.DataFrame(recs)
        df["fill_rate"] = df["bikes"] / df["capacity"].replace({0: np.nan})
        return df.dropna(subset=["lat", "lon"])

df = load_data()

with st.sidebar:
    st.markdown("""
        <style>
        .css-1d391kg, .css-16idsys, .nav-link {
            font-size: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.session_state.selected = option_menu(
        'DASHBOARD', 
        ["DONNÉES", "EXPLORATION", "PILOTAGE", "DOCUMENTATION"], 
        icons = ['database','search', 'speedometer', 'file-earmark-text'], 
        menu_icon='cast', 
        default_index=0
    )


if st.session_state.selected == "DONNÉES":

    # ─────────────────────────────────────
    # SECTION 1 – SOURCE DES DONNÉES
    # ─────────────────────────────────────

    st.markdown("## 🗂️ 1. Source des données")

    st.markdown("""
    Les données affichées dans ce dashboard proviennent de la **plateforme OpenData de la Ville de Paris**, et sont mises à jour en temps réel.

    Nous interrogeons directement l’**API REST JSON** de la ressource suivante :

    🔗 [Vélib’ - Disponibilité en temps réel](https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel)

    ---
    """)

    # Architecture d’acquisition simplifiée
    st.markdown("### 📦 Architecture simplifiée")
    st.code("""
                +---------------------------+
                |  open-data.paris.fr API   |
                +------------+--------------+
                                |
                        Requête GET (JSON)
                                |
                +-------------v--------------+
                |   Fonction `load_data()`   |
                +-------------+--------------+
                                |
                    Nettoyage & filtrage
                                |
                +-------------v--------------+
                |  DataFrame exploitable     |
                +---------------------------+
    """, language="text")

    st.markdown("---")
    st.markdown("### ⚙️ Fonction d'acquisition")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
    Voici les étapes de récupération des données :

    1. Construction de l’URL d’accès à l’API
    2. Requête HTTP `GET` avec `requests`
    3. Parsing des enregistrements JSON
    4. Création d’un `DataFrame` structuré
    5. Nettoyage et calcul du champ `fill_rate`

    Les résultats sont **mis en cache pendant 2 minutes** pour éviter des appels répétés à l’API.
        """)

    with col2:
        st.code("""
    @st.cache_data(ttl=120, show_spinner="📡 Chargement des données…")
    def load_data() -> pd.DataFrame:
        r = requests.get(DATA_URL, timeout=20)
        r.raise_for_status()
        recs = []
        for rec in r.json()["records"]:
            f = rec["fields"]
            lat, lon = f.get("coordonnees_geo", [None, None])
            recs.append({
                "code": f.get("stationcode"),
                "name": f.get("name"),
                "arr": f.get("nom_arrondissement_communes"),
                "capacity": f.get("capacity", 0),
                "bikes": f.get("numbikesavailable", 0),
                "docks": f.get("numdocksavailable", 0),
                "lat": lat,
                "lon": lon,
                "installed": str(f.get("is_installed")).lower() in {"1", "true", "yes", "oui"},
                "renting": str(f.get("is_renting")).lower() in {"1", "true", "yes", "oui"},
                "returning": str(f.get("is_returning")).lower() in {"1", "true", "yes", "oui"},
            })
        df = pd.DataFrame(recs)
        df["fill_rate"] = df["bikes"] / df["capacity"].replace({0: np.nan})
        return df.dropna(subset=["lat", "lon"])
        """, language="python")

    st.markdown("---")
    st.markdown("### 🧼 Nettoyage et transformation")

    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown("""
    Les données sont ensuite nettoyées :

    - Transformation des flags logiques (ex. `is_renting`)
    - Suppression des valeurs manquantes (`NaN`)
    - Calcul du **taux de remplissage** : `fill_rate`
    - Filtrage des stations sans coordonnées géographiques

    Cela garantit un DataFrame propre et exploitable pour les visualisations.
        """)

    with col4:
        st.code("""
    df["fill_rate"] = df["bikes"] / df["capacity"].replace({0: np.nan})
    df = df.dropna(subset=["lat", "lon"])
        """, language="python")

    st.divider()

    


    # ─────────────────────────────────────
    # SECTION 2 – VARIABLES DISPONIBLES
    # ─────────────────────────────────────

    st.markdown("## 📑 2. Variables disponibles")

    st.markdown("""
    Les données récupérées depuis l’API contiennent plusieurs **champs bruts** que nous filtrons, nettoyons et enrichissons.  
    Voici les principales **colonnes du DataFrame** exploité dans ce dashboard :
    """)

    # ► Tableau descriptif des variables
    var_data = [
        ["`code`", "str", "Code unique de la station"],
        ["`name`", "str", "Nom de la station Vélib’"],
        ["`arr`", "str", "Arrondissement administratif"],
        ["`capacity`", "int", "Capacité totale de la station (nombre de bornes)"],
        ["`bikes`", "int", "Nombre de vélos disponibles"],
        ["`docks`", "int", "Nombre d’emplacements libres pour retour"],
        ["`lat`, `lon`", "float", "Coordonnées géographiques (latitude, longitude)"],
        ["`installed`", "bool", "Station physiquement installée"],
        ["`renting`", "bool", "Location de vélo possible"],
        ["`returning`", "bool", "Retour de vélo possible"],
        ["`fill_rate`", "float", "Taux de remplissage : `bikes / capacity`"],
    ]

    df_vars = pd.DataFrame(var_data, columns=["Variable", "Type", "Description"])
    st.table(df_vars)

    st.markdown("---")

    # ► Aperçu du DataFrame brut
    st.markdown("### 🧪 Exemple d’enregistrements (5 lignes aléatoires)")

    df_sample = df.sample(5, random_state=42) if len(df) >= 5 else df.copy()
    st.dataframe(df_sample, use_container_width=True)

    st.markdown("---")

    # ► Graphe de dépendance des variables (réseau interactif)
    st.markdown("### 🧭 Relations entre les variables (vue réseau)")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("""
    Ce graphe représente les **relations logiques et calculées** entre les différentes variables utilisées dans le DataFrame.

    - `fill_rate` dépend de `bikes` et `capacity`
    - `capacity` est la somme de `bikes` et `docks`
    - `renting` et `returning` conditionnent respectivement `bikes` et `docks`
    - `installed` est nécessaire pour que la station soit active
    - `lat/lon` + `arr` sont utilisés pour la géolocalisation
        """)

    with col_right:
        # Définir le graphe NetworkX
        G = nx.DiGraph()

        nodes = [
            "bikes", "docks", "capacity", "fill_rate",
            "installed", "renting", "returning",
            "lat/lon", "arr", "name"
        ]
        edges = [
            ("bikes", "fill_rate"),
            ("capacity", "fill_rate"),
            ("bikes", "capacity"),
            ("docks", "capacity"),
            ("installed", "capacity"),
            ("renting", "bikes"),
            ("returning", "docks"),
            ("name", "arr"),
            ("lat/lon", "name"),
        ]

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # PyVis interactive graph
        net = Network(height="500px", width="100%", bgcolor="white", font_color="black", directed=True)

        net.barnes_hut()
        net.from_nx(G)

        net.set_options("""
        const options = {
        "nodes": {
            "shape": "dot",
            "size": 16,
            "font": {
            "size": 18,
            "color": "white"
            },
            "color": {
            "highlight": {
                "border": "white",
                "background": "darkorchid"
            }
            }
        },
        "edges": {
            "color": {
            "color": "gray",
            "highlight": "darkGoldenRod"
            },
            "arrows": {
            "to": {"enabled": true}
            },
            "smooth": false
        },
        "physics": {
            "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        }
        }
        """)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_path = tmp_file.name

        components.html(open(html_path, "r", encoding="utf-8").read(), height=550)
        os.remove(html_path)





















elif st.session_state.selected == "EXPLORATION":

    # ─────────────────────────────────────
    # SECTION – ANALYSE EXPLORATOIRE (EDA)
    # ─────────────────────────────────────



    # 🔑 (la clé OpenAI est déjà définie plus haut : openai.api_key)

    # ── Helper : explication automatique d’un visuel ────────────────────────────
    def explain_plot(title: str,
                    variable: str | None,
                    df_src: pd.DataFrame,
                    extra: dict | None = None,
                    model: str = "gpt-3.5-turbo") -> str:
        """
        Retourne une synthèse (4‑6 lignes) générée par OpenAI
        pour expliquer le visuel concerné.
        """
        stats_block = ""
        if variable:
            stats = df_src[variable].describe().round(2).to_dict()
            stats_block = f"\nStatistiques de `{variable}` : {stats}"
        if extra:
            stats_block += f"\nInfos supplémentaires : {extra}"

        prompt = f"""
    Tu es un expert en data‑science.
    Explique en français (4‑6 phrases) le graphique « {title} ».
    Mentionne la distribution ou tendance, les valeurs typiques,
    les outliers éventuels et l’interprétation possible.
    {stats_block}
    """
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en analyse exploratoire."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=220,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            return f"⚠️ Erreur OpenAI : {e}"

    # ── 0. En‑tête ──────────────────────────────────────────────────────────────
    st.markdown("## 📊 Analyse exploratoire des données")

    st.markdown("""
    Cette section réalise une EDA complète :

    - **Univarié** : distributions & statistiques  
    - **Bivarié / multivarié** : corrélations, regroupements  
    - **Géospatial** : carte de densité  
    - **Qualité des données** : valeurs manquantes, outliers  

    Chaque visuel est suivi d’une **interprétation générée par OpenAI**.  
    Pour certains graphiques, le **code de calcul** est affiché à droite.
    """)

    # ─────────────────────────────────────
    # 1. Statistiques descriptives globales
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧮 1 Statistiques descriptives globales")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Résumé numérique :**")
        st.dataframe(df.describe().T.round(2), use_container_width=True)
    with col2:
        st.markdown("**Types de variables :**")
        st.write(df.dtypes.value_counts())

    # ─────────────────────────────────────
    # 2. Distributions univariées
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 2 Distribution des variables numériques")

    num_cols = ["bikes", "docks", "capacity", "fill_rate"]

    for col in num_cols:
        st.markdown(f"#### 🔹 Distribution de `{col}`")

        g_col, code_col = st.columns([3, 1])  # graphe à gauche, code à droite

        # — Graphe
        with g_col:
            fig = px.histogram(df, x=col, nbins=40, marginal="box",
                            title=f"Distribution de {col}")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # — Code (uniquement pour bikes & fill_rate pour l’exemple)
        with code_col:
            if col in ["bikes", "fill_rate"]:
                st.code(f"""
    # Distribution de {col}
    fig = px.histogram(
        df,
        x="{col}",
        nbins=40,
        marginal="box",
        title="Distribution de {col}"
    )
    st.plotly_chart(fig)
    """, language="python")

        # — Interprétation OpenAI
        with st.expander("🧠 Interprétation OpenAI", expanded=False):
            st.markdown(explain_plot(f"Distribution de {col}", col, df))

    # ─────────────────────────────────────
    # 3. Outliers & incohérences
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 3 Valeurs extrêmes & incohérences")

    df_extreme = df[(df["capacity"] < 5) | (df["capacity"] > 50)]
    df_incoh   = df[df["bikes"] > df["capacity"]]

    c_tbl, c_code = st.columns([3, 1])
    with c_tbl:
        st.markdown("**Stations capacité atypique (<5 ou >50) :**")
        st.dataframe(df_extreme[["name", "capacity", "bikes", "docks"]], use_container_width=True)

        st.markdown("**Stations où `bikes` > `capacity` :**")
        st.dataframe(df_incoh[["name", "bikes", "capacity", "fill_rate"]], use_container_width=True)

    with c_code:
        st.code("""
    # Détection simple d'outliers
    df_extreme = df[(df["capacity"] < 5) | (df["capacity"] > 50)]

    # Incohérence logique
    df_incoh   = df[df["bikes"] > df["capacity"]]
    """, language="python")

    with st.expander("🧠 Interprétation OpenAI – Outliers", expanded=False):
        st.markdown(explain_plot("Analyse des valeurs extrêmes",
                                None, df,
                                extra={"cap_outliers": len(df_extreme),
                                        "inc_bikes>cap": len(df_incoh)}))

    # ─────────────────────────────────────
    # 4. Corrélations
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 4 Corrélations entre variables")

    corr_matrix = df[num_cols].corr()

    g_corr, code_corr = st.columns([3, 1])
    with g_corr:
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            colorbar=dict(title="Corrélation")
        ))
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with code_corr:
        st.code("""
    # Matrice de corrélation
    corr = df[["bikes","docks","capacity","fill_rate"]].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
    st.plotly_chart(fig)
    """, language="python")

    with st.expander("🧠 Interprétation OpenAI – Corrélations", expanded=False):
        corr_flat = (corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
                                .stack().round(2).to_dict())
        st.markdown(explain_plot("Matrice de corrélation", None, df,
                                extra={"top_pairs": dict(list(corr_flat.items())[:5])}))

    # ─────────────────────────────────────
    # 5. Analyse par arrondissement
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏙️ 5 Analyse par arrondissement")

    grouped_arr = df.groupby("arr")[["bikes", "capacity", "fill_rate"]].mean().reset_index()

    # — Vélos moyens
    ga_g, ga_c = st.columns([3, 1])
    with ga_g:
        fig_arr = px.bar(grouped_arr.sort_values("bikes", ascending=False),
                        x="arr", y="bikes",
                        title="Vélos moyens par station (arr.)")
        fig_arr.update_layout(height=400)
        st.plotly_chart(fig_arr, use_container_width=True)
    with ga_c:
        st.code("""
    # Moyenne par arrondissement
    grouped = df.groupby("arr")["bikes"].mean().reset_index()
    fig = px.bar(grouped, x="arr", y="bikes")
    """, language="python")

    with st.expander("🧠 Interprétation – Vélos moyens", expanded=False):
        st.markdown(explain_plot("Vélos moyens par arrondissement",
                                None, df,
                                extra={"mean_bikes": grouped_arr.set_index('arr')["bikes"].round(1).to_dict()}))

    # — Taux de remplissage
    ga2_g, ga2_c = st.columns([3, 1])
    with ga2_g:
        fig_rate = px.bar(grouped_arr.sort_values("fill_rate", ascending=False),
                        x="arr", y="fill_rate",
                        title="Taux de remplissage moyen (arr.)")
        fig_rate.update_layout(height=400)
        st.plotly_chart(fig_rate, use_container_width=True)
    with ga2_c:
        st.code("""
    # Taux de remplissage moyen
    grouped = df.groupby("arr")["fill_rate"].mean().reset_index()
    fig = px.bar(grouped, x="arr", y="fill_rate")
    """, language="python")

    with st.expander("🧠 Interprétation – Taux de remplissage", expanded=False):
        st.markdown(explain_plot("Taux de remplissage moyen par arrondissement",
                                None, df,
                                extra={"mean_fill": grouped_arr.set_index('arr')["fill_rate"].round(2).to_dict()}))

    # ─────────────────────────────────────
    # 6. Carte de chaleur géographique
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 6 Carte de chaleur géographique")

    g_map, code_map = st.columns([3, 1])
    with g_map:
        fig_map = px.density_mapbox(
            df, lat="lat", lon="lon", z="bikes", radius=15,
            center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
            zoom=11,
            mapbox_style="carto-positron",
            title="Carte de chaleur – vélos disponibles"
        )
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    with code_map:
        st.code("""
    fig = px.density_mapbox(
        df, lat="lat", lon="lon", z="bikes", radius=15,
        center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
        zoom=11, mapbox_style="carto-positron")
    """, language="python")

    with st.expander("🧠 Interprétation – Carte", expanded=False):
        st.markdown(explain_plot("Carte de chaleur vélos", None, df,
                                extra={"stations": len(df)}))

    # ─────────────────────────────────────
    # 7. Données manquantes & invalides
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 7 Données manquantes & invalides")

    nulls = df.isna().sum()
    invalid_cap0 = (df["capacity"] == 0).sum()

    nm_g, nm_c = st.columns([3, 1])
    with nm_g:
        st.write("**Nulls (>0) :**")
        st.write(nulls[nulls > 0])

        st.write(f"🔴 Stations `capacity == 0` : **{invalid_cap0}**")

        fig_invalid = px.histogram(df[df["capacity"] == 0], x="bikes", nbins=20,
                                title="`bikes` lorsque `capacity == 0`")
        fig_invalid.update_layout(height=350)
        st.plotly_chart(fig_invalid, use_container_width=True)

    with nm_c:
        st.code("""
    nulls = df.isna().sum()
    invalid = df[df["capacity"] == 0]
    """, language="python")

    with st.expander("🧠 Interprétation – Qualité des données", expanded=False):
        st.markdown(explain_plot("Qualité des données", None, df,
                                extra={"nulls": nulls[nulls > 0].to_dict(),
                                        "capacity0": int(invalid_cap0)}))

    # ─────────────────────────────────────
    # 8. Conclusion
    # ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔚 8 Conclusion exploratoire")

    st.markdown("""
    - **Capacités** très hétérogènes (de <10 à >50 bornes).  
    - **Outliers** rares : quelques stations « méga‑capacité » ou incohérences (`bikes > capacity`).  
    - **Corrélations** : `fill_rate` fortement lié à `bikes` et `capacity`.  
    - **Géographie** : zones à forte densité de vélos vs. zones en sous‑charge.  
    - **Qualité** : très peu de valeurs nulles, données globalement fiables.
    """)

elif st.session_state.selected == "PILOTAGE":

    # ─────────────────────────────────────
    # SIDEBAR : CONTRÔLES
    # ─────────────────────────────────────
    st.sidebar.header("⚙️ Contrôles")
    if st.sidebar.button("🔄 Rafraîchir"):
        st.cache_data.clear()

    df_all = df
    arr_all = sorted(df_all["arr"].dropna().unique())
    with st.sidebar.expander("📍 Filtrer par arrondissement", expanded=False):
        arr_sel = st.multiselect(
            "Arrondissements",
            options=arr_all,
            default=[],
            placeholder="Tous les arrondissements",
        )
        st.caption("Sélection : **" + (", ".join(arr_sel) if arr_sel else "Tous") + "**")
    df = df_all[df_all["arr"].isin(arr_sel)] if arr_sel else df_all.copy()
    n_total = len(df)

    # ─────────────────────────────────────
    # KPI CALCULS
    # ─────────────────────────────────────
    mask_empty = df["bikes"] == 0
    mask_almost_empty = (df["bikes"] <= 2) & ~mask_empty
    mask_full = df["docks"] == 0
    mask_almost_full = (df["docks"] <= 2) & ~mask_full
    mask_hs = ~(df["installed"] & df["renting"] & df["returning"])

    k_empty = mask_empty.sum()
    p_almost_empty = mask_almost_empty.mean() * 100 if n_total else 0
    k_full = mask_full.sum()
    p_almost_full = mask_almost_full.mean() * 100 if n_total else 0
    k_hs = mask_hs.sum()
    p_op = (n_total - k_hs) / n_total * 100 if n_total else 0

    summary = {
        "n_total": n_total,
        "k_empty": int(k_empty),
        "k_full": int(k_full),
        "k_hs": int(k_hs),
        "p_op": p_op,
    }


    pairs = find_rebalance_pairs(df)
    report_text = generate_report(summary, pairs)


        
    col1, col2, col3 = st.columns(3)

    # ► Stations vides
    with col1:
        st.markdown(
            f"""
            <div class="dynamic-shadow"
                style="background:linear-gradient(135deg,#6e8efb,#a777e3);padding:15px;
                        display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
                <div>
                    <h4 style="color:white;margin:0;font-size:14px;">STATIONS VIDES</h4>
                    <p style="font-size:26px;color:white;margin:0;">{k_empty}</p>
                    <p style="font-size:14px;color:white;margin:0;">Quasi-vides : {p_almost_empty:.1f}%</p>
                </div>
                <div><i class="fas fa-bicycle" style="font-size:30px;color:white;"></i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ► Stations pleines
    with col2:
        st.markdown(
            f"""
            <div class="dynamic-shadow"
                style="background:linear-gradient(135deg,#6e8efb,#a777e3);padding:15px;
                        display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
                <div>
                    <h4 style="color:white;margin:0;font-size:14px;">STATIONS PLEINES</h4>
                    <p style="font-size:26px;color:white;margin:0;">{k_full}</p>
                    <p style="font-size:14px;color:white;margin:0;">Quasi-pleines : {p_almost_full:.1f}%</p>
                </div>
                <div><i class="fas fa-parking" style="font-size:30px;color:white;"></i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ► Stations hors service
    with col3:
        st.markdown(
            f"""
            <div class="dynamic-shadow"
                style="background:linear-gradient(135deg,#6e8efb,#a777e3);padding:15px;
                        display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
                <div>
                    <h4 style="color:white;margin:0;font-size:14px;">STATIONS HORS SERVICE</h4>
                    <p style="font-size:26px;color:white;margin:0;">{k_hs}</p>
                    <p style="font-size:14px;color:white;margin:0;">Opérationnelles : {p_op:.1f}%</p>
                </div>
                <div><i class="fas fa-tools" style="font-size:30px;color:white;"></i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.divider()

    # ─────────────────────────────────────
    # MIDDLE ROW : TABLEAUX + RAPPORT
    # ─────────────────────────────────────
    mid_left, mid_right = st.columns([2, 2], gap="medium")

    # ► Tableau des situations
    with mid_left:
        with st.container(border=True):
            tab1, tab2, tab3 = st.tabs(["⚠️ Stations critiques", "📉 Peu sollicitées", "📈 Très sollicitées"])

            # 1. Stations critiques
            crit = df[
                ~mask_hs & ((df["fill_rate"] < 0.10) | (df["fill_rate"] > 0.90))
            ].copy()
            crit["Situation"] = np.where(crit["fill_rate"] < 0.10, "Quasi vide", "Quasi pleine")
            crit["Taux de remplissage"] = (crit["fill_rate"] * 100).round(1)
            crit = crit.rename(columns={
                "name": "Station",
                "capacity": "Capacité",
                "bikes": "Vélos dispo",
            })

            with tab1:
                st.data_editor(
                    crit[["Station", "Situation", "Capacité", "Vélos dispo", "Taux de remplissage"]],
                    column_config={
                        "Taux de remplissage": st.column_config.ProgressColumn(
                            "Taux de remplissage",
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

            # 2. Stations peu sollicitées
            low_demand = df[
                (df["fill_rate"] > 0.85) &
                (df["capacity"] >= 15) &
                (~mask_hs)
            ].copy()
            low_demand["Taux de remplissage"] = (low_demand["fill_rate"] * 100).round(1)
            low_demand = low_demand.rename(columns={
                "name": "Station",
                "capacity": "Capacité",
                "bikes": "Vélos dispo",
            })

            with tab2:
                st.data_editor(
                    low_demand[["Station", "Capacité", "Vélos dispo", "Taux de remplissage"]],
                    column_config={
                        "Taux de remplissage": st.column_config.ProgressColumn(
                            "Taux de remplissage",
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

            # 3. Stations très sollicitées
            high_demand = df[
                (df["fill_rate"] < 0.30) &
                (df["capacity"] >= 15) &
                (~mask_hs)
            ].copy()
            high_demand["Taux de remplissage"] = (high_demand["fill_rate"] * 100).round(1)
            high_demand = high_demand.rename(columns={
                "name": "Station",
                "capacity": "Capacité",
                "bikes": "Vélos dispo",
            })

            with tab3:
                st.data_editor(
                    high_demand[["Station", "Capacité", "Vélos dispo", "Taux de remplissage"]],
                    column_config={
                        "Taux de remplissage": st.column_config.ProgressColumn(
                            "Taux de remplissage",
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )


    st.divider()

    # ► Rapport détaillé (remplace la jauge)
    with mid_right:
            with st.container(border=True):
                bt1, bt2, bt3 = st.tabs(["🏙️ Stations", "🔴 % Rupture", "📊 Taux moyen"])

                # Supprimer les enregistrements mal catégorisés
                df_viz = df[~df["arr"].isin(["Paris"])].copy()

                # Paramètre global d'affichage
                chart_height = 500

                # ───── 🏙️ Graphique 1 : Nombre de stations ─────
                stats = df_viz.groupby("arr").size().reset_index(name="stations")
                stats = stats.sort_values("stations", ascending=False)

                with bt1:
                    fig1 = px.bar(
                        stats,
                        x="arr",
                        y="stations",
                        title="Nombre de stations par arrondissement",
                        labels={"arr": "Arrondissement", "stations": "Nombre de stations"},
                    )
                    fig1.update_traces(marker_line_width=1.5)
                    fig1.update_layout(
                        height=chart_height,
                        xaxis_tickangle=-45,
                        xaxis_title=None,
                        yaxis_title=None,
                        margin=dict(l=10, r=10, t=60, b=10),
                        bargap=0.3,
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                # ───── 🔴 Graphique 2 : Taux de rupture (stations vides uniquement) ─────
                # On ne garde que les arrondissements avec au moins 3 stations
                station_counts = df_viz.groupby("arr").size()
                valid_arrs = station_counts[station_counts >= 3].index

                rupt = (
                    df_viz.assign(rupture=mask_empty)
                    .groupby("arr")["rupture"]
                    .mean()
                    .mul(100)
                    .reset_index()
                )
                rupt = rupt[rupt["arr"].isin(valid_arrs)]
                rupt = rupt[rupt["rupture"] > 0]  # Supprimer ceux à 0%
                rupt = rupt.sort_values("rupture", ascending=False)

                with bt2:
                    fig2 = px.bar(
                        rupt,
                        x="arr",
                        y="rupture",
                        title="Pourcentage de stations en rupture",
                        labels={"arr": "Arrondissement", "rupture": "% de stations vides"},
                    )
                    fig2.update_traces(marker_line_width=1.5)
                    fig2.update_layout(
                        height=chart_height,
                        xaxis_tickangle=-45,
                        xaxis_title=None,
                        yaxis_title=None,
                        margin=dict(l=10, r=10, t=60, b=10),
                        bargap=0.3,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # ───── 📊 Graphique 3 : Taux moyen de remplissage ─────
                avg = (
                    df_viz.groupby("arr")["fill_rate"]
                    .mean()
                    .mul(100)
                    .reset_index()
                    .sort_values("fill_rate", ascending=False)
                )

                with bt3:
                    fig3 = px.bar(
                        avg,
                        x="arr",
                        y="fill_rate",
                        title="Taux moyen de remplissage par arrondissement",
                        labels={"arr": "Arrondissement", "fill_rate": "% de remplissage moyen"},
                    )
                    fig3.update_traces(marker_line_width=1.5)
                    fig3.update_layout(
                        height=chart_height,
                        xaxis_tickangle=-45,
                        xaxis_title=None,
                        yaxis_title=None,
                        margin=dict(l=10, r=10, t=60, b=10),
                        bargap=0.3,
                    )
                    st.plotly_chart(fig3, use_container_width=True)




    # ─────────────────────────────────────
    # BOTTOM ROW : GRAPHIQUES + CARTE
    # ─────────────────────────────────────
    bot_left, bot_right = st.columns([2, 2], gap="medium")


    with bot_left:
        with st.container(border=True):
            tab_geo, tab_rebal = st.tabs(["🗺️ Carte des stations", "🔁 Rééquilibrage entre stations"])

            # ▼ Onglet 1 — Carte des stations classiques
            with tab_geo:
                df["R"] = (255 * (1 - df["fill_rate"].fillna(0))).astype(int)
                df["G"] = (255 * df["fill_rate"].fillna(0)).astype(int)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[lon, lat]",
                    get_radius=80,
                    get_fill_color="[R, G, 100]",
                    pickable=True,
                )

                view = pdk.ViewState(
                    latitude=df["lat"].mean(),
                    longitude=df["lon"].mean(),
                    zoom=11,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v10",
                        layers=[layer],
                        initial_view_state=view,
                        tooltip={
                            "text": "{name}\nVélos :{bikes}\nBornes :{docks}\nRempl. :{fill_rate:.0%}",
                        },
                    )
                )

            # ▼ Onglet 2 — Carte de rééquilibrage avec arcs
            # ▼ Onglet 2 — Carte de rééquilibrage avec arcs
            with tab_rebal:
                st.markdown("**Carte de rééquilibrage — flèches vert ➝ rouge**")

                # 1. Stations en tension
                stations_empty = df[(df["bikes"] == 0) & (~mask_hs)].copy()

                # 2. Stations donneuses
                stations_donor = df[(df["bikes"] >= 5) & (~mask_hs)].copy()

                # 3. Création des paires
                pairs_data = []
                for _, row in stations_empty.iterrows():
                    dists = haversine(row["lat"], row["lon"], stations_donor["lat"], stations_donor["lon"])
                    if dists.empty:
                        continue
                    idx_min = dists.idxmin()
                    donor = stations_donor.loc[idx_min]

                    pairs_data.append({
                        "donor_name": donor["name"],
                        "donor_lat": donor["lat"],
                        "donor_lon": donor["lon"],
                        "dest_name": row["name"],
                        "dest_lat": row["lat"],
                        "dest_lon": row["lon"],
                        "weight": int(donor["bikes"]),
                    })

                rebalance_df = pd.DataFrame(pairs_data)

                if not rebalance_df.empty and "weight" in rebalance_df.columns:
                    # 4. Épaisseur relative des arcs
                    rebalance_df["weight_scaled"] = rebalance_df["weight"].clip(1, 15) / 3

                    # 5. Couches PyDeck
                    arc_layer = pdk.Layer(
                        "ArcLayer",
                        data=rebalance_df,
                        get_source_position=["donor_lon", "donor_lat"],
                        get_target_position=["dest_lon", "dest_lat"],
                        get_width="weight_scaled",
                        get_source_color=[0, 180, 0, 160],
                        get_target_color=[200, 0, 0, 160],
                        pickable=True,
                        auto_highlight=True,
                    )

                    station_codes = set(rebalance_df["donor_name"]) | set(rebalance_df["dest_name"])
                    df_subset = df[df["name"].isin(station_codes)].copy()

                    scatter_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=df_subset,
                        get_position="[lon, lat]",
                        get_radius=80,
                        get_fill_color="""
                            [bikes == 0 ? 255 : bikes >= 5 ? 0 : 150,
                            bikes == 0 ? 0 : bikes >= 5 ? 200 : 150,
                            100, 180]
                        """,
                        pickable=True,
                    )

                    view = pdk.ViewState(
                        latitude=df["lat"].mean(),
                        longitude=df["lon"].mean(),
                        zoom=11,
                    )

                    st.pydeck_chart(
                        pdk.Deck(
                            map_style="mapbox://styles/mapbox/light-v10",
                            layers=[scatter_layer, arc_layer],
                            initial_view_state=view,
                            tooltip={
                                "html": "<b>{name}</b><br/>Vélos : {bikes}<br/>Bornes : {docks}<br/>Rempl. : {fill_rate:.0%}",
                                "style": {
                                    "backgroundColor": "white",
                                    "color": "black",
                                    "fontSize": "12px"
                                }
                            },
                        )
                    )
                else:
                    st.info("Aucune paire de rééquilibrage trouvée pour les arrondissements sélectionnés.")


    with bot_right:
        st.markdown(
                f"""
                <div class="report-card">
                    <h4>Rapport détaillé</h4>
                    <p>{report_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────
    # PIED DE PAGE
    # ─────────────────────────────────────
    st.markdown("---")
    st.caption("© Ville de Paris – Données temps réel. Dashboard Streamlit 2025.")


elif st.session_state.selected == "DOCUMENTATION":

    st.markdown("## 📚 Explication de l'interface de **PILOTAGE**")

    ###############################################################################
    # 1. KPI « Stations vides », « Stations pleines », « Hors service »
    ###############################################################################
    doc1_txt, doc1_code = st.columns([2, 1])
    with doc1_txt:
        st.markdown("""
    ### 1️⃣ Cartes KPI

    Chaque carte **résume un indicateur critique** pour l’opérationnel :

    | Carte | Objectif | Métrique affichée |
    |-------|----------|-------------------|
    | **Stations vides** | Identifier les risques de rupture d’offre | Nombre brut + `%` quasi-vides |
    | **Stations pleines** | Repérer la saturation des bornes de retour | Nombre brut + `%` quasi-pleines |
    | **Stations hors service** | Suivre la disponibilité technique | Nombre HS + `%` opérationnelles |

    *Notes importantes* :
    - Les couleurs du gradient purple-blue sont purement visuelles ; les icônes FA (« fas ») renforcent le repère cognitif.
    - Le **hover** déclenche une ombre portée (`.dynamic-shadow`) pour attirer le regard.
    - Les pourcentages « quasi » utilisent un *mask* `(<= 2)` : cela agit comme un seuil d’alerte douce avant la rupture franche.
    """)

    with doc1_code:
        st.code("""
    # Calculs d’état
    mask_empty        = df["bikes"] == 0
    mask_almost_empty = (df["bikes"] <= 2) & ~mask_empty
    mask_full         = df["docks"] == 0
    mask_almost_full  = (df["docks"] <= 2) & ~mask_full
    mask_hs           = ~(df["installed"] & df["renting"] & df["returning"])

    # Agrégation
    summary = {
        "k_empty":        int(mask_empty.sum()),
        "p_almost_empty": mask_almost_empty.mean() * 100,
        "k_full":         int(mask_full.sum()),
        "p_almost_full":  mask_almost_full.mean() * 100,
        "k_hs":           int(mask_hs.sum()),
        "p_op":           (len(df) - mask_hs.sum()) / len(df) * 100,
    }

    # Carte « Stations vides » (structure générique ; seules les valeurs changent)
    st.markdown(f\"\"\"
    <div class="dynamic-shadow" style="background:linear-gradient(135deg,#6e8efb,#a777e3);
        padding:15px;display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
    <div>
        <h4 style="color:white;margin:0;font-size:14px;">STATIONS VIDES</h4>
        <p style="font-size:26px;color:white;margin:0;">{summary['k_empty']}</p>
        <p style="font-size:14px;color:white;margin:0;">Quasi-vides : {summary['p_almost_empty']:.1f}%</p>
    </div>
    <div><i class="fas fa-bicycle" style="font-size:30px;color:white;"></i></div>
    </div>
    \"\"\", unsafe_allow_html=True)
    """, language="python", line_numbers=True)

    ###############################################################################
    # 2. Tableaux interactifs (stations critiques, peu / très sollicitées)
    ###############################################################################
    st.markdown("---")
    doc2_txt, doc2_code = st.columns([2, 1])
    with doc2_txt:
        st.markdown("""
    ### 2️⃣ Tableaux interactifs

    Trois onglets filtrent dynamiquement le DataFrame :

    | Onglet | Critère métier | Finalité |
    |--------|----------------|----------|
    | **⚠️ Stations critiques** | `fill_rate < 10 %` ou `> 90 %` | Prioriser les interventions |
    | **📉 Peu sollicitées** | `fill_rate > 85 %` ET `capacity ≥ 15` | Optimiser l’offre vs. la demande |
    | **📈 Très sollicitées** | `fill_rate < 30 %` ET `capacity ≥ 15` | Déployer des rééquilibrages |

    Chaque tableau est rendu avec `st.data_editor` :  
    - Colonnes reformatées (`ProgressColumn` pour le taux).  
    - Index masqué pour plus de lisibilité.  
    - Hauteur fixe pour éviter la dérive de la mise en page.
    """)

    with doc2_code:
        st.code("""
    # Exemple – stations critiques
    crit = df[
        ~mask_hs &
        ((df["fill_rate"] < 0.10) | (df["fill_rate"] > 0.90))
    ].copy()
    crit["Situation"] = np.where(crit["fill_rate"] < 0.10, "Quasi vide", "Quasi pleine")
    crit["Taux de remplissage"] = (crit["fill_rate"] * 100).round(1)

    st.data_editor(
        crit[["name", "Situation", "capacity", "bikes", "Taux de remplissage"]],
        column_config={
            "Taux de remplissage": st.column_config.ProgressColumn(
                "Taux de remplissage", format="%.1f %%", min_value=0, max_value=100
            ),
        },
        hide_index=True, height=500, use_container_width=True,
    )
    """, language="python", line_numbers=True)

    ###############################################################################
    # 3. Graphiques comparatifs par arrondissement
    ###############################################################################
    st.markdown("---")
    doc3_txt, doc3_code = st.columns([2, 1])
    with doc3_txt:
        st.markdown("""
    ### 3️⃣ Graphiques comparatifs

    Les trois **onglets horizontaux** présentent des barplots **Plotly** :

    | Graphique | Mesure | Insight recherché |
    |-----------|--------|-------------------|
    | *Nombre de stations* | `df.groupby("arr").size()` | Volume d’infrastructure |
    | *% Stations en rupture* | `mask_empty.mean()` | Vulnérabilité par zone |
    | *Taux moyen de remplissage* | `fill_rate.mean()` | Équilibre global stock/demande |

    Points clés :
    - Les arrondissements mono-station sont exclus pour éviter les taux à 0 % biaisés.
    - `chart_height = 500` harmonise la verticalité.
    - Les tick-labels à `-45 °` limitent le chevauchement.
    """)

    with doc3_code:
        st.code("""
    # % Stations en rupture (extrait)
    mask_empty = df["bikes"] == 0

    rupt = (
        df.assign(rupture=mask_empty)
        .groupby("arr")["rupture"]
        .mean()
        .mul(100)
        .reset_index()
    )
    valid_arrs = df.groupby("arr").size().query(">=3").index
    rupt = rupt[rupt["arr"].isin(valid_arrs) & (rupt["rupture"] > 0)]

    fig2 = px.bar(
        rupt.sort_values("rupture", ascending=False),
        x="arr", y="rupture",
        title="Pourcentage de stations en rupture",
        labels={"arr": "Arrondissement", "rupture": "% de stations vides"},
    )
    st.plotly_chart(fig2, use_container_width=True)
    """, language="python", line_numbers=True)

    ###############################################################################
    # 4. Cartographie – densité & rééquilibrage
    ###############################################################################
    st.markdown("---")
    doc4_txt, doc4_code = st.columns([2, 1])
    with doc4_txt:
        st.markdown("""
    ### 4️⃣ Cartes géospatiales

    #### 🗺️ Carte densité (`density_mapbox`)
    Visualisation « heatmap » de l’offre (`bikes`).  
    Paramètres : `radius=15`, `mapbox_style="carto-positron"`, centrage moyen des coordonnées.  
    > *Astuce UX* : la couleur suit la densité ; aucun overlay de point n’est nécessaire.

    #### 🔁 Carte de rééquilibrage (`ArcLayer`)
    - **Sources** : stations ≥ 5 vélos libres (donneuses).  
    - **Cibles** : stations `bikes == 0` (en tension).  
    - Les arcs vert ➝ rouge symbolisent l’acheminement de vélos.  
    - `weight_scaled` module l’épaisseur selon la dispo du donneur (clip 1-15 ➞ /3 pour lisibilité).
    """)

    with doc4_code:
        with st.expander("Voir l’extrait PyDeck complet"):
            st.code("""
    # Construction des paires (résumé)
    pairs = []
    for _, need in df[df["bikes"] == 0].iterrows():
        donor_idx = haversine(
            need["lat"], need["lon"],
            df[df["bikes"] >= 5]["lat"],
            df[df["bikes"] >= 5]["lon"],
        ).idxmin()
        donor = df.loc[donor_idx]
        pairs.append({
            "donor_lat": donor["lat"], "donor_lon": donor["lon"],
            "dest_lat": need["lat"],  "dest_lon": need["lon"],
            "weight":   int(donor["bikes"]),
        })
    rebalance_df = pd.DataFrame(pairs)
    rebalance_df["weight_scaled"] = rebalance_df["weight"].clip(1, 15) / 3

    # Couches PyDeck
    arc_layer = pdk.Layer(
        "ArcLayer", data=rebalance_df,
        get_source_position=["donor_lon", "donor_lat"],
        get_target_position=["dest_lon", "dest_lat"],
        get_width="weight_scaled",
        get_source_color=[0, 180, 0, 160],
        get_target_color=[200, 0, 0, 160],
        pickable=True, auto_highlight=True,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer", data=df,
        get_position="[lon, lat]", get_radius=80,
        get_fill_color="[bikes == 0 ? 255 : bikes >= 5 ? 0 : 150, \
                        bikes == 0 ? 0 : bikes >= 5 ? 200 : 150, 100, 180]",
        pickable=True,
    )
    view = pdk.ViewState(latitude=df["lat"].mean(),
                        longitude=df["lon"].mean(), zoom=11)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        layers=[scatter_layer, arc_layer],
        initial_view_state=view,
    ))
    """, language="python", line_numbers=True)

    ###############################################################################
    # 5. Rapport opérationnel (OpenAI)
    ###############################################################################
    st.markdown("---")
    doc5_txt, doc5_code = st.columns([2, 1])
    with doc5_txt:
        st.markdown("""
    ### 5️⃣ Rapport opérationnel (GPT-3.5)

    Le **bloc violet** (« report-card ») affiche un *one-pager* généré via l’API OpenAI :

    1. 📊 Analyse des indicateurs agrégés (`summary`).  
    2. 🔄 Liste *inline* des paires de rééquilibrage (`pairs`).  
    3. ✍️ Rédaction ≤ 120 mots, ton proactif.  

    > Le rapport se régénère à chaque rafraîchissement de la page (ou modification du filtre arrondissement).
    """)

    with doc5_code:
        st.code("""
    def generate_report(summary_dict: dict, pairs: list[dict]) -> str:
        prompt = f\"\"\"
        Contexte : réseau Vélib' temps réel...
        Indicateurs :
        • stations totales : {summary_dict['n_total']}
        ...
        \"\"\"
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un expert logistique Vélib'."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=180, temperature=0.7, top_p=0.9,
        )
        return resp.choices[0].message["content"].strip()
    """, language="python", line_numbers=True)

    st.success("🎉 Fin de la documentation — l’utilisateur dispose désormais d’une vue complète sur la logique de la page *Pilotage*.")

    # Fin de la section documentation
