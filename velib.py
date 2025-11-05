import time
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


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def find_rebalance_pairs(df: pd.DataFrame, max_pairs: int = 5) -> list[tuple]:
    need = df[df["bikes"] == 0].copy()
    supply = df[df["bikes"] >= 5].copy()
    pairs = []
    for _, row in need.sort_values("capacity", ascending=False).head(max_pairs).iterrows():
        dists = haversine(row["lat"], row["lon"], supply["lat"], supply["lon"])
        if dists.empty:
            continue
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
    now = datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")
    pairs_txt = "\n".join(
        [
            f"- {p['donor_name']} ({p['donor_bikes']} vélos, {p['distance_km']} km) → {p['dest_name']}"
            for p in pairs
        ]
    ) or "Aucune suggestion de ré‑équilibrage (réseau stable)."
    prompt = f"""
    Contexte : réseau Vélib' temps réel le {now}.
    Indicateurs :
    • stations totales : {summary_dict['n_total']}
    • stations vides : {summary_dict['k_empty']}
    • stations pleines : {summary_dict['k_full']}
    • stations HS : {summary_dict['k_hs']}
    • taux opérationnel : {summary_dict['p_op']:.1f} %

    Voici des propositions de ré‑équilibrage :
    {pairs_txt}

    Rédige un bref rapport (≤ 120 mots) en français :
    ➊ Résume la situation générale.
    ➋ Identifie la station la plus urgente.
    ➌ Donne 2 conseils opérationnels immédiats basés sur les paires ci‑dessus.
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
        return f"Erreur OpenAI : {e}"

def explain_plot(title: str,
                variable: str | None,
                df_src: pd.DataFrame,
                extra: dict | None = None,
                model: str = "gpt-3.5-turbo") -> str:
    stats_block = ""
    if variable:
        stats = df_src[variable].describe().round(2).to_dict()
        stats_block = f"\nStatistiques de `{variable}` : {stats}"
    if extra:
        stats_block += f"\nInfos supplémentaires : {extra}"

    prompt = f"""
Tu es un expert en data‑science.
Explique en français (4‑6 phrases) le graphique « {title} ».
Mentionne la distribution ou tendance, les valeurs typiques,
les outliers éventuels et l'interprétation possible.
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
        return f"⚠️ Erreur OpenAI : {e}"


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

if "chrono_start_ts" not in st.session_state:
    st.session_state.chrono_start_ts = time.time()


with st.sidebar:
    chrono_start_ts = int(st.session_state.chrono_start_ts * 1000)  

    components.html(f"""
        <div style="font-size:17px;font-weight:bold;margin:10px 0;">
            <span style="margin-right:8px;">⏱️</span>Chrono en cours :
            <span id="elapsedTime" style="color:deepskyblue;font-family:monospace;"></span>
        </div>
        <script>
        const start = {chrono_start_ts};
        function updateElapsedTime() {{
            const now = Date.now();
            let elapsed = Math.floor((now - start) / 1000);
            const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
            elapsed %= 3600;
            const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const s = String(elapsed % 60).padStart(2, '0');
            document.getElementById("elapsedTime").textContent = h + ":" + m + ":" + s;
        }}
        setInterval(updateElapsedTime, 1000);
        updateElapsedTime();
        </script>
    """, height=60)

    st.session_state.selected = option_menu(
        'DASHBOARD',
        ["SOMMAIRE", "DONNÉES", "ANALYSE", "MODELES ET EVALUATION", "ARCHITECTURE", "PREDICTIONS", "MONITORING & MAINTENANCE"],
        icons=['house', 'database', 'speedometer', 'cpu'],
        menu_icon='cast',
        default_index=0
    )


if st.session_state.selected == "SOMMAIRE":
    
    st.markdown("""
    <style>
    .sommaire-page .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0 !important;
        min-height: 100vh;
    }

    .sommaire-page .block-container {
        max-width: 1400px;
        padding: 1rem !important;
    }

    .sommaire-container {
        background: transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 2.5rem 3rem;
        margin: 0;
        animation: fadeIn 0.6s ease-out;
    }

    .sommaire-page [data-testid="column"] {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        justify-content: flex-start;
        padding: 0 !important;
    }

    .left-column,
    .right-column {
    margin: 0 !important;
    padding-top: 0 !important;
    font-size: clamp(16px, 1vw + 0.5rem, 20px) !important;
    line-height: 1.6;
    }

    .left-column > *:first-child,
    .right-column > *:first-child {
    margin-top: 0 !important;
    }


    .left-column * {
        margin-top: 0 !important;
    }

    .sommaire-page .stMarkdown, .sommaire-page .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }

    .left-column {
        padding-right: 2rem;
        border-right: 2px solid #f0f0f0;
    }

    .sommaire-label { 
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 1rem;
        text-transform: uppercase;
        font-family: 'Poppins', sans-serif;
    }


    .main-title {
        color: #2d3748;
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .objectif-text {
        color: #4a5568;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 3rem;
    }

    .membres-section {
        padding-top: 2rem;
    }

    .membres-title {
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: inline-block;
        border-bottom: 3px solid #667eea;
        padding-bottom: .3rem;
    }

    .membres-list {
        color: #4a5568;
        font-size: 1rem;
        line-height: 1.8;
        list-style: none;
        padding-left: 0;
        margin: 0;
    }

    .membres-list li {
        padding-left: 1.5rem;
        position: relative;
        margin-bottom: .5rem;
    }

    .membres-list li:before {
        content: "●";
        color: #667eea;
        font-size: 1.2rem;
        position: absolute;
        left: 0;
        top: -2px;
    }

    .right-column {
        padding-left: 3rem;
    }

    .stepper-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.2rem;
        transition: transform .2s ease;
        padding: .8rem;
        border-radius: 10px;
    }

    .stepper-item:hover {
        background: #f7fafc;
        transform: translateX(5px);
    }

    .stepper-number {
        min-width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: 1.2rem;
        box-shadow: 0 4px 15px rgba(102,126,234,.4);
    }

    .stepper-content {
        flex: 1;
        padding-top: .3rem;
    }

    .stepper-title {
        color: #2d3748;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: .3rem;
    }

    .stepper-description {
        color: #718096;
        font-size: .95rem;
        line-height: 1.4;
    }

    @keyframes fadeIn {
        from {opacity:0; transform:translateY(20px);}
        to {opacity:1; transform:translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sommaire-page">', unsafe_allow_html=True)
    st.markdown('<div class="sommaire-container">', unsafe_allow_html=True)

    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.markdown('<div class="sommaire-label">SOMMAIRE</div>', unsafe_allow_html=True)
        st.markdown('<div class="left-column">', unsafe_allow_html=True)
        st.markdown('''
        <div class="main-title">
            PROJET VELIB – PRÉDICTION DU NOMBRE DE VÉLOS DISPONIBLES PAR STATION
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <div class="objectif-text">
            Ce projet vise à développer un système de prédiction intelligent pour anticiper la disponibilité des vélos Velib en temps réel, optimisant ainsi l'expérience utilisateur et la gestion des stations.
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('''
        <div class="membres-section">
            <div class="membres-title">MEMBRES DU GROUPE :</div>
            <ul class="membres-list">
                <li>David Graceffa</li>
                <li>Hadrien Moncomble</li>
                <li>Nour Amara</li>
                <li>Daniel AGOUNDOTE</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="right-column">', unsafe_allow_html=True)
        parties = [
            ("01", "Données & Métriques", "Sources, métriques clés et datamarts."),
            ("02", "Data Analyse", "Visualisations et tendances majeures."),
            ("03", "Modèles & Évaluation", "Modèles choisis, paramètres et performance."),
            ("04", "API & Conteneurisation", "Endpoints et image Docker."),
            ("05", "Architecture", "Architecture, orchestration et mise en prod."),
            ("06", "Prédictions", "Tentatives de prédiction par le modèle"),
            ("07", "Monitoring & Maintenance", "Suivi en production et améliorations")
        ]
        for num, title, desc in parties:
            st.markdown(f'''
            <div class="stepper-item">
                <div class="stepper-number">{num}</div>
                <div class="stepper-content">
                    <div class="stepper-title">{title}</div>
                    <div class="stepper-description">{desc}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.selected in ["DONNÉES", "ANALYSE", "MODELES ET EVALUATION", "MICROSERVICES & DEPLOIEMENT", "PREDICTIONS", "MONITORING & MAINTENANCE"]:
    
    st.markdown(
        """
        <style>
        section.main > div:first-child {padding-top:0.3rem;}
        div[data-testid="stMetric"] div {justify-content:flex-start;}
        .dynamic-shadow {
            transition: box-shadow 0.3s, transform 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        .dynamic-shadow:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,255,0.3);
        }
        .report-card {
            background: linear-gradient(135deg,#6e8efb,#a777e3);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .report-card:hover {
            box-shadow: 0 8px 16px rgba(0,0,255,0.3);
        }
        </style>
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        """,
        unsafe_allow_html=True,
    )


if st.session_state.selected == "DONNÉES":

    st.markdown(
        """
        <style>
        section.main > div:first-child {padding-top:0.3rem;}
        div[data-testid="stMetric"] div {justify-content:flex-start;}
        .dynamic-shadow {
            transition: box-shadow 0.3s, transform 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        .dynamic-shadow:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,255,0.3);
        }
        .report-card {
            background: linear-gradient(135deg,#6e8efb,#a777e3);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .report-card:hover {
            box-shadow: 0 8px 16px rgba(0,0,255,0.3);
        }
        </style>
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        """,
        unsafe_allow_html=True,
    )


if st.session_state.selected == "DONNÉES":

    st.markdown("""
    <style>
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .workflow-box {
        background: white;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        font-weight: 600;
        color: #2d3748;
        min-height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .workflow-arrow {
        text-align: center;
        font-size: 2rem;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .workflow-number {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .highlight-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 CORPUS D\'ENTRAÎNEMENT & DATA ENGINEERING</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Constitution d'un DataMart")
        st.markdown("""
        <div class="info-card">
        <p style="margin:0; font-size:1.05rem; line-height:1.6;">
        Pour réaliser ce travail, nous avons besoin de données issues des différentes stations Vélib' 
        de la métropole de Paris. Pour cela, nous nous sommes connectés à l'<strong>API OpenData Paris</strong>.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🔗 Source API")
        st.code("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records", language="text")
    
    with col2:
        st.markdown("### 📋 Schéma du DataMart")
        
        st.markdown("""
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div style="flex: 1; margin: 0 10px;">
                <div style="background: #667eea; color: white; padding: 10px; border-radius: 8px 8px 0 0; text-align: center; font-weight: 700;">
                    Station table
                </div>
                <div style="background: #f8f9fa; border: 2px solid #667eea; border-top: none; padding: 15px; border-radius: 0 0 8px 8px; font-size: 0.9rem;">
                    • station_id (primary key)<br>
                    • stationcode (foreign key)<br>
                    • capacity<br>
                    • code_insee_commune<br>
                    • latitude<br>
                    • longitude<br>
                    • name<br>
                    • nom_arrondissement_commune
                </div>
            </div>
            <div style="display: flex; align-items: center; font-size: 1.5rem; color: #667eea; font-weight: 700;">
                1 ── n
            </div>
            <div style="flex: 1; margin: 0 10px;">
                <div style="background: #764ba2; color: white; padding: 10px; border-radius: 8px 8px 0 0; text-align: center; font-weight: 700;">
                    Availability table
                </div>
                <div style="background: #f8f9fa; border: 2px solid #764ba2; border-top: none; padding: 15px; border-radius: 0 0 8px 8px; font-size: 0.9rem;">
                    • availability_id (primary key)<br>
                    • stationcode (foreign key)<br>
                    • duedate<br>
                    • ebike<br>
                    • mechanical<br>
                    • numbikesavailable<br>
                    • numdocksavailable<br>
                    • is_installed<br>
                    • is_renting<br>
                    • is_returning
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙️ PIPELINE D\'EXTRACTION</div>', unsafe_allow_html=True)

    workflow_col1, workflow_col2, workflow_col3, workflow_col4, workflow_col5 = st.columns([1, 1, 1, 1, 1])
    
    with workflow_col1:
        st.markdown('<div class="workflow-number">1</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-box">Velib API Client</div>', unsafe_allow_html=True)
    
    with workflow_col2:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-number">2</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-box">Response preprocessing and validation</div>', unsafe_allow_html=True)
    
    with workflow_col3:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-number">3</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-box">Create Velib availability database structure</div>', unsafe_allow_html=True)
    
    with workflow_col4:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-number">4</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-box">Create and update Velib availability database</div>', unsafe_allow_html=True)
    
    with workflow_col5:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-number">5</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-box">Scheduler and API request every 15 minutes</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-box" style="background: #e8f5e9; border-color: #4caf50;">Velib availability DB</div>', unsafe_allow_html=True)

    st.markdown("### 🔄 Étapes du Pipeline")
    
    pipeline_steps = [
        ("Client API", "Récupération des données de différentes ressources de l'API Vélib' (stations, disponibilité actualisée toutes les 15 minutes)", "api-client"),
        ("Extraction & Validation", "Extraction des champs d'intérêts à partir des réponses JSON et validation des types à l'aide de la librairie pydantic", "pydantic"),
        ("Création Structure DB", "Création de la structure de la base de données en deux tables : Une table Stations pour les données statiques et une table Availability pour les données dynamiques", "SQLAlchemy"),
        ("Module de Réalisation", "Module réunissant les réalisations de requêtes, la validation des données, la création selon le schéma ou la mise à jour des deux tables (boucle pour récupérer les données sur l'ensemble des ~1400 stations du réseau)", None),
        ("Scheduler", "Fonctions du module utilisées dans un scheduler programmé toutes les 15 minutes avec librairie schedule et interface CLI avec argparse pour démarrer ou stopper la mise à jour", "schedule")
    ]
    
    for i, (title, desc, lib) in enumerate(pipeline_steps, 1):
        with st.expander(f"**Bloc {i} : {title}**", expanded=False):
            st.markdown(f"**Description :** {desc}")
            if lib:
                st.markdown(f"**Librairie utilisée :** `{lib}`")

    st.markdown('<div class="section-title">📊 VARIABLES & MÉTRIQUES</div>', unsafe_allow_html=True)

    var_col1, var_col2 = st.columns([3, 2])
    
    with var_col1:
        st.markdown("### 📋 Variables Extraites")
        
        variables_data = {
            "Variable": [
                "stationcode", "name", "nom_arrondissement_communes", 
                "code_insee_commune", "latitude", "longitude", "duedate",
                "ebike", "mechanical", "numbikesavailable", "numdocksavailable"
            ],
            "Type": [
                "string", "string", "string", "string", 
                "float", "float", "datetime", "int", "int", "int", "int"
            ],
            "Description": [
                "Code unique de la station (identifiant technique)",
                "Nom lisible de la station (ex. 'Bastille - Rue Saint-Antoine')",
                "Arrondissement ou commune associée à la station",
                "Code INSEE de la commune (utile pour croiser avec météo, démographie, etc.)",
                "Latitude de la station",
                "Longitude de la station",
                "Date et heure de l'observation (instantané)",
                "Nombre de vélos à assistance électrique disponibles à cette station à ce moment",
                "Nombre de vélos mécaniques disponibles à cette station à ce moment",
                "Nombre total de vélos disponibles à cette station (souvent ebike + mechanical)",
                "Nombre de bornes vides disponibles à cette station (pour pouvoir restituer un vélo)"
            ],
            "Pertinence": [85, 95, 80, 60, 100, 100, 100, 90, 90, 95, 90]
        }
        
        df_vars = pd.DataFrame(variables_data)
        
        st.dataframe(
            df_vars,
            column_config={
                "Variable": st.column_config.TextColumn("Variable", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Pertinence": st.column_config.ProgressColumn(
                    "Pertinence",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                    width="small"
                ),
            },
            hide_index=True,
            use_container_width=True,
            height=450
        )
    
    with var_col2:
        st.markdown("### ⚙️ Variables Non Utilisées")
        st.markdown("""
        <div class="info-card">
        <p style="margin-bottom:1rem; font-weight:600;">Les champs suivants ne sont pas utilisés pour la prédiction :</p>
        <ul style="margin:0; padding-left:1.5rem; line-height:1.8;">
            <li><code>is_installed</code></li>
            <li><code>is_renting</code></li>
            <li><code>is_returning</code></li>
        </ul>
        <p style="margin-top:1rem; font-style:italic; color:#555;">
        Ces flags booléens représentent l'état opérationnel de la station mais n'apportent pas 
        d'information prédictive pour le nombre de vélos disponibles.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Période de Collecte")
        st.markdown("""
        <div class="info-card">
        <p style="margin:0; font-size:1.1rem; text-align:center;">
        <span class="highlight-text">Données extraites toutes les 15 minutes</span><br>
        <strong>Du 11 mars 2025 jusqu'à aujourd'hui</strong><br><br>
        <em>Objectif : construire un historique sur plusieurs mois pour servir de données d'entraînement au modèle</em>
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔧 GÉNÉRATION DU DATASET D\'ENTRAÎNEMENT</div>', unsafe_allow_html=True)

    gen_col1, gen_col2 = st.columns([1, 1])
    
    with gen_col1:
        st.markdown("### 🎯 Objectifs du Dataset")
        
        objectifs = [
            ("Fusionner les tables", "Extraire les données statiques de la table Stations et dynamiques de la table Availability en une seule table exploitable"),
            ("Ré-échantillonnage horaire", "Ré-échantillonner par station les points de données toutes les heures, pour obtenir une moyenne par heure au lieu de toutes les 15 minutes"),
            ("Gestion des heures manquantes", "Faire 'apparaître' les heures où il n'y a pas d'enregistrement pour chaque station, et faire commencer et terminer les séries temporelles correspondant à chaque station au même moment"),
            ("Élimination des stations incomplètes", "Jeter les stations qui présentent trop d'heures manquantes selon un ratio prédéfini"),
            ("Séparation Train/Test", "Séparer la table obtenue en dataset d'entraînement et de test, selon un ratio prédéfini (par défaut 80% pour le train et 20% pour le test)")
        ]
        
        for i, (titre, desc) in enumerate(objectifs, 1):
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div class="workflow-number" style="margin-right: 1rem; margin-bottom: 0;">{i}</div>
                    <strong style="font-size: 1.1rem; color: #2d3748;">{titre}</strong>
                </div>
                <p style="margin-left: 3.5rem; color: #4a5568; line-height: 1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with gen_col2:
        st.markdown("### 🔀 Workflow de Traitement")
        
        st.markdown("""
        <div style="margin-top: 2rem;">
            <div class="workflow-box" style="background: #e3f2fd; border-color: #2196f3;">
                run_pipeline
            </div>
            <div class="workflow-arrow">↓</div>
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        load_data_from_db
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        transform_to_tidy_with_resample
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        resample
                    </div>
                </div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        create_time_complete_df
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        format_columns
                    </div>
                </div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        remove_complete_nan_df
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        drop_parsed_NaN_series
                    </div>
                </div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        compute_split_ratio
                    </div>
                </div>
                <div style="flex: 1;">
                    <div class="workflow-box" style="font-size: 0.9rem;">
                        split_and_check_NaN_percent
                    </div>
                </div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background: #e8f5e9; border-color: #4caf50;">
                Dataset Train & Test
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card" style="margin-top: 1rem;">
        <p style="margin:0; font-size:0.95rem;">
        <strong>📊 Traitement avec Polars :</strong><br>
        Cette étape est réalisée par un seul module python et à l'aide de la librairie 
        <code>polars</code>, qui permet un traitement rapide des gros datasets avec parallélisation poussée.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Métriques de Qualité du Split")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #667eea;">80%</div>
            <div style="font-size: 1.1rem; color: #4a5568; margin-top: 0.5rem;">Dataset d'Entraînement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #764ba2;">20%</div>
            <div style="font-size: 1.1rem; color: #4a5568; margin-top: 0.5rem;">Dataset de Test</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #e91e63;">≤ 30%</div>
            <div style="font-size: 1.1rem; color: #4a5568; margin-top: 0.5rem;">Valeurs Manquantes Max</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <p style="margin:0; line-height:1.8;">
    <strong>⚠️ Gestion des valeurs manquantes :</strong><br>
    La proportion de valeurs manquantes par split est une métrique importante à surveiller, car imposer 
    une limite de 20 % (par exemple) de valeurs manquantes sur toute une série temporelle peut mener 
    à des problèmes dans les splits en aval. En effet, si la proportion de valeurs manquantes atteint 
    plus de 20 % pour le split le plus petit, le split de test, si tant est que la proportion de valeurs 
    manquantes est un peu en dessous de 20 % pour le split d'entraînement. Cette inhomogénéité a plus d'impact 
    sur le split de test est plus petite que celle du split d'entraînement et donc qu'une inhomogénéité a plus 
    d'impact sur le split de test.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧭 RELATIONS ENTRE LES VARIABLES</div>', unsafe_allow_html=True)

    relation_col1, relation_col2 = st.columns([1, 2])
    
    with relation_col1:
        st.markdown("""
        <div class="info-card">
        <h4 style="margin-top:0; color:#667eea;">Vue Réseau des Relations</h4>
        <p style="line-height:1.8; margin-bottom:1rem;">
        Ce graphe représente les <strong>relations logiques et calculées</strong> entre les différentes 
        variables utilisées dans le DataFrame.
        </p>
        <ul style="line-height:1.8; padding-left:1.5rem;">
            <li><code>fill_rate</code> dépend de <code>bikes</code> et <code>capacity</code></li>
            <li><code>capacity</code> est la somme de <code>bikes</code> et <code>docks</code></li>
            <li><code>renting</code> et <code>returning</code> conditionnent respectivement 
            <code>bikes</code> et <code>docks</code></li>
            <li><code>installed</code> est nécessaire pour que la station soit active</li>
            <li><code>lat/lon</code> + <code>arr</code> sont utilisés pour la géolocalisation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with relation_col2:
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


elif st.session_state.selected == "ANALYSE":

    st.sidebar.header("⚙️ Contrôles")
    if st.sidebar.button("🔄 Rafraîchir"):
        st.cache_data.clear()

    df_all = df
    arr_all = sorted(df_all["arr"].dropna().unique())
    with st.sidebar.expander("📍 Filtrer par arrondissement", expanded=False):
        arr_sel = st.multiselect(
            "Arrondissements",
            options=arr_all,
            default=[],
            placeholder="Tous les arrondissements",
        )
        st.caption("Sélection : **" + (", ".join(arr_sel) if arr_sel else "Tous") + "**")
    df = df_all[df_all["arr"].isin(arr_sel)] if arr_sel else df_all.copy()
    n_total = len(df)

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

    with col1:
        st.markdown(
            f"""
            <div class="dynamic-shadow"
                style="background:linear-gradient(135deg,#6e8efb,#a777e3);padding:15px;
                        display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
                <div>
                    <h4 style="color:white;margin:0;font-size:14px;">STATIONS VIDES</h4>
                    <p style="font-size:26px;color:white;margin:0;">{k_empty}</p>
                    <p style="font-size:14px;color:white;margin:0;">Quasi-vides : {p_almost_empty:.1f}%</p>
                </div>
                <div><i class="fas fa-bicycle" style="font-size:30px;color:white;"></i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="dynamic-shadow"
                style="background:linear-gradient(135deg,#6e8efb,#a777e3);padding:15px;
                        display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
                <div>
                    <h4 style="color:white;margin:0;font-size:14px;">STATIONS PLEINES</h4>
                    <p style="font-size:26px;color:white;margin:0;">{k_full}</p>
                    <p style="font-size:14px;color:white;margin:0;">Quasi-pleines : {p_almost_full:.1f}%</p>
                </div>
                <div><i class="fas fa-parking" style="font-size:30px;color:white;"></i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="dynamic-shadow"
                style="background:linear-gradient(135deg,#6e8efb,#a777e3);padding:15px;
                        display:flex;justify-content:space-between;align-items:center;border-radius:10px;">
                <div>
                    <h4 style="color:white;margin:0;font-size:14px;">STATIONS HORS SERVICE</h4>
                    <p style="font-size:26px;color:white;margin:0;">{k_hs}</p>
                    <p style="font-size:14px;color:white;margin:0;">Opérationnelles : {p_op:.1f}%</p>
                </div>
                <div><i class="fas fa-tools" style="font-size:30px;color:white;"></i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    mid_left, mid_right = st.columns([2, 2], gap="medium")

    with mid_left:
        with st.container(border=True):
            tab1, tab2, tab3 = st.tabs(["⚠️ Stations critiques", "📉 Peu sollicitées", "📈 Très sollicitées"])

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
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

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
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

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
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

    st.divider()

    with mid_right:
        with st.container(border=True):
            bt1, bt2, bt3 = st.tabs(["🏙️ Stations", "🔴 % Rupture", "📊 Taux moyen"])

            df_viz = df[~df["arr"].isin(["Paris"])].copy()
            chart_height = 500

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
            rupt = rupt[rupt["rupture"] > 0]
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

    bot_left, bot_right = st.columns([2, 2], gap="medium")

    with bot_left:
        with st.container(border=True):
            tab_geo, tab_rebal = st.tabs(["🗺️ Carte des stations", "🔁 Rééquilibrage entre stations"])

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
                            "text": "{name}\nVélos :{bikes}\nBornes :{docks}\nRempl. :{fill_rate:.0%}",
                        },
                    )
                )

            with tab_rebal:
                st.markdown("**Carte de rééquilibrage — flèches vert ➝ rouge**")

                stations_empty = df[(df["bikes"] == 0) & (~mask_hs)].copy()
                stations_donor = df[(df["bikes"] >= 5) & (~mask_hs)].copy()

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
                    rebalance_df["weight_scaled"] = rebalance_df["weight"].clip(1, 15) / 3

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

    st.markdown("---")
    st.caption("© Ville de Paris – Données temps réel. Dashboard Streamlit 2025.")




elif st.session_state.selected == "MODELES ET EVALUATION":
    
    st.markdown("""
    <style>
    .models-page {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .models-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .models-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .section-box {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .subsection-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4a5568;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .workflow-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid #e2e8f0;
    }
    
    .workflow-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.2);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .workflow-number {
        background: white;
        color: #667eea;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .workflow-arrow {
        text-align: center;
        font-size: 2rem;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.3);
    }
    
    .model-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.4rem;
    }
    
    .param-table {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
    }
    
    .comparison-box {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .approach-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
    }
    
    .info-text {
        color: #4a5568;
        font-size: 1rem;
        line-height: 1.8;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="models-header">
        <h1>🤖 MODÈLES & ÉVALUATION</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🎯 Choix des Modèles</div>
        <div class="info-text">
            Pour ce projet de prédiction de séries temporelles, nous avons sélectionné des modèles de 
            <strong>gradient boosting</strong> performants et adaptables aux séries temporelles.
        </div>
        <div class="highlight-box">
            <p style="margin:0; font-weight:600; color:#667eea; font-size:1.1rem;">Approche Globale & Multivariée</p>
            <p style="margin:0.5rem 0 0 0; color:#4a5568;">
            Modélisation de toutes les séries temporelles avec un seul modèle, en supposant qu'elles partagent 
            un mécanisme sous-jacent commun. Cette approche permet d'améliorer les prédictions pour chaque série 
            grâce aux informations de toutes les autres.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🔧 Deux Approches Complémentaires</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>📊 Approche 1 : Implémentation Personnalisée</h3>
            <ul style="line-height:2; margin:0;">
                <li>Feature engineering manuel détaillé</li>
                <li>XGBoost & LightGBM quantiles</li>
                <li>Prédiction par quantiles (5%, 50%, 95%)</li>
                <li>Intégration données météo</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>⚡ Approche 2 : Pipeline Optimisé</h3>
            <ul style="line-height:2; margin:0;">
                <li>Utilisation de Skforecast</li>
                <li>Optimisation avec Optuna</li>
                <li>Logging avec MLflow</li>
                <li>Validation croisée temporelle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🔍 Feature Engineering - Approche 1</div>
        <div class="subsection-title">Variables Créées</div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Temporelles", "⏱️ Lags & Rolling", "🌤️ Météo", "🏷️ Catégorielles"])
    
    with tab1:
        features_temporelles = pd.DataFrame([
            ["hour", "int", "Heure de la journée (0-23)"],
            ["dow", "int", "Jour de la semaine (0=lundi, 6=dimanche)"],
            ["is_weekend", "bool", "Si le jour est un week-end"],
            ["is_holiday", "bool", "Si c'est un jour férié (France)"],
            ["is_school_holiday", "bool", "Si c'est les vacances scolaires (zone C)"],
            ["is_bridge_day", "bool", "S'il s'agit d'un pont"],
            ["hour_sin, hour_cos", "float", "Encodage cyclique de l'heure"],
            ["day_sin, day_cos", "float", "Encodage cyclique du jour de l'année"]
        ], columns=["Variable", "Type", "Description"])
        st.dataframe(features_temporelles, use_container_width=True, hide_index=True)
    
    with tab2:
        features_lags = pd.DataFrame([
            ["lag_1h", "float", "Vélos disponibles 1h avant"],
            ["lag_2h", "float", "Vélos disponibles 2h avant"],
            ["lag_24h", "float", "Vélos disponibles 24h avant"],
            ["lag_48h", "float", "Vélos disponibles 48h avant"],
            ["rolling_mean_6h", "float", "Moyenne mobile sur 6h"],
            ["rolling_mean_24h", "float", "Moyenne mobile sur 24h"],
            ["rolling_std_24h", "float", "Écart-type mobile sur 24h"]
        ], columns=["Variable", "Type", "Description"])
        st.dataframe(features_lags, use_container_width=True, hide_index=True)
    
    with tab3:
        features_meteo = pd.DataFrame([
            ["temperature_2m", "float", "Température en °C"],
            ["precipitation", "float", "Précipitations (mm)"],
            ["is_rain", "bool", "Si precipitation > 0"],
            ["is_heavy_rain", "bool", "Si precipitation > 1.0"],
            ["is_dry", "bool", "Si precipitation == 0"],
            ["is_cold", "bool", "Si température < 10°C"],
            ["is_hot", "bool", "Si température > 25°C"],
            ["is_mild", "bool", "Si température entre 10 et 25°C"]
        ], columns=["Variable", "Type", "Description"])
        st.dataframe(features_meteo, use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("""
        <div class="info-text">
            <strong>stationcode :</strong> Identifiant de la station (traité comme catégorie)<br>
            <strong>target_numbikes :</strong> Nombre de vélos disponibles à t+1h (variable cible)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🔄 Pipeline de Preprocessing - Approche 2</div>
        <div class="workflow-box">
            <div class="workflow-step">
                <div class="workflow-number">1</div>
                <div>Encodage des variables catégorielles (Ordinal Encoding)</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">2</div>
                <div>Extraction des valeurs calendaires (heure, jour, mois)</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">3</div>
                <div>Transformation en features cycliques (sin/cos)</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">4</div>
                <div>Transformation Skforecast en matrices de prédiction</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">5</div>
                <div>Passage au modèle (XGBoost / LightGBM)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">⚙️ Paramètres des Modèles</div>
    """, unsafe_allow_html=True)
    
    tab_lgbm, tab_xgb_q, tab_xgb_r = st.tabs(["LightGBM Quantile", "XGBoost Quantile", "XGBoost Classique"])
    
    with tab_lgbm:
        st.markdown('<div class="subsection-title">LightGBM - Régression Quantile</div>', unsafe_allow_html=True)
        params_lgbm = pd.DataFrame([
            ["objective", "quantile", "Optimisation par quantile"],
            ["alpha", "0.05 / 0.5 / 0.95", "Quantiles ciblés"],
            ["n_estimators", "2000", "Nombre d'arbres"],
            ["learning_rate", "0.01", "Vitesse d'apprentissage"],
            ["num_leaves", "64", "Feuilles par arbre"],
            ["min_data_in_leaf", "10", "Observations min par feuille"],
            ["max_depth", "10", "Profondeur max"],
            ["subsample", "0.9", "Fraction échantillons"],
            ["colsample_bytree", "0.9", "Fraction variables"]
        ], columns=["Paramètre", "Valeur", "Description"])
        st.dataframe(params_lgbm, use_container_width=True, hide_index=True)
    
    with tab_xgb_q:
        st.markdown('<div class="subsection-title">XGBoost - Régression Quantile</div>', unsafe_allow_html=True)
        params_xgb_q = pd.DataFrame([
            ["objective", "QuantileObjective", "Loss quantile personnalisée"],
            ["alpha", "0.05 / 0.5 / 0.95", "Quantiles ciblés"],
            ["n_estimators", "2000", "Nombre d'arbres"],
            ["learning_rate", "0.01", "Vitesse d'apprentissage"],
            ["max_depth", "8", "Profondeur max"],
            ["subsample", "0.8", "Fraction échantillons"],
            ["colsample_bytree", "0.8", "Fraction variables"],
            ["random_state", "42", "Reproductibilité"]
        ], columns=["Paramètre", "Valeur", "Description"])
        st.dataframe(params_xgb_q, use_container_width=True, hide_index=True)
    
    with tab_xgb_r:
        st.markdown('<div class="subsection-title">XGBoost - Régression Classique</div>', unsafe_allow_html=True)
        params_xgb_r = pd.DataFrame([
            ["objective", "reg:squarederror", "Régression standard"],
            ["n_estimators", "2000", "Nombre d'arbres"],
            ["learning_rate", "0.01", "Vitesse d'apprentissage"],
            ["max_depth", "8", "Profondeur max"],
            ["subsample", "0.8", "Fraction échantillons"],
            ["colsample_bytree", "0.8", "Fraction variables"],
            ["random_state", "42", "Reproductibilité"]
        ], columns=["Paramètre", "Valeur", "Description"])
        st.dataframe(params_xgb_r, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">📊 Résultats d'Évaluation - Approche 1</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">LightGBM Quantile 50%</div>
            <div class="metric-value">3.04</div>
            <div class="metric-label">MAE</div>
            <div style="font-size:0.9rem; margin-top:0.5rem;">923 442 échantillons</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">XGBoost Classique</div>
            <div class="metric-value">2.57</div>
            <div class="metric-label">MAE</div>
            <div style="font-size:0.9rem; margin-top:0.5rem;">923 442 échantillons</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">XGBoost Quantile</div>
            <div class="metric-value">13-34</div>
            <div class="metric-label">MAE (range)</div>
            <div style="font-size:0.9rem; margin-top:0.5rem;">Performances limitées</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="highlight-box" style="margin-top:2rem;">
            <p style="margin:0; font-weight:600; color:#667eea; font-size:1.1rem;">✅ Meilleur Modèle : XGBoost Classique</p>
            <p style="margin:0.5rem 0 0 0; color:#4a5568;">
            Avec une MAE de 2.57, le modèle XGBoost classique offre la meilleure précision générale. 
            LightGBM quantile reste très compétitif avec une MAE de 3.04.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🎯 Méthodologie d'Évaluation - Approche 2</div>
        <div class="workflow-box">
            <div class="workflow-step">
                <div class="workflow-number">1</div>
                <div>Split Train/Test (80% / 20%)</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">2</div>
                <div>Optimisation bayésienne avec Optuna (30 runs)</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">3</div>
                <div>Validation croisée temporelle (3 folds)</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">4</div>
                <div>Évaluation : RMSE, MAE, MAPE</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">5</div>
                <div>Logging des résultats avec MLflow</div>
            </div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-step">
                <div class="workflow-number">6</div>
                <div>Ré-entraînement du meilleur modèle</div>
            </div>
        </div>
        <div class="highlight-box">
            <p style="margin:0; color:#4a5568; font-size:0.95rem;">
            <strong>📊 Horizon de prédiction :</strong> 24 heures<br>
            <strong>🔄 Total d'instances entraînées :</strong> 90 (30 runs × 3 folds)<br>
            <strong>⏱️ Temps d'exécution XGBoost :</strong> 47 minutes<br>
            <strong>⏱️ Temps d'exécution LightGBM :</strong> 12 minutes
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🏆 Meilleurs Hyper-paramètres - Approche 2</div>
    """, unsafe_allow_html=True)
    
    best_params = pd.DataFrame([
        ["Nombre d'estimateurs", "650", "700"],
        ["Profondeur max", "10", "7"],
        ["Taux d'apprentissage", "0.24", "0.79"],
        ["Fraction features", "0.61", "0.82"],
        ["Régularisation L2", "6.9", "8.9"],
        ["Régularisation L1", "7.2", "7.1"],
        ["Nombre de lags", "1", "1"]
    ], columns=["Hyper-paramètre", "XGBoost", "LightGBM"])
    
    st.dataframe(best_params, use_container_width=True, hide_index=True)
    
    st.markdown("""
        <div class="highlight-box">
            <p style="margin:0; font-weight:600; color:#667eea; font-size:1.1rem;">🔍 Observation Clé</p>
            <p style="margin:0.5rem 0 0 0; color:#4a5568;">
            Le nombre optimal de lags converge vers 1 pour les deux modèles, suggérant que les séries 
            sont globalement stationnaires avec une périodicité principale de 24h.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">📈 Performances des Meilleurs Modèles</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-title">XGBoost</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">MAE Médiane</div>
                <div class="metric-value">2.3</div>
                <div class="metric-label">vélos</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RMSE Médiane</div>
                <div class="metric-value">2.0</div>
                <div class="metric-label">vélos</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAPE Médiane</div>
                <div class="metric-value">50%</div>
                <div class="metric-label">erreur relative</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-title">LightGBM</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">MAE Médiane</div>
                <div class="metric-value">2.1</div>
                <div class="metric-label">vélos</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RMSE Médiane</div>
                <div class="metric-value">1.9</div>
                <div class="metric-label">vélos</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MAPE Médiane</div>
                <div class="metric-value">52%</div>
                <div class="metric-label">erreur relative</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="highlight-box" style="margin-top:2rem;">
            <p style="margin:0; font-weight:600; color:#667eea; font-size:1.1rem;">⚡ Avantage LightGBM</p>
            <p style="margin:0.5rem 0 0 0; color:#4a5568;">
            LightGBM présente des performances légèrement supérieures avec un temps d'exécution 4× plus rapide, 
            le rendant idéal pour l'optimisation itérative et le déploiement en production.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">🎨 Importance des Features</div>
    """, unsafe_allow_html=True)
    
    st.info("📊 **Images des feature importance à insérer ici**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <strong style="color:#667eea;">XGBoost</strong><br>
            <span class="approach-badge">Lag 1 : ~80%</span>
            <span class="approach-badge">Longitude : ~10%</span>
            <span class="approach-badge">Autres : <5%</span>
            <p style="margin-top:1rem; color:#4a5568; font-size:0.95rem;">
            XGBoost s'appuie massivement sur le premier lag, suggérant une dépendance forte 
            à la valeur précédente.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <strong style="color:#667eea;">LightGBM</strong><br>
            <span class="approach-badge">Lag 1 : Important</span>
            <span class="approach-badge">Lat/Lon : Important</span>
            <span class="approach-badge">Heure : Important</span>
            <span class="approach-badge">Station : Important</span>
            <p style="margin-top:1rem; color:#4a5568; font-size:0.95rem;">
            LightGBM exploite davantage les features exogènes (localisation, temporalité), 
            offrant une interprétabilité supérieure.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">📸 Prédictions Visuelles</div>
        <div class="info-text">
            Exemples de prédictions sur 48h pour différentes stations avec intervalles de confiance.
        </div>
    """, unsafe_allow_html=True)
    
    st.info("📊 **Image 1 : Prédiction Station 8007 (LGBM) - À insérer**")
    st.info("📊 **Image 2 : Prédictions Stations 14023 & 23105 (LGBM) - À insérer**")
    st.info("📊 **Image 3 : Observations sur les prédictions - À insérer**")
    st.info("📊 **Image 4 : Prédictions Stations 14023 & 23105 (XGB Quantile) - À insérer**")
    
    st.markdown("""
        <div class="highlight-box">
            <p style="margin:0; font-weight:600; color:#667eea; font-size:1.1rem;">🔍 Observations</p>
            <p style="margin:0.5rem 0 0 0; color:#4a5568; line-height:1.8;">
            • <strong>LightGBM :</strong> Capture bien les pics matinaux (7-8h) et vespéraux (16h), 
            avec des intervalles de confiance raisonnables<br>
            • <strong>XGBoost Quantile :</strong> Prédictions bruitées et instables, intervalles trop larges<br>
            • <strong>XGBoost Classique :</strong> Prédictions nettes avec bonne capture du rythme journalier
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">✅ Conclusions & Recommandations</div>
        <div class="workflow-box">
            <div style="background:white; border-radius:10px; padding:1.5rem; margin:1rem 0;">
                <h4 style="color:#667eea; margin:0 0 1rem 0;">🏆 Modèles Retenus</h4>
                <div style="color:#4a5568; line-height:1.8;">
                    <strong>1. LightGBM Quantile (Médiane)</strong><br>
                    • Meilleur compromis performance/vitesse<br>
                    • Intervalles de prédiction fiables<br>
                    • Temps d'entraînement optimal<br><br>
                    
                    <strong>2. XGBoost Classique</strong><br>
                    • Meilleure MAE absolue (2.57)<br>
                    • Prédictions robustes et stables<br>
                    • Alternative crédible
                </div>
            </div>
            
            <div style="background:white; border-radius:10px; padding:1.5rem; margin:1rem 0;">
                <h4 style="color:#667eea; margin:0 0 1rem 0;">📋 Axes d'Amélioration</h4>
                <div style="color:#4a5568; line-height:1.8;">
                    • Intégration complète des variables météo dans le pipeline optimisé<br>
                    • Analyse des stations à forte erreur de prédiction<br>
                    • Tests avec périodes de lags supérieures (2-7 jours)<br>
                    • Enrichissement des features calendaires (événements spéciaux)<br>
                    • Modélisation différenciée par type de station (résidentielle, professionnelle)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🤖 Modélisation & Évaluation — Projet Vélib' 2025")
