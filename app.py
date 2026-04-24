"""
INF 232 EC2 — Analyse de données de Santé Mentale
Secteur : Santé Mentale & Intelligence Émotionnelle
Version : Optimal & Robust AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, r2_score, mean_squared_error,
)
from supabase import create_client, Client

# ══════════════════════════════════════════════════════════════════════════════
#  1. CONFIG & DESIGN (THE ORIGINAL "POWER" LOOK)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="INF 232 EC2 | Dashboard Analytique",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8f9fa; color: #1a1a2e; }
.page-header { background-color: #ffffff; border: 1px solid #dde1e7; border-left: 5px solid #1a73e8; border-radius: 6px; padding: 1.6rem 2rem; margin-bottom: 1.8rem; }
.page-header h1 { font-family: 'Source Serif 4', serif; font-size: 1.75rem; font-weight: 700; color: #1a1a2e; margin: 0; }
.metric-card { background: #ffffff; border: 1px solid #dde1e7; border-top: 3px solid #1a73e8; border-radius: 6px; padding: 1.2rem; text-align: center; }
.metric-card .val { font-family: 'Source Serif 4', serif; font-size: 2rem; font-weight: 700; color: #1a73e8; }
.metric-card .lbl { font-size: .72rem; color: #7a8694; text-transform: uppercase; letter-spacing: .08em; margin-top: .2rem; }
.section-title { font-family: 'Source Serif 4', serif; font-size: 1.2rem; font-weight: 600; color: #1a1a2e; border-bottom: 2px solid #1a73e8; padding-bottom: .3rem; margin: 2rem 0 1rem 0; }
.info-box { background: #eaf2fb; border: 1px solid #b3d1f5; border-left: 4px solid #1a73e8; border-radius: 5px; padding: 1rem; color: #1a3a5c; font-size: .88rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  2. GLOBAL THEME & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=12),
)
CLR_MAIN = "#1a73e8"
CLR_ALT  = "#34a853"
FEATURES = ["sleep_hours", "stress", "social_interaction"]
TARGET   = "mood_score"

def robust_plot(fig):
    """Applies theme correctly to prevent TypeError."""
    fig.update_layout(**PLOTLY_THEME)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  3. SUPABASE & DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["connections"]["supabase"]["SUPABASE_URL"], st.secrets["connections"]["supabase"]["SUPABASE_KEY"])

def add_labels(df):
    df = df.copy()
    # Logic: Healthy = Mood >= 60 AND Stress <= 5 AND Sleep >= 7
    df["health_status"] = np.where((df[TARGET] >= 60) & (df["stress"] <= 5) & (df["sleep_hours"] >= 7), "Healthy", "At Risk")
    return df

@st.cache_data(ttl=10)
def load_data():
    try:
        data = get_supabase().table("mental_health_data").select("*").execute()
        df = pd.DataFrame(data.data)
        if df.empty: return pd.DataFrame()
        for col in FEATURES + [TARGET]: df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=FEATURES + [TARGET])
    except: return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
#  4. SIDEBAR INPUT (ORIGINAL FLOW)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📋 Formulaire d'entrée")
    with st.form("input_form", clear_on_submit=True):
        name = st.text_input("Nom Complet")
        mat  = st.text_input("Matricule")
        slp  = st.slider("Sommeil", 2.0, 12.0, 7.0, 0.5)
        strss = st.slider("Stress", 1, 10, 5)
        soc  = st.slider("Interaction", 1, 10, 5)
        md   = st.slider("Humeur", 0, 100, 60)
        if st.form_submit_button("💾 Enregistrer", use_container_width=True):
            if name and mat:
                get_supabase().table("mental_health_data").insert({"name":name, "matricule":mat, "sleep_hours":slp, "stress":strss, "social_interaction":soc, "mood_score":md}).execute()
                st.success("Données synchronisées !")
                st.cache_data.clear()

# ══════════════════════════════════════════════════════════════════════════════
#  5. MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
df_raw = load_data()

if df_raw.empty:
    st.info("👋 Bienvenue ! Veuillez ajouter une entrée dans le formulaire pour générer l'analyse.")
    st.stop()

df = add_labels(df_raw)

st.markdown('<div class="page-header"><h1>🧠 Analyse Avancée : Santé Mentale & Bien-être</h1><p>INF 232 EC2 — Système Expert Analytique</p></div>', unsafe_allow_html=True)

# KPI Section
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><div class="val">{len(df)}</div><div class="lbl">Effectif Total</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="val">{df[TARGET].mean():.1f}</div><div class="lbl">Humeur Moy.</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="val">{df["health_status"].value_counts().get("Healthy",0)}</div><div class="lbl">Profils Sains</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="val">{df["sleep_hours"].mean():.1f}h</div><div class="lbl">Repos Moyen</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Aperçu", "📈 Régression", "🔷 PCA", "🌲 Classification", "🔵 Clustering"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — APERÇU (FIXED HISTOGRAMS)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        if df[TARGET].nunique() > 1:
            fig = px.histogram(df, x=TARGET, title="Distribution de l'Humeur", color_discrete_sequence=[CLR_MAIN])
            st.plotly_chart(robust_plot(fig), use_container_width=True)
        else:
            st.info("Données d'humeur identiques : Histogramme indisponible.")
    with col_b:
        if df["sleep_hours"].nunique() > 1:
            fig = px.histogram(df, x="sleep_hours", title="Distribution du Sommeil", color_discrete_sequence=[CLR_ALT])
            st.plotly_chart(robust_plot(fig), use_container_width=True)
        else:
            st.info("Données de sommeil identiques : Histogramme indisponible.")
    
    st.markdown('<div class="section-title">Registre des Participants</div>', unsafe_allow_html=True)
    st.dataframe(df[["name", "sleep_hours", "stress", "social_interaction", "mood_score", "health_status"]], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RÉGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if len(df) > 2 and df["sleep_hours"].nunique() > 1:
        st.markdown('<div class="section-title">Régression Linéaire : Sommeil vs Humeur</div>', unsafe_allow_html=True)
        X = df[["sleep_hours"]].values
        y = df[TARGET].values
        model = LinearRegression().fit(X, y)
        
        fig = px.scatter(df, x="sleep_hours", y=TARGET, trendline="ols", trendline_color_override="red", title="Corrélation Sommeil / Humeur")
        st.plotly_chart(robust_plot(fig), use_container_width=True)
        
        ca, cb = st.columns(2)
        ca.metric("Indice de Confiance (R²)", f"{model.score(X, y):.3f}")
        cb.metric("Coefficient de Pente", f"{model.coef_[0]:.2f}")
    else:
        st.warning("Data insuffisante pour la régression (minimum 3 points variés requis).")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PCA (2D PROJECTION)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(df) >= 3:
        st.markdown('<div class="section-title">Réduction de Dimensionnalité (PCA)</div>', unsafe_allow_html=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[FEATURES])
        
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
        pca_df["Status"] = df["health_status"].values
        pca_df["Nom"] = df["name"].values
        
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Status", hover_name="Nom", symbol="Status", title="Projection des profils en 2 dimensions")
        st.plotly_chart(robust_plot(fig), use_container_width=True)
    else:
        st.info("L'analyse PCA nécessite au moins 3 participants.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION (RANDOM FOREST)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Classification par Intelligence Artificielle</div>', unsafe_allow_html=True)
    
    # ERROR FIX: Check if we have at least one of EACH class
    if len(df) >= 5 and df["health_status"].nunique() > 1:
        X = df[FEATURES]
        y = df["health_status"]
        
        # Robust AI: If dataset < 20, use all data for visualization to avoid empty split
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        importances = pd.DataFrame({'Variable': FEATURES, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
        
        fig = px.bar(importances, x='Variable', y='Importance', color='Importance', title="Quels facteurs prédisent le mieux la santé mentale ?")
        st.plotly_chart(robust_plot(fig), use_container_width=True)
        
        st.markdown(f"""
        <div class="info-box">
        💡 <b>Insight :</b> La variable <b>{importances.iloc[0]['Variable']}</b> est actuellement le prédicteur le plus influent de l'état "Sain" dans votre groupe.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Pour activer l'IA, ajoutez au moins un profil 'Healthy' (Sommeil élevé, Stress bas). Actuellement, tous vos profils sont dans la même catégorie.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — CLUSTERING (K-MEANS)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Segmentation Automatique (Profiles de Groupes)</div>', unsafe_allow_html=True)
    
    if len(df) >= 3:
        # Robust K selection: If very small data, use 2 clusters max
        k_val = 2 if len(df) < 6 else 3
        
        scaler = StandardScaler()
        X_clust = scaler.fit_transform(df[FEATURES + [TARGET]])
        
        km = KMeans(n_clusters=k_val, n_init=10, random_state=42).fit(X_clust)
        df["Cluster"] = [f"Groupe {i+1}" for i in km.labels_]
        
        fig = px.scatter(df, x="stress", y=TARGET, color="Cluster", size="sleep_hours", title=f"Segmentation des profils (K={k_val})", labels={"stress": "Niveau de Stress", "mood_score": "Score d'Humeur"})
        st.plotly_chart(robust_plot(fig), use_container_width=True)
        
        st.markdown('<div class="section-title">Analyse Moyenne par Groupe</div>', unsafe_allow_html=True)
        summary = df.groupby("Cluster")[FEATURES + [TARGET]].mean().reset_index()
        st.dataframe(summary.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
    else:
        st.info("Le clustering nécessite un minimum de 3 participants pour former des groupes.")

st.divider()
st.markdown("<p style='text-align:center;color:#9aa5b1;font-size:.78rem;'>INF 232 EC2 · Analytics Engine · 2026</p>", unsafe_allow_html=True)
