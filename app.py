"""
INF 232 EC2 — Analyse de données
Secteur : Santé Mentale & Intelligence Émotionnelle
Auteur  : [Votre Nom]
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, r2_score, mean_squared_error
)

from st_supabase_connection import SupabaseConnection

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindMetrics | INF 232",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES — clean academic light theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fa;
    color: #1a1a2e;
}

/* ── Sidebar — clean light gray ── */
section[data-testid="stSidebar"] {
    background-color: #f0f2f5 !important;
    border-right: 1px solid #dde1e7;
}
section[data-testid="stSidebar"] * {
    color: #2c3e50 !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #c8cdd5 !important;
    border-radius: 6px !important;
    color: #2c3e50 !important;
}

/* ── Main area ── */
.main .block-container {
    padding-top: 2rem;
    background-color: #f8f9fa;
}

/* ── Page header banner ── */
.page-header {
    background-color: #ffffff;
    border: 1px solid #dde1e7;
    border-left: 5px solid #1a73e8;
    border-radius: 6px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
}
.page-header h1 {
    font-family: 'Source Serif 4', serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0 0 .3rem 0;
}
.page-header p {
    color: #5f6b7a;
    font-size: .88rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: .01em;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.12rem;
    font-weight: 600;
    color: #1a1a2e;
    border-bottom: 2px solid #1a73e8;
    padding-bottom: .35rem;
    margin: 2rem 0 1rem 0;
}

/* ── KPI cards ── */
.metric-grid { display: flex; gap: .9rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #ffffff;
    border: 1px solid #dde1e7;
    border-top: 3px solid #1a73e8;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Source Serif 4', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a73e8;
}
.metric-card .lbl {
    font-size: .72rem;
    color: #7a8694;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .2rem;
}

/* ── Info / interpretation boxes ── */
.info-box {
    background: #eaf2fb;
    border: 1px solid #b3d1f5;
    border-left: 4px solid #1a73e8;
    border-radius: 5px;
    padding: .85rem 1.1rem;
    color: #1a3a5c;
    font-size: .88rem;
    margin: .8rem 0 1rem 0;
    line-height: 1.6;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border: 1px solid #dde1e7;
    border-radius: 6px;
    padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    color: #5f6b7a;
    font-family: 'Inter', sans-serif;
    font-size: .87rem;
}
.stTabs [aria-selected="true"] {
    background: #1a73e8 !important;
    color: #ffffff !important;
    border-radius: 4px;
}

/* ── Button ── */
.stButton > button {
    background-color: #1a73e8;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    padding: .5rem 1.3rem;
    transition: background-color .2s;
}
.stButton > button:hover { background-color: #1558b0; }

/* ── DataFrames ── */
div[data-testid="stDataFrame"] {
    border-radius: 6px;
    border: 1px solid #dde1e7;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Source Serif 4', serif;
    color: #1a1a2e;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME  — professional white
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#2c3e50", size=12),
)

# Colour palette — academic blues / viridis
CLR_HEALTHY   = "#1a73e8"   # blue  — Healthy
CLR_AT_RISK   = "#d93025"   # red   — At Risk
CLR_MAIN      = "#1a73e8"
CLR_SECONDARY = "#34a853"
SEQ_BLUES     = "Blues"
SEQ_VIRIDIS   = "Viridis"
QUAL_PALETTE  = ["#1a73e8", "#34a853", "#ea4335", "#fbbc04", "#4285f4", "#ff6d00"]

# ─────────────────────────────────────────────
#  SUPABASE CONNECTION
# ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return st.connection("supabase", type=SupabaseConnection)

TABLE = "mental_health_data"   # ← matches your Supabase table exactly

def save_record(conn, record: dict):
    try:
        conn.table(TABLE).insert(record).execute()
        return True, "✅ Données sauvegardées avec succès !"
    except Exception as e:
        return False, f"❌ Erreur Supabase : {e}"

@st.cache_data(ttl=30)
def load_data(_conn):
    try:
        resp = _conn.table(TABLE).select("*").execute()
        df = pd.DataFrame(resp.data)
        if df.empty:
            return pd.DataFrame()
        for col in ["sleep_hours", "stress", "social_interaction", "mood_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["sleep_hours", "stress", "social_interaction", "mood_score"])
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  DEMO DATA
# ─────────────────────────────────────────────
@st.cache_data
def generate_demo_data(n=120):
    rng = np.random.default_rng(42)
    sleep  = rng.normal(6.5, 1.4, n).clip(2, 10)
    stress = rng.normal(5.5, 2.0, n).clip(1, 10)
    social = rng.normal(5.0, 2.2, n).clip(1, 10)
    mood   = (4.5*sleep - 3.8*stress + 2.2*social + rng.normal(0, 6, n) + 30).clip(0, 100)
    return pd.DataFrame({
        "name":               [f"Étudiant_{i+1:03d}" for i in range(n)],
        "matricule":          [f"20{rng.integers(20,26):02d}{rng.integers(1000,9999)}" for _ in range(n)],
        "sleep_hours":        sleep.round(1),
        "stress":             stress.round(1),
        "social_interaction": social.round(1),
        "mood_score":         mood.round(1),
    })

# ─────────────────────────────────────────────
#  CONSTANTS & HELPERS
# ─────────────────────────────────────────────
FEATURES = ["sleep_hours", "stress", "social_interaction"]
TARGET   = "mood_score"

def add_health_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["health_status"] = np.where(
        (df["mood_score"] >= 55) & (df["stress"] <= 6) & (df["sleep_hours"] >= 6),
        "Healthy", "At Risk"
    )
    return df

# ─────────────────────────────────────────────
#  ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def run_simple_regression(df):
    X = df[["sleep_hours"]].values
    y = df[TARGET].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred

def run_multiple_regression(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    coef = dict(zip(FEATURES, model.coef_))
    return model, r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), y_pred, coef

def run_pca(df):
    X_scaled = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"])
    return components, pca.explained_variance_ratio_, loadings

def run_random_forest(df):
    df = add_health_label(df)
    X = df[FEATURES].values
    y = df["health_status"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True)
    cm_    = confusion_matrix(y_te, y_pred, labels=["Healthy", "At Risk"])
    return clf, report, cm_, dict(zip(FEATURES, clf.feature_importances_)), y_te, y_pred

def run_kmeans(df, k=3):
    X_scaled = StandardScaler().fit_transform(df[FEATURES + [TARGET]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X_scaled), km

# ─────────────────────────────────────────────
#  SIDEBAR — DATA COLLECTION FORM
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 MindMetrics")
    st.markdown("<p style='color:#7a8694;font-size:.82rem;margin-top:-.4rem;'>INF 232 · Analyse de données</p>",
                unsafe_allow_html=True)
    st.divider()
    st.markdown("**📋 Saisie des données**")

    with st.form("data_form", clear_on_submit=True):
        name        = st.text_input("Nom complet",  placeholder="ex. Dupont Marie")
        matricule   = st.text_input("Matricule",    placeholder="ex. 202312345")
        sleep_hours = st.slider("🛌 Heures de sommeil", 2.0, 12.0, 7.0, 0.5)
        stress      = st.slider("😰 Stress (1–10)", 1, 10, 5)
        social      = st.slider("🤝 Interaction sociale (1–10)", 1, 10, 5)
        mood        = st.slider("😊 Score d'humeur (0–100)", 0, 100, 60)
        submitted   = st.form_submit_button("💾 Enregistrer", use_container_width=True)

    if submitted:
        if not name.strip() or not matricule.strip():
            st.warning("Veuillez remplir le Nom et le Matricule.")
        else:
            try:
                conn = get_conn()
                ok, msg = save_record(conn, {
                    "name": name.strip(), "matricule": matricule.strip(),
                    "sleep_hours": sleep_hours, "stress": stress,
                    "social_interaction": social, "mood_score": mood,
                })
                (st.success if ok else st.error)(msg)
                if ok:
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"Connexion impossible : {e}")

    st.divider()
    st.caption("Données stockées dans Supabase — table : mental_health_data")

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
try:
    conn = get_conn()
    df_raw     = load_data(conn)
    using_demo = df_raw.empty
except Exception:
    df_raw     = pd.DataFrame()
    using_demo = True

if using_demo:
    df_raw = generate_demo_data()
    st.info("ℹ️ Aucune connexion Supabase détectée — **données de démonstration** utilisées (120 enregistrements simulés).")

df = add_health_label(df_raw.copy())

# ─────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>🧠 MindMetrics — Tableau de Bord Analytique</h1>
  <p>INF 232 EC2 · Analyse de données · Santé Mentale &amp; Intelligence Émotionnelle</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────
n_total    = len(df)
n_healthy  = (df["health_status"] == "Healthy").sum()
avg_mood   = df["mood_score"].mean()
avg_sleep  = df["sleep_hours"].mean()
avg_stress = df["stress"].mean()

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="val">{n_total}</div><div class="lbl">Participants</div></div>
  <div class="metric-card"><div class="val">{avg_mood:.1f}</div><div class="lbl">Humeur moy.</div></div>
  <div class="metric-card"><div class="val">{avg_sleep:.1f}h</div><div class="lbl">Sommeil moy.</div></div>
  <div class="metric-card"><div class="val">{avg_stress:.1f}</div><div class="lbl">Stress moy.</div></div>
  <div class="metric-card"><div class="val">{n_healthy}/{n_total}</div><div class="lbl">Profil Sain</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Aperçu",
    "📈 Régression",
    "🔷 ACP (PCA)",
    "🌲 Classification",
    "🔵 Clustering",
])

# ══════════════════════════════════════════════
#  TAB 1 — APERÇU
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Distribution des variables</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="mood_score", nbins=25,
                           color_discrete_sequence=[CLR_MAIN],
                           title="Distribution du Score d'Humeur",
                           labels={"mood_score": "Score d'humeur"}, **PLOTLY_THEME)
        fig.update_traces(opacity=0.80, marker_line_color="white", marker_line_width=0.5)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="sleep_hours", nbins=20,
                           color_discrete_sequence=[CLR_SECONDARY],
                           title="Distribution des Heures de Sommeil",
                           labels={"sleep_hours": "Heures de sommeil"}, **PLOTLY_THEME)
        fig.update_traces(opacity=0.80, marker_line_color="white", marker_line_width=0.5)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.pie(df, names="health_status",
                     color="health_status",
                     color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
                     title="Répartition Sain / À Risque", hole=0.42, **PLOTLY_THEME)
        fig.update_traces(textinfo="percent+label", pull=[0.03, 0.03])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        corr = df[FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f",
                        color_continuous_scale=SEQ_BLUES,
                        title="Matrice de Corrélation", zmin=-1, zmax=1, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Scatter Matrix — Relations entre variables</div>',
                unsafe_allow_html=True)
    fig = px.scatter_matrix(df, dimensions=FEATURES + [TARGET],
                            color="health_status",
                            color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
                            title="Matrice de dispersion (toutes variables)", **PLOTLY_THEME)
    fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.55))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Données brutes</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["name", "matricule", "sleep_hours", "stress",
            "social_interaction", "mood_score", "health_status"]],
        use_container_width=True, height=300,
    )

# ══════════════════════════════════════════════
#  TAB 2 — RÉGRESSION
# ══════════════════════════════════════════════
with tab2:
    # ── Simple ──────────────────────────────
    st.markdown('<div class="section-title">Régression Linéaire Simple — Sommeil → Humeur</div>',
                unsafe_allow_html=True)

    model_s, r2_s, rmse_s, ypred_s = run_simple_regression(df)
    x_line = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 200)
    y_line = model_s.predict(x_line.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sleep_hours"], y=df[TARGET], mode="markers",
        marker=dict(color=df[TARGET], colorscale=SEQ_VIRIDIS, size=6, opacity=0.65,
                    showscale=True, colorbar=dict(title="Humeur", thickness=14)),
        name="Observations",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color=CLR_AT_RISK, width=2.5),
        name=f"Droite de régression (R²={r2_s:.3f})",
    ))
    fig.update_layout(title="Heures de Sommeil vs Score d'Humeur",
                      xaxis_title="Heures de sommeil", yaxis_title="Score d'humeur",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_s:.4f}")
    c2.metric("RMSE", f"{rmse_s:.2f}")
    c3.metric("Coefficient (pente)", f"{model_s.coef_[0]:.3f}")
    st.markdown(
        '<div class="info-box">📌 <b>Interprétation :</b> Chaque heure de sommeil supplémentaire est associée '
        f"à une variation de <b>{model_s.coef_[0]:.2f} points</b> du score d'humeur. "
        f"Le modèle explique <b>{r2_s*100:.1f}%</b> de la variance (R²={r2_s:.4f}).</div>",
        unsafe_allow_html=True,
    )

    # ── Multiple ─────────────────────────────
    st.markdown('<div class="section-title">Régression Linéaire Multiple — Tous les facteurs → Humeur</div>',
                unsafe_allow_html=True)

    model_m, r2_m, rmse_m, ypred_m, coef_m = run_multiple_regression(df)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df[TARGET], y=ypred_m, mode="markers",
        marker=dict(color=CLR_MAIN, opacity=0.55, size=6),
        name="Prédit vs Réel",
    ))
    lims = [df[TARGET].min(), df[TARGET].max()]
    fig2.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                              line=dict(color=CLR_AT_RISK, dash="dash", width=1.5),
                              name="Ligne idéale (y = x)"))
    fig2.update_layout(title="Valeurs Réelles vs Valeurs Prédites (Régression Multiple)",
                       xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       **PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        c1, c2 = st.columns(2)
        c1.metric("R² multiple", f"{r2_m:.4f}")
        c2.metric("RMSE", f"{rmse_m:.2f}")
        st.markdown("**Tableau des coefficients**")
        coef_df = pd.DataFrame({
            "Variable":    list(coef_m.keys()),
            "Coefficient": [round(v, 4) for v in coef_m.values()],
            "Direction":   ["↑ Positif" if v > 0 else "↓ Négatif" for v in coef_m.values()],
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    with col_b:
        fig3 = px.bar(x=list(coef_m.keys()), y=list(coef_m.values()),
                      color=list(coef_m.values()),
                      color_continuous_scale="RdBu",
                      labels={"x": "Variable", "y": "Coefficient"},
                      title="Poids des variables explicatives", **PLOTLY_THEME)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Résidus ──────────────────────────────
    st.markdown('<div class="section-title">Analyse des Résidus</div>', unsafe_allow_html=True)
    residuals = df[TARGET].values - ypred_m
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=ypred_m, y=residuals, mode="markers",
        marker=dict(color=residuals, colorscale="RdBu", size=5, opacity=0.65,
                    colorbar=dict(title="Résidu", thickness=14)),
        name="Résidus",
    ))
    fig4.add_hline(y=0, line_dash="dash", line_color="#666666", line_width=1.5)
    fig4.update_layout(title="Résidus vs Valeurs Prédites",
                       xaxis_title="Valeurs prédites", yaxis_title="Résidus", **PLOTLY_THEME)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        '<div class="info-box">📌 <b>Résidus :</b> Une dispersion aléatoire autour de zéro confirme '
        'la bonne spécification du modèle et le respect des hypothèses de la régression linéaire.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
#  TAB 3 — PCA
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Analyse en Composantes Principales (ACP / PCA)</div>',
                unsafe_allow_html=True)

    components, explained, loadings = run_pca(df)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["health_status"] = df["health_status"].values
    pca_df["mood_score"]    = df["mood_score"].values
    pca_df["name"]          = df["name"].values if "name" in df.columns else "N/A"

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="health_status", size="mood_score", size_max=16,
            color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
            hover_name="name", hover_data={"mood_score": True, "health_status": True},
            title=f"Projection ACP — PC1 ({explained[0]*100:.1f}%)  ·  PC2 ({explained[1]*100:.1f}%)",
            **PLOTLY_THEME,
        )
        fig.update_traces(marker=dict(opacity=0.70, line=dict(width=0.5, color="white")))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Variance expliquée**")
        ev_df = pd.DataFrame({
            "Composante": ["PC1", "PC2"],
            "Variance (%)": [f"{v*100:.1f}%" for v in explained],
            "Cumulé (%)":   [f"{explained[:i+1].sum()*100:.1f}%" for i in range(2)],
        })
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
        st.markdown("**Loadings (corrélations variables / axes)**")
        st.dataframe(loadings.round(3), use_container_width=True)

    st.markdown('<div class="section-title">Biplot — Variables et Individus</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=pca_df["PC1"], y=pca_df["PC2"], mode="markers",
        marker=dict(
            color=pca_df["health_status"].map({"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK}),
            size=5, opacity=0.40,
        ),
        name="Individus",
    ))
    scale = max(abs(components[:, 0].max()), abs(components[:, 1].max())) * 0.6
    arrow_colors = [CLR_MAIN, CLR_AT_RISK, CLR_SECONDARY]
    for i, var in enumerate(FEATURES):
        l1 = loadings.loc[var, "PC1"] * scale
        l2 = loadings.loc[var, "PC2"] * scale
        fig5.add_annotation(
            ax=0, ay=0, x=l1, y=l2, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=2,
            arrowcolor=arrow_colors[i % len(arrow_colors)],
            font=dict(color=arrow_colors[i % len(arrow_colors)], size=11),
            text=var.replace("_", " ").title(),
        )
    fig5.update_layout(
        title="Biplot ACP — Individus et Vecteurs Variables",
        xaxis_title=f"PC1 ({explained[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({explained[1]*100:.1f}%)",
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        '<div class="info-box">📌 <b>Lecture du biplot :</b> Les flèches indiquent la direction et l\'intensité '
        'de contribution de chaque variable originale aux axes principaux. Des flèches voisines signalent '
        'des variables fortement corrélées entre elles.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Random Forest — Classification Sain / À Risque</div>',
                unsafe_allow_html=True)

    clf, report, cm, importances, y_te, y_pred_rf = run_random_forest(df)

    col1, col2 = st.columns(2)
    with col1:
        cm_df = pd.DataFrame(cm,
                             index=["Réel : Healthy", "Réel : At Risk"],
                             columns=["Prédit : Healthy", "Prédit : At Risk"])
        fig = px.imshow(cm_df, text_auto=True,
                        color_continuous_scale=SEQ_BLUES,
                        title="Matrice de Confusion", **PLOTLY_THEME)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(xaxis=dict(side="bottom"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=list(importances.keys()), y=list(importances.values()),
                     color=list(importances.values()),
                     color_continuous_scale=SEQ_VIRIDIS,
                     labels={"x": "Variable", "y": "Importance relative"},
                     title="Importance des Variables (Gini)", **PLOTLY_THEME)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Métriques de Classification</div>', unsafe_allow_html=True)
    metrics = []
    for label in ["Healthy", "At Risk"]:
        if label in report:
            r = report[label]
            metrics.append({
                "Classe":    label,
                "Précision": round(r["precision"], 3),
                "Rappel":    round(r["recall"], 3),
                "F1-Score":  round(r["f1-score"], 3),
                "Support":   int(r["support"]),
            })
    st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
    st.metric("Précision globale (Accuracy)", f"{report.get('accuracy', 0)*100:.1f}%")

    st.markdown('<div class="section-title">Frontière de Décision — Projection ACP 2D</div>',
                unsafe_allow_html=True)
    X_sc2 = StandardScaler().fit_transform(df[FEATURES])
    proj  = PCA(n_components=2).fit_transform(X_sc2)
    df_proj = pd.DataFrame(proj, columns=["PC1", "PC2"])
    df_proj["Prédit"] = clf.predict(df[FEATURES])
    df_proj["Réel"]   = df["health_status"].values

    fig6 = px.scatter(df_proj, x="PC1", y="PC2",
                      color="Prédit", symbol="Réel",
                      color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
                      title="Classification projetée sur les 2 premières composantes principales",
                      **PLOTLY_THEME)
    fig6.update_traces(marker=dict(size=7, opacity=0.70, line=dict(width=0.5, color="white")))
    fig6.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown(
        '<div class="info-box">📌 <b>Modèle :</b> Random Forest (200 arbres, split 75/25). '
        'La projection 2D via ACP permet de visualiser la séparation entre classes. '
        'Les formes distinctes différencient les vraies étiquettes des prédictions du modèle.</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
#  TAB 5 — CLUSTERING
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">K-Means — Profils Émotionnels</div>', unsafe_allow_html=True)

    k = st.slider("Nombre de clusters K", 2, 6, 3, key="k_slider")
    labels_km, km_model = run_kmeans(df, k=k)
    df_cluster = df.copy()
    df_cluster["cluster"] = [f"Profil {c+1}" for c in labels_km]

    st.markdown('<div class="section-title">Méthode du Coude (Elbow Method)</div>', unsafe_allow_html=True)
    X_sc_km  = StandardScaler().fit_transform(df[FEATURES + [TARGET]])
    inertias = [KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_sc_km).inertia_
                for ki in range(2, 9)]

    fig_elbow = px.line(x=list(range(2, 9)), y=inertias, markers=True,
                        labels={"x": "Nombre de clusters K", "y": "Inertie (Within-cluster SS)"},
                        title="Méthode du Coude — Choix optimal de K",
                        color_discrete_sequence=[CLR_MAIN], **PLOTLY_THEME)
    fig_elbow.update_traces(marker=dict(size=8, color=CLR_MAIN))
    fig_elbow.add_vline(x=k, line_dash="dash", line_color=CLR_AT_RISK,
                        annotation_text=f"K={k} sélectionné",
                        annotation_position="top right",
                        annotation_font_color=CLR_AT_RISK)
    st.plotly_chart(fig_elbow, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig7 = px.scatter(df_cluster, x="sleep_hours", y="mood_score",
                          color="cluster", size="stress",
                          hover_data=["social_interaction", "health_status"],
                          title="Clusters — Sommeil vs Score d'Humeur",
                          color_discrete_sequence=QUAL_PALETTE, **PLOTLY_THEME)
        fig7.update_traces(marker=dict(opacity=0.75, line=dict(width=0.4, color="white")))
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        fig8 = px.scatter(df_cluster, x="stress", y="social_interaction",
                          color="cluster", size="mood_score",
                          title="Clusters — Stress vs Interaction Sociale",
                          color_discrete_sequence=QUAL_PALETTE, **PLOTLY_THEME)
        fig8.update_traces(marker=dict(opacity=0.75, line=dict(width=0.4, color="white")))
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-title">Profil Radar des Centroides</div>', unsafe_allow_html=True)
    cluster_means = df_cluster.groupby("cluster")[FEATURES + [TARGET]].mean()
    vars_radar    = FEATURES + [TARGET]

    fig_radar = go.Figure()
    for i, (clust, row) in enumerate(cluster_means.iterrows()):
        vals = row[vars_radar].tolist() + [row[vars_radar[0]]]
        cats = ([v.replace("_", " ").title() for v in vars_radar]
                + [vars_radar[0].replace("_", " ").title()])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=clust,
            line=dict(color=QUAL_PALETTE[i % len(QUAL_PALETTE)], width=2),
            opacity=0.65,
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color="#666666", gridcolor="#e0e0e0"),
            angularaxis=dict(color="#444444"),
            bgcolor="#fafafa",
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#2c3e50"),
        title="Profil Radar moyen par Cluster",
        template="plotly_white",
        showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-title">Tableau Résumé des Clusters</div>', unsafe_allow_html=True)
    summary = df_cluster.groupby("cluster").agg(
        Participants        =("mood_score",          "count"),
        Humeur_moyenne      =("mood_score",          lambda x: round(x.mean(), 1)),
        Sommeil_moyen       =("sleep_hours",          lambda x: round(x.mean(), 1)),
        Stress_moyen        =("stress",               lambda x: round(x.mean(), 1)),
        Interaction_sociale =("social_interaction",   lambda x: round(x.mean(), 1)),
        Pct_Sains           =("health_status",        lambda x: f"{(x=='Healthy').mean()*100:.0f}%"),
    ).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="info-box">📌 <b>Interprétation :</b> Chaque cluster représente un profil émotionnel distinct. '
        'Le radar compare simultanément toutes les dimensions par groupe. '
        'Ajustez K avec le curseur ci-dessus pour explorer différentes granularités de segmentation.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#9aa5b1;font-size:.78rem;'>"
    "MindMetrics · INF 232 EC2 Analyse de données · "
    "Streamlit · Supabase (<code>mental_health_data</code>) · scikit-learn · Plotly"
    "</p>",
    unsafe_allow_html=True,
)
