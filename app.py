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
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, r2_score, mean_squared_error
)

from supabase import create_client
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
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] * {
    color: #e8e0ff !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
    color: #fff !important;
}
section[data-testid="stSidebar"] .stSlider > div [data-testid="stTickBar"] {
    color: #a78bfa !important;
}

/* ── Main area ── */
.main .block-container {
    padding-top: 2rem;
    background: #0d0d14;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(167,139,250,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0e8ff;
    margin: 0 0 .4rem 0;
    letter-spacing: -1px;
}
.hero p {
    color: #a78bfa;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #c4b5fd;
    border-left: 4px solid #7c3aed;
    padding-left: .75rem;
    margin: 2rem 0 1rem 0;
}

/* ── Metric cards ── */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, #1e1b4b, #2d2560);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card .lbl {
    font-size: .78rem;
    color: #8b7fa8;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .2rem;
}

/* ── Alert boxes ── */
.info-box {
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #c4b5fd;
    font-size: .9rem;
    margin-bottom: 1rem;
}

/* ── Plotly charts dark bg ── */
.js-plotly-plot { border-radius: 14px; }

/* ── Streamlit default overrides ── */
h1,h2,h3,h4 { color: #e8e0ff; font-family: 'Syne', sans-serif; }
.stTabs [data-baseweb="tab-list"] { background: #12111f; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b7fa8; font-family: 'DM Sans', sans-serif; }
.stTabs [aria-selected="true"] { background: #2d2560 !important; color: #c4b5fd !important; border-radius: 8px; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: .55rem 1.4rem;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(13,13,20,0)",
    plot_bgcolor="rgba(13,13,20,0)",
    font=dict(family="DM Sans", color="#c4b5fd"),
)
PALETTE = px.colors.sequential.Purples_r + px.colors.sequential.Blues_r

# ─────────────────────────────────────────────
#  SUPABASE CONNECTION
# ─────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return st.connection("supabase", type=SupabaseConnection)

TABLE = "mental_health_records"

def save_record(conn, record: dict):
    """Insert one row into Supabase."""
    try:
        conn.table(TABLE).insert(record).execute()
        return True, "✅ Données sauvegardées avec succès!"
    except Exception as e:
        return False, f"❌ Erreur Supabase : {e}"

@st.cache_data(ttl=30)
def load_data(_conn):
    """Fetch all rows from Supabase (cache 30 s)."""
    try:
        resp = _conn.table(TABLE).select("*").execute()
        df = pd.DataFrame(resp.data)
        if df.empty:
            return pd.DataFrame()
        # Ensure correct dtypes
        for col in ["sleep_hours", "stress", "social_interaction", "mood_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["sleep_hours", "stress", "social_interaction", "mood_score"])
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  HELPER — DEMO DATA (when DB is empty)
# ─────────────────────────────────────────────
@st.cache_data
def generate_demo_data(n=120):
    rng = np.random.default_rng(42)
    sleep  = rng.normal(6.5, 1.4, n).clip(2, 10)
    stress = rng.normal(5.5, 2.0, n).clip(1, 10)
    social = rng.normal(5.0, 2.2, n).clip(1, 10)
    mood   = (
        4.5 * sleep
        - 3.8 * stress
        + 2.2 * social
        + rng.normal(0, 6, n)
        + 30
    ).clip(0, 100)
    names  = [f"Étudiant_{i+1:03d}" for i in range(n)]
    mats   = [f"20{rng.integers(20,26):02d}{rng.integers(1000,9999)}" for _ in range(n)]
    return pd.DataFrame({
        "name": names, "matricule": mats,
        "sleep_hours": sleep.round(1),
        "stress": stress.round(1),
        "social_interaction": social.round(1),
        "mood_score": mood.round(1),
    })

# ─────────────────────────────────────────────
#  FEATURE COLUMNS
# ─────────────────────────────────────────────
FEATURES = ["sleep_hours", "stress", "social_interaction"]
TARGET   = "mood_score"

def add_health_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["health_status"] = np.where(
        (df["mood_score"] >= 55)
        & (df["stress"] <= 6)
        & (df["sleep_hours"] >= 6),
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
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return model, r2, rmse, y_pred


def run_multiple_regression(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    coef = dict(zip(FEATURES, model.coef_))
    return model, r2, rmse, y_pred, coef


def run_pca(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=["PC1", "PC2"]
    )
    return components, explained, loadings


def run_random_forest(df):
    df = add_health_label(df)
    X = df[FEATURES].values
    y = df["health_status"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True)
    cm     = confusion_matrix(y_te, y_pred, labels=["Healthy", "At Risk"])
    importances = dict(zip(FEATURES, clf.feature_importances_))
    return clf, report, cm, importances, y_te, y_pred


def run_kmeans(df, k=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES + [TARGET]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km

# ─────────────────────────────────────────────
#  SIDEBAR — DATA COLLECTION FORM
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindMetrics")
    st.markdown("<p style='color:#8b7fa8;font-size:.85rem;margin-top:-.5rem;'>INF 232 · Analyse de données</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📋 Saisie des données")

    with st.form("data_form", clear_on_submit=True):
        name        = st.text_input("Nom complet", placeholder="ex. Dupont Marie")
        matricule   = st.text_input("Matricule", placeholder="ex. 202312345")
        sleep_hours = st.slider("🛌 Heures de sommeil", 2.0, 12.0, 7.0, 0.5)
        stress      = st.slider("😰 Stress (1–10)", 1, 10, 5)
        social      = st.slider("🤝 Interaction sociale (1–10)", 1, 10, 5)
        mood        = st.slider("😊 Score d'humeur (0–100)", 0, 100, 60)

        submitted = st.form_submit_button("💾 Enregistrer")

    if submitted:
        if not name.strip() or not matricule.strip():
            st.warning("Veuillez remplir Nom et Matricule.")
        else:
            try:
                conn = get_conn()
                ok, msg = save_record(conn, {
                    "name": name.strip(),
                    "matricule": matricule.strip(),
                    "sleep_hours": sleep_hours,
                    "stress": stress,
                    "social_interaction": social,
                    "mood_score": mood,
                })
                if ok:
                    st.success(msg)
                    st.cache_data.clear()
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Connexion impossible : {e}")

    st.divider()
    st.markdown("<p style='font-size:.75rem;color:#6b6080;'>Les données sont anonymisées et stockées dans Supabase.</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
try:
    conn = get_conn()
    df_raw = load_data(conn)
    using_demo = df_raw.empty
except Exception:
    df_raw    = pd.DataFrame()
    using_demo = True

if using_demo:
    df_raw = generate_demo_data()
    st.info("ℹ️ Aucune connexion Supabase détectée — **données de démonstration** utilisées (120 enregistrements simulés).")

df = add_health_label(df_raw.copy())

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 MindMetrics</h1>
  <p>Analyse de données · Santé Mentale & Intelligence Émotionnelle · INF 232 EC2</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────
n_total   = len(df)
n_healthy = (df["health_status"] == "Healthy").sum()
avg_mood  = df["mood_score"].mean()
avg_sleep = df["sleep_hours"].mean()
avg_stress= df["stress"].mean()

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="val">{n_total}</div><div class="lbl">Participants</div></div>
  <div class="metric-card"><div class="val">{avg_mood:.1f}</div><div class="lbl">Humeur moy.</div></div>
  <div class="metric-card"><div class="val">{avg_sleep:.1f}h</div><div class="lbl">Sommeil moy.</div></div>
  <div class="metric-card"><div class="val">{avg_stress:.1f}</div><div class="lbl">Stress moy.</div></div>
  <div class="metric-card"><div class="val">{n_healthy}/{n_total}</div><div class="lbl">Sains</div></div>
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
        fig = px.histogram(
            df, x="mood_score", nbins=25,
            color_discrete_sequence=["#7c3aed"],
            title="Distribution du Score d'Humeur",
            labels={"mood_score": "Score d'humeur"},
            **PLOTLY_THEME,
        )
        fig.update_traces(opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="sleep_hours", nbins=20,
            color_discrete_sequence=["#4f46e5"],
            title="Distribution des Heures de Sommeil",
            labels={"sleep_hours": "Heures de sommeil"},
            **PLOTLY_THEME,
        )
        fig.update_traces(opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.pie(
            df, names="health_status",
            color="health_status",
            color_discrete_map={"Healthy": "#7c3aed", "At Risk": "#ef4444"},
            title="Répartition Sain / À Risque",
            hole=.42,
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        corr = df[FEATURES + [TARGET]].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Purples",
            title="Matrice de Corrélation",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Scatter Matrix</div>', unsafe_allow_html=True)
    fig = px.scatter_matrix(
        df,
        dimensions=FEATURES + [TARGET],
        color="health_status",
        color_discrete_map={"Healthy": "#7c3aed", "At Risk": "#ef4444"},
        title="Relations entre toutes les variables",
        **PLOTLY_THEME,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Données brutes</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["name", "matricule", "sleep_hours", "stress", "social_interaction", "mood_score", "health_status"]],
        use_container_width=True,
        height=300,
    )

# ══════════════════════════════════════════════
#  TAB 2 — RÉGRESSION
# ══════════════════════════════════════════════
with tab2:

    # ── Simple ──────────────────────────────
    st.markdown('<div class="section-title">Régression Linéaire Simple — Sommeil → Humeur</div>', unsafe_allow_html=True)

    model_s, r2_s, rmse_s, ypred_s = run_simple_regression(df)
    x_line = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 200)
    y_line = model_s.predict(x_line.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sleep_hours"], y=df[TARGET],
        mode="markers",
        marker=dict(color=df[TARGET], colorscale="Purples", size=6, opacity=0.7, showscale=True,
                    colorbar=dict(title="Humeur")),
        name="Observations",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#a78bfa", width=3),
        name=f"Droite de régression (R²={r2_s:.3f})",
    ))
    fig.update_layout(
        title="Sommeil vs Score d'Humeur",
        xaxis_title="Heures de sommeil",
        yaxis_title="Score d'humeur",
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_s:.4f}")
    c2.metric("RMSE", f"{rmse_s:.2f}")
    c3.metric("Coefficient (pente)", f"{model_s.coef_[0]:.3f}")

    st.markdown('<div class="info-box">📌 <b>Interprétation :</b> Chaque heure de sommeil supplémentaire est associée à une variation de <b>' +
                f"{model_s.coef_[0]:.2f} points</b> du score d'humeur (R²={r2_s:.3f}).</div>", unsafe_allow_html=True)

    # ── Multiple ─────────────────────────────
    st.markdown('<div class="section-title">Régression Linéaire Multiple — Tous les facteurs → Humeur</div>', unsafe_allow_html=True)

    model_m, r2_m, rmse_m, ypred_m, coef_m = run_multiple_regression(df)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df[TARGET], y=ypred_m,
        mode="markers",
        marker=dict(color="#7c3aed", opacity=0.6, size=6),
        name="Prédit vs Réel",
    ))
    lims = [df[TARGET].min(), df[TARGET].max()]
    fig2.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                              line=dict(color="#ef4444", dash="dash"), name="Parfait"))
    fig2.update_layout(
        title="Valeurs Réelles vs Prédites (Régression Multiple)",
        xaxis_title="Valeurs réelles",
        yaxis_title="Valeurs prédites",
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        c1, c2 = st.columns(2)
        c1.metric("R² multiple", f"{r2_m:.4f}")
        c2.metric("RMSE", f"{rmse_m:.2f}")
        st.markdown("**Coefficients :**")
        coef_df = pd.DataFrame(
            {"Variable": list(coef_m.keys()),
             "Coefficient": [f"{v:.3f}" for v in coef_m.values()]}
        )
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    with col_b:
        fig3 = px.bar(
            x=list(coef_m.keys()),
            y=list(coef_m.values()),
            color=list(coef_m.values()),
            color_continuous_scale="RdBu",
            labels={"x": "Variable", "y": "Coefficient", "color": ""},
            title="Poids des variables",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Residuals ──────────────────────────
    st.markdown('<div class="section-title">Analyse des Résidus</div>', unsafe_allow_html=True)
    residuals = df[TARGET].values - ypred_m
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=ypred_m, y=residuals,
        mode="markers",
        marker=dict(color=residuals, colorscale="RdBu", size=5, opacity=0.7),
    ))
    fig4.add_hline(y=0, line_dash="dash", line_color="#a78bfa")
    fig4.update_layout(
        title="Résidus vs Valeurs Prédites",
        xaxis_title="Valeurs prédites",
        yaxis_title="Résidus",
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — PCA
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Analyse en Composantes Principales (ACP)</div>', unsafe_allow_html=True)

    components, explained, loadings = run_pca(df)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["health_status"] = df["health_status"].values
    pca_df["mood_score"]    = df["mood_score"].values
    pca_df["name"]          = df["name"].values if "name" in df.columns else "N/A"

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="health_status",
            size="mood_score",
            size_max=18,
            color_discrete_map={"Healthy": "#7c3aed", "At Risk": "#ef4444"},
            hover_name="name",
            hover_data={"mood_score": True, "health_status": True},
            title=f"Projection ACP — PC1 ({explained[0]*100:.1f}%) · PC2 ({explained[1]*100:.1f}%)",
            **PLOTLY_THEME,
        )
        fig.update_traces(marker=dict(opacity=0.75))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Variance expliquée**")
        ev_df = pd.DataFrame({
            "Composante": ["PC1", "PC2"],
            "Variance (%)": [f"{v*100:.1f}%" for v in explained],
            "Cumulé (%)":   [f"{explained[:i+1].sum()*100:.1f}%" for i in range(2)],
        })
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

        st.markdown("**Corrélations variables / axes (loadings)**")
        st.dataframe(loadings.round(3), use_container_width=True)

    # ── Biplot ──────────────────────────────
    st.markdown('<div class="section-title">Biplot — Variables et Individus</div>', unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=pca_df["PC1"], y=pca_df["PC2"],
        mode="markers",
        marker=dict(
            color=pca_df["health_status"].map({"Healthy": "#7c3aed", "At Risk": "#ef4444"}),
            size=5, opacity=0.5
        ),
        name="Individus",
    ))
    scale = max(abs(components[:, 0].max()), abs(components[:, 1].max())) * 0.6
    for var in FEATURES:
        l1, l2 = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
        fig5.add_annotation(
            ax=0, ay=0, x=l1, y=l2,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=2,
            arrowcolor="#f0abfc",
            font=dict(color="#f0abfc", size=12),
            text=var.replace("_", " ").title(),
        )
    fig5.update_layout(title="Biplot ACP", **PLOTLY_THEME)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="info-box">📌 <b>Lecture :</b> Les flèches montrent la direction et l\'intensité de contribution '
                'de chaque variable aux composantes principales. Des flèches proches indiquent des variables corrélées.</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Random Forest — Classification Sain / À Risque</div>', unsafe_allow_html=True)

    clf, report, cm, importances, y_te, y_pred_rf = run_random_forest(df)

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix heatmap
        cm_df = pd.DataFrame(cm, index=["Healthy", "At Risk"], columns=["Prédit: Healthy", "Prédit: At Risk"])
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Purples",
            title="Matrice de Confusion",
            **PLOTLY_THEME,
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importances
        fig = px.bar(
            x=list(importances.keys()),
            y=list(importances.values()),
            color=list(importances.values()),
            color_continuous_scale="Purples",
            labels={"x": "Variable", "y": "Importance"},
            title="Importance des Variables",
            **PLOTLY_THEME,
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Classification metrics
    st.markdown('<div class="section-title">Métriques de Classification</div>', unsafe_allow_html=True)
    metrics = []
    for label in ["Healthy", "At Risk"]:
        if label in report:
            r = report[label]
            metrics.append({
                "Classe": label,
                "Précision": f"{r['precision']:.3f}",
                "Rappel":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r["support"]),
            })
    st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

    overall_acc = report.get("accuracy", 0)
    st.metric("Précision globale (Accuracy)", f"{overall_acc*100:.1f}%")

    # Decision boundary projection (2D via PCA)
    st.markdown('<div class="section-title">Frontière de Décision (projection 2D via ACP)</div>', unsafe_allow_html=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    pca2 = PCA(n_components=2).fit(X_scaled)
    proj = pca2.transform(X_scaled)

    df_proj = pd.DataFrame(proj, columns=["PC1", "PC2"])
    df_proj["Prédit"] = clf.predict(df[FEATURES])
    df_proj["Réel"]   = df["health_status"].values

    fig6 = px.scatter(
        df_proj, x="PC1", y="PC2",
        color="Prédit",
        symbol="Réel",
        color_discrete_map={"Healthy": "#7c3aed", "At Risk": "#ef4444"},
        title="Classification projetée sur les 2 premières composantes principales",
        **PLOTLY_THEME,
    )
    fig6.update_traces(marker=dict(size=7, opacity=0.75))
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="info-box">📌 <b>Modèle :</b> Random Forest (200 arbres). '
                'La frontière de décision est visualisée via une projection ACP 2D. '
                'Les différentes formes distinguent les vrais labels des prédictions.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 5 — CLUSTERING
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">K-Means — Profils Émotionnels</div>', unsafe_allow_html=True)

    k = st.slider("Nombre de clusters K", 2, 6, 3, key="k_slider")
    labels, km_model = run_kmeans(df, k=k)
    df_cluster = df.copy()
    df_cluster["cluster"] = [f"Profil {c+1}" for c in labels]

    # Elbow method
    st.markdown('<div class="section-title">Méthode du Coude (Elbow)</div>', unsafe_allow_html=True)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(df[FEATURES + [TARGET]])
    inertias = []
    ks = range(2, 9)
    for ki in ks:
        inertias.append(KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_sc).inertia_)

    fig_elbow = px.line(
        x=list(ks), y=inertias,
        markers=True,
        labels={"x": "Nombre de clusters K", "y": "Inertie"},
        title="Méthode du Coude",
        color_discrete_sequence=["#a78bfa"],
        **PLOTLY_THEME,
    )
    fig_elbow.add_vline(x=k, line_dash="dash", line_color="#ef4444",
                        annotation_text=f"K={k} sélectionné", annotation_position="top right")
    st.plotly_chart(fig_elbow, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig7 = px.scatter(
            df_cluster, x="sleep_hours", y="mood_score",
            color="cluster",
            size="stress",
            hover_data=["social_interaction", "health_status"],
            title="Clusters — Sommeil vs Humeur",
            color_discrete_sequence=px.colors.qualitative.Bold,
            **PLOTLY_THEME,
        )
        fig7.update_traces(marker=dict(opacity=0.8))
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        fig8 = px.scatter(
            df_cluster, x="stress", y="social_interaction",
            color="cluster",
            size="mood_score",
            title="Clusters — Stress vs Interaction Sociale",
            color_discrete_sequence=px.colors.qualitative.Bold,
            **PLOTLY_THEME,
        )
        fig8.update_traces(marker=dict(opacity=0.8))
        st.plotly_chart(fig8, use_container_width=True)

    # Radar chart of cluster centroids
    st.markdown('<div class="section-title">Profil Radar des Centroides</div>', unsafe_allow_html=True)
    cluster_means = df_cluster.groupby("cluster")[FEATURES + [TARGET]].mean()
    vars_radar = FEATURES + [TARGET]

    fig_radar = go.Figure()
    colors_radar = px.colors.qualitative.Bold
    for i, (clust, row) in enumerate(cluster_means.iterrows()):
        vals = row[vars_radar].tolist()
        vals += [vals[0]]
        cats = [v.replace("_", " ").title() for v in vars_radar] + [vars_radar[0].replace("_", " ").title()]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=clust,
            line=dict(color=colors_radar[i % len(colors_radar)], width=2),
            opacity=0.72,
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color="#8b7fa8"),
            angularaxis=dict(color="#8b7fa8"),
            bgcolor="rgba(13,13,20,0.5)",
        ),
        title="Profil Radar par Cluster",
        showlegend=True,
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Cluster summary table
    st.markdown('<div class="section-title">Résumé des Clusters</div>', unsafe_allow_html=True)
    summary = df_cluster.groupby("cluster").agg(
        Participants=("mood_score", "count"),
        Humeur_moy=("mood_score", lambda x: f"{x.mean():.1f}"),
        Sommeil_moy=("sleep_hours", lambda x: f"{x.mean():.1f}h"),
        Stress_moy=("stress", lambda x: f"{x.mean():.1f}"),
        Social_moy=("social_interaction", lambda x: f"{x.mean():.1f}"),
        Pct_Sains=("health_status", lambda x: f"{(x=='Healthy').mean()*100:.0f}%"),
    ).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="info-box">📌 <b>Interprétation :</b> Chaque cluster représente un profil émotionnel distinct. '
                'Les centroides du radar révèlent les dimensions dominantes de chaque groupe. '
                'Ajustez K avec le curseur pour explorer différentes granularités.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#4a4268;font-size:.8rem;'>"
    "MindMetrics · INF 232 EC2 Analyse de données · Propulsé par Streamlit + Supabase + scikit-learn + Plotly"
    "</p>",
    unsafe_allow_html=True,
)