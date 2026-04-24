"""
INF 232 EC2 — Analyse de données de Santé Mentale
Secteur : Santé Mentale & Intelligence Émotionnelle
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
    classification_report, confusion_matrix, r2_score, mean_squared_error,
)
from supabase import create_client, Client

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="INF 232 EC2 | Santé Mentale",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — clean academic light theme
# ══════════════════════════════════════════════════════════════════════════════
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
section[data-testid="stSidebar"] * { color: #2c3e50 !important; }
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #c8cdd5 !important;
    border-radius: 6px !important;
    color: #2c3e50 !important;
}

/* ── Main area ── */
.main .block-container { padding-top: 2rem; background-color: #f8f9fa; }

/* ── Page header ── */
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
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0 0 .3rem 0;
}
.page-header p {
    color: #5f6b7a;
    font-size: .88rem;
    margin: 0;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a2e;
    border-bottom: 2px solid #1a73e8;
    padding-bottom: .3rem;
    margin: 2rem 0 1rem 0;
}

/* ── KPI metric cards ── */
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

/* ── Interpretation boxes ── */
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
.stTabs [data-baseweb="tab"] { color: #5f6b7a; font-size: .87rem; }
.stTabs [aria-selected="true"] {
    background: #1a73e8 !important;
    color: #ffffff !important;
    border-radius: 4px;
}

/* ── Submit button ── */
.stButton > button {
    background-color: #1a73e8;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    padding: .5rem 1.3rem;
    transition: background-color .2s;
}
.stButton > button:hover { background-color: #1558b0; }

/* ── DataFrames ── */
div[data-testid="stDataFrame"] { border-radius: 6px; border: 1px solid #dde1e7; }

h1, h2, h3, h4 { font-family: 'Source Serif 4', serif; color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME — plotly_white, professional palette
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#2c3e50", size=12),
)

CLR_HEALTHY   = "#1a73e8"
CLR_AT_RISK   = "#d93025"
CLR_MAIN      = "#1a73e8"
CLR_SECONDARY = "#34a853"
SEQ_BLUES     = "Blues"
SEQ_VIRIDIS   = "Viridis"
QUAL_PALETTE  = ["#1a73e8", "#34a853", "#ea4335", "#fbbc04", "#4285f4", "#ff6d00"]

# ══════════════════════════════════════════════════════════════════════════════
#  FIX 1 & 2 — SUPABASE: read from st.secrets, use correct table name
# ══════════════════════════════════════════════════════════════════════════════
TABLE = "mental_health_data"   # FIX 2 — single source of truth for table name

@st.cache_resource
def get_supabase_client() -> Client:
    """
    Reads credentials from the exact secrets path the user configured:
        [connections.supabase]
        SUPABASE_URL = "..."
        SUPABASE_KEY = "..."
    Shows a clear, actionable error instead of silently falling back to demo data.
    """
    try:
        url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
        key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    except KeyError as e:
        st.error(
            f"❌ Clé manquante dans les Secrets Streamlit : `{e}`\n\n"
            "Ajoutez ce bloc dans **Settings → Secrets** :\n"
            "```toml\n"
            "[connections.supabase]\n"
            'SUPABASE_URL = "https://VOTRE_ID.supabase.co"\n'
            'SUPABASE_KEY = "votre_anon_key"\n'
            "```"
        )
        st.stop()
    return create_client(url, key)


def save_record(client: Client, record: dict) -> tuple[bool, str]:
    try:
        client.table(TABLE).insert(record).execute()
        return True, "✅ Données sauvegardées avec succès !"
    except Exception as e:
        return False, f"❌ Erreur lors de l'insertion : {e}"


@st.cache_data(ttl=30)
def load_data(_client: Client) -> pd.DataFrame:
    """Fetch all rows from Supabase (cached 30 s). Returns empty DataFrame on failure."""
    try:
        resp = _client.table(TABLE).select("*").execute()
        df = pd.DataFrame(resp.data)
        if df.empty:
            return pd.DataFrame()
        for col in ["sleep_hours", "stress", "social_interaction", "mood_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["sleep_hours", "stress", "social_interaction", "mood_score"])
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
FEATURES    = ["sleep_hours", "stress", "social_interaction"]
TARGET      = "mood_score"
MIN_ROWS_ML = 3   # minimum rows required by ML algorithms (KMeans, RF, PCA)


def add_health_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["health_status"] = np.where(
        (df["mood_score"] >= 55) & (df["stress"] <= 6) & (df["sleep_hours"] >= 6),
        "Healthy", "At Risk",
    )
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_simple_regression(df):
    X = df[["sleep_hours"]].values
    y = df[TARGET].values
    m = LinearRegression().fit(X, y)
    yp = m.predict(X)
    return m, r2_score(y, yp), np.sqrt(mean_squared_error(y, yp)), yp


def run_multiple_regression(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    m = LinearRegression().fit(X, y)
    yp = m.predict(X)
    return m, r2_score(y, yp), np.sqrt(mean_squared_error(y, yp)), yp, dict(zip(FEATURES, m.coef_))


def run_pca(df):
    Xs = StandardScaler().fit_transform(df[FEATURES])
    pca = PCA(n_components=2)
    comps = pca.fit_transform(Xs)
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"])
    return comps, pca.explained_variance_ratio_, loadings


def run_random_forest(df):
    df = add_health_label(df)
    X, y = df[FEATURES].values, df["health_status"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    return (
        clf,
        classification_report(y_te, yp, output_dict=True),
        confusion_matrix(y_te, yp, labels=["Healthy", "At Risk"]),
        dict(zip(FEATURES, clf.feature_importances_)),
        y_te, yp,
    )


def run_kmeans(df, k=3):
    Xs = StandardScaler().fit_transform(df[FEATURES + [TARGET]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(Xs), km

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — data collection form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 INF 232 EC2")
    st.markdown(
        "<p style='color:#7a8694;font-size:.82rem;margin-top:-.4rem;'>"
        "Analyse de données · Santé Mentale</p>",
        unsafe_allow_html=True,
    )
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
            supabase_client = get_supabase_client()
            ok, msg = save_record(supabase_client, {
                "name":               name.strip(),
                "matricule":          matricule.strip(),
                "sleep_hours":        sleep_hours,
                "stress":             stress,
                "social_interaction": social,
                "mood_score":         mood,
            })
            (st.success if ok else st.error)(msg)
            if ok:
                st.cache_data.clear()

    st.divider()
    st.caption(f"Table Supabase : {TABLE}")

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA FROM SUPABASE
# ══════════════════════════════════════════════════════════════════════════════
supabase_client = get_supabase_client()
df_raw = load_data(supabase_client)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER  (FIX 4 — exact required title)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
  <h1>🧠 INF 232 EC2 : Analyse de données de Santé Mentale</h1>
  <p>Tableau de bord analytique · Intelligence Émotionnelle · Santé Mentale</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3 — EMPTY DATABASE GUARD
#  Stops all chart rendering if the table is empty, preventing TypeError crashes.
# ══════════════════════════════════════════════════════════════════════════════
if df_raw.empty:
    st.warning(
        "La base de données est vide. "
        "Veuillez ajouter une entrée via le formulaire pour activer l'analyse."
    )
    st.info(
        "👈 Utilisez le formulaire dans la barre latérale pour saisir vos premières données. "
        "Le tableau de bord s'activera automatiquement dès qu'une ligne est enregistrée."
    )
    st.stop()   # ← halts execution; no chart code below runs on an empty DataFrame

# ── At this point df_raw has at least one valid row ──────────────────────────
df = add_health_label(df_raw.copy())

# ══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
n_total    = len(df)
n_healthy  = (df["health_status"] == "Healthy").sum()
avg_mood   = df["mood_score"].mean()
avg_sleep  = df["sleep_hours"].mean()
avg_stress = df["stress"].mean()

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="val">{n_total}</div>
    <div class="lbl">Participants</div>
  </div>
  <div class="metric-card">
    <div class="val">{avg_mood:.1f}</div>
    <div class="lbl">Humeur moy.</div>
  </div>
  <div class="metric-card">
    <div class="val">{avg_sleep:.1f}h</div>
    <div class="lbl">Sommeil moy.</div>
  </div>
  <div class="metric-card">
    <div class="val">{avg_stress:.1f}</div>
    <div class="lbl">Stress moy.</div>
  </div>
  <div class="metric-card">
    <div class="val">{n_healthy}/{n_total}</div>
    <div class="lbl">Profil Sain</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Aperçu",
    "📈 Régression",
    "🔷 ACP (PCA)",
    "🌲 Classification",
    "🔵 Clustering",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — APERÇU
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Distribution des variables</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if not df.empty and df['mood_score'].nunique() > 1:
            fig = px.histogram(df, x="mood_score", nbins=10,
                               color_discrete_sequence=[CLR_MAIN],
                               title="Distribution du Score d'Humeur",
                               labels={"mood_score": "Score d'humeur"}, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Plus de données variées nécessaires.")

    with col2:
        if not df.empty and df['sleep_hours'].nunique() > 1:
            fig = px.histogram(df, x="sleep_hours", nbins=10,
                               color_discrete_sequence=[CLR_SECONDARY],
                               title="Distribution des Heures de Sommeil",
                               labels={"sleep_hours": "Heures de sommeil"}, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 En attente de données.")
            
    col3, col4 = st.columns(2)
    with col3:
        fig = px.pie(df, names="health_status", color="health_status",
                     color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
                     title="Répartition Sain / À Risque", hole=0.42, **PLOTLY_THEME)
        fig.update_traces(textinfo="percent+label", pull=[0.03, 0.03])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        corr = df[FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=SEQ_BLUES,
                        title="Matrice de Corrélation", zmin=-1, zmax=1, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    if len(df) >= MIN_ROWS_ML:
        st.markdown('<div class="section-title">Scatter Matrix — Relations entre variables</div>',
                    unsafe_allow_html=True)
        fig = px.scatter_matrix(df, dimensions=FEATURES + [TARGET],
                                color="health_status",
                                color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
                                title="Matrice de dispersion (toutes variables)", **PLOTLY_THEME)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.55))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Données brutes</div>', unsafe_allow_html=True)
    display_cols = [c for c in
                    ["name", "matricule", "sleep_hours", "stress",
                     "social_interaction", "mood_score", "health_status"]
                    if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=300)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — RÉGRESSION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if len(df) < 2:
        st.warning("Au moins 2 enregistrements sont nécessaires pour la régression.")
    else:
        # ── Régression Linéaire Simple ──
        st.markdown('<div class="section-title">Régression Linéaire Simple — Sommeil → Humeur</div>',
                    unsafe_allow_html=True)

        model_s, r2_s, rmse_s, ypred_s = run_simple_regression(df)
        x_line = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 200)
        y_line = model_s.predict(x_line.reshape(-1, 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["sleep_hours"], y=df[TARGET], mode="markers",
            marker=dict(color=df[TARGET], colorscale=SEQ_VIRIDIS,
                        size=7, opacity=0.70, showscale=True,
                        colorbar=dict(title="Humeur", thickness=14)),
            name="Observations",
        ))
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=CLR_AT_RISK, width=2.5),
            name=f"Droite de régression (R²={r2_s:.3f})",
        ))
        fig.update_layout(
            title="Heures de Sommeil vs Score d'Humeur",
            xaxis_title="Heures de sommeil", yaxis_title="Score d'humeur",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2_s:.4f}")
        c2.metric("RMSE", f"{rmse_s:.2f}")
        c3.metric("Coefficient (pente)", f"{model_s.coef_[0]:.3f}")
        st.markdown(
            '<div class="info-box">📌 <b>Interprétation :</b> Chaque heure de sommeil supplémentaire '
            f"est associée à une variation de <b>{model_s.coef_[0]:.2f} points</b> du score d'humeur. "
            f"Le modèle explique <b>{r2_s*100:.1f}%</b> de la variance (R²={r2_s:.4f}).</div>",
            unsafe_allow_html=True,
        )

        # ── Régression Linéaire Multiple ──
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
        fig2.update_layout(
            title="Valeurs Réelles vs Valeurs Prédites (Régression Multiple)",
            xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig2, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            ca, cb = st.columns(2)
            ca.metric("R² multiple", f"{r2_m:.4f}")
            cb.metric("RMSE", f"{rmse_m:.2f}")
            st.markdown("**Tableau des coefficients**")
            coef_df = pd.DataFrame({
                "Variable":    list(coef_m.keys()),
                "Coefficient": [round(v, 4) for v in coef_m.values()],
                "Direction":   ["↑ Positif" if v > 0 else "↓ Négatif" for v in coef_m.values()],
            })
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        with col_b:
            fig3 = px.bar(x=list(coef_m.keys()), y=list(coef_m.values()),
                          color=list(coef_m.values()), color_continuous_scale="RdBu",
                          labels={"x": "Variable", "y": "Coefficient"},
                          title="Poids des variables explicatives", **PLOTLY_THEME)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Résidus ──
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
                           xaxis_title="Valeurs prédites", yaxis_title="Résidus",
                           **PLOTLY_THEME)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(
            '<div class="info-box">📌 <b>Résidus :</b> Une dispersion aléatoire autour de zéro confirme '
            'la bonne spécification du modèle et le respect des hypothèses de la régression.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — ACP / PCA
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    # FIX 5: minimum row check for PCA
    if len(df) < MIN_ROWS_ML:
        st.warning(
            f"Au moins {MIN_ROWS_ML} enregistrements sont nécessaires pour l'ACP. "
            f"Actuellement : {len(df)} ligne(s). Veuillez ajouter plus de données via le formulaire."
        )
    else:
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
                hover_name="name",
                hover_data={"mood_score": True, "health_status": True},
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

        st.markdown('<div class="section-title">Biplot — Variables et Individus</div>',
                    unsafe_allow_html=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=pca_df["PC1"], y=pca_df["PC2"], mode="markers",
            marker=dict(
                color=pca_df["health_status"].map(
                    {"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK}
                ),
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
                ax=0, ay=0, x=l1, y=l2,
                xref="x", yref="y", axref="x", ayref="y",
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
            '<div class="info-box">📌 <b>Lecture du biplot :</b> Les flèches indiquent la direction et '
            "l'intensité de contribution de chaque variable aux axes principaux. "
            'Des flèches proches signalent des variables fortement corrélées.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — CLASSIFICATION  (FIX 5: len(df) < 3 guard)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Random Forest — Classification Sain / À Risque</div>',
                unsafe_allow_html=True)

    # FIX 5: need at least MIN_ROWS_ML rows for train/test split + RF
    if len(df) < MIN_ROWS_ML:
        st.warning(
            f"Au moins {MIN_ROWS_ML} enregistrements sont nécessaires pour la classification. "
            f"Actuellement : {len(df)} ligne(s). Veuillez ajouter plus de données via le formulaire."
        )
    else:
        clf, report, cm, importances, y_te, y_pred_rf = run_random_forest(df)

        col1, col2 = st.columns(2)
        with col1:
            cm_df = pd.DataFrame(
                cm,
                index=["Réel : Healthy", "Réel : At Risk"],
                columns=["Prédit : Healthy", "Prédit : At Risk"],
            )
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=SEQ_BLUES,
                            title="Matrice de Confusion", **PLOTLY_THEME)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(xaxis=dict(side="bottom"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(x=list(importances.keys()), y=list(importances.values()),
                         color=list(importances.values()), color_continuous_scale=SEQ_VIRIDIS,
                         labels={"x": "Variable", "y": "Importance relative"},
                         title="Importance des Variables (Gini)", **PLOTLY_THEME)
            fig.update_coloraxes(showscale=False)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Métriques de Classification</div>',
                    unsafe_allow_html=True)
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
            'La projection ACP 2D permet de visualiser la séparation entre classes. '
            'Les formes distinctes différencient les vrais labels des prédictions du modèle.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — CLUSTERING  (FIX 5: len(df) < 3 guard)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">K-Means — Profils Émotionnels</div>',
                unsafe_allow_html=True)

    # FIX 5: need at least MIN_ROWS_ML rows for KMeans
    if len(df) < MIN_ROWS_ML:
        st.warning(
            f"Au moins {MIN_ROWS_ML} enregistrements sont nécessaires pour le clustering. "
            f"Actuellement : {len(df)} ligne(s). Veuillez ajouter plus de données via le formulaire."
        )
    else:
        # Cap k so it never exceeds the number of samples
        max_k = min(6, len(df))
        k = st.slider("Nombre de clusters K", 2, max_k, min(3, max_k), key="k_slider")
        labels_km, _ = run_kmeans(df, k=k)
        df_cluster = df.copy()
        df_cluster["cluster"] = [f"Profil {c+1}" for c in labels_km]

        # ── Elbow ──
        st.markdown('<div class="section-title">Méthode du Coude (Elbow Method)</div>',
                    unsafe_allow_html=True)
        X_sc_km   = StandardScaler().fit_transform(df[FEATURES + [TARGET]])
        max_elbow = min(9, len(df) + 1)
        inertias  = [
            KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X_sc_km).inertia_
            for ki in range(2, max_elbow)
        ]
        fig_elbow = px.line(x=list(range(2, max_elbow)), y=inertias, markers=True,
                            labels={"x": "Nombre de clusters K",
                                    "y": "Inertie (Within-cluster SS)"},
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

        # ── Radar ──
        st.markdown('<div class="section-title">Profil Radar des Centroides</div>',
                    unsafe_allow_html=True)
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
            font=dict(family="Inter, sans-serif", color="#2c3e50"),
            title="Profil Radar moyen par Cluster",
            template="plotly_white",
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Summary table ──
        st.markdown('<div class="section-title">Tableau Résumé des Clusters</div>',
                    unsafe_allow_html=True)
        summary = df_cluster.groupby("cluster").agg(
            Participants        =("mood_score",         "count"),
            Humeur_moyenne      =("mood_score",         lambda x: round(x.mean(), 1)),
            Sommeil_moyen       =("sleep_hours",         lambda x: round(x.mean(), 1)),
            Stress_moyen        =("stress",              lambda x: round(x.mean(), 1)),
            Interaction_sociale =("social_interaction",  lambda x: round(x.mean(), 1)),
            Pct_Sains           =("health_status",       lambda x: f"{(x=='Healthy').mean()*100:.0f}%"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="info-box">📌 <b>Interprétation :</b> Chaque cluster représente un profil '
            'émotionnel distinct. Le radar compare toutes les dimensions par groupe. '
            'Ajustez K avec le curseur ci-dessus pour explorer différentes granularités.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<p style='text-align:center;color:#9aa5b1;font-size:.78rem;'>"
    "INF 232 EC2 · Analyse de données de Santé Mentale · "
    f"Streamlit · Supabase (<code>{TABLE}</code>) · scikit-learn · Plotly"
    "</p>",
    unsafe_allow_html=True,
)
