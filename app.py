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
.main .block-container { padding-top: 2rem; background-color: #f8f9fa; }
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
.page-header p { color: #5f6b7a; font-size: .88rem; margin: 0; }
.section-title {
    font-family: 'Source Serif 4', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a2e;
    border-bottom: 2px solid #1a73e8;
    padding-bottom: .3rem;
    margin: 2rem 0 1rem 0;
}
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
div[data-testid="stDataFrame"] { border-radius: 6px; border: 1px solid #dde1e7; }
h1, h2, h3, h4 { font-family: 'Source Serif 4', serif; color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
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
#  SAFETY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _can_plot(df: pd.DataFrame, *cols: str) -> bool:
    if df is None or df.empty:
        return False
    return all(df[c].nunique() > 1 for c in cols if c in df.columns)

def _no_data() -> None:
    st.info("Données insuffisantes pour générer ce graphique. Continuez la collecte !")

# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE
# ══════════════════════════════════════════════════════════════════════════════
TABLE = "mental_health_data"

@st.cache_resource
def get_supabase_client() -> Client:
    try:
        url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
        key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    except KeyError as e:
        st.error(f"❌ Clé manquante : `{e}`")
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
#  CONSTANTS & ML
# ══════════════════════════════════════════════════════════════════════════════
FEATURES    = ["sleep_hours", "stress", "social_interaction"]
TARGET      = "mood_score"
MIN_ROWS_ML = 3

def add_health_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["health_status"] = np.where(
        (df["mood_score"] >= 55) & (df["stress"] <= 6) & (df["sleep_hours"] >= 6),
        "Healthy", "At Risk",
    )
    return df

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
#  SIDEBAR
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
            _cl = get_supabase_client()
            ok, msg = save_record(_cl, {
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
#  LOAD DATA & HEADER
# ══════════════════════════════════════════════════════════════════════════════
supabase_client = get_supabase_client()
df_raw = load_data(supabase_client)

st.markdown("""
<div class="page-header">
  <h1>🧠 INF 232 EC2 : Analyse de données de Santé Mentale</h1>
  <p>Tableau de bord analytique · Intelligence Émotionnelle · Santé Mentale</p>
</div>
""", unsafe_allow_html=True)

if df_raw.empty:
    st.warning("La base de données est vide. Veuillez ajouter une entrée via le formulaire.")
    st.stop()

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
  <div class="metric-card"><div class="val">{n_total}</div><div class="lbl">Participants</div></div>
  <div class="metric-card"><div class="val">{avg_mood:.1f}</div><div class="lbl">Humeur moy.</div></div>
  <div class="metric-card"><div class="val">{avg_sleep:.1f}h</div><div class="lbl">Sommeil moy.</div></div>
  <div class="metric-card"><div class="val">{avg_stress:.1f}</div><div class="lbl">Stress moy.</div></div>
  <div class="metric-card"><div class="val">{n_healthy}/{n_total}</div><div class="lbl">Profil Sain</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Aperçu", "📈 Régression", "🔷 ACP (PCA)", "🌲 Classification", "🔵 Clustering",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — APERÇU
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Distribution des variables</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if not df.empty and df["mood_score"].nunique() > 1:
            fig = px.histogram(
                df, x="mood_score",
                color_discrete_sequence=[CLR_MAIN],
                title="Distribution du Score d'Humeur",
                labels={"mood_score": "Score d'humeur"}
            )
            fig.update_traces(opacity=0.80, marker_line_color="white", marker_line_width=0.5)
            fig.update_layout(showlegend=False, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_data()

    with col2:
        if not df.empty and df["sleep_hours"].nunique() > 1:
            fig = px.histogram(
                df, x="sleep_hours",
                color_discrete_sequence=[CLR_SECONDARY],
                title="Distribution des Heures de Sommeil",
                labels={"sleep_hours": "Heures de sommeil"}
            )
            fig.update_traces(opacity=0.80, marker_line_color="white", marker_line_width=0.5)
            fig.update_layout(showlegend=False, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_data()

    col3, col4 = st.columns(2)
    with col3:
        if not df.empty and df["health_status"].nunique() > 0:
            fig = px.pie(
                df, names="health_status", color="health_status",
                color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
                title="Répartition Sain / À Risque", hole=0.42
            )
            fig.update_traces(textinfo="percent+label", pull=[0.03, 0.03])
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_data()

    with col4:
        if _can_plot(df, *FEATURES, TARGET):
            corr = df[FEATURES + [TARGET]].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=SEQ_BLUES,
                title="Matrice de Corrélation", zmin=-1, zmax=1
            )
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _no_data()

    if len(df) >= MIN_ROWS_ML and _can_plot(df, *FEATURES, TARGET):
        st.markdown('<div class="section-title">Scatter Matrix — Relations entre variables</div>', unsafe_allow_html=True)
        fig = px.scatter_matrix(
            df, dimensions=FEATURES + [TARGET],
            color="health_status",
            color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK},
            title="Matrice de dispersion (toutes variables)"
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.55))
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Données brutes</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["name", "matricule", "sleep_hours", "stress", "social_interaction", "mood_score", "health_status"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RÉGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if len(df) < 2:
        st.warning("Au moins 2 enregistrements sont nécessaires pour la régression.")
    else:
        st.markdown('<div class="section-title">Régression Linéaire Simple — Sommeil → Humeur</div>', unsafe_allow_html=True)
        if not df.empty and df["sleep_hours"].nunique() > 1 and df[TARGET].nunique() > 1:
            model_s, r2_s, rmse_s, ypred_s = run_simple_regression(df)
            x_line = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 200)
            y_line = model_s.predict(x_line.reshape(-1, 1))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["sleep_hours"], y=df[TARGET], mode="markers",
                marker=dict(color=df[TARGET], colorscale=SEQ_VIRIDIS, size=7, opacity=0.70, showscale=True), name="Observations"
            ))
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color=CLR_AT_RISK, width=2.5), name=f"Droite (R²={r2_s:.3f})"))
            fig.update_layout(title="Heures de Sommeil vs Score d'Humeur", xaxis_title="Heures de sommeil", yaxis_title="Score d'humeur", legend=dict(orientation="h", yanchor="bottom", y=1.02), **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2_s:.4f}")
            c2.metric("RMSE", f"{rmse_s:.2f}")
            c3.metric("Coefficient (pente)", f"{model_s.coef_[0]:.3f}")
        else:
            _no_data()

        st.markdown('<div class="section-title">Régression Linéaire Multiple — Tous facteurs → Humeur</div>', unsafe_allow_html=True)
        if _can_plot(df, *FEATURES, TARGET):
            model_m, r2_m, rmse_m, ypred_m, coef_m = run_multiple_regression(df)
            
            if not df.empty and df[TARGET].nunique() > 1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df[TARGET], y=ypred_m, mode="markers", marker=dict(color=CLR_MAIN, opacity=0.55, size=6), name="Prédit vs Réel"))
                fig2.add_trace(go.Scatter(x=[df[TARGET].min(), df[TARGET].max()], y=[df[TARGET].min(), df[TARGET].max()], mode="lines", line=dict(color=CLR_AT_RISK, dash="dash"), name="Idéal"))
                fig2.update_layout(title="Valeurs Réelles vs Prédites", xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites", **PLOTLY_THEME)
                st.plotly_chart(fig2, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R² multiple", f"{r2_m:.4f}")
            with col_b:
                fig3 = px.bar(x=list(coef_m.keys()), y=list(coef_m.values()), color=list(coef_m.values()), color_continuous_scale="RdBu", title="Poids des variables")
                fig3.update_layout(showlegend=False, **PLOTLY_THEME)
                st.plotly_chart(fig3, use_container_width=True)

            residuals = df[TARGET].values - ypred_m
            if len(np.unique(residuals)) > 1:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=ypred_m, y=residuals, mode="markers", marker=dict(color=residuals, colorscale="RdBu", size=5)))
                fig4.update_layout(title="Résidus vs Prédites", **PLOTLY_THEME)
                st.plotly_chart(fig4, use_container_width=True)
        else:
            _no_data()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ACP / PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if len(df) < MIN_ROWS_ML:
        st.warning(f"Au moins {MIN_ROWS_ML} enregistrements nécessaires pour l'ACP.")
    else:
        st.markdown('<div class="section-title">Analyse en Composantes Principales (ACP / PCA)</div>', unsafe_allow_html=True)
        components, explained, loadings = run_pca(df)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["health_status"] = df["health_status"].values
        pca_df["mood_score"]    = df["mood_score"].values
        pca_df["name"]          = df["name"].values if "name" in df.columns else "N/A"

        col1, col2 = st.columns([2, 1])
        with col1:
            if not pca_df.empty and pca_df["PC1"].nunique() > 1 and pca_df["PC2"].nunique() > 1:
                fig = px.scatter(
                    pca_df, x="PC1", y="PC2", color="health_status", size="mood_score", size_max=16,
                    color_discrete_map={"Healthy": CLR_HEALTHY, "At Risk": CLR_AT_RISK}, hover_name="name", title="Projection ACP"
                )
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02), **PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)
            else:
                _no_data()

        st.markdown('<div class="section-title">Biplot</div>', unsafe_allow_html=True)
        if not pca_df.empty and pca_df["PC1"].nunique() > 1:
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=pca_df["PC1"], y=pca_df["PC2"], mode="markers", marker=dict(color=CLR_MAIN, size=5, opacity=0.40)))
            fig5.update_layout(title="Biplot ACP", **PLOTLY_THEME)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            _no_data()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if len(df) < MIN_ROWS_ML:
        st.warning(f"Au moins {MIN_ROWS_ML} enregistrements nécessaires pour la classification.")
    else:
        clf, report, cm, importances, y_te, y_pred_rf = run_random_forest(df)
        col1, col2 = st.columns(2)
        with col1:
            cm_df = pd.DataFrame(cm, index=["Réel:Healthy", "Réel:Risk"], columns=["Prédit:Healthy", "Prédit:Risk"])
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=SEQ_BLUES, title="Matrice de Confusion")
            fig.update_layout(xaxis=dict(side="bottom"), **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(x=list(importances.keys()), y=list(importances.values()), color=list(importances.values()), color_continuous_scale=SEQ_VIRIDIS, title="Importance Variables")
            fig.update_layout(showlegend=False, **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Précision globale (Accuracy)", f"{report.get('accuracy', 0)*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    if len(df) < MIN_ROWS_ML:
        st.warning(f"Au moins {MIN_ROWS_ML} enregistrements nécessaires pour le clustering.")
    else:
        k = st.slider("Nombre de clusters K", 2, min(6, len(df)), 3)
        labels_km, _ = run_kmeans(df, k=k)
        df_cluster = df.copy()
        df_cluster["cluster"] = [f"Profil {c+1}" for c in labels_km]

        col1, col2 = st.columns(2)
        with col1:
            if not df_cluster.empty and df_cluster["sleep_hours"].nunique() > 1:
                fig7 = px.scatter(df_cluster, x="sleep_hours", y="mood_score", color="cluster", size="stress", title="Clusters — Sommeil vs Humeur", color_discrete_sequence=QUAL_PALETTE)
                fig7.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig7, use_container_width=True)
            else:
                _no_data()

        with col2:
            if not df_cluster.empty and df_cluster["stress"].nunique() > 1:
                fig8 = px.scatter(df_cluster, x="stress", y="social_interaction", color="cluster", size="mood_score", title="Clusters — Stress vs Social", color_discrete_sequence=QUAL_PALETTE)
                fig8.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig8, use_container_width=True)
            else:
                _no_data()

st.divider()
st.markdown("<p style='text-align:center;color:#9aa5b1;font-size:.78rem;'>INF 232 EC2 · Supabase · scikit-learn · Plotly</p>", unsafe_allow_html=True)
