"""
SMART-PDM: AI-Based Predictive Maintenance for Home Appliances
Streamlit portfolio application
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GroupKFold, cross_val_predict,
)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
)
from sklearn.decomposition import PCA

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMART-PDM | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "2-washing_machines")
FRIDGE_DIR = os.path.join(BASE_DIR, "..", "3-fridges")
WINDOWED_FEATURES_PATH = os.path.join(DATA_DIR, "WM_WindowedFeatures.csv")

METADATA_COLS = [
    "cycle_id", "begin_ts", "end_ts", "brand", "model",
    "program", "temperature", "spin_speed", "load",
    "fault_condition", "notes",
]

FAULT_COLORS = {
    "Working":  "#2ecc71",
    "Heating":  "#e74c3c",
    "Bearings": "#3498db",
    "Motor":    "#9b59b6",
}

FAULT_ICONS = {
    "Working":  "✅",
    "Heating":  "🔥",
    "Bearings": "⚙️",
    "Motor":    "⚡",
}

# Refrigerator health labels (binary: healthy vs faulty).
FRIDGE_COLORS = {
    "Normal":      "#3498db",
    "Malfunction": "#e74c3c",
}
FRIDGE_ICONS = {
    "Normal":      "✅",
    "Malfunction": "⚠️",
}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0;
        line-height: 1.15;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }
    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #7c4dff;
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    /* Fault badges */
    .fault-card {
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 6px 0;
    }
    /* Metric overrides */
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #6c757d !important;
    }
    /* Divider */
    hr { border-color: #e9ecef; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (all cached) ─────────────────────────────────────────────────

@st.cache_data
def load_metadata():
    path = os.path.join(DATA_DIR, "washing_machine_metadata.csv")
    df = pd.read_csv(path, header=None, names=METADATA_COLS)
    df["notes"] = df["notes"].fillna("")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["spin_speed"] = pd.to_numeric(df["spin_speed"], errors="coerce")
    return df


@st.cache_data
def load_features():
    path = os.path.join(DATA_DIR, "WM_ExtractedFeatures.csv")
    return pd.read_csv(path)


@st.cache_data
def load_merged():
    features = load_features()
    meta = load_metadata().drop_duplicates(subset="cycle_id")
    merged = features.merge(
        meta[["cycle_id", "fault_condition", "brand", "model",
              "program", "temperature", "spin_speed", "load"]],
        left_on="Id",
        right_on="cycle_id",
        how="inner",
    )
    return merged


@st.cache_data
def load_slow_signal(cycle_id: str):
    path = os.path.join(DATA_DIR, cycle_id, "slow.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_fast_signal(cycle_id: str, nrows: int = 300_000):
    """Read fast CSV, capped at nrows to keep load time acceptable.

    Fast files can be ~22 M rows (2 kHz × 3 h). We read the first `nrows`
    rows (~150 s of data), which are then downsampled again for plotting.
    """
    path = os.path.join(DATA_DIR, cycle_id, "fast.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, nrows=nrows)
    df.columns = df.columns.str.strip()
    return df


# ── Fridge raw-signal loaders (mirror the washing-machine loaders above) ───────
@st.cache_data
def load_fridge_slow(cycle_id: str):
    """Slow electrical stream (1 Hz) for one fridge cycle — same schema as WM."""
    path = os.path.join(FRIDGE_DIR, cycle_id, "slow.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_fridge_fast(cycle_id: str, nrows: int = 300_000):
    """High-frequency Current/Vibration burst for one fridge cycle, capped."""
    path = os.path.join(FRIDGE_DIR, cycle_id, "fast.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, nrows=nrows)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_fridge_24h(cycle_id: str):
    """24-hour Active-Power trace (1 Hz) — unique to fridges; shows duty cycling."""
    path = os.path.join(FRIDGE_DIR, cycle_id, "24h.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Which cycles have raw signals available on disk ───────────────────────────
# The full multi-GB raw dataset isn't bundled in the deployed demo, so the
# signal-explorer pages restrict their pickers to cycles that are actually
# present. This is adaptive: locally you get every cycle; online you get the
# curated demo subset — with no code change.
@st.cache_data
def wm_cycles_with_signal():
    if not os.path.isdir(DATA_DIR):
        return set()
    return {d for d in os.listdir(DATA_DIR)
            if os.path.exists(os.path.join(DATA_DIR, d, "slow.csv"))}


@st.cache_data
def fridge_cycles_with_signal():
    if not os.path.isdir(FRIDGE_DIR):
        return set()
    return {d for d in os.listdir(FRIDGE_DIR)
            if os.path.exists(os.path.join(FRIDGE_DIR, d, "slow.csv"))
            or os.path.exists(os.path.join(FRIDGE_DIR, d, "24h.csv"))}


@st.cache_data
def train_model():
    merged = load_merged()
    meta_cols = {"Id", "cycle_id", "fault_condition", "brand", "model",
                 "program", "temperature", "spin_speed", "load"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].values
    y_raw = merged["fault_condition"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="f1_macro")

    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    return clf, scaler, le, feature_cols, cv_scores, y, y_pred


# ── Windowed dataset (per-window features) ────────────────────────────────────
@st.cache_data
def load_windowed():
    """Load the windowed feature table and attach labels via cycle_id.

    Returns None if the windowing pipeline hasn't been run yet.
    """
    if not os.path.exists(WINDOWED_FEATURES_PATH):
        return None
    wf = pd.read_csv(WINDOWED_FEATURES_PATH)
    meta = load_metadata().drop_duplicates(subset="cycle_id")
    merged = wf.merge(
        meta[["cycle_id", "fault_condition", "brand", "model"]],
        on="cycle_id",
        how="inner",
    )
    return merged


@st.cache_data
def train_windowed_model():
    """Train RF on windowed features, evaluated with GroupKFold grouped by cycle.

    Grouping by cycle is essential: windows from one cycle are near-duplicates
    sharing a label, so a plain split would leak them across train/test and
    inflate the score. GroupKFold guarantees a cycle is never in both.
    """
    merged = load_windowed()
    if merged is None:
        return None

    meta_cols = {"Id", "cycle_id", "fault_condition", "brand", "model"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].values
    y_raw = merged["fault_condition"].values
    groups = merged["cycle_id"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                 class_weight="balanced", n_jobs=-1)

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = cross_val_score(clf, X_scaled, y, groups=groups, cv=gkf,
                                scoring="f1_macro")
    # Out-of-fold predictions for an honest (leak-free) confusion matrix.
    y_oof = cross_val_predict(clf, X_scaled, y, groups=groups, cv=gkf)

    # Deliberately-wrong baseline: a plain shuffled split ignores cycle groups,
    # letting near-duplicate windows leak across folds. Shown only to expose how
    # much such leakage inflates the reported score.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    leaky_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="f1_macro")

    clf.fit(X_scaled, y)
    return {
        "clf": clf, "scaler": scaler, "le": le, "feature_cols": feature_cols,
        "cv_scores": cv_scores, "leaky_scores": leaky_scores,
        "y_true": y, "y_oof": y_oof,
        "n_windows": len(merged), "n_cycles": len(np.unique(groups)),
        "n_splits": n_splits, "merged": merged,
    }


# ── Fridge anomaly detection ──────────────────────────────────────────────────
@st.cache_data
def load_fridge():
    """Load fridge features merged with Normal/Malfunction labels."""
    feat = pd.read_csv(os.path.join(FRIDGE_DIR, "F_ExtractedFeatures.csv"))
    meta = pd.read_csv(os.path.join(FRIDGE_DIR, "fridge_metadata.csv"))
    merged = feat.merge(meta[["begin_end", "brand", "model", "failure"]],
                        left_on="Id", right_on="begin_end", how="inner")
    return merged


@st.cache_data
def run_fridge_anomaly():
    """Unsupervised IsolationForest on fridge cycles; labels used only to score.

    Trained without labels (pure anomaly detection). The 24 known Malfunctions
    are then used purely to *evaluate* how well the anomaly score separates them
    from the 1,087 Normal cycles.
    """
    merged = load_fridge()
    meta_cols = {"Id", "begin_end", "brand", "model", "failure"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = (merged["failure"] != "Normal").astype(int).values  # 1 = malfunction

    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=400, contamination=float(y.mean()),
                          random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    score = -iso.score_samples(X_scaled)  # higher = more anomalous

    roc_auc = roc_auc_score(y, score)
    pr_auc = average_precision_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    prec, rec, _ = precision_recall_curve(y, score)

    # precision/recall if we flag the top-k% most anomalous cycles
    budget = []
    for frac in (0.02, 0.05, 0.10, 0.20):
        thr = np.quantile(score, 1 - frac)
        pred = score >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        budget.append({
            "Flag top": f"{int(frac*100)}%",
            "Cycles reviewed": int(pred.sum()),
            "Malfunctions caught": f"{tp} / {int(y.sum())}",
            "Recall": tp / max(int(y.sum()), 1),
            "Precision": tp / max(tp + fp, 1),
        })

    merged = merged.copy()
    merged["anomaly_score"] = score
    return {
        "merged": merged, "y": y, "score": score,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
        "budget": pd.DataFrame(budget), "base_rate": float(y.mean()),
        "n_total": len(y), "n_anom": int(y.sum()),
    }


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SMART-PDM")
    st.markdown("*AI-Powered Predictive Maintenance*")
    st.markdown("---")

    # Level 1 — pick the appliance domain.
    appliance = st.radio(
        "Appliance",
        ["🏠  Overview", "🧺  Washing Machines", "🧊  Refrigerators"],
        label_visibility="collapsed",
    )

    # Level 2 — pick a section within that appliance. The resulting `page` values
    # reuse the original washing-machine keys unchanged, so their page blocks below
    # need no edits; fridge pages use their own namespaced keys.
    if appliance == "🏠  Overview":
        page = "🏠  Overview"
    elif appliance == "🧺  Washing Machines":
        st.markdown("**🧺 Washing Machines**")
        page = st.radio(
            "Washing-machine section",
            [
                "🔍  Signal Explorer",
                "⚙️  Cycle Detection",
                "📊  Feature Analysis",
                "🤖  Fault Classifier",
                "🪟  Windowed Model",
            ],
            label_visibility="collapsed",
        )
    else:  # Refrigerators
        st.markdown("**🧊 Refrigerators**")
        page = st.radio(
            "Refrigerator section",
            [
                "🔍  Fridge Signal Explorer",
                "📊  Fridge Feature Analysis",
                "🧊  Fridge Anomalies",
            ],
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown(
        "<small>Dataset: SMART-PDM (EU project)<br>"
        "Built with Streamlit · scikit-learn · Plotly</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<p class="main-header">⚙️ SMART-PDM</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Based Predictive Maintenance for Home Appliances</p>',
        unsafe_allow_html=True,
    )

    meta = load_metadata()
    unique = meta.drop_duplicates(subset="cycle_id")
    fridge = load_fridge()

    wm_cycles = len(unique)
    fridge_cycles = len(fridge)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Appliance Types", "2", help="Washing machines · Refrigerators")
    c2.metric("Cycles Monitored", f"{wm_cycles + fridge_cycles:,}")
    c3.metric("Washing-Machine Cycles", f"{wm_cycles}")
    c4.metric("Refrigerator Cycles", f"{fridge_cycles:,}")

    st.divider()

    st.markdown('<p class="section-title">What is Predictive Maintenance?</p>', unsafe_allow_html=True)
    st.markdown("""
    **Predictive Maintenance (PdM)** uses real-time sensor data and machine learning to
    detect faults *before* a machine fails — reducing downtime and repair costs compared
    to scheduled or reactive maintenance.

    This project applies PdM to **household appliances** using the SMART-PDM dataset,
    monitoring their electrical consumption (Active Power, Current) and mechanical
    **Vibration**. It covers two appliance types, each with the modelling approach that
    best fits its data:
    """)

    ac_wm, ac_fr = st.columns(2, gap="large")
    with ac_wm:
        st.markdown(f"""
        <div style="background:#7c4dff10; border-left:5px solid #7c4dff;
                    border-radius:8px; padding:16px 18px; height:100%;">
            <div style="font-size:1.2rem; font-weight:800;">🧺 Washing Machines</div>
            <div style="color:#444; font-size:0.92rem; margin-top:6px;">
              <b>{wm_cycles}</b> wash cycles · <b>4</b> brands ·
              <b>{meta['fault_condition'].nunique()}</b> health states<br>
              <b>Approach:</b> supervised <b>fault classification</b>
              (Working / Heating / Bearings / Motor) with a Random Forest, plus a
              window-based model evaluated leak-free with GroupKFold.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ac_fr:
        n_mal = int((fridge["failure"] != "Normal").sum())
        st.markdown(f"""
        <div style="background:#3498db10; border-left:5px solid #3498db;
                    border-radius:8px; padding:16px 18px; height:100%;">
            <div style="font-size:1.2rem; font-weight:800;">🧊 Refrigerators</div>
            <div style="color:#444; font-size:0.92rem; margin-top:6px;">
              <b>{fridge_cycles:,}</b> cycles · only <b>{n_mal}</b> malfunctions
              (<b>{n_mal/fridge_cycles*100:.1f}%</b>)<br>
              <b>Approach:</b> unsupervised <b>anomaly detection</b> — with so few faults,
              the model learns "normal" and flags outliers (IsolationForest).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    **Shared end-to-end pipeline:**

    | Step | Tool |
    |---|---|
    | 1. Acquire sensor data | Smart plug + accelerometer |
    | 2. Detect working cycles | Threshold state machine |
    | 3. Extract features | tsfresh (25 features × 3 sensors = 75) |
    | 4. Dimensionality reduction | PCA |
    | 5. Model | Random Forest (WM) · IsolationForest (fridge) |
    """)

    # ── Washing-machine dataset ────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">🧺 Washing Machine Dataset</p>', unsafe_allow_html=True)

    fault_desc = {
        "Working":  "Healthy, normal operation",
        "Heating":  "Heating element malfunction",
        "Bearings": "Bearing wear or damage",
        "Motor":    "Motor fault or degradation",
    }
    fcols = st.columns(len(fault_desc))
    for col, (fault, desc) in zip(fcols, fault_desc.items()):
        color = FAULT_COLORS[fault]
        icon = FAULT_ICONS[fault]
        count = len(unique[unique["fault_condition"] == fault])
        col.markdown(f"""
        <div style="background:{color}15; border-left:5px solid {color};
                    border-radius:8px; padding:12px 14px; margin-bottom:10px;">
            <span style="font-size:1.05rem; font-weight:700;">{icon} {fault}</span>
            <span style="float:right; background:{color}30; color:{color};
                         border-radius:12px; padding:1px 9px; font-size:0.78rem;
                         font-weight:600;">{count}</span><br>
            <span style="color:#555; font-size:0.85rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">Cycle Distribution by Fault &amp; Wash Program</p>', unsafe_allow_html=True)
    dist = (
        unique.groupby(["fault_condition", "program"])
        .size()
        .reset_index(name="cycles")
    )
    fig = px.bar(
        dist, x="program", y="cycles", color="fault_condition",
        color_discrete_map=FAULT_COLORS,
        barmode="group",
        labels={"program": "Wash Program", "cycles": "Number of Cycles",
                "fault_condition": "Condition"},
        height=380,
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend_title_text="Condition",
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">Machines in the Dataset</p>', unsafe_allow_html=True)
    machines = (
        meta.groupby(["brand", "model"])
        .agg(
            cycles=("cycle_id", "nunique"),
            programs=("program", lambda x: " · ".join(sorted(x.dropna().unique()))),
            faults=("fault_condition", lambda x: " · ".join(sorted(x.unique()))),
        )
        .reset_index()
    )
    machines.columns = ["Brand", "Model", "Cycles", "Programs", "Conditions"]
    st.dataframe(machines, use_container_width=True, hide_index=True)

    # ── Refrigerator dataset ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">🧊 Refrigerator Dataset</p>', unsafe_allow_html=True)

    fr_left, fr_right = st.columns([1, 1.1], gap="large")
    with fr_left:
        fr_counts = (fridge["failure"].value_counts()
                     .rename_axis("Condition").reset_index(name="Cycles"))
        fig_fr = px.bar(
            fr_counts, x="Condition", y="Cycles", color="Condition",
            color_discrete_map=FRIDGE_COLORS, text="Cycles", height=340,
            title="Cycles by Health Condition",
            log_y=True,
            category_orders={"Condition": ["Normal", "Malfunction"]},
        )
        fig_fr.update_layout(showlegend=False, plot_bgcolor="white",
                             paper_bgcolor="white")
        fig_fr.update_yaxes(title="Cycles (log scale)", showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_fr, use_container_width=True)
    with fr_right:
        n_mal = int((fridge["failure"] != "Normal").sum())
        brand = fridge["brand"].mode().iloc[0] if "brand" in fridge else "SMEG"
        st.markdown(f"""
        The refrigerator dataset is large but **extremely imbalanced**:
        **{fridge_cycles:,} cycles** from a single brand (**{brand}**), with only
        **{n_mal} malfunctions** ({n_mal/fridge_cycles*100:.1f}%). That is far too few
        faulty examples to train a reliable supervised classifier.

        Instead it is framed as **anomaly detection** — the model learns what a *normal*
        cooling cycle looks like and flags outliers, so the imbalance becomes the point
        rather than a weakness. See the **Refrigerators → Anomaly Detection** page for
        the results (ROC-AUC ≈ 0.88).

        Refrigerators also record a **24-hour Active-Power trace** per cycle, revealing the
        compressor's duty-cycling — explore it under **Refrigerators → Signal Explorer**.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIGNAL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Signal Explorer":
    st.markdown('<p class="main-header">🔍 Signal Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Browse raw sensor signals for individual wash cycles</p>',
        unsafe_allow_html=True,
    )

    meta = load_metadata()

    # Restrict to cycles whose raw signals are actually present (full set locally,
    # curated subset in the deployed demo).
    avail = wm_cycles_with_signal()
    meta = meta[meta["cycle_id"].astype(str).isin(avail)]
    if meta.empty:
        st.info("No raw-signal cycles are bundled in this demo. The analysis pages "
                "(Feature Analysis, Fault Classifier, Windowed Model) run on the full "
                "precomputed feature set — try those instead.")
        st.stop()
    st.caption(f"Showing {meta['cycle_id'].nunique()} cycle(s) with raw signals available.")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand_f = st.selectbox("Brand", ["All"] + sorted(meta["brand"].unique().tolist()))
    with c2:
        fault_f = st.selectbox("Fault Condition", ["All"] + sorted(meta["fault_condition"].unique().tolist()))
    with c3:
        prog_f = st.selectbox("Program", ["All"] + sorted(meta["program"].dropna().unique().tolist()))

    filt = meta.copy()
    if brand_f != "All":
        filt = filt[filt["brand"] == brand_f]
    if fault_f != "All":
        filt = filt[filt["fault_condition"] == fault_f]
    if prog_f != "All":
        filt = filt[filt["program"] == prog_f]
    filt = filt.drop_duplicates(subset="cycle_id")

    if filt.empty:
        st.warning("No cycles match the selected filters.")
        st.stop()

    cycle_id = st.selectbox(
        f"Select Cycle  ({len(filt)} found)",
        filt["cycle_id"].tolist(),
    )
    row = filt[filt["cycle_id"] == cycle_id].iloc[0]

    # Metadata strip
    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("Brand", row["brand"])
    cb.metric("Program", row["program"])
    cc.metric("Temperature", f"{int(row['temperature'])}°C" if pd.notna(row["temperature"]) else "—")
    cd.metric("Load", row["load"])
    color = FAULT_COLORS.get(row["fault_condition"], "#888")
    ce.markdown(f"""
    <div style='text-align:center; padding-top:0.9rem;'>
      <span style='background:{color}20; color:{color}; border:2px solid {color};
                   border-radius:20px; padding:5px 18px; font-weight:700; font-size:0.95rem;'>
        {FAULT_ICONS.get(row["fault_condition"], "")} {row["fault_condition"]}
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Slow stream (electrical) ──────────────────────────────────────────────
    slow_df = load_slow_signal(cycle_id)
    if slow_df is not None:
        slow_df["datetime"] = pd.to_datetime(slow_df["Ts"], unit="s")
        fault_color = FAULT_COLORS.get(row["fault_condition"], "#667eea")

        # Downsample to ≤ 2 000 points so the browser stays responsive
        ds_step = max(1, len(slow_df) // 2000)
        sd = slow_df.iloc[::ds_step]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Active Power (W)", "Voltage (mV)", "Current (cA)"),
            vertical_spacing=0.07,
        )
        fig.add_trace(go.Scatter(x=sd["datetime"], y=sd["ActP"],
                                 mode="lines", name="Active Power",
                                 line=dict(color=fault_color, width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sd["datetime"], y=sd["V"],
                                 mode="lines", name="Voltage",
                                 line=dict(color="#f39c12", width=1.3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=sd["datetime"], y=sd["A"],
                                 mode="lines", name="Current",
                                 line=dict(color="#16a085", width=1.3)), row=3, col=1)
        fig.update_layout(
            height=530,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            title_text="Electrical Signals — Slow Stream (1 Hz)",
            title_font_size=14,
        )
        for r in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", row=r, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", row=r, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Slow stream data not available for this cycle.")

    # ── Fast stream — loaded only when user requests it ───────────────────────
    if st.checkbox("⚡ Show high-frequency signals (Current & Vibration — large file)"):
        fast_df = load_fast_signal(cycle_id)
        if fast_df is not None:
            time_col = fast_df.columns[0]
            fast_df["time_s"] = fast_df[time_col] / 1e6  # µs → s

            # Downsample to ≤ 5 000 points for browser performance
            step = max(1, len(fast_df) // 5000)
            fd = fast_df.iloc[::step].reset_index(drop=True)

            fig2 = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=("Current — Fast Stream (ADC units)", "Vibration (ADC units)"),
                vertical_spacing=0.1,
            )
            fig2.add_trace(go.Scatter(x=fd["time_s"], y=fd["Current"],
                                      mode="lines", name="Current",
                                      line=dict(color="#16a085", width=0.9)), row=1, col=1)
            fig2.add_trace(go.Scatter(x=fd["time_s"], y=fd["Vibration"],
                                      mode="lines", name="Vibration",
                                      line=dict(color="#8e44ad", width=0.9)), row=2, col=1)
            fig2.update_layout(
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                title_text="High-Frequency Signals — Fast Stream (first ~150 s shown)",
                title_font_size=14,
            )
            fig2.update_xaxes(title_text="Time (s)", row=2, col=1,
                              showgrid=True, gridcolor="#f0f0f0")
            fig2.update_xaxes(showgrid=True, gridcolor="#f0f0f0", row=1, col=1)
            fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Fast stream data not available for this cycle.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CYCLE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Cycle Detection":
    st.markdown('<p class="main-header">⚙️ Cycle Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">How working cycles are identified from raw power signals</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.markdown('<p class="section-title">Algorithm</p>', unsafe_allow_html=True)
        st.markdown("""
        Cycles are detected from the **slow stream** (1 Hz Active Power readings) using a
        state machine with two thresholds and a debounce counter:

        1. **Cycle starts** when power rises above the **ON threshold** while the machine is idle
        2. The counter increments each second power stays **below the OFF threshold**
        3. **Cycle ends** once the counter reaches the debounce limit (≥ 30 s)
        4. Short events (< 12 min) are discarded as noise

        The debounce counter **resets** whenever power recovers above the OFF threshold
        — making the detector robust to brief dips during spin or rinse pauses.
        """)

    with right:
        st.markdown('<p class="section-title">Detection Parameters</p>', unsafe_allow_html=True)
        on_threshold  = st.slider("ON threshold (W)", 1, 50, 10, 1)
        off_threshold = st.slider("OFF threshold (W)", 1, 20, 3, 1)
        debounce_s    = st.slider("Debounce (seconds)", 5, 60, 30, 5)
        min_dur_min   = st.slider("Minimum cycle duration (min)", 1, 30, 12, 1)

    meta = load_metadata()
    avail = wm_cycles_with_signal()
    sessions = [c for c in meta.drop_duplicates(subset="cycle_id")["cycle_id"].tolist()
                if str(c) in avail]
    if not sessions:
        st.info("No raw-signal sessions are bundled in this demo.")
        st.stop()
    cycle_id = st.selectbox(
        f"Select a monitoring session  ({len(sessions)} available)",
        sessions,
    )

    slow_df = load_slow_signal(cycle_id)
    if slow_df is None:
        st.error(f"Could not load slow stream for: {cycle_id}")
        st.stop()

    slow_df["datetime"] = pd.to_datetime(slow_df["Ts"], unit="s")

    # Run detection using numpy arrays (fast, avoids iterrows overhead)
    ts_arr   = slow_df["Ts"].values
    actp_arr = slow_df["ActP"].values
    min_margin = min_dur_min * 60

    lst_cycles = []
    flag = 0
    counter = 0
    begin_ts = None

    for i in range(len(actp_arr)):
        v = actp_arr[i]
        t = ts_arr[i]
        if v > on_threshold and flag == 0:
            flag = 1
            counter = 0
            begin_ts = t
        elif flag == 1:
            if v < off_threshold:
                if counter >= debounce_s:
                    flag = 0
                    if (t - begin_ts) >= min_margin:
                        lst_cycles.append((begin_ts, t))
                else:
                    counter += 1
            else:
                counter = 0

    # Downsample for plotting (detection ran on full data above)
    ds = max(1, len(slow_df) // 2000)
    sd = slow_df.iloc[::ds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sd["datetime"],
        y=sd["ActP"],
        mode="lines",
        name="Active Power (W)",
        line=dict(color="#667eea", width=1.5),
    ))

    for i, (b, e) in enumerate(lst_cycles):
        b_dt = pd.to_datetime(b, unit="s")
        e_dt = pd.to_datetime(e, unit="s")
        dur_min = (e - b) / 60
        fig.add_vrect(
            x0=b_dt, x1=e_dt,
            fillcolor="#2ecc71", opacity=0.12,
            line_width=1.5, line_color="#2ecc71",
            annotation_text=f"#{i+1} ({dur_min:.0f} min)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#27ae60"),
        )

    fig.add_hline(
        y=on_threshold,
        line_dash="dash", line_color="#e74c3c",
        annotation_text=f"ON threshold = {on_threshold} W",
        annotation_position="top right",
        annotation_font_size=11,
    )
    fig.add_hline(
        y=off_threshold,
        line_dash="dot", line_color="#f39c12",
        annotation_text=f"OFF threshold = {off_threshold} W",
        annotation_position="bottom right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=f"Power Signal — {len(lst_cycles)} cycle(s) detected",
        xaxis_title="Time",
        yaxis_title="Active Power (W)",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    if lst_cycles:
        st.success(f"**{len(lst_cycles)} cycle(s) detected.** "
                   f"Durations: "
                   + ", ".join(f"**{(e-b)/60:.1f} min**" for b, e in lst_cycles))
    else:
        st.warning("No valid cycles detected with the current parameters.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Feature Analysis":
    st.markdown('<p class="main-header">📊 Feature Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Statistical and frequency-domain features extracted per cycle</p>',
        unsafe_allow_html=True,
    )

    merged = load_merged()
    meta_cols = {"Id", "cycle_id", "fault_condition", "brand", "model",
                 "program", "temperature", "spin_speed", "load"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    tab_dist, tab_pca = st.tabs(["📦 Feature Distributions", "🔵 PCA"])

    # ── Tab 1: distributions ─────────────────────────────────────────────────
    with tab_dist:
        st.markdown('<p class="section-title">Distribution by Fault Class</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sensor = st.selectbox("Sensor", ["ActP", "Current", "Vibration"])
        sensor_feats = [f for f in feature_cols if f.startswith(sensor + "__")]
        labels = [f.replace(sensor + "__", "").replace("__", " ") for f in sensor_feats]
        with c2:
            sel_label = st.selectbox("Feature", labels)
        sel_col = sensor_feats[labels.index(sel_label)]

        fig_box = px.box(
            merged,
            x="fault_condition",
            y=sel_col,
            color="fault_condition",
            color_discrete_map=FAULT_COLORS,
            points="all",
            labels={"fault_condition": "Condition", sel_col: f"{sensor} — {sel_label}"},
            title=f"{sensor} · {sel_label}  —  Distribution by Fault Condition",
            height=470,
            category_orders={"fault_condition": list(FAULT_COLORS.keys())},
        )
        fig_box.update_traces(jitter=0.3, marker_size=5)
        fig_box.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        fig_box.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("Summary statistics"):
            stats = merged.groupby("fault_condition")[sel_col].describe().round(4)
            st.dataframe(stats, use_container_width=True)

    # ── Tab 2: PCA ───────────────────────────────────────────────────────────
    with tab_pca:
        st.markdown('<p class="section-title">Principal Component Analysis</p>', unsafe_allow_html=True)

        max_pcs = min(10, len(feature_cols))
        n_components = st.slider("Number of components to compute", 2, max_pcs, 2)

        X_raw = merged[feature_cols].values.astype(float)
        valid = ~np.isnan(X_raw).any(axis=1) & ~np.isinf(X_raw).any(axis=1)
        X_clean = X_raw[valid]
        labels_clean = merged["fault_condition"].values[valid]

        X_scaled = StandardScaler().fit_transform(X_clean)
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X_scaled)

        col_pca, col_var = st.columns([1.6, 1], gap="large")

        with col_pca:
            pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
            pc_df["Condition"] = labels_clean

            fig_pca = px.scatter(
                pc_df, x="PC1", y="PC2",
                color="Condition",
                color_discrete_map=FAULT_COLORS,
                title="First 2 Principal Components",
                height=460,
                opacity=0.8,
                category_orders={"Condition": list(FAULT_COLORS.keys())},
            )
            fig_pca.update_traces(
                marker=dict(size=9, line=dict(width=0.8, color="white"))
            )
            fig_pca.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(title="Condition", itemsizing="constant"),
            )
            fig_pca.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ddd")
            fig_pca.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ddd")
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_var:
            var_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(n_components)],
                "Variance (%)": (pca.explained_variance_ratio_ * 100).round(2),
            })
            fig_var = px.bar(
                var_df, x="Component", y="Variance (%)",
                title="Explained Variance per Component",
                color="Variance (%)",
                color_continuous_scale="Purples",
                text="Variance (%)",
                height=460,
            )
            fig_var.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_var.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                coloraxis_showscale=False,
            )
            fig_var.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig_var, use_container_width=True)

        total_2pc = pca.explained_variance_ratio_[:2].sum() * 100
        st.info(
            f"The first 2 PCs explain **{total_2pc:.1f}%** of total variance.  "
            f"The first {n_components} PCs explain "
            f"**{pca.explained_variance_ratio_.sum()*100:.1f}%**."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FAULT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Fault Classifier":
    st.markdown('<p class="main-header">🤖 Fault Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Random Forest trained on 75 tsfresh features to detect appliance faults</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Training classifier…"):
        clf, scaler, le, feature_cols, cv_scores, y_true, y_pred = train_model()

    classes = le.classes_

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Trees", "200")
    c3.metric("CV F1-Macro (5-fold)", f"{cv_scores.mean():.3f}")
    c4.metric("CV Std", f"± {cv_scores.std():.3f}")

    st.divider()

    tab_cm, tab_fi, tab_demo = st.tabs(["📊 Confusion Matrix", "📈 Feature Importance", "🔮 Live Prediction"])

    # ── Confusion matrix ──────────────────────────────────────────────────────
    with tab_cm:
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Samples"),
            x=classes,
            y=classes,
            color_continuous_scale="Purples",
            text_auto=True,
            title="Confusion Matrix",
            height=450,
        )
        fig_cm.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cm, use_container_width=True)

        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    with tab_fi:
        fi_df = (
            pd.DataFrame({"feature": feature_cols, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        # Readable label: "ActP → mean_abs_change"
        fi_df["label"] = (
            fi_df["feature"]
            .str.replace(r"__aggtype_", " ", regex=True)
            .str.replace(r"__bins_", " bins=", regex=True)
            .str.replace("__", " → ", regex=False)
            .str.replace('"', "", regex=False)
        )

        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="label",
            orientation="h",
            title="Top 20 Most Important Features",
            color="importance",
            color_continuous_scale="Purples",
            text=fi_df["importance"].round(4),
            height=620,
        )
        fig_fi.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_fi.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        fig_fi.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Live prediction demo ──────────────────────────────────────────────────
    with tab_demo:
        st.markdown('<p class="section-title">Predict Fault for a Cycle</p>', unsafe_allow_html=True)
        st.markdown("Select any cycle from the dataset. The model returns a fault prediction and confidence scores.")

        merged = load_merged()
        options = merged["Id"].tolist()
        cycle_id = st.selectbox("Cycle", options)
        row = merged[merged["Id"] == cycle_id].iloc[0]

        X_single = row[feature_cols].values.reshape(1, -1)
        X_sc = scaler.transform(X_single)
        pred_class = le.inverse_transform(clf.predict(X_sc))[0]
        pred_probs = clf.predict_proba(X_sc)[0]
        true_label = row["fault_condition"]

        pred_color = FAULT_COLORS.get(pred_class, "#888")
        true_color = FAULT_COLORS.get(true_label, "#888")

        col_pred, col_vs, col_true = st.columns([1, 0.15, 1])
        col_pred.markdown(f"""
        <div style="background:{pred_color}15; border:2.5px solid {pred_color};
                    border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:0.8rem; color:#888; margin-bottom:4px;">MODEL PREDICTION</div>
            <div style="font-size:2rem;">{FAULT_ICONS.get(pred_class, "")}</div>
            <div style="font-size:1.6rem; font-weight:800; color:{pred_color};">{pred_class}</div>
        </div>""", unsafe_allow_html=True)

        col_vs.markdown("<div style='text-align:center;padding-top:2.8rem;font-size:1.4rem;color:#aaa;'>vs</div>",
                        unsafe_allow_html=True)

        col_true.markdown(f"""
        <div style="background:{true_color}15; border:2.5px solid {true_color};
                    border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:0.8rem; color:#888; margin-bottom:4px;">GROUND TRUTH</div>
            <div style="font-size:2rem;">{FAULT_ICONS.get(true_label, "")}</div>
            <div style="font-size:1.6rem; font-weight:800; color:{true_color};">{true_label}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if pred_class == true_label:
            st.success(f"Correct prediction — model identified **{pred_class}**.")
        else:
            st.error(f"Incorrect — predicted **{pred_class}**, actual is **{true_label}**.")

        st.markdown('<p class="section-title">Prediction Confidence</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Condition": le.classes_, "Probability": pred_probs})
        fig_prob = px.bar(
            prob_df,
            x="Condition",
            y="Probability",
            color="Condition",
            color_discrete_map=FAULT_COLORS,
            range_y=[0, 1],
            text=prob_df["Probability"].round(3),
            height=340,
            category_orders={"Condition": list(FAULT_COLORS.keys())},
        )
        fig_prob.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_prob.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        fig_prob.update_yaxes(showgrid=True, gridcolor="#f0f0f0", title="Probability")
        st.plotly_chart(fig_prob, use_container_width=True)

        with st.expander("Cycle metadata"):
            display_cols = ["brand", "model", "program", "temperature", "spin_speed", "load", "fault_condition"]
            avail = [c for c in display_cols if c in row.index]
            st.dataframe(row[avail].to_frame("Value").T, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WINDOWED MODEL (GroupKFold)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🪟  Windowed Model":
    st.markdown('<p class="main-header">🪟 Windowed Model</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Expanding the dataset by slicing each cycle into short '
        'time windows — evaluated leak-free with GroupKFold</p>',
        unsafe_allow_html=True,
    )

    result = train_windowed_model()

    if result is None:
        st.warning(
            "The windowed feature file hasn't been generated yet.\n\n"
            "Run the pipeline first:\n\n"
            "```bash\npython window_features.py\n```\n\n"
            "It slices every cycle's high-frequency Current/Vibration burst into "
            f"{120}-second windows and extracts tsfresh features per window, writing "
            "`WM_WindowedFeatures.csv`."
        )
    else:
        st.markdown(
            "The base classifier only had **108 cycle-level samples**. Here each "
            "cycle's high-frequency burst is sliced into short windows, so one cycle "
            "produces many feature vectors — a much larger training set."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Windows (samples)", f"{result['n_windows']:,}")
        c2.metric("Source cycles", result["n_cycles"])
        c3.metric("GroupKFold F1-Macro", f"{result['cv_scores'].mean():.3f}")
        c4.metric("CV Std", f"± {result['cv_scores'].std():.3f}")

        st.divider()

        # ── The leakage lesson ────────────────────────────────────────────────
        st.markdown('<p class="section-title">Why grouping by cycle matters</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "Windows from the same cycle are near-duplicates that share one label. "
            "If they're allowed to fall on both sides of a train/test split, the model "
            "effectively sees the test data during training — **data leakage** — and the "
            "score looks far better than reality. **GroupKFold** keeps every cycle "
            "entirely within one fold, giving the honest number."
        )
        leaky = result["leaky_scores"].mean()
        honest = result["cv_scores"].mean()
        cmp_df = pd.DataFrame({
            "Evaluation": ["Naive shuffled split (LEAKY ❌)", "GroupKFold by cycle (HONEST ✅)"],
            "Macro-F1": [leaky, honest],
        })
        fig_cmp = px.bar(
            cmp_df, x="Macro-F1", y="Evaluation", orientation="h",
            text=cmp_df["Macro-F1"].round(3), range_x=[0, 1], height=240,
            color="Evaluation",
            color_discrete_map={
                "Naive shuffled split (LEAKY ❌)": "#e74c3c",
                "GroupKFold by cycle (HONEST ✅)": "#2ecc71",
            },
        )
        fig_cmp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_cmp.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", yaxis_title="")
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.info(
            f"The leaky split reports **{leaky:.3f}** — about "
            f"**{(leaky - honest):.3f} higher** than the leak-free **{honest:.3f}**. "
            "The lower number is the one to trust and to put on a portfolio."
        )

        st.divider()

        tab_cm, tab_dist = st.tabs(["📊 Confusion Matrix (out-of-fold)", "🧮 Window Distribution"])

        with tab_cm:
            classes = result["le"].classes_
            cm = confusion_matrix(result["y_true"], result["y_oof"])
            fig_cm = px.imshow(
                cm, labels=dict(x="Predicted", y="Actual", color="Windows"),
                x=classes, y=classes, color_continuous_scale="Purples",
                text_auto=True, title="Out-of-fold confusion matrix (leak-free)",
                height=450,
            )
            fig_cm.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_cm, use_container_width=True)
            report = classification_report(result["y_true"], result["y_oof"],
                                           target_names=classes, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        with tab_dist:
            dist = (result["merged"]["fault_condition"].value_counts()
                    .rename_axis("Condition").reset_index(name="Windows"))
            fig_d = px.bar(
                dist, x="Condition", y="Windows", color="Condition",
                color_discrete_map=FAULT_COLORS, text="Windows", height=380,
                category_orders={"Condition": list(FAULT_COLORS.keys())},
            )
            fig_d.update_layout(showlegend=False, plot_bgcolor="white",
                                paper_bgcolor="white")
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption(
                "Windowing multiplies *samples*, not independent fault *events* — "
                "there are still only ~20 physical faults behind these windows. It "
                "gives the model more to learn from, but the honest headline metric "
                "remains the GroupKFold score above."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — FRIDGE SIGNAL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Fridge Signal Explorer":
    st.markdown('<p class="main-header">🔍 Fridge Signal Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Browse raw sensor signals for individual refrigerator cycles</p>',
        unsafe_allow_html=True,
    )

    fridge = load_fridge()

    # Restrict to cycles whose raw signals are present (full set locally, curated
    # subset in the deployed demo).
    avail = fridge_cycles_with_signal()
    fridge = fridge[fridge["begin_end"].astype(str).isin(avail)]
    if fridge.empty:
        st.info("No raw-signal cycles are bundled in this demo. See the "
                "**Refrigerators → Anomaly Detection** and **Feature Analysis** pages, "
                "which run on the full precomputed feature set.")
        st.stop()
    st.caption(f"Showing {len(fridge)} cycle(s) with raw signals available.")

    c1, c2 = st.columns(2)
    with c1:
        fail_f = st.selectbox("Health Condition",
                              ["All"] + sorted(fridge["failure"].unique().tolist()))
    with c2:
        model_f = st.selectbox("Model",
                               ["All"] + sorted(fridge["model"].dropna().unique().tolist()))

    filt = fridge.copy()
    if fail_f != "All":
        filt = filt[filt["failure"] == fail_f]
    if model_f != "All":
        filt = filt[filt["model"] == model_f]

    if filt.empty:
        st.warning("No cycles match the selected filters.")
        st.stop()

    cycle_id = st.selectbox(
        f"Select Cycle  ({len(filt)} found)",
        filt["begin_end"].tolist(),
    )
    row = filt[filt["begin_end"] == cycle_id].iloc[0]

    # Metadata strip with health badge
    ca, cb, cc = st.columns([1, 1, 1.4])
    ca.metric("Brand", row.get("brand", "—"))
    cb.metric("Model", row.get("model", "—"))
    cond = row["failure"]
    color = FRIDGE_COLORS.get(cond, "#888")
    cc.markdown(f"""
    <div style='text-align:center; padding-top:0.9rem;'>
      <span style='background:{color}20; color:{color}; border:2px solid {color};
                   border-radius:20px; padding:5px 18px; font-weight:700; font-size:0.95rem;'>
        {FRIDGE_ICONS.get(cond, "")} {cond}
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 24-hour power trace (distinctive to fridges) ──────────────────────────
    st.markdown('<p class="section-title">24-Hour Power Trace — Compressor Duty Cycling</p>',
                unsafe_allow_html=True)
    day_df = load_fridge_24h(cycle_id)
    if day_df is not None and not day_df.empty:
        day_df["datetime"] = pd.to_datetime(day_df["Ts"], unit="s")
        ds = max(1, len(day_df) // 3000)
        dd = day_df.iloc[::ds]
        fig_day = go.Figure()
        fig_day.add_trace(go.Scatter(
            x=dd["datetime"], y=dd["ActP"], mode="lines",
            name="Active Power", line=dict(color=color, width=1.1)))
        fig_day.update_layout(
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            title_text="Active Power over 24 h (1 Hz)", title_font_size=14,
            showlegend=False)
        fig_day.update_xaxes(title_text="Time", showgrid=True, gridcolor="#f0f0f0")
        fig_day.update_yaxes(title_text="Active Power (W)", showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_day, use_container_width=True)
        st.caption("The repeating on/off pattern is the compressor cycling to hold "
                   "temperature — a fingerprint of refrigerator health.")
    else:
        st.info("24-hour trace not available for this cycle.")

    # ── Slow stream (electrical) ──────────────────────────────────────────────
    st.markdown('<p class="section-title">Cycle Electrical Signals</p>', unsafe_allow_html=True)
    slow_df = load_fridge_slow(cycle_id)
    if slow_df is not None:
        slow_df["datetime"] = pd.to_datetime(slow_df["Ts"], unit="s")
        ds_step = max(1, len(slow_df) // 2000)
        sd = slow_df.iloc[::ds_step]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Active Power (W)", "Voltage (mV)", "Current (cA)"),
            vertical_spacing=0.07,
        )
        fig.add_trace(go.Scatter(x=sd["datetime"], y=sd["ActP"], mode="lines",
                                 name="Active Power", line=dict(color=color, width=1.3)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=sd["datetime"], y=sd["V"], mode="lines",
                                 name="Voltage", line=dict(color="#f39c12", width=1.3)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sd["datetime"], y=sd["A"], mode="lines",
                                 name="Current", line=dict(color="#16a085", width=1.3)),
                      row=3, col=1)
        fig.update_layout(height=530, plot_bgcolor="white", paper_bgcolor="white",
                          showlegend=False,
                          title_text="Electrical Signals — Slow Stream (1 Hz)",
                          title_font_size=14)
        for r in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", row=r, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", row=r, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Slow stream data not available for this cycle.")

    # ── Fast stream — on demand ───────────────────────────────────────────────
    if st.checkbox("⚡ Show high-frequency signals (Current & Vibration — large file)"):
        fast_df = load_fridge_fast(cycle_id)
        if fast_df is not None:
            time_col = fast_df.columns[0]
            fast_df["time_s"] = fast_df[time_col] / 1e6
            step = max(1, len(fast_df) // 5000)
            fd = fast_df.iloc[::step].reset_index(drop=True)

            fig2 = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=("Current — Fast Stream (ADC units)", "Vibration (ADC units)"),
                vertical_spacing=0.1,
            )
            fig2.add_trace(go.Scatter(x=fd["time_s"], y=fd["Current"], mode="lines",
                                      name="Current", line=dict(color="#16a085", width=0.9)),
                           row=1, col=1)
            fig2.add_trace(go.Scatter(x=fd["time_s"], y=fd["Vibration"], mode="lines",
                                      name="Vibration", line=dict(color="#8e44ad", width=0.9)),
                           row=2, col=1)
            fig2.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                               showlegend=False,
                               title_text="High-Frequency Signals — Fast Stream",
                               title_font_size=14)
            fig2.update_xaxes(title_text="Time (s)", row=2, col=1,
                              showgrid=True, gridcolor="#f0f0f0")
            fig2.update_xaxes(showgrid=True, gridcolor="#f0f0f0", row=1, col=1)
            fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Fast stream data not available for this cycle.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — FRIDGE FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Fridge Feature Analysis":
    st.markdown('<p class="main-header">📊 Fridge Feature Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Statistical and frequency-domain features extracted per '
        'refrigerator cycle — Normal vs Malfunction</p>',
        unsafe_allow_html=True,
    )

    fridge = load_fridge()
    meta_cols = {"Id", "begin_end", "brand", "model", "failure"}
    feature_cols = [c for c in fridge.columns if c not in meta_cols]

    tab_dist, tab_pca = st.tabs(["📦 Feature Distributions", "🔵 PCA"])

    # ── Distributions ─────────────────────────────────────────────────────────
    with tab_dist:
        st.markdown('<p class="section-title">Distribution by Health Condition</p>',
                    unsafe_allow_html=True)
        sensors = sorted({c.split("__")[0] for c in feature_cols if "__" in c})
        c1, c2 = st.columns(2)
        with c1:
            sensor = st.selectbox("Sensor", sensors)
        sensor_feats = [f for f in feature_cols if f.startswith(sensor + "__")]
        labels = [f.replace(sensor + "__", "").replace("__", " ") for f in sensor_feats]
        with c2:
            sel_label = st.selectbox("Feature", labels)
        sel_col = sensor_feats[labels.index(sel_label)]

        plot_df = fridge[[sel_col, "failure"]].copy()
        plot_df[sel_col] = pd.to_numeric(plot_df[sel_col], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[sel_col])

        fig_box = px.box(
            plot_df, x="failure", y=sel_col, color="failure",
            color_discrete_map=FRIDGE_COLORS, points="all",
            labels={"failure": "Condition", sel_col: f"{sensor} — {sel_label}"},
            title=f"{sensor} · {sel_label}  —  Distribution by Health Condition",
            height=470,
            category_orders={"failure": ["Normal", "Malfunction"]},
        )
        fig_box.update_traces(jitter=0.3, marker_size=4)
        fig_box.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        fig_box.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("With only ~2% malfunctions, the red group is small — hover to inspect "
                   "individual faulty cycles against the healthy distribution.")

    # ── PCA ───────────────────────────────────────────────────────────────────
    with tab_pca:
        st.markdown('<p class="section-title">Principal Component Analysis</p>',
                    unsafe_allow_html=True)
        max_pcs = min(10, len(feature_cols))
        n_components = st.slider("Number of components to compute", 2, max_pcs, 2)

        X_raw = fridge[feature_cols].apply(pd.to_numeric, errors="coerce")
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
        valid = ~X_raw.isna().any(axis=1)
        X_clean = X_raw[valid].values
        labels_clean = fridge["failure"].values[valid]

        X_scaled = StandardScaler().fit_transform(X_clean)
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X_scaled)

        col_pca, col_var = st.columns([1.6, 1], gap="large")
        with col_pca:
            pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
            pc_df["Condition"] = labels_clean
            fig_pca = px.scatter(
                pc_df, x="PC1", y="PC2", color="Condition",
                color_discrete_map=FRIDGE_COLORS,
                title="First 2 Principal Components", height=460, opacity=0.7,
                category_orders={"Condition": ["Normal", "Malfunction"]},
            )
            fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.6, color="white")))
            fig_pca.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  legend=dict(title="Condition", itemsizing="constant"))
            fig_pca.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ddd")
            fig_pca.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ddd")
            st.plotly_chart(fig_pca, use_container_width=True)
        with col_var:
            var_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(n_components)],
                "Variance (%)": (pca.explained_variance_ratio_ * 100).round(2),
            })
            fig_var = px.bar(
                var_df, x="Component", y="Variance (%)",
                title="Explained Variance per Component",
                color="Variance (%)", color_continuous_scale="Blues",
                text="Variance (%)", height=460,
            )
            fig_var.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_var.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  coloraxis_showscale=False)
            fig_var.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig_var, use_container_width=True)

        total_2pc = pca.explained_variance_ratio_[:2].sum() * 100
        st.info(
            f"The first 2 PCs explain **{total_2pc:.1f}%** of total variance.  "
            f"The first {n_components} PCs explain "
            f"**{pca.explained_variance_ratio_.sum()*100:.1f}%**. Malfunctions that sit "
            "apart from the healthy cluster are exactly what the anomaly detector exploits."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — FRIDGE ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧊  Fridge Anomalies":
    st.markdown('<p class="main-header">🧊 Fridge Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Unsupervised health monitoring across 1,000+ refrigerator '
        'cycles — finding the rare malfunctions without training on labels</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Running IsolationForest…"):
        r = run_fridge_anomaly()

    st.markdown(
        "The fridge dataset is huge but extremely imbalanced: "
        f"**{r['n_total']:,} cycles**, only **{r['n_anom']} malfunctions** "
        f"(**{r['base_rate']*100:.1f}%**). That's too few faults for a supervised "
        "classifier, so this is framed as **anomaly detection** — the model learns "
        "what *normal* looks like and flags the outliers. Labels are used **only to "
        "evaluate**, never to train."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cycles", f"{r['n_total']:,}")
    c2.metric("Malfunctions", r["n_anom"])
    c3.metric("ROC-AUC", f"{r['roc_auc']:.3f}")
    c4.metric("PR-AUC", f"{r['pr_auc']:.3f}")

    st.divider()

    tab_budget, tab_curves, tab_scores = st.tabs(
        ["🎯 Inspection Budget", "📈 ROC & PR Curves", "🔬 Score Distribution"])

    # ── Inspection budget ─────────────────────────────────────────────────────
    with tab_budget:
        st.markdown('<p class="section-title">If a technician inspects the top-N% most anomalous cycles</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "In practice you can only inspect so many units. This shows how many of "
            "the 24 real malfunctions you'd catch for a given inspection budget."
        )
        bd = r["budget"].copy()
        bd["Recall"] = (bd["Recall"] * 100).round(0).astype(int).astype(str) + "%"
        bd["Precision"] = (bd["Precision"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(bd, use_container_width=True, hide_index=True)
        st.info(
            "Random inspection would catch malfunctions at the ~2% base rate. "
            "The anomaly score concentrates them into the top slice — that lift is "
            "the practical value of the model."
        )

    # ── ROC & PR curves ───────────────────────────────────────────────────────
    with tab_curves:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                         name="Model", line=dict(color="#7c4dff", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         name="Random", line=dict(color="#bbb", dash="dash")))
            fig_roc.update_layout(
                title=f"ROC curve (AUC = {r['roc_auc']:.3f})",
                xaxis_title="False positive rate", yaxis_title="True positive rate",
                height=420, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_roc, use_container_width=True)
        with col_b:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=r["rec"], y=r["prec"], mode="lines",
                                        name="Model", line=dict(color="#e67e22", width=3)))
            fig_pr.add_hline(y=r["base_rate"], line_dash="dash", line_color="#bbb",
                             annotation_text="Base rate")
            fig_pr.update_layout(
                title=f"Precision–Recall (AP = {r['pr_auc']:.3f})",
                xaxis_title="Recall", yaxis_title="Precision",
                height=420, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_pr, use_container_width=True)
        st.caption(
            "ROC-AUC rewards ranking malfunctions above normals (strong here). "
            "PR-AUC looks low because with a 2% base rate, precision is inherently "
            "hard — this is an honest reflection of extreme class imbalance, not a "
            "broken model."
        )

    # ── Score distribution ────────────────────────────────────────────────────
    with tab_scores:
        m = r["merged"]
        fig_h = px.histogram(
            m, x="anomaly_score", color="failure", nbins=60, barmode="overlay",
            color_discrete_map=FRIDGE_COLORS,
            height=420, title="Anomaly score: Normal vs Malfunction",
        )
        fig_h.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_title="Anomaly score (higher = more anomalous)")
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown('<p class="section-title">Most anomalous cycles</p>',
                    unsafe_allow_html=True)
        top = (m.nlargest(15, "anomaly_score")
               [["Id", "model", "failure", "anomaly_score"]]
               .round({"anomaly_score": 3}))
        st.dataframe(top, use_container_width=True, hide_index=True)
