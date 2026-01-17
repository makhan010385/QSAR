# ==========================================================
# QSAR Suite — NO RDKit VERSION
# TASK 1: Upload Descriptors (2D/3D/ECFP/Fingerprint)
# TASK 2: KNN Similarity + MCC
# TASK 3: ML (LR, SVM, DT, RF, NB) + IC50 ≤ 3 µM
# ==========================================================

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
)
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="QSAR Suite (No RDKit)", layout="wide")
st.title("🧬 QSAR Suite — NO RDKit VERSION")

# ------------------------------------------------
# ============ HELPER FUNCTIONS =================
# ------------------------------------------------

def auto_detect_id_column(df: pd.DataFrame):
    """
    Automatically pick the best ID column:
    - Prefer text/object column
    - Avoid all-numeric columns
    - Prefer mostly unique names
    """
    candidates = []

    for col in df.columns:
        if df[col].dtype == object:
            unique_ratio = df[col].nunique() / max(1, len(df))
            null_ratio = df[col].isna().mean()

            if null_ratio < 0.2 and unique_ratio > 0.5:
                candidates.append((col, unique_ratio))

    if candidates:
        candidates = sorted(candidates, key=lambda x: -x[1])
        return candidates[0][0]

    # Fallback: first column
    return df.columns[0]


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric descriptor columns"""
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.columns[num.notna().any()]]


def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace("μ", "u").replace("µ", "u")
    match = re.findall(r"[-+]?(\d+(\.\d+)?([eE][-+]?\d+)?)", s)
    return float(match[0][0]) if match else np.nan


def convert_to_uM(values: pd.Series, unit: str) -> pd.Series:
    unit = (unit or "").strip().lower()
    factor = 1.0
    if unit in ["nm"]:
        factor = 1/1000
    elif unit in ["pm"]:
        factor = 1/1_000_000
    elif unit in ["mm"]:
        factor = 1000
    return values.astype(float) * factor


# ------------------------------------------------
# ================== LAYOUT =====================
# ------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🧪 TASK 1: Upload Descriptors",
    "🧭 TASK 2: KNN Similarity + MCC",
    "🤖 TASK 3: ML + IC50 Filtering"
])

# ======================================================
# TASK 1 — UPLOAD DESCRIPTORS (NO RDKit)
# ======================================================
with tab1:
    st.subheader("📥 TASK 1: Upload Descriptor File (2D/3D/ECFP/Fingerprint)")

    desc_file = st.file_uploader(
        "Upload descriptor file (2D3D.xlsx / Drugfingerprint.xlsx / ECFP_Fingerprint-2.xlsx)",
        type=["csv", "xlsx"],
        key="task1_desc"
    )

    if desc_file:
        df_desc = pd.read_excel(desc_file) if desc_file.name.endswith(".xlsx") else pd.read_csv(desc_file)
        st.success(f"Descriptors loaded: {df_desc.shape}")
        st.dataframe(df_desc.head())
        st.info("You can use this file directly in TASK 2 and TASK 3.")

# ======================================================
# TASK 2 — KNN SIMILARITY + MCC (NO RDKit)
# ======================================================
with tab2:
    st.subheader("🧭 TASK 2: KNN Similarity + MCC")

    tr_file = st.file_uploader("Upload TRAINING descriptor file", type=["csv","xlsx"], key="tr")
    qr_file = st.file_uploader("Upload QUERY descriptor file", type=["csv","xlsx"], key="qr")

    if tr_file and qr_file:
        df_tr = pd.read_excel(tr_file) if tr_file.name.endswith(".xlsx") else pd.read_csv(tr_file)
        df_qr = pd.read_excel(qr_file) if qr_file.name.endswith(".xlsx") else pd.read_csv(qr_file)

        tr_id = auto_detect_id_column(df_tr)
        qr_id = auto_detect_id_column(df_qr)

        st.info(f"Auto-detected TRAINING ID: **{tr_id}**")
        st.info(f"Auto-detected QUERY ID: **{qr_id}**")

        # Select IC50 column in training
        ic50_cols = [c for c in df_tr.columns if "ic50" in c.lower()]
        tr_ic50_col = ic50_cols[0] if ic50_cols else st.selectbox("Select IC50 column (Training)", df_tr.columns)

        tr_ic50_unit = st.selectbox("Training IC50 unit", ["µM","nM","pM","mM"], index=0)

        tr_ic50 = convert_to_uM(df_tr[tr_ic50_col].apply(extract_numeric), tr_ic50_unit)
        y_tr = (tr_ic50 <= 3).astype(int)   # 1 = Strong, 0 = Weak

        # Prepare numeric matrices
        X_tr = safe_numeric(df_tr).fillna(0).values
        X_qr = safe_numeric(df_qr).fillna(0).values

        top_k = st.number_input("K for KNN", 1, 50, 5)

        # Cosine similarity
        sims = cosine_similarity(X_qr, X_tr)
        nn_idx = np.argsort(-sims, axis=1)[:, :top_k]

        pred_labels = [
            1 if y_tr.iloc[idx].sum() >= (len(idx)/2) else 0
            for idx in nn_idx
        ]

        df_out = pd.DataFrame({
            qr_id: df_qr[qr_id],
            "MaxSimilarity": sims.max(axis=1),
            "Pred_Label_by_Similarity": ["Active" if p==1 else "Inactive" for p in pred_labels]
        })

        st.write("### Similarity Results (Preview)")
        st.dataframe(df_out.head())

        st.download_button(
            "Download Similarity Results",
            df_out.to_csv(index=False).encode(),
            "Similarity_Results.csv"
        )

        # ---- MCC Calculation ----
        if st.checkbox("Calculate MCC (if Query has IC50)"):
            qr_ic50_col = st.selectbox("Select Query IC50 column", df_qr.columns)
            qr_ic50_unit = st.selectbox("Query IC50 unit", ["µM","nM","pM","mM"], index=0)

            y_true = (convert_to_uM(df_qr[qr_ic50_col].apply(extract_numeric), qr_ic50_unit) <= 3).astype(int)
            mcc = matthews_corrcoef(y_true, pred_labels)

            st.success(f"✅ MCC = **{mcc:.3f}**")

# ======================================================
# TASK 3 — ML + IC50 FILTERING (NO RDKit)
# ======================================================
with tab3:
    st.subheader("🤖 TASK 3: ML + IC50 Filtering (≤ 3 µM)")

    desc_file = st.file_uploader("Upload Descriptor file", type=["csv","xlsx"], key="ml")

    if desc_file:
        df = pd.read_excel(desc_file) if desc_file.name.endswith(".xlsx") else pd.read_csv(desc_file)
        st.success(f"Descriptors Loaded — shape: {df.shape}")
        st.dataframe(df.head())

        id_col = auto_detect_id_column(df)
        st.info(f"Auto-detected Compound ID: **{id_col}**")

        # Prepare numeric data
        X = safe_numeric(df).fillna(0)

        # Unsupervised clustering for pseudo-labels
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        y = kmeans.fit_predict(X)

        # Train ML models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB()
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        best_model = None
        best_f1 = -1
        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            f1 = f1_score(y_test, pred, average="weighted")
            results.append([name, f1])

            if f1 > best_f1:
                best_f1 = f1
                best_model = model

        results_df = pd.DataFrame(results, columns=["Model", "F1-Score"])
        st.write("Model Comparison")
        st.dataframe(results_df)

        st.success(f"🏆 Best Model: {results_df.iloc[results_df['F1-Score'].idxmax(), 0]}")

        # Predict labels for all compounds
        df["Predicted_Label"] = best_model.predict(X)
        st.write("Labeled Dataset (Preview)")
        st.dataframe(df.head())

        # ---- IC50 FILTERING ----
        ic50_file = st.file_uploader("Upload IC50 file", type=["csv","xlsx"], key="ic")

        if ic50_file:
            df_ic = pd.read_excel(ic50_file) if ic50_file.name.endswith(".xlsx") else pd.read_csv(ic50_file)

            ic50_cols = [c for c in df_ic.columns if "ic50" in c.lower()]
            ic50_col = ic50_cols[0] if ic50_cols else st.selectbox("Select IC50 column", df_ic.columns)

            df_ic["IC50_uM"] = convert_to_uM(df_ic[ic50_col].apply(extract_numeric), "µM")

            df_final = df.merge(df_ic[[id_col, "IC50_uM"]], on=id_col, how="left")
            df_final["Activity"] = np.where(df_final["IC50_uM"] <= 3, "Strong", "Weak")

            st.write("Final Results (Preview)")
            st.dataframe(df_final.head())

            st.download_button(
                "Download Final Results",
                df_final.to_csv(index=False).encode(),
                "Final_QSAR_Results.csv"
            )
