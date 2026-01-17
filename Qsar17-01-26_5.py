# ===========================================
# QSAR Suite (ERROR-PROOF VERSION)
# SMILES → Descriptors | KNN Similarity + MCC | ML + IC50 (≤ 3 µM)
# ===========================================

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

# Try RDKit (may or may not be installed)
RDKit_OK = True
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:
    RDKit_OK = False

st.set_page_config(page_title="QSAR Suite (Robust Version)", layout="wide")
st.title("🧬 QSAR Suite (Robust / Auto-ID Version)")

# ------------------------------------------------
# ============ HELPER FUNCTIONS =================
# ------------------------------------------------

def auto_detect_id_column(df: pd.DataFrame):
    """
    Automatically pick a good ID column:
    - Prefer text/object column
    - Must NOT be all unique numbers
    - Prefer low cardinality but mostly unique names
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
    "🧪 TASK 1: SMILES → Descriptors",
    "🧭 TASK 2: KNN Similarity + MCC",
    "🤖 TASK 3: ML + IC50 Filtering"
])

# ======================================================
# TASK 1 — SMILES → DESCRIPTORS
# ======================================================
with tab1:
    st.subheader("Convert SMILES → Descriptors")

    if not RDKit_OK:
        st.error("RDKit not available. Skip TASK 1 or install RDKit.")
    else:
        smiles_file = st.file_uploader("Upload SMILES file", type=["csv","xlsx"])

        if smiles_file:
            df = pd.read_excel(smiles_file) if smiles_file.name.endswith(".xlsx") else pd.read_csv(smiles_file)
            st.write("Preview:", df.head())

            smiles_col = [c for c in df.columns if "smiles" in c.lower()]
            smiles_col = smiles_col[0] if smiles_col else st.selectbox("Select SMILES column", df.columns)

            def smiles_to_desc(name, smi):
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    return None

                d = {
                    "Compound_ID": name,
                    "MW": Descriptors.MolWt(mol),
                    "LogP": Descriptors.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    "HBD": rdMolDescriptors.CalcNumHBD(mol),
                    "HBA": rdMolDescriptors.CalcNumHBA(mol),
                    "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
                }

                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fp_arr = np.frombuffer(fp.ToBitString().encode(), "S1").astype(int)

                for i, bit in enumerate(fp_arr):
                    d[f"FP_{i}"] = int(bit)

                return d

            id_col = auto_detect_id_column(df)
            rows = [smiles_to_desc(r[id_col], r[smiles_col]) for _, r in df.iterrows()]
            df_desc = pd.DataFrame([r for r in rows if r])

            st.success("Descriptors generated!")
            st.dataframe(df_desc.head())

            st.download_button(
                "Download Descriptors",
                df_desc.to_csv(index=False).encode(),
                "SMILES_Descriptors.csv"
            )

# ======================================================
# TASK 2 — KNN SIMILARITY + MCC (AUTO-ID)
# ======================================================
with tab2:
    st.subheader("KNN Similarity + MCC")

    tr_file = st.file_uploader("Upload TRAINING descriptors", type=["csv","xlsx"], key="tr")
    qr_file = st.file_uploader("Upload QUERY descriptors", type=["csv","xlsx"], key="qr")

    if tr_file and qr_file:
        df_tr = pd.read_excel(tr_file) if tr_file.name.endswith(".xlsx") else pd.read_csv(tr_file)
        df_qr = pd.read_excel(qr_file) if qr_file.name.endswith(".xlsx") else pd.read_csv(qr_file)

        tr_id = auto_detect_id_column(df_tr)
        qr_id = auto_detect_id_column(df_qr)

        st.info(f"Auto-detected TRAINING ID: {tr_id}")
        st.info(f"Auto-detected QUERY ID: {qr_id}")

        ic50_col = [c for c in df_tr.columns if "ic50" in c.lower()]
        ic50_col = ic50_col[0] if ic50_col else st.selectbox("Select IC50 column", df_tr.columns)

        ic50_unit = st.selectbox("IC50 unit", ["µM","nM","pM","mM"], index=0)

        tr_ic50 = convert_to_uM(df_tr[ic50_col].apply(extract_numeric), ic50_unit)
        y_tr = (tr_ic50 <= 3).astype(int)  # 1 = Strong

        X_tr = safe_numeric(df_tr).fillna(0).values
        X_qr = safe_numeric(df_qr).fillna(0).values

        top_k = st.number_input("K for KNN", 1, 50, 5)

        sims = cosine_similarity(X_qr, X_tr)
        nn_idx = np.argsort(-sims, axis=1)[:, :top_k]

        pred_labels = [
            1 if y_tr.iloc[idx].sum() >= (len(idx)/2) else 0
            for idx in nn_idx
        ]

        df_out = pd.DataFrame({
            qr_id: df_qr[qr_id],
            "MaxSimilarity": sims.max(axis=1),
            "Pred_Label": ["Active" if p==1 else "Inactive" for p in pred_labels]
        })

        st.dataframe(df_out.head())

        if st.checkbox("Calculate MCC (if query has IC50)"):
            qr_ic50_col = st.selectbox("Select Query IC50 column", df_qr.columns)
            qr_ic50_unit = st.selectbox("Query IC50 unit", ["µM","nM","pM","mM"], index=0)

            y_true = (convert_to_uM(df_qr[qr_ic50_col].apply(extract_numeric), qr_ic50_unit) <= 3).astype(int)
            mcc = matthews_corrcoef(y_true, pred_labels)

            st.success(f"MCC = {mcc:.3f}")

# ======================================================
# TASK 3 — ML + IC50 FILTERING
# ======================================================
with tab3:
    st.subheader("ML Classification + IC50 ≤ 3 µM")

    desc_file = st.file_uploader("Upload Descriptor file", type=["csv","xlsx"], key="ml")

    if desc_file:
        df = pd.read_excel(desc_file) if desc_file.name.endswith(".xlsx") else pd.read_csv(desc_file)
        st.write("Preview:", df.head())

        id_col = auto_detect_id_column(df)
        st.info(f"Auto-detected ID: {id_col}")

        X = safe_numeric(df).fillna(0)
        y = KMeans(n_clusters=2, random_state=42).fit_predict(X)

        models = {
            "LR": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "DT": DecisionTreeClassifier(),
            "RF": RandomForestClassifier(),
            "NB": GaussianNB()
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        best_model = None
        best_f1 = -1

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            f1 = f1_score(y_test, pred, average="weighted")
            st.write(f"{name} F1: {f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

        df["Pred_Label"] = best_model.predict(X)
        st.dataframe(df.head())

        ic50_file = st.file_uploader("Upload IC50 file", type=["csv","xlsx"], key="ic")

        if ic50_file:
            df_ic = pd.read_excel(ic50_file) if ic50_file.name.endswith(".xlsx") else pd.read_csv(ic50_file)

            ic50_col = [c for c in df_ic.columns if "ic50" in c.lower()][0]
            df_ic["IC50_uM"] = convert_to_uM(df_ic[ic50_col].apply(extract_numeric), "µM")

            df_final = df.merge(df_ic[[id_col, "IC50_uM"]], on=id_col, how="left")
            df_final["Activity"] = np.where(df_final["IC50_uM"] <= 3, "Strong", "Weak")

            st.dataframe(df_final.head())
            st.download_button("Download Results", df_final.to_csv(index=False).encode(), "Final_Results.csv")
