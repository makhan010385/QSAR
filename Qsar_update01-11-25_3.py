# ===========================================
# QSAR Suite: SMILES→Descriptors + Similarity Filtering (MCC) + ML & IC50 (≤ 6.45 µM)
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
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ---- RDKit (TASK 1) ----
RDKit_OK = True
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem import AllChem, MACCSkeys
except Exception:
    RDKit_OK = False

st.set_page_config(page_title="QSAR Suite (SMILES→Descriptors • Similarity+MCC • ML+IC50)", layout="wide")
st.title("🧬 QSAR Suite")

# -----------------------------
# Helpers (shared)
# -----------------------------
def normalize_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace(r"[’'`\"]", "", regex=True)
         .str.replace(r"\s*\(.*?\)\s*", " ", regex=True)
         .str.strip()
    )

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.columns[num.notna().any()]]

def safe_pca_fit_transform(X_train, X_test, min_components=2):
    n_features = X_train.shape[1]
    n_comp = max(1, min(min_components, n_features))
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return pca, X_train_pca, X_test_pca

def plot_pca_scatter(X_pca, labels, title):
    if X_pca.shape[1] >= 2:
        x, y = X_pca[:, 0], X_pca[:, 1]
        fig = px.scatter(x=x, y=y, color=labels.astype(str),
                         title=title, labels={"x": "PCA1", "y": "PCA2", "color": "Cluster"})
    else:
        x, y = X_pca[:, 0], np.zeros_like(X_pca[:, 0])
        fig = px.scatter(x=x, y=y, color=labels.astype(str),
                         title=title + " (PCA2 not available — PCA1 shown)",
                         labels={"x": "PCA1", "y": "0", "color": "Cluster"})
    return fig

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    s = s.replace("μ", "u").replace("µ", "u")
    match = re.findall(r"[-+]?(\d+(\.\d+)?([eE][-+]?\d+)?)", s)
    if not match:
        return np.nan
    try:
        return float(match[0][0])
    except Exception:
        return np.nan

def convert_to_uM(values: pd.Series, unit: str) -> pd.Series:
    unit = (unit or "").strip().lower()
    if unit in ["µm", "um", "um ", "micromolar", "micromole", "micromoles", "micro molar"]:
        factor = 1.0
    elif unit in ["nm", "nanomolar", "nanomole", "nanomoles"]:
        factor = 1.0 / 1000.0
    elif unit in ["pm", "picomolar", "picomole", "picomoles"]:
        factor = 1.0 / 1_000_000.0
    elif unit in ["mm", "millimolar", "millimole", "millimoles"]:
        factor = 1000.0
    else:
        factor = 1.0
    return values.astype(float) * factor

def diagnostics_after_merge(df_final, ic50_uM_col):
    total = len(df_final)
    matched = df_final[ic50_uM_col].notna().sum()
    st.info(f"🔗 Merge summary: matched IC50 for **{matched} / {total}** compounds.")
    if matched > 0:
        desc = df_final[ic50_uM_col].dropna().describe()
        st.write("📊 IC50 (µM) summary on matched rows:", desc.to_frame().T)
        fig = px.histogram(df_final, x=ic50_uM_col, nbins=50, title="IC50 (µM) Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🧪 TASK 1: SMILES → Descriptors", "🧭 TASK 2: Similarity Filter + MCC", "🤖 TASK 3: ML + IC50 Filtering"])

# ===========================================
# TASK 1 — SMILES → DESCRIPTORS (RDKit)
# ===========================================
with tab1:
    st.subheader("SMILES → 2D/FP Descriptors")
    if not RDKit_OK:
        st.error("RDKit not available in this environment. Please install RDKit to use TASK 1.")
    else:
        smiles_file = st.file_uploader("📥 Upload SMILES File (CSV/XLSX with columns e.g. Name, SMILES)", type=["csv", "xlsx"], key="smiles_upload")
        id_col_name = st.text_input("Identifier column name (optional, default = first column)", value="")
        smiles_col_name = st.text_input("SMILES column name", value="SMILES")
        fp_bits = st.number_input("Morgan fingerprint bits", min_value=256, max_value=4096, value=2048, step=256)
        radius = st.number_input("Morgan FP radius", min_value=1, max_value=3, value=2, step=1)

        def smiles_to_descriptors_row(name, smi):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            desc = {
                "Name": name,
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBD": rdMolDescriptors.CalcNumHBD(mol),
                "HBA": rdMolDescriptors.CalcNumHBA(mol),
                "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "AromRings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            }
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=int(fp_bits))
            fp_arr = np.frombuffer(fp.ToBitString().encode("utf-8"), "S1").astype(int)
            for i in range(fp_bits):
                desc[f"FP_{i}"] = int(fp_arr[i])
            return desc

        if smiles_file is not None:
            df_smiles = pd.read_excel(smiles_file) if smiles_file.name.endswith(".xlsx") else pd.read_csv(smiles_file)
            st.success(f"SMILES file loaded: shape {df_smiles.shape}")
            st.dataframe(df_smiles.head())

            # Columns
            if not smiles_col_name or smiles_col_name not in df_smiles.columns:
                st.error("❌ Please set a valid SMILES column name present in the file.")
            else:
                id_col = id_col_name if id_col_name and id_col_name in df_smiles.columns else df_smiles.columns[0]
                rows = []
                for _, r in df_smiles.iterrows():
                    rec = smiles_to_descriptors_row(r[id_col], r[smiles_col_name])
                    if rec is not None:
                        rows.append(rec)
                if len(rows) == 0:
                    st.error("No valid SMILES parsed.")
                else:
                    df_desc = pd.DataFrame(rows)
                    st.success("✅ Descriptors generated!")
                    st.dataframe(df_desc.head())

                    st.download_button("💾 Download Generated Descriptors",
                                       df_desc.to_csv(index=False).encode("utf-8"),
                                       "SMILES_Descriptors.csv", "text/csv")
                    st.info("You can feed this file directly into TASK 2 or TASK 3.")

# ===========================================
# TASK 2 — SIMILARITY FILTER + MCC
# ===========================================
with tab2:
    st.subheader("Similarity-based Filtering vs Training Set + MCC")

    st.markdown("**Input A (Training set, 207 compounds):** must contain either SMILES or numeric descriptors and an IC50 column.")
    tr_file = st.file_uploader("📥 Upload TRAINING set (CSV/XLSX)", type=["csv", "xlsx"], key="train_file")
    st.markdown("**Input B (Query/Filtered set):** compounds to compare; can also include IC50 to evaluate MCC.")
    qr_file = st.file_uploader("📥 Upload QUERY set (CSV/XLSX)", type=["csv", "xlsx"], key="query_file")

    if tr_file is not None and qr_file is not None:
        df_tr = pd.read_excel(tr_file) if tr_file.name.endswith(".xlsx") else pd.read_csv(tr_file)
        df_qr = pd.read_excel(qr_file) if qr_file.name.endswith(".xlsx") else pd.read_csv(qr_file)
        st.success(f"Training loaded: {df_tr.shape} | Query loaded: {df_qr.shape}")

        st.write("**Training preview**")
        st.dataframe(df_tr.head())
        st.write("**Query preview**")
        st.dataframe(df_qr.head())

        # ID columns
        tr_id = st.selectbox("🔗 Training ID column", list(df_tr.columns), index=0)
        qr_id = st.selectbox("🔗 Query ID column", list(df_qr.columns), index=0)

        # IC50 in training
        tr_ic50_col = st.selectbox("🧪 Training IC50 column", list(df_tr.columns), index=min(1, len(df_tr.columns)-1))
        tr_ic50_unit = st.selectbox("📏 Training IC50 units", ["µM", "nM", "pM", "mM"], index=1)

        # (Optional) IC50 in query for MCC
        use_qr_ic50 = st.checkbox("My QUERY set has IC50 too (for MCC calculation)", value=False)
        if use_qr_ic50:
            qr_ic50_col = st.selectbox("🧪 Query IC50 column", list(df_qr.columns), index=min(1, len(df_qr.columns)-1))
            qr_ic50_unit = st.selectbox("📏 Query IC50 units", ["µM", "nM", "pM", "mM"], index=1)

        # Similarity options
        sim_method = st.radio("Similarity method",
                              ["Tanimoto (Morgan FP, needs SMILES in both)", "Cosine (shared numeric descriptors)"],
                              index=0)

        # Try to detect SMILES
        def guess_smiles_col(df):
            for c in df.columns:
                if c.strip().lower() in ["smiles", "smile", "smiles_str"]:
                    return c
            return None

        tr_smiles_col = guess_smiles_col(df_tr)
        qr_smiles_col = guess_smiles_col(df_qr)

        # Prepare training labels (Active/Inactive)
        tr_ic50_raw = df_tr[tr_ic50_col].apply(extract_numeric)
        tr_ic50_uM = convert_to_uM(tr_ic50_raw, tr_ic50_unit)
        tr_label = np.where(tr_ic50_uM <= 6.45, 1, 0)  # 1=Active, 0=Inactive

        # Build feature matrices
        if sim_method.startswith("Tanimoto"):
            if not RDKit_OK:
                st.error("RDKit not available; switch to Cosine similarity.")
                st.stop()
            if tr_smiles_col is None or qr_smiles_col is None:
                st.error("SMILES column not found in one or both files. Add SMILES or switch to Cosine similarity.")
                st.stop()

            def morgan_fp_mat(smiles_series, nbits=2048, radius=2):
                fps = []
                idx_keep = []
                for i, smi in enumerate(smiles_series):
                    m = Chem.MolFromSmiles(str(smi))
                    if m is None:
                        fps.append(None)
                    else:
                        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
                        arr = np.frombuffer(bv.ToBitString().encode("utf-8"), "S1").astype(int)
                        fps.append(arr.astype(np.uint8))
                        idx_keep.append(i)
                return np.vstack([fp for fp in fps if fp is not None]), idx_keep

            fp_bits = st.number_input("Morgan bits", 256, 4096, 2048, 256)
            radius = st.number_input("Morgan radius", 1, 3, 2, 1)

            X_tr, keep_tr = morgan_fp_mat(df_tr[tr_smiles_col], nbits=int(fp_bits), radius=int(radius))
            X_qr, keep_qr = morgan_fp_mat(df_qr[qr_smiles_col], nbits=int(fp_bits), radius=int(radius))

            df_tr_f = df_tr.iloc[keep_tr].reset_index(drop=True)
            df_qr_f = df_qr.iloc[keep_qr].reset_index(drop=True)
            y_tr = tr_label[keep_tr]
        else:
            # cosine on shared numeric columns
            num_tr = safe_numeric(df_tr)
            num_qr = safe_numeric(df_qr)
            shared_cols = sorted(list(set(num_tr.columns).intersection(set(num_qr.columns))))
            if len(shared_cols) == 0:
                st.error("No shared numeric descriptor columns between training and query for cosine similarity.")
                st.stop()
            X_tr = num_tr[shared_cols].fillna(0).values
            X_qr = num_qr[shared_cols].fillna(0).values
            df_tr_f = df_tr.copy()
            df_qr_f = df_qr.copy()
            y_tr = tr_label

        # Similarity computation
        st.markdown("---")
        st.subheader("Similarity filtering")
        top_k = st.number_input("k for nearest neighbors (label vote)", min_value=1, max_value=50, value=5, step=1)
        thr = st.slider("Similarity threshold to KEEP query compounds", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

        if sim_method.startswith("Tanimoto"):
            # Manual Tanimoto on bit vectors
            # X_tr, X_qr are binary {0,1}
            X_tr_bool = X_tr.astype(bool)
            X_qr_bool = X_qr.astype(bool)

            def tanimoto_row(qrow, T):
                # qrow: (bits,), T: (n_train, bits)
                inter = (T & qrow).sum(axis=1)
                union = (T | qrow).sum(axis=1)
                sim = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union!=0)
                return sim

            sims = []
            nn_idx = []
            pred_label = []
            max_sim = []
            for i in range(X_qr_bool.shape(0) if callable(getattr(X_qr_bool, "shape", None)) else X_qr_bool.shape[0]):
                s = tanimoto_row(X_qr_bool[i], X_tr_bool)
                sims.append(s)
                max_sim.append(s.max() if s.size else 0.0)
                # vote by top-k
                top_idx = np.argsort(-s)[:int(top_k)]
                nn_idx.append(top_idx)
                vote = y_tr[top_idx]
                pred = 1 if (vote.sum() >= (len(vote)/2.0)) else 0
                pred_label.append(pred)
            sims = np.vstack(sims)
        else:
            sims = cosine_similarity(X_qr, X_tr)  # (n_qr, n_tr)
            nn_idx = np.argsort(-sims, axis=1)[:, :int(top_k)]
            pred_label = [1 if (y_tr[idx].sum() >= (len(idx)/2.0)) else 0 for idx in nn_idx]
            max_sim = sims.max(axis=1)

        df_out = df_qr_f[[qr_id]].copy()
        df_out["MaxSimilarity"] = max_sim
        df_out["Pred_Label_bySimilarity"] = np.where(np.array(pred_label) == 1, "Active", "Inactive")
        df_out["Keep_by_threshold"] = df_out["MaxSimilarity"] >= thr

        st.write("### Filtered by Similarity (Keep=True)")
        st.dataframe(df_out[df_out["Keep_by_threshold"]].head(30))

        st.download_button("💾 Download Similarity Predictions",
                           df_out.to_csv(index=False).encode("utf-8"),
                           "Similarity_Predictions.csv", "text/csv")

        # MCC if Query IC50 available
        if use_qr_ic50 and qr_ic50_col in df_qr_f.columns:
            q_ic50_raw = df_qr_f[qr_ic50_col].apply(extract_numeric)
            q_ic50_uM = convert_to_uM(q_ic50_raw, qr_ic50_unit)
            y_true = np.where(q_ic50_uM <= 6.45, 1, 0)

            # Evaluate only on rows that passed threshold (or all? you asked during TASK 2 -> use passed)
            mask_keep = df_out["Keep_by_threshold"].values
            if mask_keep.sum() == 0:
                st.warning("No query compounds passed the similarity threshold; MCC cannot be computed.")
            else:
                y_pred_eval = (np.array(pred_label)[mask_keep]).astype(int)
                y_true_eval = (y_true[mask_keep]).astype(int)
                mcc = matthews_corrcoef(y_true_eval, y_pred_eval)
                st.success(f"✅ MCC on similarity-kept set: **{mcc:.3f}**")
                st.write(f"Support (n): {mask_keep.sum()}")

# ===========================================
# TASK 3 — PSEUDO-LABEL ML + IC50 FILTERING
# ===========================================
with tab3:
    st.subheader("Pseudo-Label (Unsupervised→Supervised) + IC50 Filtering (≤ 6.45 µM)")

    uploaded_file = st.file_uploader("📤 Upload 2D/3D Descriptor File", type=["xlsx", "csv"], key="ml_file")

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        st.success(f"✅ Descriptors Loaded — shape: {df.shape}")
        st.dataframe(df.head())

        non_numeric_cols = [c for c in df.columns if df[c].dtype == object]
        if not non_numeric_cols:
            df.insert(0, "compound_id", [f"cmp_{i}" for i in range(len(df))])
            non_numeric_cols = ["compound_id"]
            st.info("No text ID column found; created synthetic `compound_id`.")
        compound_col = st.selectbox("🧬 Choose compound ID column (Descriptor file):", non_numeric_cols, index=0)

        df_numeric = safe_numeric(df)
        st.write(f"🔢 Numeric descriptor columns: **{df_numeric.shape[1]}**")
        if df_numeric.shape[1] == 0:
            st.error("No numeric descriptor columns found.")
            st.stop()

        X_train, X_test = train_test_split(df_numeric, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca, X_train_pca, X_test_pca = safe_pca_fit_transform(X_train_scaled, X_test_scaled, min_components=2)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        y_train = kmeans.fit_predict(X_train_pca)
        y_test_kmeans = kmeans.predict(X_test_pca)
        anticancer_cluster = np.argmax(np.bincount(y_train))

        st.plotly_chart(plot_pca_scatter(X_train_pca, y_train, "PCA (2D) Clustering by KMeans"), use_container_width=True)
        st.success(f"🧠 Cluster {anticancer_cluster} auto-tagged as *Anticancer-like* (pseudo-label).")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Machine": SVC(probability=True),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB(),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
        }

        results, best_model, best_name, best_f1 = [], None, None, -1.0
        for name, model in models.items():
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            acc  = accuracy_score(y_test_kmeans, y_pred)
            prec = precision_score(y_test_kmeans, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test_kmeans, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test_kmeans, y_pred, average='weighted', zero_division=0)
            results.append([name, acc, prec, rec, f1])
            if f1 > best_f1:
                best_f1, best_model, best_name = f1, model, name

        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
        st.dataframe(results_df.style.background_gradient(cmap="Greens", subset=["Accuracy", "Precision", "Recall", "F1-Score"]))
        st.success(f"🏆 Best Model: **{best_name}**  (F1 = {best_f1:.3f})")

        X_full = scaler.transform(df_numeric)
        X_full_pca = pca.transform(X_full)
        pred_cluster = kmeans.predict(X_full_pca)
        pred_labels  = best_model.predict(X_full_pca)

        try:
            probs = best_model.predict_proba(X_full_pca)[:, anticancer_cluster]
        except Exception:
            probs = np.zeros(len(df))

        df_pred = df.copy()
        df_pred["Predicted_Cluster"] = pred_cluster
        df_pred["Predicted_Label"] = np.where(pred_labels == anticancer_cluster, "Anticancer", "Non-Anticancer")
        df_pred["Probability"] = probs
        df_pred["Model"] = best_name

        st.subheader("🧾 Labeled QSAR Dataset (Preview)")
        st.dataframe(df_pred.head(12))
        st.download_button("💾 Download Labeled CSV", df_pred.to_csv(index=False).encode("utf-8"),
                           "QSAR_Labeled_Anticancer.csv", "text/csv")

        st.markdown("---")
        st.subheader("🔬 IC50-Based Filtering (≤ 6.45 µM)")
        ic50_file = st.file_uploader("📥 Upload IC50 File (e.g., DrugPAAD IC50.xlsx/CSV)", type=["xlsx", "csv"], key="ml_ic50")
        if ic50_file is not None:
            df_ic50_raw = pd.read_excel(ic50_file) if ic50_file.name.endswith(".xlsx") else pd.read_csv(ic50_file)
            st.success("✅ IC50 file loaded")
            st.dataframe(df_ic50_raw.head(12))

            ic50_id_col = st.selectbox("🔗 Compound ID column in IC50 file:", list(df_ic50_raw.columns), index=0)
            ic50_col = st.selectbox("🧪 IC50 value column:", [c for c in df_ic50_raw.columns if c != ic50_id_col], index=0)
            ic50_unit = st.selectbox("📏 Units for IC50 column:", ["µM", "nM", "pM", "mM"], index=1)

            df_ic50 = df_ic50_raw[[ic50_id_col, ic50_col]].copy()
            df_ic50["_join_id"] = normalize_id_series(df_ic50[ic50_id_col])
            df_ic50["_raw_value"] = df_ic50[ic50_col].apply(extract_numeric)
            df_ic50["IC50_uM"] = convert_to_uM(df_ic50["_raw_value"], ic50_unit)

            df_pred_norm = df_pred.copy()
            df_pred_norm["_join_id"] = normalize_id_series(df_pred_norm[compound_col])
            df_final = df_pred_norm.merge(df_ic50[["_join_id", "IC50_uM"]], on="_join_id", how="left").drop(columns=["_join_id"])

            diagnostics_after_merge(df_final, "IC50_uM")

            df_final["IC50_Activity"] = np.where(df_final["IC50_uM"] <= 6.45, "Active",
                                                 np.where(df_final["IC50_uM"].notna(), "Inactive", "Unknown"))
            df_active = df_final[df_final["IC50_Activity"] == "Active"]

            st.write("### ✅ Final (Prediction + IC50)")
            st.dataframe(df_final.head(20))
            st.write("### ✅ Filtered Strong-Active (IC50 ≤ 6.45 µM)")
            st.dataframe(df_active)

            st.download_button("💾 Download (All with IC50)", df_final.to_csv(index=False).encode("utf-8"),
                               "QSAR_with_IC50.csv", "text/csv")
            st.download_button("💾 Download Strong-Active (≤ 6.45 µM)", df_active.to_csv(index=False).encode("utf-8"),
                               "StrongActiveCompounds.csv", "text/csv")

            if len(df_active) == 0:
                st.warning("No compounds passed IC50 ≤ 6.45 µM. Check IC50 column/units and IDs.")
    else:
        st.info("Upload descriptor file to run TASK 3.")
