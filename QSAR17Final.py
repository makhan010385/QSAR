# =========================================================
# PAAD QSAR STREAMLIT APP (FULL VERSION)
# =========================================================

import pandas as pd
import numpy as np
import streamlit as st
import re
from io import BytesIO
import matplotlib.pyplot as plt

# ---------------- ML IMPORTS ----------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------- RDKIT (optional) ----------------
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def find_id_column(df):
    candidates = []
    for c in df.columns:
        normalized = str(c).strip().lower().replace(" ", "_")
        if normalized in {"id", "name", "compound", "molecule", "cid", "drug"}:
            candidates.append((c, normalized, 0))
        elif any(keyword in normalized for keyword in ["drug_name", "drugname", "drug_id", "drugid", "compound_id", "compoundid", "molecule_id", "mol_id", "chembl_id", "pubchem_id"]):
            candidates.append((c, normalized, 1))
        elif "name" in normalized or "id" in normalized:
            candidates.append((c, normalized, 2))
    if candidates:
        candidates.sort(key=lambda x: x[2])
        return candidates[0][0]
    return None

def merge_by_id(df_left, df_right):
    id_left = find_id_column(df_left)
    id_right = find_id_column(df_right)
    if id_left and id_right:
        if df_left[id_left].duplicated().any():
            st.warning(f"Duplicate IDs found in left file column '{id_left}'. Merge may create duplicated rows.")
        if df_right[id_right].duplicated().any():
            st.warning(f"Duplicate IDs found in right file column '{id_right}'. Merge may create duplicated rows.")
        left = df_left.copy()
        right = df_right.copy()
        left[id_left] = left[id_left].astype(str)
        right[id_right] = right[id_right].astype(str)
        return pd.merge(left, right, left_on=id_left, right_on=id_right, how="inner")
    return None

def numeric_features(df):
    id_col = find_id_column(df)
    if id_col:
        df = df.drop(columns=[id_col], errors="ignore")
    return df.select_dtypes(include=np.number)

def smiles_to_simple_descriptors(smiles):
    s = str(smiles).strip()
    return {
        "smiles_length": len(s),
        "num_c": s.count("C") + s.count("c"),
        "num_n": s.count("N") + s.count("n"),
        "num_o": s.count("O") + s.count("o"),
        "num_f": s.count("F"),
        "num_cl": s.count("Cl"),
        "num_br": s.count("Br"),
        "num_i": s.count("I"),
        "num_p": s.count("P") + s.count("p"),
        "num_s": s.count("S") + s.count("s"),
        "num_rings": sum(s.count(str(i)) for i in range(1, 10)),
        "num_brackets": s.count("(") + s.count(")"),
        "num_double_bonds": s.count("="),
        "num_triple_bonds": s.count("#"),
        "num_aromatic": len(re.findall(r"[a-z]", s)),
        "num_atoms": len(re.findall(r"[A-Z][a-z]?", s))
    }

def smiles_to_rdkit_descriptors(smiles):
    desc_names = [name for name, _ in Descriptors.descList]
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {name: np.nan for name in desc_names}
    out = {}
    for name, func in Descriptors.descList:
        try:
            out[name] = float(func(mol))
        except Exception:
            out[name] = np.nan
    return out

def process_ic50(ic50):
    s = ic50.astype(str).str.replace(">", "", regex=False).str.replace("<", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PAAD QSAR App (IC50 ≤ 3 µM)",
    layout="wide",
    page_icon="🧪"
)

st.title("🧪 PAAD QSAR App (IC50 ≤ 3 µM)")
st.markdown("""
**End-to-end QSAR pipeline**
- SMILES → Descriptors  
- Similarity + MCC  
- Machine Learning (9 models)  
""")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ SMILES → Descriptors",
    "2️⃣ Similarity + MCC",
    "3️⃣ ML Prediction (IC50 ≤ 3 µM)"
])

# =========================================================
# TASK 1: SMILES → DESCRIPTORS
# =========================================================

with tab1:
    st.subheader("🔬 TASK 1: Convert SMILES to Descriptors")

    desc_options = ["Simple"]
    if RDKIT_AVAILABLE:
        desc_options.append("RDKit")
    desc_type = st.selectbox("Descriptor engine", desc_options, index=0)

    file = st.file_uploader("Upload SMILES file", type=["csv","xlsx"])

    if file:
        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
        smiles_col = next((c for c in df.columns if "smiles" in str(c).lower()), None)

        if smiles_col is None:
            st.error("SMILES column not found")
        else:
            df = df.rename(columns={smiles_col:"SMILES"})
            if desc_type == "RDKit":
                desc = [smiles_to_rdkit_descriptors(s) for s in df["SMILES"]]
            else:
                desc = [smiles_to_simple_descriptors(s) for s in df["SMILES"]]
            df_desc = pd.DataFrame(desc)
            invalid = df_desc.isna().all(axis=1).sum()
            if invalid > 0:
                st.warning(f"{invalid} SMILES could not be parsed and produced NaN descriptors.")
            id_col = find_id_column(df)
            if id_col:
                df_desc.insert(0, "id", df[id_col].astype(str).values)
            else:
                df_desc.insert(0, "id", range(1, len(df_desc)+1))
            df_desc["smiles"] = df["SMILES"].values

            st.write(f"{len(df_desc)} descriptors generated ({len(df_desc.columns)-2} features)")
            st.dataframe(df_desc.head())

            buffer = BytesIO()
            df_desc.to_excel(buffer, index=False)
            st.download_button(
                "⬇ Download Descriptors",
                buffer.getvalue(),
                "descriptors.xlsx"
            )

# =========================================================
# TASK 2: SIMILARITY + MCC
# =========================================================

with tab2:
    st.subheader("📊 TASK 2: Similarity Filtering + MCC")

    train_file = st.file_uploader("Training IC50 file", type=["csv","xlsx"], key="t2a")
    query_file = st.file_uploader("Query descriptor file", type=["csv","xlsx"], key="t2b")

    if train_file and query_file:
        df_train = pd.read_excel(train_file) if train_file.name.endswith("xlsx") else pd.read_csv(train_file)
        df_query = pd.read_excel(query_file) if query_file.name.endswith("xlsx") else pd.read_csv(query_file)

        df_train = standardize_columns(df_train)
        df_query = standardize_columns(df_query)

        ic50_col = next((c for c in df_train.columns if "ic50" in c), None)
        if ic50_col is None:
            st.error("IC50 column not found in training file")
        else:
            train_values = process_ic50(df_train[ic50_col])
            valid_train = train_values.notna()
            if valid_train.sum() == 0:
                st.error("No valid IC50 values found in training file")
            else:
                df_train = df_train[valid_train].copy()
                df_train["label"] = (train_values[valid_train] <= 3).astype(int)

                X_train = numeric_features(df_train).drop(columns=["label"], errors="ignore")
                X_query = numeric_features(df_query)

                common_cols = list(X_train.columns.intersection(X_query.columns))
                if len(common_cols) == 0:
                    merged = merge_by_id(df_train, df_query)
                    if merged is not None and len(merged) > 0:
                        X_train_merged = numeric_features(merged).drop(columns=["label"], errors="ignore")
                        X_query = numeric_features(df_query)
                        common_cols = list(X_train_merged.columns.intersection(X_query.columns))
                        if len(common_cols) > 0:
                            X_train = X_train_merged[common_cols]
                            X_query = X_query[common_cols]
                            st.info(f"Merged training and query by ID; using {len(common_cols)} common descriptors.")
                        else:
                            st.error("No common descriptor columns after ID merge.")
                            merged = None
                    if not common_cols:
                        st.warning("No common ID column found. Falling back to row-order alignment.")
                        min_len = min(len(df_train), len(df_query))
                        if min_len > 0:
                            df_train_aligned = df_train.iloc[:min_len]
                            df_query_aligned = df_query.iloc[:min_len]
                            X_train = numeric_features(df_train_aligned).drop(columns=["label"], errors="ignore")
                            X_query = numeric_features(df_query_aligned)
                            common_cols = list(X_train.columns.intersection(X_query.columns))
                            if len(common_cols) > 0:
                                X_train = X_train[common_cols]
                                X_query = X_query[common_cols]
                                st.info(f"Aligned by row order; using {len(common_cols)} common descriptors.")
                    if not common_cols:
                        st.error("No common descriptor columns between training and query files. The training IC50 file must contain descriptor columns, or both files must share an ID/name column.")

                if len(common_cols) > 0:
                    X_train = X_train.dropna()
                    X_query = X_query.dropna()
                    if len(X_train) == 0 or len(X_query) == 0:
                        st.error("No valid descriptors after removing NaN values. Check your SMILES/input data.")
                    else:
                        st.write(f"Training rows after dropping NaN: {len(X_train)}; Query rows: {len(X_query)}")
                        sim = cosine_similarity(X_train, X_query)
                        st.write("Similarity Matrix (first 5x5)")
                        st.dataframe(sim[:5, :5])

                        if "label" in df_query.columns:
                            y_true = df_query.loc[X_query.index, "label"].values
                            y_pred = (sim.mean(axis=0) > 0.5).astype(int)
                            st.metric("MCC", matthews_corrcoef(y_true, y_pred))

# =========================================================
# TASK 3: ML MODELS (ALL)
# =========================================================

with tab3:
    st.subheader("🤖 TASK 3: ML Prediction (IC50 ≤ 3 µM)")

    desc_file = st.file_uploader("Descriptor file", type=["csv","xlsx"], key="t3a")
    ic50_file = st.file_uploader("IC50 file", type=["csv","xlsx"], key="t3b")

    if desc_file and ic50_file:
        df_desc = pd.read_excel(desc_file) if desc_file.name.endswith("xlsx") else pd.read_csv(desc_file)
        df_ic50 = pd.read_excel(ic50_file) if ic50_file.name.endswith("xlsx") else pd.read_csv(ic50_file)

        df_desc = standardize_columns(df_desc)
        df_ic50 = standardize_columns(df_ic50)

        merged = merge_by_id(df_desc, df_ic50)
        if merged is None:
            st.warning("No common id/name/compound column found. Falling back to row-order alignment.")
            min_len = min(len(df_desc), len(df_ic50))
            if min_len == 0:
                st.error("One of the uploaded files is empty.")
            else:
                merged = pd.concat([df_desc.iloc[:min_len],
                                    df_ic50.iloc[:min_len]], axis=1)

        if merged is not None and len(merged) > 0:
            ic50_col = next((c for c in merged.columns if "ic50" in c), None)
            if ic50_col is None:
                st.error("IC50 column not found")
            else:
                y_raw = process_ic50(merged[ic50_col])
                valid = y_raw.notna()
                if valid.sum() == 0:
                    st.error("No valid IC50 values found")
                else:
                    merged = merged[valid]
                    y = (y_raw[valid] <= 3).astype(int)
                    X = numeric_features(merged).dropna()
                    y = y.loc[X.index]

                    if len(X) == 0:
                        st.error("No valid descriptors after removing NaN values. Check your SMILES/input data.")
                    elif len(np.unique(y)) < 2:
                        st.error("Only one class present in labels. Cannot train classifiers.")
                    elif min(np.bincount(y)) < 2:
                        st.error("At least 2 samples are required in each class for SMOTE.")
                    else:
                        st.write(f"Samples after merge and NaN drop: {len(merged)}; Active (IC50 ≤ 3 µM): {(y == 1).sum()}; Inactive: {(y == 0).sum()}")

                        mcc_scorer = make_scorer(matthews_corrcoef)
                        scoring = {
                            "accuracy": "accuracy",
                            "precision": "precision",
                            "recall": "recall",
                            "f1": "f1",
                            "roc_auc": "roc_auc",
                            "mcc": mcc_scorer
                        }

                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                            "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
                            "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
                            "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
                            "Naive Bayes": GaussianNB(),
                            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
                            "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
                            "LightGBM": lgb.LGBMClassifier(random_state=42),
                            "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
                        }

                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        results = []
                        progress = st.progress(0.0)

                        for i, (name, model) in enumerate(models.items()):
                            pipe = ImbPipeline([
                                ("smote", SMOTE(random_state=42)),
                                ("scaler", StandardScaler()),
                                ("model", model)
                            ])
                            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
                            results.append({
                                "Model": name,
                                "Accuracy": scores["test_accuracy"].mean(),
                                "Precision": scores["test_precision"].mean(),
                                "Recall": scores["test_recall"].mean(),
                                "F1": scores["test_f1"].mean(),
                                "ROC-AUC": scores["test_roc_auc"].mean(),
                                "MCC": scores["test_mcc"].mean()
                            })
                            progress.progress((i + 1) / len(models))

                        res_df = pd.DataFrame(results)
                        st.dataframe(res_df)

                        buffer = BytesIO()
                        res_df.to_excel(buffer, index=False)
                        st.download_button("⬇ Download Results", buffer.getvalue(), "ml_results.xlsx")

# =========================================================
# END OF FILE
# =========================================================
