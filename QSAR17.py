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
from sklearn.metrics import matthews_corrcoef
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

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def smiles_to_simple_descriptors(smiles):
    s = str(smiles)
    return {
        "smiles_length": len(s),
        "num_c": s.count("C"),
        "num_n": s.count("N"),
        "num_o": s.count("O"),
        "num_cl": s.count("Cl"),
        "num_br": s.count("Br"),
        "num_rings": sum(s.count(str(i)) for i in range(1,10)),
        "num_brackets": s.count("(") + s.count(")"),
        "num_double_bonds": s.count("="),
        "num_triple_bonds": s.count("#"),
        "num_aromatic": len(re.findall(r"[cnos]", s)),
        "num_atoms": len(re.findall(r"[A-Z]", s)),
        "num_lowercase": len(re.findall(r"[a-z]", s))
    }

def process_ic50(ic50):
    return ic50.astype(str).str.replace(">", "").astype(float)

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

    file = st.file_uploader("Upload SMILES file", type=["csv","xlsx"])

    if file:
        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
        smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)

        if smiles_col is None:
            st.error("SMILES column not found")
        else:
            df = df.rename(columns={smiles_col:"SMILES"})
            desc = [smiles_to_simple_descriptors(s) for s in df["SMILES"]]
            df_desc = pd.DataFrame(desc)
            df_desc.insert(0,"id",range(1,len(df_desc)+1))

            st.dataframe(df_desc.head())

            buffer = BytesIO()
            df_desc.to_excel(buffer,index=False)
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

        ic50_col = next(c for c in df_train.columns if "ic50" in c)
        df_train["label"] = (process_ic50(df_train[ic50_col]) <= 3).astype(int)

        X_train = df_train.select_dtypes(include=np.number).drop(columns=["label"])
        X_query = df_query.select_dtypes(include=np.number)

        min_cols = min(X_train.shape[1], X_query.shape[1])
        X_train = X_train.iloc[:,:min_cols]
        X_query = X_query.iloc[:,:min_cols]

        sim = cosine_similarity(X_train, X_query)
        st.write("Similarity Matrix")
        st.dataframe(sim[:5,:5])

        if "label" in df_query.columns:
            y_true = df_query["label"].values
            y_pred = (sim.mean(axis=0) > 0.5).astype(int)
            st.metric("MCC", matthews_corrcoef(y_true,y_pred))

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

        ic50_col = next(c for c in df_ic50.columns if "ic50" in c)
        y = (process_ic50(df_ic50[ic50_col]) <= 3).astype(int)

        X = df_desc.select_dtypes(include=np.number)
        X = X.loc[:len(y)-1]

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000,class_weight="balanced"),
            "SVM": SVC(probability=True,class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200,class_weight="balanced"),
            "Naive Bayes": GaussianNB(),
            "MLP": MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500),
            "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
            "LightGBM": lgb.LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0)
        }

        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        results = []

        for name,model in models.items():
            pipe = ImbPipeline([
                ("smote",SMOTE()),
                ("scaler",StandardScaler()),
                ("model",model)
            ])
            scores = cross_validate(pipe,X,y,cv=cv,
                                    scoring=["accuracy","precision","recall","f1","roc_auc"])
            results.append({
                "Model":name,
                "Accuracy":scores["test_accuracy"].mean(),
                "Precision":scores["test_precision"].mean(),
                "Recall":scores["test_recall"].mean(),
                "F1":scores["test_f1"].mean(),
                "ROC-AUC":scores["test_roc_auc"].mean()
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df)

# =========================================================
# END OF FILE
# =========================================================
