import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ---------------------------
# Helpers
# ---------------------------
def extract_numeric(v):
    if pd.isna(v): return np.nan
    m = re.findall(r"[-+]?\d*\.\d+|\d+", str(v))
    return float(m[0]) if m else np.nan

def convert_to_uM(v, unit):
    u = unit.lower()
    if u=="nm": return v/1000
    if u=="pm": return v/1e6
    if u=="mm": return v*1000
    return v

st.set_page_config(page_title="Strong Anticancer Filter", layout="wide")
st.title("🧬 Strong Anticancer Compound Filter (IC50 ≤ 3 µM)")

# ---------------------------
# Initialize session state
# ---------------------------
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "num_cols" not in st.session_state:
    st.session_state.num_cols = None

# ---------------------------
# Training
# ---------------------------
st.subheader("1️⃣ Upload Training Data (with IC50)")
train_file = st.file_uploader("Upload training dataset", type=["csv","xlsx"], key="train")

if train_file:
    df_tr = pd.read_excel(train_file) if train_file.name.endswith(".xlsx") else pd.read_csv(train_file)
    st.write("Training data preview:")
    st.dataframe(df_tr.head())

    id_col = st.selectbox("Compound ID", df_tr.columns)
    ic50_col = st.selectbox("IC50 column", df_tr.columns)
    ic50_unit = st.selectbox("IC50 unit", ["µM","nM","pM","mM"])

    df_tr["IC50_raw"] = df_tr[ic50_col].apply(extract_numeric)
    df_tr["IC50_uM"] = convert_to_uM(df_tr["IC50_raw"], ic50_unit)
    df_tr["Label"] = (df_tr["IC50_uM"] <= 3).astype(int)

    num_cols = df_tr.select_dtypes(include=[np.number]).columns.drop(
        ["IC50_raw","IC50_uM","Label"], errors="ignore"
    )
    X = df_tr[num_cols]
    y = df_tr["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }

    trained_models = {}
    st.subheader("📊 Validation Results")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        score = model.score(X_val, y_val)
        st.write(f"{name} Accuracy: {score:.3f}")

    st.session_state.trained_models = trained_models
    st.session_state.scaler = scaler
    st.session_state.num_cols = num_cols

    st.success("Training completed and stored.")

# ---------------------------
# Prediction
# ---------------------------
st.subheader("2️⃣ Upload New Compounds (No IC50)")
test_file = st.file_uploader("Upload new compounds file", type=["csv","xlsx"], key="test")

if test_file:
    if st.session_state.trained_models is None:
        st.warning("Please upload and train on training data first.")
    else:
        df_new = pd.read_excel(test_file) if test_file.name.endswith(".xlsx") else pd.read_csv(test_file)
        st.write("New compounds preview:")
        st.dataframe(df_new.head())

        X_new = df_new[st.session_state.num_cols]
        X_new_scaled = st.session_state.scaler.transform(X_new)

        for name, model in st.session_state.trained_models.items():
            df_new[name+"_Prob"] = model.predict_proba(X_new_scaled)[:,1]

        df_new["Mean_Prob"] = df_new[[c for c in df_new.columns if c.endswith("_Prob")]].mean(axis=1)
        df_new["Strong_Anticancer"] = df_new["Mean_Prob"] >= 0.7

        st.subheader("✅ Filtered Strong Anticancer Compounds")
        df_strong = df_new[df_new["Strong_Anticancer"]]
        st.dataframe(df_strong)

        st.download_button("Download Strong Actives", df_strong.to_csv(index=False), "Strong_Actives.csv")
