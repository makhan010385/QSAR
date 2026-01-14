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
from sklearn.neighbors import KNeighborsClassifier

# ---------------- UI ----------------
st.set_page_config(page_title="Strong Anticancer Filter", layout="wide")
st.title("🧬 Structure-Aware Strong Anticancer Filter (IC50 ≤ 3 µM)")

# ---------------- Session State ----------------
if "models" not in st.session_state: st.session_state.models = None
if "scaler" not in st.session_state: st.session_state.scaler = None
if "num_cols" not in st.session_state: st.session_state.num_cols = None
if "knn" not in st.session_state: st.session_state.knn = None

# ---------------- Training ----------------
st.subheader("1️⃣ Upload Training Dataset (IC50 already in µM)")
train_file = st.file_uploader("Upload training dataset", type=["csv","xlsx"], key="train")

if train_file:
    df = pd.read_excel(train_file) if train_file.name.endswith(".xlsx") else pd.read_csv(train_file)
    st.dataframe(df.head())

    id_col = st.selectbox("Compound ID column", df.columns)
    ic50_col = st.selectbox("IC50 column (µM)", df.columns)

    df["Label"] = (df[ic50_col] <= 3).astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.drop(["Label"], errors="ignore")
    X = df[num_cols]
    y = df["Label"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.3, random_state=42)

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "NB": GaussianNB()
    }

    trained = {}
    st.subheader("📊 Validation Accuracy")
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        st.write(f"{name}: {acc:.3f}")
        trained[name] = model

    # Train KNN for structural similarity
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xs, y)

    st.session_state.models = trained
    st.session_state.scaler = scaler
    st.session_state.num_cols = num_cols
    st.session_state.knn = knn

    st.success("Training completed successfully.")

# ---------------- Screening ----------------
st.subheader("2️⃣ Upload New Compounds")
test_file = st.file_uploader("Upload test compounds", type=["csv","xlsx"], key="test")

ml_threshold = st.slider("ML strong-like probability threshold", 0.0, 1.0, 0.7)
knn_ratio = st.slider("KNN strong-neighbor ratio threshold", 0.0, 1.0, 0.6)

if test_file and st.session_state.models:
    df_new = pd.read_excel(test_file) if test_file.name.endswith(".xlsx") else pd.read_csv(test_file)
    st.dataframe(df_new.head())

    X_new = df_new[st.session_state.num_cols]
    X_new_scaled = st.session_state.scaler.transform(X_new)

    # ML ensemble
    probs = []
    for model in st.session_state.models.values():
        probs.append(model.predict_proba(X_new_scaled)[:,1])
    df_new["ML_Mean_Prob"] = np.mean(probs, axis=0)

    # KNN similarity vote
    knn_preds = st.session_state.knn.predict(X_new_scaled)
    knn_probs = st.session_state.knn.predict_proba(X_new_scaled)[:,1]
    df_new["KNN_Strong_Ratio"] = knn_probs

    # Final decision
    df_new["Strong_Like"] = (df_new["ML_Mean_Prob"] >= ml_threshold) & (df_new["KNN_Strong_Ratio"] >= knn_ratio)

    st.subheader("✅ Final Selected Strong Compounds")
    df_keep = df_new[df_new["Strong_Like"]]
    st.dataframe(df_keep)

    st.download_button("Download Strong Compounds", df_keep.to_csv(index=False), "Strong_Compounds.csv")
