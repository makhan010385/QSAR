# ===========================================
# Unsupervised + Supervised QSAR Pseudo-Labeling with Export
# Updated with XGBoost, LightGBM, CatBoost, MLP
# ===========================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import plotly.express as px

# -----------------------------
# Streamlit App Title
# -----------------------------
st.set_page_config(page_title="2D/3D QSAR Unsupervised + ML Pipeline", layout="wide")
st.title("🧬 Analysis of Anticancer or Cancer Drug using QSAR Modelling")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload your 2D/3D Descriptor File", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read dataset
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success(f"✅ Dataset Loaded! Shape: {df.shape}")
    st.write("### Raw Data Preview", df.head())

    # Drop non-numeric columns but preserve compound names (if any)
    compound_col = None
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            compound_col = col
            break

    df_numeric = df.select_dtypes(include=[np.number])
    st.write(f"### Numeric Descriptors Used: {df_numeric.shape[1]} columns")

    # -----------------------------
    # Split, Scale, PCA
    # -----------------------------
    X_train, X_test = train_test_split(df_numeric, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # -----------------------------
    # KMeans Pseudo-label Generation
    # -----------------------------
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_train = kmeans.fit_predict(X_train_pca)
    y_test_kmeans = kmeans.predict(X_test_pca)

    # Determine which cluster is “anticancer-like”
    anticancer_cluster = np.argmax(np.bincount(y_train))
    st.success(f"🧠 Cluster {anticancer_cluster} identified as *Anticancer-like*")

    st.subheader("🔹 PCA Visualization of Clusters")
    fig_pca = px.scatter(
        x=X_train_pca[:, 0],
        y=X_train_pca[:, 1],
        color=y_train.astype(str),
        title="PCA (2D) Clustering by KMeans",
        labels={'x': 'PCA1', 'y': 'PCA2', 'color': 'Cluster'}
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # -----------------------------
    # Train Supervised Models
    # -----------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    results = []
    best_model_name = None
    best_f1 = 0
    best_model = None

    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        acc = accuracy_score(y_test_kmeans, y_pred)
        prec = precision_score(y_test_kmeans, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_kmeans, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_kmeans, y_pred, average='weighted', zero_division=0)
        results.append([name, acc, prec, rec, f1])

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # -----------------------------
    # Display Results
    # -----------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    st.write("### ⚙️ Model Evaluation Results")
    st.dataframe(results_df.style.background_gradient(cmap="Greens", subset=["Accuracy", "Precision", "Recall", "F1-Score"]))
    st.success(f"🏆 Best Performing Model: **{best_model_name}** (F1 = {best_f1:.3f})")

    # -----------------------------
    # Apply Best Model to Full Dataset
    # -----------------------------
    df_scaled_full = scaler.transform(df_numeric)
    df_pca_full = pca.transform(df_scaled_full)

    predicted_clusters = kmeans.predict(df_pca_full)
    model_labels = best_model.predict(df_pca_full)
    probabilities = None
    try:
        probabilities = best_model.predict_proba(df_pca_full)[:, anticancer_cluster]
    except:
        probabilities = np.zeros(len(df))

    df_result = df.copy()
    df_result["Predicted_Cluster"] = predicted_clusters
    df_result["Predicted_Label"] = np.where(model_labels == anticancer_cluster, "Anticancer", "Non-Anticancer")
    df_result["Probability"] = probabilities
    df_result["Model"] = best_model_name

    st.subheader("🧾 Labeled QSAR Dataset Preview")
    st.dataframe(df_result.head(10))

    # -----------------------------
    # Download CSV
    # -----------------------------
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Labeled CSV", csv, "QSAR_Labeled_Anticancer.csv", "text/csv")

    # -----------------------------
    # Notes
    # -----------------------------
    st.markdown("""
    ---
    ### 🧾 Notes
    - Anti-cancer or Cancer Drug Classification using KMeans clustering.
    - The best supervised model was selected  out of 9  models using Accuracy, F1-score, Precision  and Recall.
    - Exported CSV contains:
        - `Predicted_Cluster`
        - `Predicted_Label` (Anticancer / Non-Anticancer)
        - `Probability`
        - `Model`
    - This helps identify potential anticancer compounds from unlabeled QSAR data.
    """)

else:
    st.info("📂 Please upload your `2D3D.xlsx` or `2D3D.csv` descriptor file to begin.")
