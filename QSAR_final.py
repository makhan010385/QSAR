# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="Drug Descriptor Analysis & Cancer Protection Prediction", layout="wide")
st.title("💊 Drug Descriptor Clustering & Anticancer Prediction (with Auto-Cluster Labeling)")

st.sidebar.header("📂 Upload Main Descriptor Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file (.csv or .xlsx)", type=["csv", "xlsx"])

# -------------------------------
# Load and preprocess dataset
# -------------------------------
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Separate numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.error("Dataset must have multiple numeric descriptor columns.")
        st.stop()

    # Scale descriptors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # -------------------------------
    # Unsupervised Clustering
    # -------------------------------
    st.sidebar.subheader("🔹 Clustering Configuration")
    k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='rainbow', alpha=0.7)
    ax.set_title("Drug Clusters (PCA Visualization)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster Assignments")
    st.dataframe(df[['Cluster']])

    # -------------------------------
    # 🧬 Automatic Anticancer Cluster Detection
    # -------------------------------
    anticancer_cluster = None
    if 'Label' in df.columns:
        if df['Label'].nunique() == 2:
            cluster_stats = df.groupby('Cluster')['Label'].mean()
            anticancer_cluster = cluster_stats.idxmax()
            st.markdown(f"🧠 **Automatically identified Cluster {anticancer_cluster} as 'Anticancer-like'** (highest % of anticancer drugs).")
        else:
            st.warning("Column 'Label' found, but it must contain binary values (0 and 1).")
    else:
        st.sidebar.subheader("🧬 Manual Selection (if no Label column)")
        anticancer_cluster = st.sidebar.selectbox(
            "Select cluster representing anticancer-like drugs:",
            sorted(df['Cluster'].unique())
        )

    # -------------------------------
    # Feature Importance
    # -------------------------------
    st.subheader("📊 Feature Importance (Cluster Differentiation)")
    rf_cluster = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cluster.fit(numeric_df, df['Cluster'])
    importances = pd.Series(rf_cluster.feature_importances_, index=numeric_df.columns)
    importances = importances.sort_values(ascending=False)
    st.bar_chart(importances.head(10))

    # -------------------------------
    # Save / Load Model
    # -------------------------------
    st.sidebar.subheader("💾 Model Management")
    save_model = st.sidebar.checkbox("Save trained model")
    load_model = st.sidebar.checkbox("Load existing model")

    if save_model:
        joblib.dump((scaler, kmeans, rf_cluster, anticancer_cluster), "drug_cluster_model.pkl")
        st.sidebar.success("Model saved as drug_cluster_model.pkl")

    if load_model:
        try:
            scaler, kmeans, rf_cluster, anticancer_cluster = joblib.load("drug_cluster_model.pkl")
            st.sidebar.success("✅ Model loaded successfully.")
        except:
            st.sidebar.error("❌ No saved model found!")

    # -------------------------------
    # 🔍 Predict New Drug Descriptor
    # -------------------------------
    st.subheader("🔍 Predict if New Drug is Protective Against Cancer")

    new_input = {}
    for col in numeric_df.columns[:10]:
        new_input[col] = st.number_input(f"{col}", value=float(numeric_df[col].mean()), step=0.01)

    if st.button("Predict Cluster"):
        new_data = np.array(list(new_input.values())).reshape(1, -1)
        if new_data.shape[1] < numeric_df.shape[1]:
            padded = np.zeros((1, numeric_df.shape[1]))
            padded[0, :new_data.shape[1]] = new_data
            new_data = padded
        new_data_scaled = scaler.transform(new_data)
        new_cluster = kmeans.predict(new_data_scaled)[0]

        st.success(f"Predicted Cluster: **{new_cluster}**")

        if new_cluster == anticancer_cluster:
            st.markdown("🧬 **Prediction:** This drug is **PROTECTIVE AGAINST CANCER** ✅")
        else:
            st.markdown("⚠️ **Prediction:** This drug is **NOT classified as anticancer-like**")

    # -------------------------------
    # 🧪 Supervised Cross-Validation
    # -------------------------------
    st.markdown("---")
    st.header("🧪 Supervised Validation with Labeled Dataset (GDSC/ChEMBL)")

    labeled_file = st.file_uploader("Upload labeled dataset (.csv/.xlsx) with target column `Label` (1=Anticancer, 0=Non-Anticancer)", type=["csv", "xlsx"])

    if labeled_file:
        if labeled_file.name.endswith(".csv"):
            labeled_df = pd.read_csv(labeled_file)
        else:
            labeled_df = pd.read_excel(labeled_file)

        st.write("Labeled Data Preview:")
        st.dataframe(labeled_df.head())

        if 'Label' not in labeled_df.columns:
            st.error("The dataset must contain a 'Label' column (1=Anticancer, 0=Non-Anticancer).")
        else:
            X = labeled_df.select_dtypes(include=[np.number]).drop(columns=['Label'], errors='ignore')
            y = labeled_df['Label']

            rf_supervised = RandomForestClassifier(n_estimators=200, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(rf_supervised, X, y, cv=cv, scoring='accuracy')

            st.write(f"✅ **5-Fold Cross-Validation Accuracy:** {scores.mean():.3f}")

            rf_supervised.fit(X, y)
            joblib.dump((scaler, rf_supervised), "anticancer_predictor.pkl")
            st.success("Supervised model trained and saved as anticancer_predictor.pkl")

            if st.button("Predict with Supervised Model"):
                new_data_scaled = scaler.transform(new_data)
                pred = rf_supervised.predict(new_data_scaled)[0]
                prob = rf_supervised.predict_proba(new_data_scaled)[0][1]

                if pred == 1:
                    st.markdown(f"🧬 **Supervised Prediction:** Protective Against Cancer ✅ (Confidence: {prob:.2f})")
                else:
                    st.markdown(f"⚠️ **Supervised Prediction:** Not Protective (Confidence: {prob:.2f})")

else:
    st.info("👆 Upload your main descriptor dataset to begin.")

st.markdown("---")
st.caption("Developed using KMeans, PCA, RandomForest & Cross-Validation for Anticancer Prediction.")
