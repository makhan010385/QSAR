# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="QSAR Anticancer Classifier", layout="wide")
st.title("🧬 QSAR Anticancer Compound Analysis & ML Model Comparison")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload your descriptor file (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.success(f"✅ File loaded successfully! Shape: {df.shape}")

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric.fillna(df_numeric.mean(), inplace=True)

    # -------------------------------
    # Unsupervised KMeans Clustering
    # -------------------------------
    scaler_all = StandardScaler()
    X_scaled_all = scaler_all.fit_transform(df_numeric)

    pca_all = PCA(n_components=0.95)
    X_pca_all = pca_all.fit_transform(X_scaled_all)

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca_all)
    df["Cluster"] = clusters

    sil_score = silhouette_score(X_pca_all, clusters)
    st.metric("Silhouette Score", f"{sil_score:.3f}")

    # -------------------------------
    # Feature Importance via Random Forest
    # -------------------------------
    rf_unsup = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_unsup.fit(df_numeric, clusters)

    importances = pd.Series(rf_unsup.feature_importances_, index=df_numeric.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    top_features_list = top_features.index.tolist()  # ✅ Top 10 descriptors

    st.subheader("📊 Top 10 Important Descriptors for Cluster Separation")
    st.bar_chart(top_features)

    # -------------------------------
    # Identify anticancer-like cluster
    # -------------------------------
    descriptor_summary = df.groupby("Cluster")[top_features_list].mean()
    anticancer_like_cluster = descriptor_summary.mean(axis=1).idxmax()
    df["Predicted_Label"] = np.where(df["Cluster"] == anticancer_like_cluster,
                                     "Anticancer-like", "Non-anticancer-like")

    st.success(f"🧠 Cluster {anticancer_like_cluster} identified as *Anticancer-like*")

    # -------------------------------
    # 3D PCA Visualization
    # -------------------------------
    scaler_top = StandardScaler()
    X_scaled_top = scaler_top.fit_transform(df_numeric[top_features_list])

    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit_transform(X_scaled_top)

    pca_df = pd.DataFrame({
        "PC1": X_3d[:, 0],
        "PC2": X_3d[:, 1],
        "PC3": X_3d[:, 2],
        "Label": df["Predicted_Label"]
    })

    fig = px.scatter_3d(
        pca_df, x="PC1", y="PC2", z="PC3", color="Label",
        title="3D PCA Visualization (Anticancer vs Non-Anticancer)",
        template="plotly_white", opacity=0.8
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Supervised Model Training using Top 10 features
    # -------------------------------
    st.subheader("🤖 Model Comparison (SVM, Logistic Regression, Random Forest)")

    X = df_numeric[top_features_list]
    y = df["Predicted_Label"]  # ✅ Use biologically meaningful label

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    trained_models = {}
    results = []
    fig_cm, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        acc = report["accuracy"]
        prec = report["weighted avg"]["precision"]
        rec = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        results.append([name, acc, prec, rec, f1])

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{name}\nConfusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    st.pyplot(fig_cm)

    # -------------------------------
    # Show Results Table
    # -------------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    st.dataframe(results_df.style.highlight_max(color="lightgreen", axis=0))

    # -------------------------------
    # Feature Input for Prediction
    # -------------------------------
    st.subheader("🧪 Predict a New Compound (Top 10 Descriptor Input)")

    input_data = {}
    cols = st.columns(2)
    for idx, feature in enumerate(top_features_list):
        with cols[idx % 2]:
            input_data[feature] = st.number_input(f"{feature}", value=float(df_numeric[feature].mean()))

    input_df = pd.DataFrame([input_data])
    st.write("Input Descriptor Values:")
    st.dataframe(input_df)

    selected_model_name = st.selectbox("Select Model for Prediction", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]

    if st.button("🔍 Predict Anticancer Potential"):
        pred_label = selected_model.predict(input_df[top_features_list])[0]
        st.success(f"🧬 Prediction: This compound is **{pred_label}** according to {selected_model_name}.")

    # -------------------------------
    # Download Results
    # -------------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Predictions CSV", csv, "anticancer_predictions.csv", "text/csv")

else:
    st.info("👆 Upload your molecular descriptor dataset to start analysis.")

st.markdown("---")
st.markdown("📘 Developed by Makhan Kumbhkar — *AI-driven QSAR & Anticancer Research*")
