import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import os

st.set_page_config(page_title="QSAR Transfer Learning App", layout="wide")
st.title("🧪 QSAR Model with Transfer Learning")

# -------------------------
# Model saving/loading
# -------------------------
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

def save_model(model, name):
    """Save a trained model to disk."""
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path

def load_model(name):
    """Load a saved model from disk."""
    path = MODEL_DIR / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None

# -------------------------
# Dataset management
# -------------------------
st.sidebar.header("Dataset Management")

# Source dataset (for pre-training)
st.sidebar.subheader("Source Dataset")
source_file = st.sidebar.file_uploader("📂 Upload Source Dataset (CSV)", type=["csv"])

# Target dataset (for fine-tuning)
st.sidebar.subheader("Target Dataset")
target_file = st.sidebar.file_uploader("📂 Upload Target Dataset (CSV)", type=["csv"])

# If no files uploaded, use default
if source_file is None and target_file is None:
    try:
        default_data = pd.read_csv("qsar_androgen_receptor.csv", sep=';')
        source_data = default_data
        target_data = default_data.sample(frac=0.3, random_state=42)  # Simulate different target dataset
        st.sidebar.info("Using default dataset for both source and target")
    except FileNotFoundError:
        st.sidebar.error("Please upload dataset files or place qsar_androgen_receptor.csv in the directory")
        st.stop()
else:
    if source_file:
        source_data = pd.read_csv(source_file, sep=';')
        # If only source file is provided, use a subset for target
        if not target_file:
            target_data = source_data.sample(frac=0.3, random_state=42)
            st.sidebar.info("Using a subset of the source data as target dataset")
    if target_file:
        target_data = pd.read_csv(target_file, sep=';')
        # If only target file is provided, use the same for source
        if not source_file:
            source_data = target_data.sample(frac=0.7, random_state=42)
            st.sidebar.info("Using a subset of the target data as source dataset")

# Ensure both datasets are defined
if 'source_data' not in locals() or 'target_data' not in locals():
    st.sidebar.error("Error initializing datasets. Please check your input files.")
    st.stop()

# -------------------------
# Model training and evaluation
# -------------------------
def train_model(X, y, model_type, pretrained_model=None):
    """Train a model with optional transfer learning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance
    sampler = SMOTE(random_state=42)
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    
    # Initialize model
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    
    # If using transfer learning
    if pretrained_model:
        # Fine-tune the pre-trained model
        model = pretrained_model
        model.fit(X_train, y_train)  # Continue training
    else:
        # Train from scratch
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    return model, X_test, y_test, y_pred, y_prob

# -------------------------
# Main app
# -------------------------
st.header("Transfer Learning for QSAR Modeling")

# Sidebar controls
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox("Model Type", ["Logistic Regression", "Random Forest", "Neural Network"])
use_transfer = st.sidebar.checkbox("Use Transfer Learning", value=True)

# Data preprocessing
st.subheader("Data Preprocessing")

# Let user select target column (assuming same for both datasets)
target_col = st.selectbox("Select target column", options=source_data.columns, index=len(source_data.columns)-1)

# Prepare source data
X_source = source_data.drop(columns=[target_col])
y_source = source_data[target_col]

# Prepare target data
X_target = target_data.drop(columns=[target_col])
y_target = target_data[target_col]

# Show data info
col1, col2 = st.columns(2)
with col1:
    st.metric("Source Samples", len(source_data))
    st.metric("Source Features", X_source.shape[1])
    st.write("Source class distribution:", y_source.value_counts().to_dict())

with col2:
    st.metric("Target Samples", len(target_data))
    st.metric("Target Features", X_target.shape[1])
    st.write("Target class distribution:", y_target.value_counts().to_dict())

# Model training
st.subheader("Model Training")

if st.button("🚀 Train Models"):
    # Train source model
    with st.spinner("Training source model..."):
        source_model, X_test_source, y_test_source, y_pred_source, y_prob_source = train_model(
            X_source, y_source, model_type
        )
        source_accuracy = accuracy_score(y_test_source, y_pred_source)
        st.session_state.source_model = source_model
        st.session_state.source_metrics = {
            'accuracy': source_accuracy,
            'confusion_matrix': confusion_matrix(y_test_source, y_pred_source),
            'classification_report': classification_report(y_test_source, y_pred_source, output_dict=True)
        }
    
    # Train target model with/without transfer learning
    with st.spinner("Training target model..."):
        if use_transfer and 'source_model' in st.session_state:
            target_model, X_test_target, y_test_target, y_pred_target, y_prob_target = train_model(
                X_target, y_target, model_type, pretrained_model=source_model
            )
            transfer_type = "with Transfer Learning"
        else:
            target_model, X_test_target, y_test_target, y_pred_target, y_prob_target = train_model(
                X_target, y_target, model_type
            )
            transfer_type = "from Scratch"
        
        target_accuracy = accuracy_score(y_test_target, y_pred_target)
        st.session_state.target_model = target_model
        st.session_state.target_metrics = {
            'accuracy': target_accuracy,
            'confusion_matrix': confusion_matrix(y_test_target, y_pred_target),
            'classification_report': classification_report(y_test_target, y_pred_target, output_dict=True),
            'transfer_type': transfer_type
        }
    
    # Save models
    save_model(source_model, "source_model")
    save_model(target_model, "target_model")

# Display results
if 'source_metrics' in st.session_state and 'target_metrics' in st.session_state:
    st.subheader("Model Performance")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "Metrics Comparison", "Detailed Report"])
    
    with tab1:
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Source Model Accuracy", f"{st.session_state.source_metrics['accuracy']:.2%}")
            st.subheader("Source Model Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.source_metrics['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.metric("Target Model Accuracy", 
                     f"{st.session_state.target_metrics['accuracy']:.2%}",
                     f"{st.session_state.target_metrics['transfer_type']}")
            st.subheader("Target Model Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.target_metrics['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Greens', ax=ax)
            st.pyplot(fig)
    
    with tab2:
        # Extract metrics for visualization
        source_metrics = st.session_state.source_metrics['classification_report']
        target_metrics = st.session_state.target_metrics['classification_report']
        
        # Get class names (handle both string and numeric labels)
        classes = [str(cls) for cls in source_metrics.keys() 
                  if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        # Prepare data for plotting
        data = []
        for cls in classes:
            for metric in metrics:
                data.append({
                    'Class': str(cls).capitalize(),
                    'Metric': metric.capitalize(),
                    'Source Model': source_metrics[cls][metric],
                    'Target Model': target_metrics[cls][metric]
                })
        
        df_metrics = pd.DataFrame(data)
        
        # Melt for easier plotting
        df_melted = df_metrics.melt(id_vars=['Class', 'Metric'], 
                                   var_name='Model', 
                                   value_name='Score')
        
        # Plot metrics comparison
        st.subheader("Metrics Comparison")
        
        # Plot for each class
        for cls in classes:
            st.write(f"### {str(cls).capitalize()} Class")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Filter data for current class
            plot_data = df_melted[df_melted['Class'] == str(cls).capitalize()]
            
            # Create bar plot
            sns.barplot(data=plot_data, x='Metric', y='Score', hue='Model', 
                        palette=['#1f77b4', '#2ca02c'], ax=ax)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', 
                           xytext=(0, 10), 
                           textcoords='offset points')
            
            plt.ylim(0, 1.1)
            plt.title(f"{str(cls).capitalize()} Class Metrics Comparison")
            plt.legend(title='Model', loc='upper right')
            st.pyplot(fig)
        
        # Add accuracy comparison
        st.write("### Model Accuracy Comparison")
        acc_data = pd.DataFrame({
            'Model': ['Source Model', 'Target Model'],
            'Accuracy': [source_metrics['accuracy'], target_metrics['accuracy']]
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(data=acc_data, x='Model', y='Accuracy', 
                          palette=['#1f77b4', '#2ca02c'])
        
        # Add value labels
        for p in bars.patches:
            bars.annotate(f"{p.get_height():.2%}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', 
                         xytext=(0, 10), 
                         textcoords='offset points')
        
        plt.ylim(0, 1.1)
        plt.title("Model Accuracy Comparison")
        st.pyplot(fig)
    
    with tab3:
        # Detailed report in expandable sections
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📊 Source Model Detailed Report"):
                st.json(st.session_state.source_metrics['classification_report'])
        
        with col2:
            with st.expander(f"📊 Target Model Detailed Report ({st.session_state.target_metrics['transfer_type']})"):
                st.json(st.session_state.target_metrics['classification_report'])

# Feature importance for tree-based models
if 'target_model' in st.session_state and hasattr(st.session_state.target_model, 'feature_importances_'):
    st.subheader("Feature Importance")
    
    # Get feature importances
    importances = st.session_state.target_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(10, 6))
    plt.title("Top 20 Important Features")
    plt.bar(range(20), importances[indices][:20], align='center')
    plt.xticks(range(20), X_target.columns[indices][:20], rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

# Save/Load models
st.sidebar.header("Model Management")
if st.sidebar.button("💾 Save Current Models"):
    if 'source_model' in st.session_state and 'target_model' in st.session_state:
        save_model(st.session_state.source_model, "source_model")
        save_model(st.session_state.target_model, "target_model")
        st.sidebar.success("Models saved successfully!")
    else:
        st.sidebar.warning("Train models first before saving")

if st.sidebar.button("🔍 Load Saved Models"):
    source_model = load_model("source_model")
    target_model = load_model("target_model")
    
    if source_model and target_model:
        st.session_state.source_model = source_model
        st.session_state.target_model = target_model
        st.sidebar.success("Models loaded successfully!")
    else:
        st.sidebar.error("No saved models found")

# Add some styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)
