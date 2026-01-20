import pandas as pd
import numpy as np
import streamlit as st
import re
from io import BytesIO
import matplotlib.pyplot as plt

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

# Imbalanced-learn for handling class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def standardize_columns(df):
    """Standardize column names by converting to lowercase and stripping whitespace"""
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df

def find_common_numeric_columns(df1, df2):
    """Find common numeric columns between two dataframes"""
    # Get numeric columns from both dataframes
    num_cols1 = set(df1.select_dtypes(include=[np.number]).columns)
    num_cols2 = set(df2.select_dtypes(include=[np.number]).columns)
    
    # Find intersection
    common_cols = list(num_cols1.intersection(num_cols2))
    return common_cols

def smiles_to_simple_descriptors(smiles):
    s = str(smiles)
    desc = {
        "SMILES_Length": len(s),
        "Num_C": s.count("C"),
        "Num_N": s.count("N"),
        "Num_O": s.count("O"),
        "Num_Cl": s.count("Cl"),
        "Num_Br": s.count("Br"),
        "Num_Rings": s.count("1") + s.count("2") + s.count("3"),
        "Num_Brackets": s.count("(") + s.count(")"),
        "Num_Double_Bonds": s.count("="),
        "Num_Triple_Bonds": s.count("#"),
        "Num_Aromatic": len(re.findall(r"[cnos]", s)),
        "Num_Atoms": len(re.findall(r"[A-Z]", s)),
        "Num_Lowercase": len(re.findall(r"[a-z]", s))
    }
    return desc

def calculate_similarity(descriptors1, descriptors2):
    # Ensure both matrices have the same features
    common_cols = list(set(descriptors1.columns) & set(descriptors2.columns))
    if not common_cols:
        return None
    return cosine_similarity(
        descriptors1[common_cols].values,
        descriptors2[common_cols].values
    )

def process_ic50_values(ic50_series):
    """Convert IC50 values with '>' to numeric, handling various formats"""
    try:
        # Convert to string, remove '>', then to float
        return ic50_series.astype(str).str.replace('>', '').astype(float)
    except Exception as e:
        st.error(f"Error processing IC50 values: {str(e)}")
        return None

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PAAD QSAR App (IC50 ≤ 3 µM)",
    page_icon="🧪",
    layout="wide"
)

# -------------------------------------------------
# APP TITLE
# -------------------------------------------------
st.title("🧪 PAAD QSAR App (IC50 ≤ 3 µM)")
st.markdown("""
This app performs QSAR analysis with three main tasks:
1. **SMILES to Descriptors**: Convert SMILES to molecular descriptors
2. **Similarity + MCC**: Calculate similarity and Matthews Correlation Coefficient
3. **ML + IC50 ≤ 3 µM**: Machine learning prediction of active compounds
""")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1️⃣ SMILES to Descriptors",
    "2️⃣ Similarity + MCC",
    "3️⃣ ML + IC50 ≤ 3 µM"
])

# =========================================================
# TASK 1: SMILES to Descriptors
# =========================================================
with tab1:
    st.subheader("🔬 TASK 1: Convert SMILES to Descriptors")
    smiles_file = st.file_uploader("Upload SMILES file (CSV or Excel)", 
                                 type=["csv", "xlsx", "xls"],
                                 key="smiles_uploader")

    if smiles_file:
        try:
            # Read file
            if smiles_file.name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(smiles_file)
            else:
                df = pd.read_csv(smiles_file)

            # Check for SMILES column (case insensitive)
            smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
            
            if smiles_col is None:
                st.error("Error: No column containing 'SMILES' found in the uploaded file.")
                st.write("Available columns:", df.columns.tolist())
            else:
                # Rename to standardize
                df = df.rename(columns={smiles_col: 'SMILES'})
                
                # Calculate descriptors
                desc_list = []
                for smi in df['SMILES']:
                    try:
                        desc_list.append(smiles_to_simple_descriptors(smi))
                    except Exception as e:
                        st.warning(f"Could not process SMILES: {smi}. Error: {str(e)}")
                        desc_list.append({k: 0 for k in smiles_to_simple_descriptors("C").keys()})
                
                # Create new dataframe with descriptors
                df_desc = pd.DataFrame(desc_list)
                
                # Preserve original ID if exists, otherwise create one
                id_col = df.columns[0] if len(df.columns) > 0 else 'ID'
                if id_col != 'SMILES':  # Don't use SMILES as ID
                    df_desc.insert(0, 'ID', df[id_col])
                else:
                    df_desc.insert(0, 'ID', range(1, len(df) + 1))
                
                # Show preview
                st.subheader("Generated Descriptors")
                st.dataframe(df_desc.head())
                
                # Show descriptor statistics
                st.subheader("Descriptor Statistics")
                st.dataframe(df_desc.describe())
                
                # Download buttons
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_desc.to_excel(writer, index=False, sheet_name='Descriptors')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download as Excel",
                        data=output.getvalue(),
                        file_name="descriptors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        "📥 Download as CSV",
                        data=df_desc.to_csv(index=False).encode('utf-8'),
                        file_name="descriptors.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# =========================================================
# TASK 2: Similarity + MCC
# =========================================================
with tab2:
    st.subheader("📊 TASK 2: Similarity + MCC")
    
    st.markdown("""
    **Upload:**
    - Training IC50 file: `IC50 value PAAD celline specific.xlsx` 
    - Query descriptors: the file downloaded from TASK 1
    """)

    train_file = st.file_uploader("Upload Training IC50 File", 
                                 type=["xlsx", "xls", "csv"], 
                                 key="train_file")
    query_file = st.file_uploader("Upload Query Descriptors (from TASK 1)", 
                                 type=["csv", "xlsx", "xls"], 
                                 key="query_file")

    if train_file and query_file:
        with st.spinner("Calculating similarities and MCC..."):
            try:
                # Read files with error handling
                try:
                    if train_file.name.lower().endswith(('.xlsx', '.xls')):
                        df_train = pd.read_excel(train_file)
                    else:
                        df_train = pd.read_csv(train_file)
                except Exception as e:
                    st.error(f"Error reading training file: {str(e)}")
                    st.stop()
                
                try:
                    if query_file.name.lower().endswith(('.xlsx', '.xls')):
                        df_query = pd.read_excel(query_file)
                    else:
                        df_query = pd.read_csv(query_file)
                except Exception as e:
                    st.error(f"Error reading query file: {str(e)}")
                    st.stop()
                
                # Basic validation
                if df_train.empty or df_query.empty:
                    st.error("Error: One or both files are empty")
                    st.stop()
                
                # Standardize column names
                df_train = standardize_columns(df_train)
                df_query = standardize_columns(df_query)
                
                # Find IC50 column (case insensitive)
                ic50_col = next((col for col in df_train.columns if 'ic50' in col.lower()), None)
                if ic50_col is None:
                    st.error("Error: Could not find IC50 column in the training file.")
                    st.write("Available columns in training file:", df_train.columns.tolist())
                    st.stop()
                
                st.write("### File Information")
                st.write(f"- Training data shape: {df_train.shape}")
                st.write(f"- Query data shape: {df_query.shape}")
                st.write(f"Using IC50 column: '{ic50_col}'")
                
                # Process IC50 values
                ic50_values = process_ic50_values(df_train[ic50_col])
                if ic50_values is None:
                    st.error("Could not process IC50 values. Please check the format.")
                    st.stop()
                
                # Create binary labels
                df_train['label'] = (ic50_values <= 3).astype(int)
                
                # Show class distribution
                st.write("### Class Distribution in Training Data")
                class_counts = df_train['label'].value_counts()
                st.write(f"- Active (IC50 ≤ 3): {class_counts.get(1, 0)} compounds")
                st.write(f"- Inactive (IC50 > 3): {class_counts.get(0, 0)} compounds")
                
                if class_counts.get(1, 0) == 0 or class_counts.get(0, 0) == 0:
                    st.warning("Warning: One of the classes has no samples. MCC calculation may not be meaningful.")
                
                # Find common numeric features
                common_cols = find_common_numeric_columns(df_train, df_query)
                
                if not common_cols:
                    st.warning("No common numeric columns found. Attempting to align by position...")
                    
                    # Try to align by position if column counts match
                    train_num_cols = df_train.select_dtypes(include=[np.number]).columns
                    query_num_cols = df_query.select_dtypes(include=[np.number]).columns
                    
                    if len(train_num_cols) > 0 and len(query_num_cols) > 0:
                        min_cols = min(len(train_num_cols), len(query_num_cols))
                        X_train = df_train[train_num_cols[:min_cols]].fillna(0)
                        X_query = df_query[query_num_cols[:min_cols]].fillna(0)
                        common_cols = [f"feature_{i}" for i in range(min_cols)]
                        X_train.columns = common_cols
                        X_query.columns = common_cols
                        st.warning(f"Aligned first {min_cols} numeric columns by position")
                    else:
                        st.error("No numeric columns found in one or both datasets.")
                        st.write("Training numeric columns:", train_num_cols.tolist())
                        st.write("Query numeric columns:", query_num_cols.tolist())
                        st.stop()
                else:
                    st.write(f"Found {len(common_cols)} common numeric features")
                    X_train = df_train[common_cols].fillna(0)
                    X_query = df_query[common_cols].fillna(0)
                
                # Calculate similarity
                similarity = cosine_similarity(X_train, X_query)
                
                # Get top-k most similar compounds
                k = min(5, len(X_train))
                top_k_indices = np.argsort(similarity, axis=0)[-k:].T
                
                # Display results
                st.subheader(f"Top {k} Most Similar Compounds")
                
                # Create a DataFrame for better display
                results = []
                for i, idx in enumerate(top_k_indices):
                    for j, comp_idx in enumerate(idx):
                        results.append({
                            'Query': f"Query {i+1}",
                            'Rank': j+1,
                            'Training_ID': df_train.iloc[comp_idx][df_train.columns[0]],
                            'Similarity': similarity[comp_idx, i],
                            'IC50_Value': df_train.iloc[comp_idx][ic50_col],
                            'Label': 'Active' if df_train.iloc[comp_idx]['label'] == 1 else 'Inactive'
                        })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df.pivot(index='Training_ID', columns='Query', 
                                            values=['Similarity', 'IC50_Value', 'Label']))
                
                # Calculate MCC if we have labels for query
                if 'label' in df_query.columns:
                    y_true = df_query['label'].values
                    if len(y_true) == similarity.shape[1]:  # Check if lengths match
                        y_pred = (np.mean(similarity, axis=0) > 0.5).astype(int)
                        mcc = matthews_corrcoef(y_true, y_pred)
                        st.metric("Matthews Correlation Coefficient", f"{mcc:.3f}")
                    else:
                        st.warning("Label length doesn't match query size. Skipping MCC calculation.")
                
                # Download results
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Similarity matrix
                    sim_df = pd.DataFrame(
                        similarity,
                        index=df_train[df_train.columns[0]],
                        columns=df_query[df_query.columns[0]]
                    )
                    sim_df.to_excel(writer, sheet_name='Similarity_Matrix')
                    
                    # Top matches
                    results_df.to_excel(writer, sheet_name='Top_Matches', index=False)
                    
                    # Training data with labels
                    df_train[[df_train.columns[0], ic50_col, 'label']].to_excel(
                        writer, sheet_name='Training_Data', index=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Full Report (Excel)",
                        data=output.getvalue(),
                        file_name="similarity_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Similarity Matrix (CSV)",
                        data=pd.DataFrame(similarity, 
                                        index=df_train[df_train.columns[0]], 
                                        columns=df_query[df_query.columns[0]]).to_csv(),
                        file_name="similarity_matrix.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

# =========================================================
# TASK 3: ML + IC50 ≤ 3 (Enhanced Version)
# =========================================================
with tab3:
    st.subheader("🤖 TASK 3: ML + Filter IC50 ≤ 3 µM")

    st.markdown("""
    **Upload:**
    - Descriptor file (from TASK 1)
    - IC50 file: `IC50 value PAAD celline specific.xlsx` 
    """)

    desc_file_ml = st.file_uploader("Upload Descriptor File", 
                                   type=["csv", "xlsx", "xls"], 
                                   key="ml_desc")
    ic50_file_ml = st.file_uploader("Upload IC50 File", 
                                   type=["xlsx", "xls", "csv"], 
                                   key="ml_ic50")

    if desc_file_ml and ic50_file_ml:
        with st.spinner("Processing data and training models..."):
            try:
                # Read and validate files
                try:
                    if desc_file_ml.name.lower().endswith(('.xlsx', '.xls')):
                        df_desc = pd.read_excel(desc_file_ml)
                    else:
                        df_desc = pd.read_csv(desc_file_ml)
                    
                    if ic50_file_ml.name.lower().endswith(('.xlsx', '.xls')):
                        df_ic50 = pd.read_excel(ic50_file_ml)
                    else:
                        df_ic50 = pd.read_csv(ic50_file_ml)
                except Exception as e:
                    st.error(f"Error reading files: {str(e)}")
                    st.stop()
                
                # Basic data validation
                if df_desc.empty or df_ic50.empty:
                    st.error("Error: One or both files are empty")
                    st.stop()
                
                # Standardize column names
                df_desc = standardize_columns(df_desc)
                df_ic50 = standardize_columns(df_ic50)
                
                st.write("### Data Summary")
                st.write(f"- Descriptors shape: {df_desc.shape}")
                st.write(f"- IC50 data shape: {df_ic50.shape}")
                
                # Find IC50 column (case insensitive)
                id_col = df_desc.columns[0]  # First column is ID
                ic50_col = next((col for col in df_ic50.columns if 'ic50' in col.lower()), None)
                
                if ic50_col is None:
                    st.error("Error: Could not find IC50 column in the uploaded file.")
                    st.write("Available columns in IC50 file:", df_ic50.columns.tolist())
                    st.stop()
                
                st.write(f"Using IC50 column: '{ic50_col}'")
                
                # Process IC50 values
                ic50_values = process_ic50_values(df_ic50[ic50_col])
                if ic50_values is None:
                    st.error("Could not process IC50 values. Please check the format.")
                    st.stop()
                
                # Create binary labels
                df_ic50 = df_ic50.copy()
                df_ic50["label"] = (ic50_values <= 3).astype(int)
                
                # Show class distribution
                st.write("### Class Distribution")
                class_counts = df_ic50['label'].value_counts()
                st.write(f"- Active (IC50 ≤ 3): {class_counts.get(1, 0)} compounds")
                st.write(f"- Inactive (IC50 > 3): {class_counts.get(0, 0)} compounds")
                
                if class_counts.get(1, 0) < 10 or class_counts.get(0, 0) < 10:
                    st.warning("Warning: One or both classes have very few samples. Model performance may be poor.")
                
                # Align data
                min_len = min(len(df_desc), len(df_ic50))
                if len(df_desc) != len(df_ic50):
                    st.warning(f"Data length mismatch. Using first {min_len} samples from each file.")
                    df_ic50 = df_ic50.iloc[:min_len].copy()
                    df_desc = df_desc.iloc[:min_len].copy()
                
                # Prepare features and target
                X = df_desc.select_dtypes(include=[np.number]).fillna(0)
                y = df_ic50["label"].values
                
                # Check for constant features
                constant_columns = X.columns[X.nunique() <= 1]
                if not constant_columns.empty:
                    st.warning(f"Removing {len(constant_columns)} constant features: {', '.join(constant_columns)}")
                    X = X.drop(columns=constant_columns)
                
                # Define models with balanced class weights
                models = {
                    "Logistic Regression": LogisticRegression(
                        class_weight='balanced', 
                        max_iter=1000, 
                        random_state=42
                    ),
                    "SVM": SVC(
                        class_weight='balanced', 
                        probability=True, 
                        random_state=42
                    ),
                    "Decision Tree": DecisionTreeClassifier(
                        class_weight='balanced', 
                        random_state=42
                    ),
                    "Random Forest": RandomForestClassifier(
                        class_weight='balanced', 
                        random_state=42, 
                        n_jobs=-1
                    ),
                    "Naive Bayes": GaussianNB()
                }
                
                # Evaluate models using cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                results = []
                
                for name, model in models.items():
                    with st.spinner(f"Training {name}..."):
                        try:
                            # Create pipeline with SMOTE and model
                            pipeline = ImbPipeline([
                                ('smote', SMOTE(random_state=42)),
                                ('scaler', StandardScaler()),
                                ('model', model)
                            ])
                            
                            # Cross-validate
                            cv_scores = cross_validate(
                                pipeline, X, y, 
                                cv=cv,
                                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                return_train_score=False,
                                n_jobs=-1
                            )
                            
                            # Store results
                            results.append({
                                'Model': name,
                                'Accuracy': f"{np.mean(cv_scores['test_accuracy']):.3f} ± {np.std(cv_scores['test_accuracy']):.3f}",
                                'Precision': f"{np.mean(cv_scores['test_precision']):.3f} ± {np.std(cv_scores['test_precision']):.3f}",
                                'Recall': f"{np.mean(cv_scores['test_recall']):.3f} ± {np.std(cv_scores['test_recall']):.3f}",
                                'F1': f"{np.mean(cv_scores['test_f1']):.3f} ± {np.std(cv_scores['test_f1']):.3f}",
                                'ROC-AUC': f"{np.mean(cv_scores['test_roc_auc']):.3f} ± {np.std(cv_scores['test_roc_auc']):.3f}"
                            })
                            
                        except Exception as e:
                            st.error(f"Error with {name}: {str(e)}")
                            continue
                
                if not results:
                    st.error("No models were successfully trained. Please check your data and try again.")
                    st.stop()
                
                # Display results
                st.subheader("📊 Model Performance (5-fold CV)")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Train final model on full data (Random Forest)
                st.subheader("🔍 Feature Importance (Random Forest)")
                rf = RandomForestClassifier(
                    class_weight='balanced', 
                    random_state=42,
                    n_jobs=-1
                )
                
                # Scale features
                X_scaled = StandardScaler().fit_transform(X)
                rf.fit(X_scaled, y)
                
                # Feature importance
                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot top 20 important features
                fig, ax = plt.subplots(figsize=(12, 8))
                top_n = min(20, len(X.columns))
                ax.barh(range(top_n), importances[indices][:top_n], align='center')
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([X.columns[i] for i in indices[:top_n]])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 20 Important Features')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Make predictions
                df_desc = df_desc.copy()
                df_desc["Prediction_Prob"] = rf.predict_proba(X_scaled)[:, 1]
                df_desc["Prediction"] = (df_desc["Prediction_Prob"] > 0.5).astype(int)
                df_desc["Predicted_Label"] = np.where(
                    df_desc["Prediction"] == 1, 
                    "Active (≤3 µM)", 
                    "Inactive (>3 µM)"
                )
                
                # Show predictions
                st.subheader("📋 Predictions Summary")
                pred_summary = df_desc["Prediction"].value_counts().to_frame("Count")
                pred_summary["Percentage"] = (pred_summary["Count"] / len(df_desc) * 100).round(1)
                st.dataframe(pred_summary)
                
                # Show sample predictions
                st.subheader("🔍 Sample Predictions")
                st.dataframe(df_desc[[id_col, "Predicted_Label", "Prediction_Prob"]].head(10))
                
                # Prepare downloads
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = {
                        "Metric": ["Total Compounds", "Predicted Active", "Predicted Inactive"],
                        "Count": [
                            len(df_desc), 
                            (df_desc["Prediction"] == 1).sum(), 
                            (df_desc["Prediction"] == 0).sum()
                        ],
                        "Percentage": [
                            "100%", 
                            f"{(df_desc['Prediction'] == 1).mean()*100:.1f}%", 
                            f"{(df_desc['Prediction'] == 0).mean()*100:.1f}%"
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
                    
                    # Model performance
                    results_df.to_excel(writer, sheet_name="Model_Performance", index=False)
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
                    
                    # Predictions
                    df_desc.to_excel(writer, sheet_name="All_Predictions", index=False)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Full Report (Excel)",
                        data=output.getvalue(),
                        file_name="ML_IC50_Predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Predictions (CSV)",
                        data=df_desc.to_csv(index=False).encode('utf-8'),
                        file_name="ML_IC50_Predictions.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)