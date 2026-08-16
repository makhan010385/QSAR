# ============================================================
# PAAD QSAR DRUG DISCOVERY SYSTEM
# ============================================================
#
# TASK 1:
#   SMILES -> Molecular Descriptors
#
# TASK 2:
#   IC50 -> pIC50 (conversion only)
#   pIC50 -> Active / Inactive
#   Download PAAD_IC50_pIC50.xlsx
#
# TASK 3:
#   Descriptor file + PAAD_IC50_pIC50.xlsx
#   -> Match compounds
#   -> Supervised ML
#   -> 5-Fold Cross Validation
#   -> MCC / Accuracy / Precision / Recall / F1 / ROC-AUC
#   -> Out-of-fold predictions
#   -> Final model
#   -> Virtual screening
#   -> Active candidate ranking
#
# ============================================================

import re
import difflib
import time
from io import BytesIO
from urllib.parse import quote

import requests

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Optional gradient boosting libraries
XGBOOST_OK = False
LIGHTGBM_OK = False
CATBOOST_OK = False

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_OK = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except Exception:
    pass

# ------------------------------------------------------------
# Optional imbalanced-learn
# ------------------------------------------------------------

IMBLEARN_OK = True

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    IMBLEARN_OK = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PAAD QSAR Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

# IC50 = 3 µM corresponds to pIC50 = -log10(3e-6) = 5.522879...
# Activity classification is performed using pIC50 only.
PIC50_THRESHOLD = 5.522879


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    .download-button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .section-header {
        color: #1e293b;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    .step-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        margin-right: 8px;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 1.25rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1rem;">
            <div style="font-size:2.5rem;">🧬</div>
            <div style="font-size:1.3rem; font-weight:700; color:#1e40af;">
                PAAD QSAR
            </div>
            <div style="font-size:0.9rem; color:#64748b;">
                Drug Discovery Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### 📋 Workflow Steps")
    st.markdown(
        """
        <div class="info-card" style="border-left-color:#10b981;">
            <span class="step-badge">1</span>
            <strong>SMILES → Descriptors</strong><br>
            <span style="font-size:0.85rem; color:#64748b;">
            Generate molecular descriptors from SMILES strings.
            </span>
        </div>
        <div class="info-card" style="border-left-color:#f59e0b;">
            <span class="step-badge">2</span>
            <strong>IC50 → pIC50</strong><br>
            <span style="font-size:0.85rem; color:#64748b;">
            Convert IC50 to pIC50 and classify activity using pIC50.
            </span>
        </div>
        <div class="info-card" style="border-left-color:#8b5cf6;">
            <span class="step-badge">3</span>
            <strong>ML + Screening</strong><br>
            <span style="font-size:0.85rem; color:#64748b;">
            Train models, evaluate with CV-MCC, and screen candidates.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### ⚗️ Activity Rule")
    st.markdown(
        f"""
        - **Active:** pIC50 ≥ {PIC50_THRESHOLD:.4f}
        - **Inactive:** pIC50 < {PIC50_THRESHOLD:.4f}
        - **pIC50 threshold:** {PIC50_THRESHOLD:.4f}
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="footer" style="border-top:none; margin-top:1rem;">
            Built with Streamlit & scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="hero-title">🧬 PAAD QSAR Drug Discovery</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'A complete computational workflow for PAAD drug discovery: '
    'from SMILES descriptors to machine-learning-driven virtual screening.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem;">
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value">🧪</div>
            <div class="metric-label">TASK 1</div>
            <div style="font-size:0.95rem; font-weight:600;">SMILES → Descriptors</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value">📊</div>
            <div class="metric-label">TASK 2</div>
            <div style="font-size:0.95rem; font-weight:600;">pIC50 → Activity</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value">🤖</div>
            <div class="metric-label">TASK 3</div>
            <div style="font-size:0.95rem; font-weight:600;">ML + Screening</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_uploaded_file(uploaded_file):

    if uploaded_file.name.lower().endswith(
        (".xlsx", ".xls")
    ):
        return pd.read_excel(uploaded_file)

    return pd.read_csv(uploaded_file)


# ------------------------------------------------------------
# Normalize column names
# ------------------------------------------------------------

def clean_column_name(col):

    return (
        str(col)
        .strip()
        .lower()
        .replace("\n", " ")
    )


# ------------------------------------------------------------
# Find column
# ------------------------------------------------------------

def find_column(df, keywords):

    columns = list(df.columns)

    # Exact / partial search
    for keyword in keywords:

        keyword = keyword.lower()

        for col in columns:

            c = clean_column_name(col)

            if keyword in c:

                return col

    return None


# ------------------------------------------------------------
# Normalize compound name
# ------------------------------------------------------------

def normalize_name(value):

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    # Remove spaces and special characters
    value = re.sub(
        r"[^a-z0-9]+",
        "",
        value
    )

    return value


# ------------------------------------------------------------
# Extract numeric value
# ------------------------------------------------------------

def extract_numeric(value):

    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    s = (
        s
        .replace("μ", "u")
        .replace("µ", "u")
        .replace(",", "")
    )

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        s
    )

    if match:

        try:
            return float(
                match.group()
            )

        except Exception:
            return np.nan

    return np.nan


# ------------------------------------------------------------
# Convert IC50 to µM
# ------------------------------------------------------------

def convert_to_uM(values, unit):

    unit = (
        str(unit)
        .strip()
        .lower()
    )

    if unit in ["µm", "um"]:

        factor = 1.0

    elif unit == "nm":

        factor = 1 / 1000

    elif unit == "pm":

        factor = 1 / 1_000_000

    elif unit == "mm":

        factor = 1000

    else:

        factor = 1.0

    return (
        pd.to_numeric(
            values,
            errors="coerce"
        )
        * factor
    )


# ------------------------------------------------------------
# IC50 µM -> pIC50
# ------------------------------------------------------------

def ic50_to_pic50(ic50_um):

    values = pd.to_numeric(
        ic50_um,
        errors="coerce"
    )

    return np.where(
        values > 0,
        -np.log10(
            values * 1e-6
        ),
        np.nan
    )


# ------------------------------------------------------------
# pIC50 -> activity
# ------------------------------------------------------------

def create_activity_from_pic50(pic50):
    """Classify compounds using pIC50 only.

    Active   : pIC50 >= 5.522879
    Inactive : pIC50 <  5.522879

    The threshold is equivalent to IC50 <= 3 µM, but the actual
    classification variable used by TASK 2 and TASK 3 is pIC50.
    """
    pic50 = pd.to_numeric(pic50, errors="coerce")

    label = np.where(
        pic50 >= PIC50_THRESHOLD,
        1,
        0
    )

    activity = np.where(
        label == 1,
        "Active",
        "Inactive"
    )

    return label, activity


# ============================================================
# SIMPLE MOLECULAR DESCRIPTORS
# ============================================================

def calculate_descriptors(smiles):

    if pd.isna(smiles):

        return None

    smiles = str(smiles).strip()

    if smiles == "":

        return None

    # -----------------------------------------
    # Basic molecular information from SMILES
    # -----------------------------------------

    descriptor = {

        "SMILES_Length":
            len(smiles),

        "Num_C":
            len(
                re.findall(
                    r"C(?!l)",
                    smiles
                )
            ),

        "Num_N":
            len(
                re.findall(
                    r"N",
                    smiles
                )
            ),

        "Num_O":
            len(
                re.findall(
                    r"O",
                    smiles
                )
            ),

        "Num_S":
            len(
                re.findall(
                    r"S",
                    smiles
                )
            ),

        "Num_P":
            len(
                re.findall(
                    r"P",
                    smiles
                )
            ),

        "Num_F":
            len(
                re.findall(
                    r"F",
                    smiles
                )
            ),

        "Num_Cl":
            len(
                re.findall(
                    r"Cl",
                    smiles
                )
            ),

        "Num_Br":
            len(
                re.findall(
                    r"Br",
                    smiles
                )
            ),

        "Num_I":
            len(
                re.findall(
                    r"I",
                    smiles
                )
            ),

        "Num_RingDigits":
            len(
                re.findall(
                    r"[0-9]",
                    smiles
                )
            ),

        "Num_Brackets":
            smiles.count("(")
            +
            smiles.count(")"),

        "Num_Double_Bonds":
            smiles.count("="),

        "Num_Triple_Bonds":
            smiles.count("#"),

        "Num_Aromatic":
            len(
                re.findall(
                    r"[cnops]",
                    smiles
                )
            ),

        "Num_Lowercase":
            len(
                re.findall(
                    r"[a-z]",
                    smiles
                )
            )
    }

    atom_tokens = re.findall(
        r"Cl|Br|[A-Z][a-z]?|[cnops]",
        smiles
    )

    descriptor[
        "Num_Atoms"
    ] = len(atom_tokens)

    return descriptor


# ============================================================
# DESCRIPTOR FEATURE SELECTION
# ============================================================

def get_descriptor_columns(df):
    """
    Return ONLY genuine numeric molecular descriptor columns.

    Identity/matching/helper columns are never ML features, even if they
    happen to be numeric (for example _cid_key or _descriptor_row).
    """
    excluded_exact = {
        "cid", "id", "compound_id",
        "ic50", "ic50_value", "ic50_um",
        "pic50", "label", "activity",
        "prediction", "predicted_activity",
        "active_probability",
        "_cid_key", "_row_key", "_name_key", "_smiles_key",
        "_descriptor_row", "_pubchem_cid", "_pubchem_smiles",
        "_matched", "_matched_pic50_row", "_matched_descriptor_index",
        "_match_score"
    }

    excluded_contains = (
        "prediction",
        "probability",
        "activity",
        "label",
        "ic50",
        "pic50"
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    features = []

    for col in numeric_columns:
        name = str(col).strip().lower()

        # Any internal helper column is forbidden.
        if name.startswith("_"):
            continue

        if name in excluded_exact:
            continue

        if any(token in name for token in excluded_contains):
            continue

        features.append(col)

    return features


# ============================================================
# CREATE ML MODELS
# ============================================================

def get_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ),

        "SVM":
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "Naive Bayes":
            GaussianNB(),

        "MLP":
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1500,
                random_state=42,
                early_stopping=False,
                validation_fraction=0.1
            )
    }

    if XGBOOST_OK:

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    if LIGHTGBM_OK:

        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    if CATBOOST_OK:

        models["CatBoost"] = CatBoostClassifier(
            iterations=200,
            random_seed=42,
            verbose=False
        )

    return models


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_pipeline(model, min_train_minority, use_smote=False):
    """
    Build a leakage-safe preprocessing/model pipeline.

    SMOTE is applied only when there are enough minority samples in EVERY
    CV training fold. This prevents the common error:
    "Expected n_neighbors <= n_samples".
    """
    steps = []

    if use_smote and IMBLEARN_OK and min_train_minority >= 3:
        # k_neighbors must be <= (minority samples in the smallest
        # CV training fold - 1).
        k_neighbors = min(5, min_train_minority - 1)
        k_neighbors = max(1, k_neighbors)

        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors
                )
            )
        )

    steps.extend(
        [
            ("scaler", StandardScaler()),
            ("model", model)
        ]
    )

    if IMBLEARN_OK:
        return ImbPipeline(steps)

    # If imbalanced-learn is unavailable, use a normal sklearn pipeline.
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)


# ============================================================
# PUBCHEM CID RESOLUTION
# ============================================================

@st.cache_data(show_spinner=False)
def pubchem_resolve_name(drug_name):
    """
    Resolve a common drug name to PubChem CID and canonical/isomeric SMILES.
    This is used only as an identity bridge in TASK 3; PubChem data are not
    used as ML features.
    """
    if pd.isna(drug_name):
        return None, None

    name = str(drug_name).strip()

    if not name:
        return None, None

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + quote(name, safe="")
        + "/property/Title,CanonicalSMILES,IsomericSMILES/JSON"
    )

    try:
        response = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "PAAD-QSAR-Streamlit/1.0"}
        )

        if response.status_code != 200:
            return None, None

        data = response.json()
        props = data["PropertyTable"]["Properties"][0]

        cid = props.get("CID")
        smiles = (
            props.get("IsomericSMILES")
            or props.get("CanonicalSMILES")
        )

        return cid, smiles

    except Exception:
        return None, None


def resolve_paad_names_to_cid(pic_df, drug_col):
    """
    Resolve all PAAD drug names to PubChem CID.
    Cached requests prevent repeated calls during Streamlit reruns.
    """
    out = pic_df.copy()

    cids = []
    smiles = []

    progress = st.progress(0)
    status = st.empty()

    values = (
        out[drug_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    total = len(values)

    for i, name in enumerate(values):
        cid, smi = pubchem_resolve_name(name)
        cids.append(cid)
        smiles.append(smi)

        if i % 5 == 0 or i == total - 1:
            progress.progress((i + 1) / max(1, total))
            status.text(
                f"Resolving PAAD drug identity: {i + 1:,}/{total:,}"
            )

        # Be polite to the public service.
        time.sleep(0.05)

    progress.empty()
    status.empty()

    out["_PubChem_CID"] = pd.to_numeric(
        pd.Series(cids, index=out.index),
        errors="coerce"
    )
    out["_PubChem_SMILES"] = smiles

    return out



# ============================================================
# MATCH DESCRIPTORS WITH PIC50
# ============================================================

def match_descriptor_and_pic50(
    descriptor_df,
    pic50_df,
    reference_df=None,
    use_pubchem=True
):
    """
    Match TASK 1 descriptors with TASK 2 pIC50 data.

    Priority:
      1. CID if TASK 2/reference data contain CID
      2. PubChem drug-name -> CID -> descriptor CID
      3. Exact normalized drug-name
      4. Fuzzy drug-name
      5. SMILES matching when available

    This is an identity-resolution step only. pIC50/Activity are never
    included in the descriptor feature matrix.
    """

    desc = descriptor_df.copy()
    pic = pic50_df.copy()
    ref = reference_df.copy() if reference_df is not None else None

    desc_name = find_column(
        desc,
        [
            "drug_name", "drug name", "drugname",
            "compound_name", "compound name",
            "compound", "iupacname", "iupac name", "name"
        ]
    )
    if desc_name is None:
        desc_name = desc.columns[0]

    pic_name = find_column(
        pic,
        [
            "drug_name", "drug name", "drugname",
            "compound_name", "compound name",
            "compound", "name"
        ]
    )
    if pic_name is None:
        pic_name = pic.columns[0]

    pic50_col = find_column(pic, ["pic50"])
    if pic50_col is None:
        return None, desc_name, pic_name

    desc_cid = find_column(desc, ["cid", "pubchem cid", "pubchem_cid"])
    pic_cid = find_column(pic, ["cid", "pubchem cid", "pubchem_cid"])

    desc_smiles = find_column(
        desc, ["smiles", "smile", "canonical_smiles", "isomeric_smiles"]
    )
    pic_smiles = find_column(
        pic, ["smiles", "smile", "canonical_smiles", "isomeric_smiles"]
    )

    # Prepare normalized names.
    desc["_name_key"] = desc[desc_name].map(normalize_name)
    pic["_name_key"] = pic[pic_name].map(normalize_name)

    # Descriptor CID index.
    if desc_cid:
        desc["_cid_key"] = pd.to_numeric(
            desc[desc_cid], errors="coerce"
        )
    else:
        desc["_cid_key"] = np.nan

    # Direct CID in TASK 2, if it exists.
    if pic_cid:
        pic["_cid_key"] = pd.to_numeric(
            pic[pic_cid], errors="coerce"
        )
    else:
        pic["_cid_key"] = np.nan

    # Prepare SMILES keys.
    if desc_smiles:
        desc["_smiles_key"] = (
            desc[desc_smiles].fillna("").astype(str).str.strip().str.lower()
        )
    else:
        desc["_smiles_key"] = ""

    if pic_smiles:
        pic["_smiles_key"] = (
            pic[pic_smiles].fillna("").astype(str).str.strip().str.lower()
        )
    else:
        pic["_smiles_key"] = ""

    # --------------------------------------------------------
    # Optional PubChem identity bridge.
    # --------------------------------------------------------
    if use_pubchem and not pic_cid:
        unresolved = pic["_cid_key"].isna().sum()
        if unresolved > 0:
            st.info(
                "🔎 TASK 3 is resolving PAAD drug names to PubChem CIDs "
                "so they can be matched against the CID column in the "
                "65,482-compound descriptor library."
            )
            pic = resolve_paad_names_to_cid(pic, pic_name)

            pic["_cid_key"] = pd.to_numeric(
                pic["_PubChem_CID"], errors="coerce"
            )

            if pic_smiles is None:
                pic["_smiles_key"] = (
                    pic["_PubChem_SMILES"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

    # --------------------------------------------------------
    # Optional reference file bridge.
    # --------------------------------------------------------
    if ref is not None:
        ref_name = find_column(
            ref,
            [
                "drug_name", "drug name", "drugname",
                "compound_name", "compound name",
                "compound", "name"
            ]
        )
        ref_cid = find_column(ref, ["cid", "pubchem cid", "pubchem_cid"])
        ref_smiles = find_column(
            ref, ["smiles", "smile", "canonical_smiles", "isomeric_smiles"]
        )

        if ref_name is None:
            ref_name = ref.columns[0]

        ref["_ref_name_key"] = ref[ref_name].map(normalize_name)

        if ref_cid:
            ref["_ref_cid"] = pd.to_numeric(
                ref[ref_cid], errors="coerce"
            )
        else:
            ref["_ref_cid"] = np.nan

        if ref_smiles:
            ref["_ref_smiles"] = (
                ref[ref_smiles].fillna("").astype(str).str.strip().str.lower()
            )
        else:
            ref["_ref_smiles"] = ""

        # Fill missing CID/SMILES in the PAAD table from the reference.
        ref_map = (
            ref.drop_duplicates("_ref_name_key")
            .set_index("_ref_name_key")
        )

        if "_ref_cid" in ref.columns:
            missing_cid = pic["_cid_key"].isna()
            pic.loc[missing_cid, "_cid_key"] = (
                pic.loc[missing_cid, "_name_key"]
                .map(ref_map["_ref_cid"])
            )

        if "_ref_smiles" in ref.columns:
            missing_smiles = pic["_smiles_key"].eq("")
            pic.loc[missing_smiles, "_smiles_key"] = (
                pic.loc[missing_smiles, "_name_key"]
                .map(ref_map["_ref_smiles"])
                .fillna("")
            )

    # --------------------------------------------------------
    # Build a single matched training table, CID first.
    # --------------------------------------------------------
    desc["_descriptor_row"] = np.arange(len(desc))
    pic["_pic_row"] = np.arange(len(pic))

    # Keep one PAAD record per CID where possible.
    pic_valid_cid = pic[pic["_cid_key"].notna()].copy()
    pic_valid_cid = pic_valid_cid.drop_duplicates(
        "_cid_key", keep="first"
    )

    matched = pd.DataFrame()
    cid_count = 0
    name_count = 0
    smiles_count = 0
    fuzzy_count = 0

    if not pic_valid_cid.empty and desc["_cid_key"].notna().any():
        cid_merge = desc.merge(
            pic_valid_cid[
                [
                    "_cid_key", pic_name, pic50_col,
                    "_name_key", "_smiles_key"
                ]
            ],
            on="_cid_key",
            how="inner",
            suffixes=("", "_pIC50")
        )

        if not cid_merge.empty:
            cid_count = len(cid_merge)
            matched = cid_merge.copy()

    # --------------------------------------------------------
    # Match remaining PAAD compounds by exact normalized name.
    # --------------------------------------------------------
    matched_desc_rows = set()
    if not matched.empty and "_descriptor_row" in matched.columns:
        matched_desc_rows = set(
            matched["_descriptor_row"].tolist()
        )

    remaining_desc = desc[
        ~desc["_descriptor_row"].isin(matched_desc_rows)
    ].copy()

    remaining_pic = pic.copy()

    # Do not duplicate PAAD records already used by CID matching.
    if not matched.empty and "_pic_row" in matched.columns:
        used_pic_rows = set(matched["_pic_row"].tolist())
        remaining_pic = remaining_pic[
            ~remaining_pic["_pic_row"].isin(used_pic_rows)
        ]

    exact_desc_keys = set(
        remaining_desc["_name_key"]
    )
    exact_pic_keys = set(
        remaining_pic["_name_key"]
    )
    exact_keys = exact_desc_keys.intersection(exact_pic_keys)

    if exact_keys:
        exact_merge = remaining_desc[
            remaining_desc["_name_key"].isin(exact_keys)
        ].merge(
            remaining_pic[
                [
                    "_name_key", pic_name, pic50_col,
                    "_pic_row", "_smiles_key"
                ]
            ].drop_duplicates("_name_key"),
            on="_name_key",
            how="inner",
            suffixes=("", "_pIC50")
        )
        if not exact_merge.empty:
            name_count = len(exact_merge)
            matched = pd.concat(
                [matched, exact_merge],
                ignore_index=True
            )

    # --------------------------------------------------------
    # Match remaining by SMILES.
    # --------------------------------------------------------
    if desc_smiles or "_smiles_key" in remaining_desc.columns:
        used_desc = set(
            matched["_descriptor_row"].tolist()
        ) if "_descriptor_row" in matched.columns else set()

        used_pic = set(
            matched["_pic_row"].tolist()
        ) if "_pic_row" in matched.columns else set()

        rd = desc[
            ~desc["_descriptor_row"].isin(used_desc)
        ].copy()
        rp = pic[
            ~pic["_pic_row"].isin(used_pic)
        ].copy()

        rd = rd[rd["_smiles_key"] != ""]
        rp = rp[rp["_smiles_key"] != ""]

        if not rd.empty and not rp.empty:
            smi_merge = rd.merge(
                rp[
                    [
                        "_smiles_key", pic_name, pic50_col,
                        "_pic_row", "_name_key"
                    ]
                ].drop_duplicates("_smiles_key"),
                on="_smiles_key",
                how="inner",
                suffixes=("", "_pIC50")
            )
            if not smi_merge.empty:
                smiles_count = len(smi_merge)
                matched = pd.concat(
                    [matched, smi_merge],
                    ignore_index=True
                )

    # --------------------------------------------------------
    # Fuzzy name matching only for still-unmatched PAAD names.
    # --------------------------------------------------------
    used_pic = set(
        matched["_pic_row"].tolist()
    ) if not matched.empty and "_pic_row" in matched.columns else set()

    used_desc = set(
        matched["_descriptor_row"].tolist()
    ) if not matched.empty and "_descriptor_row" in matched.columns else set()

    rd = desc[
        ~desc["_descriptor_row"].isin(used_desc)
    ].copy()

    rp = pic[
        ~pic["_pic_row"].isin(used_pic)
    ].copy()

    if not rd.empty and not rp.empty:
        desc_keys = rd["_name_key"].dropna().unique().tolist()

        fuzzy_rows = []

        for _, prow in rp.iterrows():
            pk = prow["_name_key"]
            if not pk:
                continue

            best = difflib.get_close_matches(
                pk,
                desc_keys,
                n=1,
                cutoff=0.92
            )

            if best:
                candidate_key = best[0]
                drows = rd[
                    rd["_name_key"] == candidate_key
                ]
                if not drows.empty:
                    drow = drows.iloc[0].copy()
                    drow[pic_name] = prow[pic_name]
                    drow[pic50_col] = prow[pic50_col]
                    drow["_pic_row"] = prow["_pic_row"]
                    drow["_fuzzy_match"] = True
                    fuzzy_rows.append(drow)

        if fuzzy_rows:
            fuzzy_merge = pd.DataFrame(fuzzy_rows)
            fuzzy_count = len(fuzzy_merge)
            matched = pd.concat(
                [matched, fuzzy_merge],
                ignore_index=True
            )

    if matched.empty:
        return (
            pd.DataFrame(),
            desc_name,
            pic_name
        )

    # --------------------------------------------------------
    # Remove duplicate descriptor rows.
    # --------------------------------------------------------
    matched = matched.drop_duplicates(
        "_descriptor_row",
        keep="first"
    )

    # --------------------------------------------------------
    # pIC50 numeric and activity.
    # --------------------------------------------------------
    matched["pIC50"] = pd.to_numeric(
        matched[pic50_col],
        errors="coerce"
    )

    matched = matched.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=["pIC50"]
    )

    matched["Label"], matched["Activity"] = (
        create_activity_from_pic50(
            matched["pIC50"]
        )
    )

    # --------------------------------------------------------
    # Matching method.
    # --------------------------------------------------------
    if "_fuzzy_match" in matched.columns:
        matched["Match_Method"] = np.where(
            matched["_fuzzy_match"].fillna(False),
            "Fuzzy name",
            "CID/Name/SMILES"
        )
    else:
        matched["Match_Method"] = "CID/Name/SMILES"

    # Diagnostic summary.
    st.markdown("### 🔗 Identity Matching Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CID matches", f"{cid_count:,}")
    m2.metric("Exact-name matches", f"{name_count:,}")
    m3.metric("SMILES matches", f"{smiles_count:,}")
    m4.metric("Fuzzy-name matches", f"{fuzzy_count:,}")

    return (
        matched,
        desc_name,
        pic_name
    )


# ============================================================
# CREATE EXCEL
# ============================================================

def dataframe_to_excel(
    sheets
):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for sheet_name, dataframe in sheets.items():

            dataframe.to_excel(
                writer,
                index=False,
                sheet_name=sheet_name[:31]
            )

    return output.getvalue()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ SMILES → Descriptors",
        "2️⃣ IC50 → pIC50",
        "3️⃣ ML + CV-MCC + Screening"
    ]
)


# ============================================================
# TASK 1
# ============================================================

with tab1:

    with st.container():
        st.markdown(
            '<div class="section-header">'
            '🧪 TASK 1: SMILES → Molecular Descriptors'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card" style="border-left-color:#10b981;">
                <strong>Upload your drug library</strong>
                (e.g. <em>New drug library</em> or <em>PDAC Ligand Library</em>)
                containing a <strong>SMILES</strong> column.
                The application will generate simple molecular descriptors for
                each compound.
            </div>
            """,
            unsafe_allow_html=True
        )

        smiles_file = st.file_uploader(
            "📁 Upload Drug Library",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
            key="task1_upload",
            help="CSV or Excel file with a SMILES column"
        )

    if smiles_file:

        try:

            df = read_uploaded_file(
                smiles_file
            )

            st.subheader(
                "📄 Input Data Preview"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Detect SMILES
            # ------------------------------------------------

            smiles_col = find_column(
                df,
                [
                    "smiles"
                ]
            )

            if smiles_col is None:

                st.error(
                    "⚠️ SMILES column was not found."
                )

                st.write(
                    "Available columns:",
                    df.columns.tolist()
                )

                st.stop()

            # ------------------------------------------------
            # Detect name
            # ------------------------------------------------

            name_col = find_column(
                df,
                [
                    "drug_name",
                    "drug name",
                    "drugname",
                    "compound_name",
                    "compound name",
                    "compound",
                    "iupacname",
                    "iupac name",
                    "name"
                ]
            )

            if name_col is None:

                name_col = df.columns[0]

            # ------------------------------------------------
            # Detect CID
            # ------------------------------------------------

            cid_col = find_column(
                df,
                [
                    "cid"
                ]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.success(
                    f"✅ SMILES column: **{smiles_col}**"
                )
            with col2:
                st.info(
                    f"🔤 Drug/Compound column: **{name_col}**"
                )

            # ------------------------------------------------
            # Generate descriptors
            # ------------------------------------------------

            records = []

            invalid = 0

            progress = st.progress(
                0
            )

            status_text = st.empty()

            total = len(df)

            for i, (_, row) in enumerate(
                df.iterrows()
            ):

                try:

                    smi = row[
                        smiles_col
                    ]

                    desc = calculate_descriptors(
                        smi
                    )

                    if desc is None:

                        invalid += 1

                        continue

                    record = {}

                    # Keep original identifiers

                    record[
                        "Drug_Name"
                    ] = row[
                        name_col
                    ]

                    if cid_col:

                        record[
                            "CID"
                        ] = row[
                            cid_col
                        ]

                    record[
                        "SMILES"
                    ] = smi

                    # Add descriptors

                    record.update(
                        desc
                    )

                    records.append(
                        record
                    )

                except Exception:

                    invalid += 1

                progress.progress(
                    (i + 1) / max(
                        1,
                        total
                    )
                )

                if i % max(1, total // 20) == 0:
                    status_text.text(
                        f"Processing compound {i + 1:,} of {total:,}..."
                    )

            progress.empty()
            status_text.empty()

            df_desc = pd.DataFrame(
                records
            )

            if df_desc.empty:

                st.error(
                    "❌ No descriptors were generated."
                )

                st.stop()

            # ------------------------------------------------
            # Result metrics
            # ------------------------------------------------

            st.subheader(
                "✨ Generation Summary"
            )

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Total Input Compounds",
                f"{len(df):,}"
            )
            m2.metric(
                "Descriptors Generated",
                f"{len(df_desc):,}"
            )
            m3.metric(
                "Invalid SMILES",
                f"{invalid:,}"
            )

            if invalid > 0:

                st.warning(
                    f"⚠️ {invalid:,} compounds could not "
                    f"be processed."
                )

            with st.expander(
                "🔍 View Generated Descriptors",
                expanded=True
            ):
                st.dataframe(
                    df_desc.head(20),
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            excel_data = dataframe_to_excel(
                {
                    "Descriptors":
                        df_desc
                }
            )

            st.subheader(
                "⬇️ Download Results"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "📥 Download Descriptor Excel",

                    data=excel_data,

                    file_name=
                    "PAAD_Descriptors.xlsx",

                    mime=
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                    use_container_width=True
                )

            with col2:

                st.download_button(
                    "📥 Download Descriptor CSV",

                    data=
                    df_desc
                    .to_csv(
                        index=False
                    )
                    .encode(),

                    file_name=
                    "PAAD_Descriptors.csv",

                    mime="text/csv",

                    use_container_width=True
                )

            st.success(
                "✅ TASK 1 complete. Use the downloaded descriptor file in TASK 3."
            )

        except Exception as e:

            st.error(
                f"❌ Task 1 error: {e}"
            )

            st.exception(e)


# ============================================================
# TASK 2
# ============================================================

with tab2:

    with st.container():
        st.markdown(
            '<div class="section-header">'
            '📊 TASK 2: IC50 → pIC50 → Activity (pIC50-based)'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card" style="border-left-color:#f59e0b;">
                <strong>Upload your PAAD IC50 file</strong><br>
                Example: <code>IC50 value PAAD celline specific.xlsx</code><br>
                Expected columns: <strong>DRUG NAME</strong> and <strong>IC50 VALUE</strong>.<br>
                IC50 is used only to calculate pIC50. Activity and Label are assigned from the pIC50 threshold.
            </div>
            """,
            unsafe_allow_html=True
        )

        ic50_file = st.file_uploader(
            "📁 Upload PAAD IC50 File",
            type=[
                "xlsx",
                "xls",
                "csv"
            ],
            key="task2_upload",
            help="Excel or CSV file with drug names and IC50 values"
        )

    if ic50_file:

        try:

            df_ic = read_uploaded_file(
                ic50_file
            )

            if df_ic.empty:

                st.error(
                    "❌ The uploaded file is empty."
                )

                st.stop()

            st.subheader(
                "📄 Original PAAD IC50 Data"
            )

            st.dataframe(
                df_ic.head(20),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Find drug name
            # ------------------------------------------------

            drug_col = find_column(
                df_ic,
                [
                    "drug_name",
                    "drug name",
                    "drugname",
                    "compound_name",
                    "compound name",
                    "compound",
                    "name"
                ]
            )

            if drug_col is None:

                drug_col = df_ic.columns[0]

            # ------------------------------------------------
            # Find optional SMILES column
            # ------------------------------------------------

            smiles_col = find_column(
                df_ic,
                [
                    "smiles",
                    "smile",
                    "canonical_smiles",
                    "isosmiles",
                    "structure"
                ]
            )

            # ------------------------------------------------
            # Find IC50
            # ------------------------------------------------

            ic50_col = find_column(
                df_ic,
                [
                    "ic50"
                ]
            )

            if ic50_col is None:

                st.error(
                    "⚠️ Could not find IC50 column."
                )

                st.write(
                    "Available columns:",
                    df_ic.columns.tolist()
                )

                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                st.success(
                    f"✅ Drug name column: **{drug_col}**"
                )
            with col2:
                st.success(
                    f"✅ IC50 column: **{ic50_col}**"
                )

            # ------------------------------------------------
            # Unit
            # ------------------------------------------------

            unit = st.selectbox(
                "⚗️ Select IC50 Unit",
                [
                    "µM",
                    "nM",
                    "pM",
                    "mM"
                ],
                index=0,
                key="task2_ic50_unit",
                help="Choose the concentration unit used in your input file"
            )

            # ------------------------------------------------
            # Convert
            # ------------------------------------------------

            df_ic[
                "IC50_original"
            ] = df_ic[
                ic50_col
            ]

            df_ic[
                "IC50_numeric"
            ] = (
                df_ic[
                    ic50_col
                ]
                .apply(
                    extract_numeric
                )
            )

            df_ic[
                "IC50_uM"
            ] = convert_to_uM(
                df_ic[
                    "IC50_numeric"
                ],
                unit
            )

            # ------------------------------------------------
            # pIC50
            # ------------------------------------------------

            df_ic[
                "pIC50"
            ] = ic50_to_pic50(
                df_ic[
                    "IC50_uM"
                ]
            )

            # ------------------------------------------------
            # Activity — pIC50 ONLY
            # ------------------------------------------------

            (
                df_ic["Label"],
                df_ic["Activity"]
            ) = create_activity_from_pic50(
                df_ic["pIC50"]
            )

            st.info(
                f"""
**Activity is classified from pIC50, not directly from IC50.**

- **pIC50 ≥ {PIC50_THRESHOLD:.4f} → Active**
- **pIC50 < {PIC50_THRESHOLD:.4f} → Inactive**

The threshold **5.522879** is the pIC50 equivalent of **IC50 = 3 µM**.
"""
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            st.subheader(
                "📈 Converted PAAD Dataset"
            )

            display_cols = [
                drug_col
            ]

            if smiles_col:

                display_cols.append(
                    smiles_col
                )

            display_cols.extend(
                [
                    "IC50_uM",
                    "pIC50",
                    "Activity",
                    "Label"
                ]
            )

            st.dataframe(
                df_ic[
                    display_cols
                ].head(30),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            active_count = int(
                (
                    df_ic["Label"] == 1
                ).sum()
            )

            inactive_count = int(
                (
                    df_ic["Label"] == 0
                ).sum()
            )

            st.subheader(
                "📊 Activity Summary (pIC50-based)"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total compounds",
                len(df_ic)
            )

            c2.metric(
                "🟢 Active",
                active_count
            )

            c3.metric(
                "🔴 Inactive",
                inactive_count
            )

            st.info(
                f"""
**Activity rule (pIC50 only):**

- **pIC50 ≥ {PIC50_THRESHOLD:.4f} → Active**
- **pIC50 < {PIC50_THRESHOLD:.4f} → Inactive**

IC50 is used only for the mathematical conversion:
**pIC50 = -log10(IC50 in M)**.
"""
            )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            excel_data = dataframe_to_excel(
                {
                    "IC50_pIC50":
                        df_ic
                }
            )

            st.subheader(
                "⬇️ Download Results"
            )

            st.download_button(
                "📥 Download PAAD IC50 + pIC50 File",

                data=excel_data,

                file_name=
                "PAAD_IC50_pIC50.xlsx",

                mime=
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                use_container_width=True
            )

            st.success(
                """
✅ TASK 2 complete. Download this file and use it in TASK 3.
Do NOT upload the original IC50 file in TASK 3.
"""
            )

        except Exception as e:

            st.error(
                f"❌ Task 2 error: {e}"
            )

            st.exception(e)


# ============================================================
# TASK 3
# ============================================================

with tab3:

    with st.container():
        st.markdown(
            '<div class="section-header">'
            '🤖 TASK 3: ML + Cross-Validation MCC + Virtual Screening'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card" style="border-left-color:#8b5cf6;">
                <strong>Upload TWO files</strong><br>
                <strong>File 1:</strong> Descriptor file downloaded from TASK 1<br>
                <strong>File 2:</strong> <code>PAAD_IC50_pIC50.xlsx</code> downloaded from TASK 2<br>
                The pIC50 file creates the true Active/Inactive labels,
                and the descriptors are the ML input features.<br><br>
                <em>Matching order:</em> (1) normalized drug names, (2) fuzzy name matching,
                (3) SMILES strings if available in both files.
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            descriptor_file = st.file_uploader(
                "📁 Upload Descriptor File from TASK 1",
                type=[
                    "csv",
                    "xlsx",
                    "xls"
                ],
                key="task3_descriptor",
                help="PAAD_Descriptors.xlsx from TASK 1"
            )

        with col2:
            pic50_file = st.file_uploader(
                "📁 Upload PAAD_IC50_pIC50.xlsx from TASK 2",
                type=[
                    "xlsx",
                    "xls",
                    "csv"
                ],
                key="task3_pic50",
                help="PAAD_IC50_pIC50.xlsx from TASK 2"
            )

        st.markdown(
            """
            <div class="info-card" style="border-left-color:#06b6d4; margin-top:1rem;">
                <strong>Optional:</strong> If drug names do not match between the
                descriptor and pIC50 files, upload a <strong>SMILES reference file</strong>
                that contains the same drug names used in the pIC50 file plus their
                SMILES strings. The app will use it to bridge the two datasets.
            </div>
            """,
            unsafe_allow_html=True
        )

        reference_file = st.file_uploader(
            "📁 Upload SMILES Reference File (optional)",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
            key="task3_reference",
            help="Optional: columns 'Drug_Name' and 'SMILES' to bridge pIC50 and descriptors"
        )

        use_pubchem = st.checkbox(
            "🔎 Use PubChem CID resolution for PAAD drug names (recommended)",
            value=True,
            key="task3_pubchem",
            help=(
                "Resolves common PAAD drug names such as Romidepsin/Gemcitabine "
                "to PubChem CIDs so they can be matched to the CID in TASK 1. "
                "Requires internet access while running TASK 3."
            )
        )

    if descriptor_file and pic50_file:

        try:

            # =================================================
            # READ FILES
            # =================================================

            with st.spinner(
                "Reading and matching datasets..."
            ):

                df_desc = read_uploaded_file(
                    descriptor_file
                )

                df_pic = read_uploaded_file(
                    pic50_file
                )

            st.subheader(
                "📄 Uploaded Dataset Information"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Descriptor rows",
                f"{len(df_desc):,}"
            )

            c2.metric(
                "pIC50 rows",
                f"{len(df_pic):,}"
            )

            # =================================================
            # FIND pIC50
            # =================================================

            pic50_col = find_column(
                df_pic,
                [
                    "pic50"
                ]
            )

            if pic50_col is None:

                st.error(
                    """
The uploaded Task 2 file does not contain
a pIC50 column.

Please upload the file downloaded from TASK 2:
PAAD_IC50_pIC50.xlsx
"""
                )

                st.write(
                    "Available columns:",
                    df_pic.columns.tolist()
                )

                st.stop()

            # =================================================
            # MATCH DATA
            # =================================================

            df_ref = None

            if reference_file is not None:

                df_ref = read_uploaded_file(
                    reference_file
                )

            result = match_descriptor_and_pic50(
                df_desc,
                df_pic,
                df_ref,
                use_pubchem=use_pubchem
            )

            if result[0] is None:

                st.error(
                    "Could not find pIC50 column."
                )

                st.stop()

            (
                training_df,
                descriptor_name_col,
                pic50_name_col
            ) = result

            if training_df.empty:

                st.error(
                    """
❌ No compounds matched between the descriptor
file and the pIC50 file.

The app tried CID, PubChem CID resolution,
normalized names, SMILES, and fuzzy names.
Check the identity columns or provide an
optional SMILES reference file.
"""
                )

                st.write(
                    "Descriptor columns:",
                    df_desc.columns.tolist()
                )

                st.write(
                    "pIC50 columns:",
                    df_pic.columns.tolist()
                )

                with st.expander(
                    "🔍 Sample names for diagnosis",
                    expanded=True
                ):

                    st.write(
                        "Sample descriptor names:",
                        df_desc[descriptor_name_col]
                        .dropna()
                        .astype(str)
                        .head(20)
                        .tolist()
                    )

                    st.write(
                        "Sample pIC50 names:",
                        df_pic[pic50_name_col]
                        .dropna()
                        .astype(str)
                        .head(20)
                        .tolist()
                    )

                st.stop()

            st.success(
                f"""
✅ Successfully matched **{len(training_df):,} compounds**
between descriptor and pIC50 datasets.
"""
            )

            # =================================================
            # CHECK CLASSES
            # =================================================

            active_count = int(
                (
                    training_df["Label"] == 1
                ).sum()
            )

            inactive_count = int(
                (
                    training_df["Label"] == 0
                ).sum()
            )

            st.subheader(
                "📊 Class Distribution"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Matched compounds",
                f"{len(training_df):,}"
            )

            c2.metric(
                "🟢 Active",
                active_count
            )

            c3.metric(
                "🔴 Inactive",
                inactive_count
            )

            # =================================================
            # SHOW MATCHED DATA
            # =================================================

            with st.expander(
                "🔍 View Matched Training Dataset",
                expanded=True
            ):

                show_cols = [
                    descriptor_name_col,
                    "pIC50",
                    "Activity",
                    "Label"
                ]

                if "IC50_uM" in training_df.columns:

                    show_cols.insert(
                        1,
                        "IC50_uM"
                    )

                st.dataframe(
                    training_df[
                        show_cols
                    ].head(30),
                    use_container_width=True,
                    hide_index=True
                )

            if active_count < 2:

                st.error(
                    "At least two Active compounds are required."
                )

                st.stop()

            if inactive_count < 2:

                st.error(
                    "At least two Inactive compounds are required."
                )

                st.stop()

            # =================================================
            # SELECT DESCRIPTORS
            # =================================================

            feature_cols = get_descriptor_columns(
                training_df
            )

            if not feature_cols:

                st.error(
                    """
No numeric molecular descriptor columns
were found.
"""
                )

                st.stop()

            X = (
                training_df[
                    feature_cols
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                .replace(
                    [
                        np.inf,
                        -np.inf
                    ],
                    np.nan
                )
                .fillna(0)
            )

            y = (
                training_df[
                    "Label"
                ]
                .astype(int)
                .values
            )

            # =================================================
            # REMOVE CONSTANT FEATURES
            # =================================================

            constant_features = [
                col
                for col in X.columns
                if X[col].nunique() <= 1
            ]

            if constant_features:

                X = X.drop(
                    columns=constant_features
                )

                feature_cols = (
                    list(
                        X.columns
                    )
                )

                st.warning(
                    f"""
Removed {len(constant_features)}
constant descriptor columns.
"""
                )

            st.info(
                f"""
ML features used:
**{len(feature_cols):,}**
"""
            )

            # =================================================
            # CROSS VALIDATION
            # =================================================

            smallest_class = min(
                active_count,
                inactive_count
            )

            n_splits = min(
                5,
                smallest_class
            )

            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42
            )

            # Determine the smallest minority-class size that will actually
            # be present in ANY CV training fold. SMOTE must be compatible
            # with this value, not just with the full dataset.
            min_train_minority = min(
                min(
                    np.bincount(y[train_idx], minlength=2)
                )
                for train_idx, _ in cv.split(X, y)
            )

            # For very small matched datasets (such as 4 Active + 5 Inactive),
            # SMOTE is deliberately disabled. Synthetic sampling with only a
            # few compounds can be unstable and was the direct cause of the
            # previous "n_neighbors <= n_samples" failures.
            use_smote = (
                IMBLEARN_OK
                and min_train_minority >= 5
            )

            st.subheader(
                f"📊 {n_splits}-Fold Cross-Validation"
            )

            if not IMBLEARN_OK:
                st.warning(
                    """
⚠️ imbalanced-learn is not installed. The application will continue without SMOTE.

If you want SMOTE for larger datasets, install: `pip install imbalanced-learn`
"""
                )
            elif use_smote:
                smote_k = min(5, min_train_minority - 1)
                st.info(
                    f"✅ SMOTE enabled safely. Minimum minority samples in a CV "
                    f"training fold: {min_train_minority}; k_neighbors={smote_k}."
                )
            else:
                st.warning(
                    f"""
⚠️ SMOTE disabled for this dataset because the smallest class contains only
{smallest_class} matched compounds and the smallest CV training fold contains
{min_train_minority} minority samples.

The models will still run with class weighting where supported. This avoids
the previous SMOTE error and is more appropriate for a very small training set.
"""
                )

            if len(training_df) < 30:
                st.warning(
                    f"""
⚠️ Only {len(training_df)} matched compounds are available for ML.
Cross-validation results are exploratory because the training set is very small.
"""
                )

            missing_libs = []

            if not XGBOOST_OK:
                missing_libs.append("xgboost")

            if not LIGHTGBM_OK:
                missing_libs.append("lightgbm")

            if not CATBOOST_OK:
                missing_libs.append("catboost")

            if missing_libs:
                st.warning(
                    f"""
⚠️ Some requested models are unavailable because their libraries are not installed:
{', '.join(missing_libs)}

Install with: `pip install {' '.join(missing_libs)}`

The available models will still run.
"""
                )
            else:
                st.info("✅ All requested gradient boosting libraries are available.")

            models = get_models()

            performance = []

            oof_results = {}

            # =================================================
            # MODEL LOOP
            # =================================================

            for model_name, model in models.items():

                try:

                    pipeline = create_pipeline(
                        model,
                        min_train_minority,
                        use_smote=use_smote
                    )

                    # -----------------------------------------
                    # Cross validation metrics
                    # -----------------------------------------

                    scores = cross_validate(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring={
                            "accuracy":
                                "accuracy",

                            "precision":
                                "precision",

                            "recall":
                                "recall",

                            "f1":
                                "f1",

                            "roc_auc":
                                "roc_auc"
                        },
                        n_jobs=-1,
                        error_score="raise"
                    )

                    # -----------------------------------------
                    # Out-of-fold prediction
                    # -----------------------------------------

                    oof_prediction = (
                        cross_val_predict(
                            pipeline,
                            X,
                            y,
                            cv=cv,
                            method="predict",
                            n_jobs=-1
                        )
                    )

                    # -----------------------------------------
                    # Out-of-fold probability
                    # -----------------------------------------

                    oof_probability = (
                        cross_val_predict(
                            pipeline,
                            X,
                            y,
                            cv=cv,
                            method="predict_proba",
                            n_jobs=-1
                        )[:, 1]
                    )

                    # -----------------------------------------
                    # OOF metrics (same fold scheme)
                    # -----------------------------------------

                    oof_accuracy = accuracy_score(
                        y,
                        oof_prediction
                    )

                    oof_precision = precision_score(
                        y,
                        oof_prediction,
                        zero_division=0
                    )

                    oof_recall = recall_score(
                        y,
                        oof_prediction,
                        zero_division=0
                    )

                    oof_f1 = f1_score(
                        y,
                        oof_prediction,
                        zero_division=0
                    )

                    mcc = (
                        matthews_corrcoef(
                            y,
                            oof_prediction
                        )
                    )

                    performance.append(
                        {
                            "Model":
                                model_name,

                            "Accuracy":
                                oof_accuracy,

                            "Precision":
                                oof_precision,

                            "Recall":
                                oof_recall,

                            "F1":
                                oof_f1,

                            "ROC_AUC":
                                roc_auc_score(
                                    y,
                                    oof_probability
                                ),

                            "MCC":
                                mcc
                        }
                    )

                    cm = confusion_matrix(
                        y,
                        oof_prediction
                    )

                    oof_results[
                        model_name
                    ] = {

                        "prediction":
                            oof_prediction,

                        "probability":
                            oof_probability,

                        "confusion_matrix":
                            cm
                    }

                except Exception as e:

                    st.warning(
                        f"⚠️ {model_name} failed: {e}"
                    )

            # =================================================
            # PERFORMANCE TABLE
            # =================================================

            if not performance:

                st.error(
                    "All ML models failed."
                )

                st.stop()

            performance_df = (
                pd.DataFrame(
                    performance
                )
                .sort_values(
                    [
                        "MCC",
                        "F1",
                        "ROC_AUC"
                    ],
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            st.subheader(
                "🏆 Model Performance"
            )

            def highlight_best(row):
                if row["Model"] == performance_df.iloc[0]["Model"]:
                    return [
                        "background-color: #dbeafe; font-weight: 600"
                    ] * len(row)
                return [""] * len(row)

            styled_performance = (
                performance_df
                .style
                .format(
                    {
                        "Accuracy":
                            "{:.3f}",

                        "Precision":
                            "{:.3f}",

                        "Recall":
                            "{:.3f}",

                        "F1":
                            "{:.3f}",

                        "ROC_AUC":
                            "{:.3f}",

                        "MCC":
                            "{:.3f}"
                    }
                )
                .apply(highlight_best, axis=1)
            )

            st.dataframe(
                styled_performance,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # FOLD-WISE MCC
            # =================================================
            st.subheader("📌 Fold-wise MCC for Each Model")

            fold_mcc_rows = []

            for model_name, model in models.items():
                try:
                    pipeline = create_pipeline(
                        model,
                        min_train_minority,
                        use_smote=use_smote
                    )

                    fold_number = 0

                    for train_idx, test_idx in cv.split(X, y):
                        fold_number += 1

                        pipeline.fit(
                            X.iloc[train_idx],
                            y[train_idx]
                        )

                        fold_pred = pipeline.predict(
                            X.iloc[test_idx]
                        )

                        fold_mcc_rows.append(
                            {
                                "Model": model_name,
                                "Fold": f"Fold {fold_number}",
                                "MCC": matthews_corrcoef(
                                    y[test_idx],
                                    fold_pred
                                )
                            }
                        )

                except Exception:
                    continue

            if fold_mcc_rows:
                fold_mcc_df = pd.DataFrame(
                    fold_mcc_rows
                )

                fold_mcc_pivot = (
                    fold_mcc_df
                    .pivot(
                        index="Model",
                        columns="Fold",
                        values="MCC"
                    )
                    .reset_index()
                )

                fold_cols = [
                    c for c in fold_mcc_pivot.columns
                    if str(c).startswith("Fold")
                ]

                fold_mcc_pivot["Mean_MCC"] = (
                    fold_mcc_pivot[fold_cols]
                    .mean(axis=1)
                )

                fold_mcc_pivot["SD_MCC"] = (
                    fold_mcc_pivot[fold_cols]
                    .std(axis=1)
                    .fillna(0)
                )

                fold_mcc_pivot = (
                    fold_mcc_pivot
                    .sort_values(
                        "Mean_MCC",
                        ascending=False
                    )
                    .reset_index(drop=True)
                )

                st.dataframe(
                    fold_mcc_pivot.style.format(
                        {
                            c: "{:.3f}"
                            for c in fold_mcc_pivot.columns
                            if c != "Model"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # PER-MODEL CONFUSION MATRICES & METRICS
            # =================================================

            st.subheader(
                "📊 Per-Model Confusion Matrices & Metrics"
            )

            st.info(
                """
Each card shows the out-of-fold (OOF) metrics and confusion matrix
for one supervised model. Metrics are computed from the full OOF
prediction set across all CV folds.
"""
            )

            model_names = list(
                performance_df["Model"]
            )

            for model_name in model_names:

                if model_name not in oof_results:
                    continue

                oof = oof_results[model_name]

                cm = oof.get(
                    "confusion_matrix"
                )

                row = (
                    performance_df[
                        performance_df["Model"] == model_name
                    ]
                    .iloc[0]
                )

                with st.expander(
                    f"🔎 {model_name} — "
                    f"MCC {row['MCC']:.3f} | "
                    f"Accuracy {row['Accuracy']:.3f}"
                ):

                    metric_cols = st.columns(5)

                    metric_cols[0].metric(
                        "Accuracy",
                        f"{row['Accuracy']:.3f}"
                    )

                    metric_cols[1].metric(
                        "Precision",
                        f"{row['Precision']:.3f}"
                    )

                    metric_cols[2].metric(
                        "Recall",
                        f"{row['Recall']:.3f}"
                    )

                    metric_cols[3].metric(
                        "F1",
                        f"{row['F1']:.3f}"
                    )

                    metric_cols[4].metric(
                        "ROC-AUC",
                        f"{row['ROC_AUC']:.3f}"
                    )

                    if cm is not None:

                        fig, ax = plt.subplots()

                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=[
                                "Inactive",
                                "Active"
                            ]
                        )

                        disp.plot(
                            ax=ax,
                            cmap="Blues",
                            colorbar=False
                        )

                        ax.set_title(
                            f"{model_name} — OOF Confusion Matrix"
                        )

                        st.pyplot(fig)

                    else:

                        st.warning(
                            "Confusion matrix not available."
                        )

            # =================================================
            # BEST MODEL
            # =================================================

            best_model_name = (
                performance_df.iloc[0][
                    "Model"
                ]
            )

            st.markdown(
                f"""
                <div class="info-card" style="border-left-color:#10b981;">
                    <div style="font-size:1.1rem; font-weight:700; color:#065f46;">
                        🥇 Best Model (by CV-MCC)
                    </div>
                    <div style="font-size:1.5rem; font-weight:800; color:#1e40af;">
                        {best_model_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            best_oof_prediction = (
                oof_results[
                    best_model_name
                ]["prediction"]
            )

            best_oof_probability = (
                oof_results[
                    best_model_name
                ]["probability"]
            )

            overall_mcc = (
                matthews_corrcoef(
                    y,
                    best_oof_prediction
                )
            )

            # =================================================
            # MCC DISPLAY
            # =================================================

            st.subheader(
                "📈 Best Model Cross-Validation Metrics"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "CV MCC",
                f"{overall_mcc:.3f}"
            )

            c2.metric(
                "CV F1",
                f"{performance_df.iloc[0]['F1']:.3f}"
            )

            c3.metric(
                "CV Accuracy",
                f"{performance_df.iloc[0]['Accuracy']:.3f}"
            )

            c4.metric(
                "CV ROC-AUC",
                f"{performance_df.iloc[0]['ROC_AUC']:.3f}"
            )

            # =================================================
            # OUT OF FOLD CANDIDATE RESULTS
            # =================================================

            st.subheader(
                "🔍 Candidate-Level Out-of-Fold Results (pIC50-based Activity)"
            )

            candidate_results = (
                training_df[
                    [
                        descriptor_name_col,
                        "pIC50",
                        "Activity"
                    ]
                ].copy()
            )

            candidate_results[
                "Actual_Label"
            ] = y

            candidate_results[
                "OOF_Active_Probability"
            ] = best_oof_probability

            candidate_results[
                "OOF_Predicted_Label"
            ] = best_oof_prediction

            candidate_results[
                "OOF_Predicted_Activity"
            ] = np.where(
                best_oof_prediction == 1,
                "Active",
                "Inactive"
            )

            candidate_results[
                "Correct"
            ] = (
                candidate_results[
                    "Actual_Label"
                ]
                ==
                candidate_results[
                    "OOF_Predicted_Label"
                ]
            )

            st.info(
                """
MCC is calculated for the complete out-of-fold prediction set.
For each compound, the table shows its OOF prediction and probability of being Active.
"""
            )

            with st.expander(
                "🔍 View Candidate-Level OOF Results",
                expanded=True
            ):
                st.dataframe(
                    candidate_results.head(50),
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # FINAL MODEL
            # =================================================

            st.subheader(
                "🧠 Final Model"
            )

            final_model = get_models()[
                best_model_name
            ]

            final_pipeline = create_pipeline(
                final_model,
                min_train_minority,
                use_smote=use_smote
            )

            final_pipeline.fit(
                X,
                y
            )

            st.success(
                "✅ Final model trained using all matched PAAD compounds."
            )

            # =================================================
            # PREPARE REMAINING DRUG LIBRARY
            # =================================================

            screening_df = df_desc.copy()

            # Preserve the original descriptor-row identity created during
            # matching. This is much safer than matching by names because the
            # training compounds may have been matched through CID/SMILES.
            screening_df["_descriptor_row"] = np.arange(len(screening_df))

            training_descriptor_rows = set(
                pd.to_numeric(
                    training_df["_descriptor_row"],
                    errors="coerce"
                ).dropna().astype(int).tolist()
            )

            remaining_df = (
                screening_df[
                    ~screening_df["_descriptor_row"].isin(
                        training_descriptor_rows
                    )
                ].copy()
            )

            # =================================================
            # SCREENING
            # =================================================

            st.subheader(
                "🔬 Virtual Screening"
            )

            st.markdown(
                f"""
                <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;">
                    <div class="metric-card" style="flex:1; min-width:140px;">
                        <div class="metric-value">{len(df_desc):,}</div>
                        <div class="metric-label">Total Descriptor Compounds</div>
                    </div>
                    <div class="metric-card" style="flex:1; min-width:140px;">
                        <div class="metric-value">{len(training_df):,}</div>
                        <div class="metric-label">Known Training Compounds</div>
                    </div>
                    <div class="metric-card" style="flex:1; min-width:140px;">
                        <div class="metric-value">{len(remaining_df):,}</div>
                        <div class="metric-label">Remaining to Screen</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if remaining_df.empty:

                st.warning(
                    "⚠️ No remaining compounds are available for virtual screening."
                )

            else:

                # ---------------------------------------------
                # Make sure screening has same descriptors
                # ---------------------------------------------

                # IMPORTANT:
                # feature_cols came from the training dataset. Never assume
                # every helper column exists in the remaining library.
                # Use ONLY columns that are present in BOTH datasets.
                screen_feature_cols = [
                    c for c in feature_cols
                    if c in remaining_df.columns
                    and not str(c).startswith("_")
                ]

                missing_screen_features = [
                    c for c in feature_cols
                    if c not in remaining_df.columns
                ]

                if missing_screen_features:
                    st.warning(
                        f"⚠️ {len(missing_screen_features)} training "
                        "feature(s) were not present in the screening file "
                        "and were excluded from screening."
                    )

                if not screen_feature_cols:
                    st.error(
                        "❌ No common molecular descriptor columns were "
                        "found between the training and screening datasets."
                    )
                    st.stop()

                # Keep EXACTLY the same feature order used for training.
                X_screen = (
                    remaining_df[
                        screen_feature_cols
                    ]
                    .apply(
                        pd.to_numeric,
                        errors="coerce"
                    )
                    .replace(
                        [
                            np.inf,
                            -np.inf
                        ],
                        np.nan
                    )
                    .fillna(0)
                )

                # Final defensive check: helper columns can NEVER reach the model.
                forbidden_screen_features = [
                    c for c in X_screen.columns
                    if str(c).startswith("_")
                ]

                if forbidden_screen_features:
                    X_screen = X_screen.drop(
                        columns=forbidden_screen_features
                    )

                # Re-align to the training feature order wherever possible.
                screen_feature_cols = [
                    c for c in feature_cols
                    if c in X_screen.columns
                ]
                X_screen = X_screen[screen_feature_cols]

                # ---------------------------------------------
                # Predict probability
                # ---------------------------------------------

                active_probability = (
                    final_pipeline
                    .predict_proba(
                        X_screen
                    )[:, 1]
                )

                prediction = (
                    active_probability >= 0.5
                ).astype(int)

                remaining_df[
                    "Active_Probability"
                ] = active_probability

                remaining_df[
                    "Predicted_Label"
                ] = prediction

                remaining_df[
                    "Predicted_Activity"
                ] = np.where(
                    prediction == 1,
                    "Active",
                    "Inactive"
                )

                # ---------------------------------------------
                # Ranking
                # ---------------------------------------------

                remaining_df = (
                    remaining_df
                    .sort_values(
                        "Active_Probability",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                remaining_df.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(
                            remaining_df
                        ) + 1
                    )
                )

                # =================================================
                # SCREENING SUMMARY
                # =================================================

                predicted_active = int(
                    (
                        remaining_df[
                            "Predicted_Label"
                        ] == 1
                    ).sum()
                )

                predicted_inactive = int(
                    (
                        remaining_df[
                            "Predicted_Label"
                        ] == 0
                    ).sum()
                )

                st.subheader(
                    "📊 Screening Summary"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Screened",
                    f"{len(remaining_df):,}"
                )

                c2.metric(
                    "🟢 Predicted Active",
                    predicted_active
                )

                c3.metric(
                    "🔴 Predicted Inactive",
                    predicted_inactive
                )

                # =================================================
                # TOP CANDIDATES
                # =================================================

                st.subheader(
                    "🏆 Top 200 Predicted Active Drug Candidates"
                )

                top_active = (
                    remaining_df[
                        remaining_df[
                            "Predicted_Label"
                        ] == 1
                    ]
                    .head(200)
                )

                if top_active.empty:

                    st.warning(
                        "⚠️ No compounds were predicted as Active in the screening library."
                    )

                else:

                    st.info(
                        f"Only the top {min(200, len(top_active)):,} predicted Active "
                        "candidates are displayed here. The application does not "
                        "display all screened compounds in this table."
                    )

                    top_columns = [
                        "Rank",
                        descriptor_name_col
                    ]

                    if "CID" in top_active.columns:

                        top_columns.append(
                            "CID"
                        )

                    top_columns.extend(
                        [
                            "Active_Probability",
                            "Predicted_Activity"
                        ]
                    )

                    st.dataframe(
                        top_active[
                            top_columns
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                # =================================================
                # DOWNLOAD VIRTUAL SCREENING
                # =================================================

                screening_download = (
                    remaining_df
                    .drop(
                        columns=[
                            "_descriptor_row"
                        ],
                        errors="ignore"
                    )
                )

                st.subheader(
                    "⬇️ Download Results"
                )

                dl_col1, dl_col2 = st.columns(2)

                with dl_col1:
                    st.download_button(
                        "📥 Download Virtual Screening Results",

                        data=
                        screening_download
                        .to_csv(
                            index=False
                        )
                        .encode(),

                        file_name=
                        "PAAD_Virtual_Screening.csv",

                        mime="text/csv",

                        use_container_width=True
                    )

                # =================================================
                # COMPLETE REPORT
                # =================================================

                summary_df = pd.DataFrame(
                    {
                        "Metric": [

                            "Total Descriptor Compounds",

                            "Total pIC50 Compounds",

                            "Matched Training Compounds",

                            "Active Training Compounds",

                            "Inactive Training Compounds",

                            "Remaining Compounds",

                            "Predicted Active",

                            "Predicted Inactive",

                            "Cross Validation Folds",

                            "Best Model",

                            "Overall CV MCC"
                        ],

                        "Value": [

                            len(df_desc),

                            len(df_pic),

                            len(training_df),

                            active_count,

                            inactive_count,

                            len(remaining_df),

                            predicted_active,

                            predicted_inactive,

                            n_splits,

                            best_model_name,

                            overall_mcc
                        ]
                    }
                )

                # ------------------------------------------------
                # Complete Excel
                # ------------------------------------------------

                report_data = dataframe_to_excel(
                    {

                        "Summary":
                            summary_df,

                        "Training_Data":
                            training_df[
                                [
                                    descriptor_name_col,
                                    "pIC50",
                                    "Activity",
                                    "Label"
                                ]
                            ],

                        "Model_Performance":
                            performance_df,

                        "Fold_Wise_MCC":
                            fold_mcc_pivot if "fold_mcc_pivot" in locals()
                            else pd.DataFrame(),

                        "OOF_Validation":
                            candidate_results,

                        "Virtual_Screening":
                            screening_download,

                        "Top_200_Active":
                            top_active
                    }
                )

                with dl_col2:
                    st.download_button(
                        "📥 Download Complete QSAR Report",

                        data=report_data,

                        file_name=
                        "PAAD_QSAR_Complete_Report.xlsx",

                        mime=
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                        use_container_width=True
                    )

                # =================================================
                # FINAL MESSAGE
                # =================================================

                st.markdown(
                    """
                    <div class="info-card" style="border-left-color:#10b981; background-color:#f0fdf4;">
                        <div style="font-size:1.2rem; font-weight:700; color:#065f46;">
                            ✅ QSAR Workflow Completed
                        </div>
                        <div style="color:#334155; margin-top:0.5rem;">
                            The final report contains: IC50 → pIC50 conversion,
                            pIC50-based Activity labels, ML model comparison, Cross-validation MCC,
                            Out-of-fold predictions, Final trained model,
                            Virtual screening results, and Ranked active candidates.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(
                f"❌ Task 3 error: {e}"
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PAAD QSAR Drug Discovery System &nbsp;|&nbsp; Streamlit + scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)