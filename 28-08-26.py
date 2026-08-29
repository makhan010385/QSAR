import pandas as pd
import numpy as np
import streamlit as st
import re
import json
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Optional models
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_OK = True
except Exception:
    LIGHTGBM_OK = False

# Imbalanced-learn for handling class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def standardize_columns(df):
    """Normalize headers and remove duplicate normalized headers safely."""
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    # Prevent pandas ambiguity when files contain pic50/pIC50,
    # Activity/activity, Label/label, etc.
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df

def drop_derived_activity_columns(df):
    """Remove previously generated pIC50/activity/label columns."""
    df = df.copy()
    derived = {"pic50", "activity", "label"}
    return df.drop(
        columns=[c for c in df.columns if str(c).strip().lower() in derived],
        errors="ignore"
    )


def is_missing_value(value):
    """Robust missing-value check for mixed text/numeric identity fields."""
    if value is None:
        return True

    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)) and result:
            return True
    except Exception:
        pass

    return str(value).strip().lower() in {
        "", "nan", "none", "null", "na", "n/a"
    }

# -------------------------------------------------
# LOCAL PAAD DRUG IDENTITY MAP
# -------------------------------------------------
# This map is embedded so Task 2 does NOT depend on PubChem/network access.
# It contains the PAAD IC50 compounds used in the supplied dataset and maps
# common drug name -> CID + SMILES.  The mapping is used only for identity;
# CID/SMILES are never ML features.
_LOCAL_PAAD_IDENTITY_RECORDS = json.loads(r'[{"DRUG NAME": "Romidepsin", "CID": 5352062, "SMILES": "CC=C1C(=O)NC(C(=O)OC2CC(=O)NC(C(=O)NC(CSSCCC=C2)C(=O)N1)C(C)C)C(C)C"}, {"DRUG NAME": "Gemcitabine", "CID": 60750, "SMILES": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F"}, {"DRUG NAME": "Bortezomib", "CID": 387447, "SMILES": "B(C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O"}, {"DRUG NAME": "Vinblastine", "CID": 13342, "SMILES": "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O"}, {"DRUG NAME": "AZD5582", "CID": 49847690, "SMILES": "CN[C@@H](C)C(N[C@H](C(N1[C@H](C(N[C@H]2C3=CC=CC=C3C[C@H]2OCC#CC#CCO[C@H]4[C@@H](NC([C@@H]5CCCN5C([C@@H](NC([C@H](C)NC)=O)C6CCCCC6)=O)=O)C(C=CC=C7)=C7C4)=O)CCC1)=O)C8CCCCC8)=O"}, {"DRUG NAME": "Luminespib", "CID": 135539077, "SMILES": "CCNC(=O)C1=NOC(=C1C2=CC=C(C=C2)CN3CCOCC3)C4=CC(=C(C=C4O)O)C(C)C"}, {"DRUG NAME": "Docetaxel", "CID": 148124, "SMILES": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)O)C)O"}, {"DRUG NAME": "Vinorelbine", "CID": 5311497, "SMILES": "CCC1=CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC"}, {"DRUG NAME": "Dinaciclib", "CID": 46926350, "SMILES": "CCC1=C2N=C(C=C(N2N=C1)NCC3=C[N+](=CC=C3)[O-])N4CCCCC4CCO"}, {"DRUG NAME": "Staurosporine", "CID": 44259, "SMILES": "CC12C(C(CC(O1)N3C4=CC=CC=C4C5=C6C(=C7C8=CC=CC=C8N2C7=C53)CNC6=O)NC)OC"}, {"DRUG NAME": "Camptothecin", "CID": 24360, "SMILES": "CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O"}, {"DRUG NAME": "Paclitaxel", "CID": 36314, "SMILES": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"}, {"DRUG NAME": "Vincristine", "CID": 5978, "SMILES": "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C=O)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O"}, {"DRUG NAME": "Dactinomycin", "CID": 457193, "SMILES": "CC1C(C(=O)NC(C(=O)N2CCCC2C(=O)N(CC(=O)N(C(C(=O)O1)C(C)C)C)C)C(C)C)NC(=O)C3=C4C(=C(C=C3)C)OC5=C(C(=O)C(=C(C5=N4)C(=O)NC6C(OC(=O)C(N(C(=O)CN(C(=O)C7CCCN7C(=O)C(NC6=O)C(C)C)C)C)C(C)C)C)N)C"}, {"DRUG NAME": "Rapamycin", "CID": 5284616, "SMILES": "CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)O)C)C)O)OC)C)C)C)OC"}, {"DRUG NAME": "Dactolisib", "CID": 11977753, "SMILES": "CC(C)(C#N)C1=CC=C(C=C1)N2C3=C4C=C(C=CC4=NC=C3N(C2=O)C)C5=CC6=CC=CC=C6N=C5"}, {"DRUG NAME": "Obatoclax Mesylate", "CID": 16681698, "SMILES": "CC1=CC(=C(N1)C=C2C(=CC(=C3C=C4C=CC=CC4=N3)N2)OC)C.CS(=O)(=O)O"}, {"DRUG NAME": "JQ1", "CID": 46907787, "SMILES": "CC1=C(SC2=C1C(=NC(C3=NN=C(N32)C)CC(=O)OC(C)(C)C)C4=CC=C(C=C4)Cl)C"}, {"DRUG NAME": "Trametinib", "CID": 11707110, "SMILES": "CC1=C2C(=C(N(C1=O)C)NC3=C(C=C(C=C3)I)F)C(=O)N(C(=O)N2C4=CC=CC(=C4)NC(=O)C)C5CC5"}, {"DRUG NAME": "Tanespimycin", "CID": 6505803, "SMILES": "CC1CC(C(C(C=C(C(C(C=CC=C(C(=O)NC2=CC(=O)C(=C(C1)C2=O)NCC=C)C)OC)OC(=O)N)C)C)O)OC"}, {"DRUG NAME": "LMP744", "CID": 397888, "SMILES": "COC1=C(C=C2C(=C1)C3=C(C4=CC5=C(C=C4C3=O)OCO5)N(C2=O)CCCNCCO)OC"}, {"DRUG NAME": "Methotrexate", "CID": 126941, "SMILES": "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"}, {"DRUG NAME": "Podophyllotoxin bromide", "CID": 234820, "SMILES": "COC1=CC(=CC(=C1OC)OC)C2C3C(COC3=O)C(C4=CC5=C(C=C24)OCO5)Br"}, {"DRUG NAME": "AZD7762", "CID": 11152667, "SMILES": "C1CC(CNC1)NC(=O)C2=C(C=C(S2)C3=CC(=CC=C3)F)NC(=O)N"}, {"DRUG NAME": "Bleomycin", "CID": 5360373, "SMILES": "CC1=C(N=C(N=C1N)C(CC(=O)N)NCC(C(=O)N)N)C(=O)NC(C(C2=CN=CN2)OC3C(C(C(C(O3)CO)O)O)OC4C(C(C(C(O4)CO)O)OC(=O)N)O)C(=O)NC(C)C(C(C)C(=O)NC(C(C)O)C(=O)NCCC5=NC(=CS5)C6=NC(=CS6)C(=O)NCCC[S+](C)C)O"}, {"DRUG NAME": "Epirubicin", "CID": 41867, "SMILES": "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O"}, {"DRUG NAME": "AZD5153", "CID": 118693659, "SMILES": "CC1C(=O)N(CCN1CCOC2=CC=C(C=C2)C3CCN(CC3)C4=NN5C(=NN=C5OC)C=C4)C"}, {"DRUG NAME": "MG-132", "CID": 462382, "SMILES": "CC(C)CC(C=O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)OCC1=CC=CC=C1"}, {"DRUG NAME": "Sabutoclax", "CID": 46236925, "SMILES": "CC1=CC2=C(C(=C(C=C2C(=C1C3=C(C4=CC(=C(C(=C4C=C3C)C(=O)NCC(C)C5=CC=CC=C5)O)O)O)O)O)O)C(=O)NCC(C)C6=CC=CC=C6"}, {"DRUG NAME": "AZD8055", "CID": 25262965, "SMILES": "CC1COCCN1C2=NC(=NC3=C2C=CC(=N3)C4=CC(=C(C=C4)OC)CO)N5CCOCC5C"}, {"DRUG NAME": "Topotecan", "CID": 60700, "SMILES": "CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=C(C=CC(=C5CN(C)C)O)N=C4C3=C2)O"}, {"DRUG NAME": "Mitoxantrone", "CID": 4212, "SMILES": "C1=CC(=C2C(=C1NCCNCCO)C(=O)C3=C(C=CC(=C3C2=O)O)O)NCCNCCO"}, {"DRUG NAME": "PD0325901", "CID": 9826528, "SMILES": "C1=CC(=C(C=C1I)F)NC2=C(C=CC(=C2F)F)C(=O)NOCC(CO)O"}, {"DRUG NAME": "BI-2536", "CID": 11364421, "SMILES": "CCC1C(=O)N(C2=CN=C(N=C2N1C3CCCC3)NC4=C(C=C(C=C4)C(=O)NC5CCN(CC5)C)OC)C"}, {"DRUG NAME": "Teniposide", "CID": 452548, "SMILES": "COC1=CC(=CC(=C1O)OC)C2C3C(COC3=O)C(C4=CC5=C(C=C24)OCO5)OC6C(C(C7C(O6)COC(O7)C8=CC=CS8)O)O"}, {"DRUG NAME": "MK-1775", "CID": 24856436, "SMILES": "CC(C)(C1=NC(=CC=C1)N2C3=NC(=NC=C3C(=O)N2CC=C)NC4=CC=C(C=C4)N5CCN(CC5)C)O"}, {"DRUG NAME": "Schweinfurthin A", "CID": 643462, "SMILES": "CC(=CCCC(=CCC1=C(C=C(C=C1O)C=CC2=CC3=C(C(=C2)O)OC4(CC(C(C(C4C3)(C)C)O)O)C)O)C)C"}, {"DRUG NAME": "Lestaurtinib", "CID": 126565, "SMILES": "CC12C(CC(O1)N3C4=CC=CC=C4C5=C6C(=C7C8=CC=CC=C8N2C7=C53)CNC6=O)(CO)O"}, {"DRUG NAME": "Dasatinib", "CID": 3062316, "SMILES": "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO"}, {"DRUG NAME": "Refametinib", "CID": 44182295, "SMILES": "COC1=CC(=C(C(=C1NS(=O)(=O)C2(CC2)CC(CO)O)NC3=C(C=C(C=C3)I)F)F)F"}, {"DRUG NAME": "Telomerase Inhibitor IX", "CID": 10385095, "SMILES": "C1=CC(=CC(=C1)NC(=O)C2=C(C(=CC=C2)O)O)NC(=O)C3=C(C(=CC=C3)O)O"}, {"DRUG NAME": "Pevonedistat", "CID": 16720766, "SMILES": "C1CC2=CC=CC=C2C1NC3=C4C=CN(C4=NC=N3)C5CC(C(C5)O)COS(=O)(=O)N"}, {"DRUG NAME": "POMHEX", "CID": 122540908, "SMILES": "CC(C)(C)C(=O)OCOP(=O)(C1CCCN(C1=O)O)OCOC(=O)C(C)(C)C"}, {"DRUG NAME": "AZD6738", "CID": 121596701, "SMILES": "CC1COCCN1C2=NC(=NC(=C2)C3(CC3)S(=N)(=O)C)C4=CN=CC5=C4C=CN5"}, {"DRUG NAME": "Foretinib", "CID": 42642645, "SMILES": "COC1=CC2=C(C=CN=C2C=C1OCCCN3CCOCC3)OC4=C(C=C(C=C4)NC(=O)C5(CC5)C(=O)NC6=CC=C(C=C6)F)F"}, {"DRUG NAME": "Cytarabine", "CID": 6253, "SMILES": "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O"}, {"DRUG NAME": "AZD8186", "CID": 52913813, "SMILES": "CC(C1=CC(=CC2=C1OC(=CC2=O)N3CCOCC3)C(=O)N(C)C)NC4=CC(=CC(=C4)F)F"}, {"DRUG NAME": "Alisertib", "CID": 24771867, "SMILES": "COC1=C(C(=CC=C1)F)C2=NCC3=CN=C(N=C3C4=C2C=C(C=C4)Cl)NC5=CC(=C(C=C5)C(=O)O)OC"}, {"DRUG NAME": "Afatinib", "CID": 10184653, "SMILES": "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4"}, {"DRUG NAME": "GNE-317", "CID": 70676303, "SMILES": "CC1=C(SC2=C1N=C(N=C2N3CCOCC3)C4=CN=C(N=C4)N)C5(COC5)OC"}, {"DRUG NAME": "SGC0946", "CID": 56962337, "SMILES": "CC(C)N(CCCNC(=O)NC1=CC=C(C=C1)C(C)(C)C)CC2C(C(C(O2)N3C=C(C4=C(N=CN=C43)N)Br)O)O"}, {"DRUG NAME": "Pictilisib", "CID": 17755052, "SMILES": "CS(=O)(=O)N1CCN(CC1)CC2=CC3=C(S2)C(=NC(=N3)C4=C5C=NNC5=CC=C4)N6CCOCC6"}, {"DRUG NAME": "Acetalax", "CID": 8269, "SMILES": "CC(=O)OC1=CC=C(C=C1)C2(C3=CC=CC=C3NC2=O)C4=CC=C(C=C4)OC(=O)C"}, {"DRUG NAME": "I-BET-762", "CID": 46943432, "SMILES": "CCNC(=O)CC1C2=NN=C(N2C3=C(C=C(C=C3)OC)C(=N1)C4=CC=C(C=C4)Cl)C"}, {"DRUG NAME": "AGI-6780", "CID": 71299339, "SMILES": "C1CC1NS(=O)(=O)C2=CC(=C(C=C2)C3=CSC=C3)NC(=O)NC4=CC=CC(=C4)C(F)(F)F"}, {"DRUG NAME": "Buparlisib", "CID": 16654980, "SMILES": "C1COCCN1C2=NC(=NC(=C2)C3=CN=C(C=C3C(F)(F)F)N)N4CCOCC4"}, {"DRUG NAME": "Vorinostat", "CID": 5311, "SMILES": "C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO"}, {"DRUG NAME": "Temsirolimus", "CID": 6918289, "SMILES": "CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)OC(=O)C(C)(CO)CO)C)C)O)OC)C)C)C)OC"}, {"DRUG NAME": "Dihydrorotenone", "CID": 243725, "SMILES": "CC(C)C1CC2=C(O1)C=CC3=C2OC4COC5=CC(=C(C=C5C4C3=O)OC)OC"}, {"DRUG NAME": "AZD5438", "CID": 16747683, "SMILES": "CC1=NC=C(N1C(C)C)C2=NC(=NC=C2)NC3=CC=C(C=C3)S(=O)(=O)C"}, {"DRUG NAME": "5-Fluorouracil", "CID": 3385, "SMILES": "C1=C(C(=O)NC(=O)N1)F"}, {"DRUG NAME": "Bosutinib", "CID": 5328940, "SMILES": "CN1CCN(CC1)CCCOC2=C(C=C3C(=C2)N=CC(=C3NC4=CC(=C(C=C4Cl)Cl)OC)C#N)OC"}, {"DRUG NAME": "Irinotecan", "CID": 60838, "SMILES": "CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)OC(=O)N6CCC(CC6)N7CCCCC7"}, {"DRUG NAME": "Mycophenolic acid", "CID": 446541, "SMILES": "CC1=C2COC(=O)C2=C(C(=C1OC)CC=C(C)CCC(=O)O)O"}, {"DRUG NAME": "Wee1 Inhibitor", "CID": 10384072, "SMILES": "C1=CC=C(C(=C1)C2=CC3=C(C4=C(N3)C=CC(=C4)O)C5=C2C(=O)NC5=O)Cl"}, {"DRUG NAME": "5-azacytidine", "CID": 9444, "SMILES": "C1=NC(=NC(=O)N1C2C(C(C(O2)CO)O)O)N"}, {"DRUG NAME": "SCH772984", "CID": 24866313, "SMILES": "C1CN(CC1C(=O)NC2=CC3=C(C=C2)NN=C3C4=CC=NC=C4)CC(=O)N5CCN(CC5)C6=CC=C(C=C6)C7=NC=CC=N7"}, {"DRUG NAME": "Osimertinib", "CID": 71496458, "SMILES": "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC"}, {"DRUG NAME": "Bromosporine", "CID": 72943187, "SMILES": "CCOC(=O)NC1=CC(=NN2C1=NN=C2C)C3=CC(=C(C=C3)C)NS(=O)(=O)C"}, {"DRUG NAME": "BMS-754807", "CID": 24785538, "SMILES": "CC1(CCCN1C2=NN3C=CC=C3C(=N2)NC4=NNC(=C4)C5CC5)C(=O)NC6=CN=C(C=C6)F"}, {"DRUG NAME": "AZ6102", "CID": 135905416, "SMILES": "CC1CN(CC(N1)C)C2=NC=C(C(=C2)C)C3=CC=C(C=C3)C4=NC5=C(C=CN5C)C(=O)N4"}, {"DRUG NAME": "AZD3759", "CID": 78209992, "SMILES": "CC1CN(CCN1C(=O)OC2=C(C=C3C(=C2)C(=NC=N3)NC4=C(C(=CC=C4)Cl)F)OC)C"}, {"DRUG NAME": "GSK-LSD1-2HCl", "CID": 91826516, "SMILES": "C1CNCCC1NC2CC2C3=CC=CC=C3.Cl.Cl"}, {"DRUG NAME": "BMS-536924", "CID": 135440466, "SMILES": "CC1=CC(=CC2=C1N=C(N2)C3=C(C=CNC3=O)NCC(C4=CC(=CC=C4)Cl)O)N5CCOCC5"}, {"DRUG NAME": "Sinularin", "CID": 5458571, "SMILES": "CC1=CCCC(C2CC(CCC3(C(O3)CC1)C)C(=C)C(=O)O2)(C)O"}, {"DRUG NAME": "AZD2014", "CID": 25262792, "SMILES": "CC1COCCN1C2=NC(=NC3=C2C=CC(=N3)C4=CC(=CC=C4)C(=O)NC)N5CCOCC5C"}, {"DRUG NAME": "Uprosertib", "CID": 51042438, "SMILES": "CN1C(=C(C=N1)Cl)C2=C(OC(=C2)C(=O)NC(CC3=CC(=C(C=C3)F)F)CN)Cl"}, {"DRUG NAME": "Gefitinib", "CID": 123631, "SMILES": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"}, {"DRUG NAME": "MK-8776", "CID": 46239015, "SMILES": "CN1C=C(C=N1)C2=C3N=C(C(=C(N3N=C2)N)Br)C4CCCNC4"}, {"DRUG NAME": "YK-4-279", "CID": 44632017, "SMILES": "COC1=CC=C(C=C1)C(=O)CC2(C3=C(C=CC(=C3NC2=O)Cl)Cl)O"}, {"DRUG NAME": "Talazoparib", "CID": 135565082, "SMILES": "CN1C(=NC=N1)C2C(NC3=CC(=CC4=C3C2=NNC4=O)F)C5=CC=C(C=C5)F"}, {"DRUG NAME": "LGK974", "CID": 46926973, "SMILES": "CC1=CC(=CN=C1C2=CC(=NC=C2)C)CC(=O)NC3=NC=C(C=C3)C4=NC=CN=C4"}, {"DRUG NAME": "IWP-2", "CID": 2155128, "SMILES": "CC1=CC2=C(C=C1)N=C(S2)NC(=O)CSC3=NC4=C(C(=O)N3C5=CC=CC=C5)SCC4"}, {"DRUG NAME": "Cediranib", "CID": 9933475, "SMILES": "CC1=CC2=C(N1)C=CC(=C2F)OC3=NC=NC4=CC(=C(C=C43)OC)OCCCN5CCCC5"}, {"DRUG NAME": "BMS-345541", "CID": 9926054, "SMILES": "CC1=CC2=C(C=C1)N=C(C3=NC=C(N23)C)NCCN.Cl"}, {"DRUG NAME": "ZM447439", "CID": 9914412, "SMILES": "COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=C(C=C3)NC(=O)C4=CC=CC=C4)OCCCN5CCOCC5"}, {"DRUG NAME": "UMI-77", "CID": 992586, "SMILES": "C1=CC=C2C(=C1)C(=CC(=C2O)SCC(=O)O)NS(=O)(=O)C3=CC=C(C=C3)Br"}, {"DRUG NAME": "Taselisib", "CID": 51001932, "SMILES": "CC1=NN(C(=N1)C2=CN3CCOC4=C(C3=N2)C=CC(=C4)C5=CN(N=C5)C(C)(C)C(=O)N)C(C)C"}, {"DRUG NAME": "Entinostat", "CID": 4261, "SMILES": "C1=CC=C(C(=C1)N)NC(=O)C2=CC=C(C=C2)CNC(=O)OCC3=CN=CC=C3"}, {"DRUG NAME": "Savolitinib", "CID": 68289010, "SMILES": "CC(C1=CN2C=CN=C2C=C1)N3C4=NC(=CN=C4N=N3)C5=CN(N=C5)C"}, {"DRUG NAME": "VX-11e", "CID": 11634725, "SMILES": "CC1=CN=C(N=C1C2=CNC(=C2)C(=O)NC(CO)C3=CC(=CC=C3)Cl)NC4=C(C=C(C=C4)F)Cl"}, {"DRUG NAME": "GSK-LSD1", "CID": 71522234, "SMILES": "C1CNCCC1NC2CC2C3=CC=CC=C3"}, {"DRUG NAME": "AZ960", "CID": 25099184, "SMILES": "CC1=CC(=NN1)NC2=C(C=C(C(=N2)NC(C)C3=CC=C(C=C3)F)C#N)F"}, {"DRUG NAME": "Pyridostatin", "CID": 25227847, "SMILES": "C1=CC=C2C(=C1)C(=CC(=N2)NC(=O)C3=CC(=CC(=N3)C(=O)NC4=NC5=CC=CC=C5C(=C4)OCCN)OCCN)OCCN"}, {"DRUG NAME": "Ulixertinib", "CID": 11719003, "SMILES": "CC(C)NC1=NC=C(C(=C1)C2=CNC(=C2)C(=O)NC(CO)C3=CC(=CC=C3)Cl)Cl"}, {"DRUG NAME": "Navitoclax", "CID": 24978538, "SMILES": "CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)NC(CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C"}, {"DRUG NAME": "Gallibiscoquinazole", "CID": 353472, "SMILES": "COC1=C(C(=C2C(=C1)C(=O)NC(N2)(NC(=O)OC)NC(=O)OC)OC)OC"}, {"DRUG NAME": "Selumetinib", "CID": 10127622, "SMILES": "CN1C=NC2=C1C=C(C(=C2F)NC3=C(C=C(C=C3)Br)Cl)C(=O)NOCCO"}, {"DRUG NAME": "Sorafenib", "CID": 216239, "SMILES": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F"}, {"DRUG NAME": "RO-3306", "CID": 136240579, "SMILES": "C1=CC2=C(C=CC(=C2)C=C3C(=O)NC(=NCC4=CC=CS4)S3)N=C1"}, {"DRUG NAME": "Sapitinib", "CID": 11488320, "SMILES": "CNC(=O)CN1CCC(CC1)OC2=C(C=C3C(=C2)C(=NC=N3)NC4=C(C(=CC=C4)Cl)F)OC"}, {"DRUG NAME": "AGK2", "CID": 2130404, "SMILES": "C1=CC2=C(C=CC=N2)C(=C1)NC(=O)C(=CC3=CC=C(O3)C4=C(C=CC(=C4)Cl)Cl)C#N"}, {"DRUG NAME": "PRIMA-1MET", "CID": 52918385, "SMILES": "COCC1(C(=O)C2CCN1CC2)CO"}, {"DRUG NAME": "GSK343", "CID": 71268957, "SMILES": "CCCC1=C(C(=O)NC(=C1)C)CNC(=O)C2=C3C=NN(C3=CC(=C2)C4=CC(=NC=C4)N5CCN(CC5)C)C(C)C"}, {"DRUG NAME": "WEHI-539", "CID": 71297207, "SMILES": "C1CC2=C(C=C(C=C2)C3=NC(=C(S3)CCCOC4=CC=C(C=C4)CN)C(=O)O)C(=NNC5=NC6=CC=CC=C6S5)C1"}, {"DRUG NAME": "AZD4547", "CID": 51039095, "SMILES": "CC1CN(CC(N1)C)C2=CC=C(C=C2)C(=O)NC3=NNC(=C3)CCC4=CC(=CC(=C4)OC)OC"}, {"DRUG NAME": "MIM1", "CID": 135691163, "SMILES": "CC1=CSC(=NC2CCCCC2)N1N=CC3=C(C(=C(C=C3)O)O)O"}, {"DRUG NAME": "GSK2606414", "CID": 53469448, "SMILES": "NC1=C2C(N(C)C=C2C3=CC4=C(N(C(CC5=CC=CC(C(F)(F)F)=C5)=O)CC4)C=C3)=NC=N1"}, {"DRUG NAME": "Erlotinib", "CID": 176870, "SMILES": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"}, {"DRUG NAME": "MK-2206", "CID": 24964624, "SMILES": "C1CC(C1)(C2=CC=C(C=C2)C3=C(C=C4C(=N3)C=CN5C4=NNC5=O)C6=CC=CC=C6)N"}, {"DRUG NAME": "PRT062607", "CID": 44462758, "SMILES": "C1CCC(C(C1)N)NC2=NC=C(C(=N2)NC3=CC(=CC=C3)N4N=CC=N4)C(=O)N"}, {"DRUG NAME": "WIKI4", "CID": 2984337, "SMILES": "COC1=CC=C(C=C1)N2C(=NN=C2SCCCN3C(=O)C4=CC=CC5=C4C(=CC=C5)C3=O)C6=CC=NC=C6"}, {"DRUG NAME": "Lapatinib", "CID": 208908, "SMILES": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl"}, {"DRUG NAME": "Cisplatin", "CID": 5702198, "SMILES": "N.N.Cl[Pt]Cl"}, {"DRUG NAME": "WZ4003", "CID": 72200024, "SMILES": "CCC(=O)NC1=CC(=CC=C1)OC2=NC(=NC=C2Cl)NC3=C(C=C(C=C3)N4CCN(CC4)C)OC"}, {"DRUG NAME": "ABT737", "CID": 11228183, "SMILES": "CN(C)CCC(CSC1=CC=CC=C1)NC2=C(C=C(C=C2)S(=O)(=O)NC(=O)C3=CC=C(C=C3)N4CCN(CC4)CC5=CC=CC=C5C6=CC=C(C=C6)Cl)[N+](=O)[O-]"}, {"DRUG NAME": "AZD1332", "CID": 49831044, "SMILES": "CC(C)OC1=NNC(=C1)NC2=NC(=NC=C2Cl)NC(C)C3=NC=C(C=C3)F"}, {"DRUG NAME": "NVP-ADW742", "CID": 9825149, "SMILES": "C1CCN(C1)CC2CC(C2)N3C=C(C4=C(N=CN=C43)N)C5=CC(=CC=C5)OCC6=CC=CC=C6"}, {"DRUG NAME": "L-Oxonoreleagnine", "CID": 87371, "SMILES": "C1CNC(=O)C2=C1C3=CC=CC=C3N2"}, {"DRUG NAME": "Alpelisib", "CID": 56649450, "SMILES": "CC1=C(SC(=N1)NC(=O)N2CCCC2C(=O)N)C3=CC(=NC=C3)C(C)(C)C(F)(F)F"}, {"DRUG NAME": "VE-822", "CID": 59472121, "SMILES": "CC(C)S(=O)(=O)C1=CC=C(C=C1)C2=CN=C(C(=N2)C3=CC(=NO3)C4=CC=C(C=C4)CNC)N"}, {"DRUG NAME": "Nilotinib", "CID": 644241, "SMILES": "CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"}, {"DRUG NAME": "LCL161", "CID": 24737642, "SMILES": "CC(C(=O)NC(C1CCCCC1)C(=O)N2CCCC2C3=NC(=CS3)C(=O)C4=CC=C(C=C4)F)NC"}, {"DRUG NAME": "EHT-1864", "CID": 9938202, "SMILES": "C1COCCN1CC2=CC(=O)C(=CO2)OCCCCCSC3=C4C=CC(=CC4=NC=C3)C(F)(F)F.Cl.Cl"}, {"DRUG NAME": "Rucaparib", "CID": 9931954, "SMILES": "CNCC1=CC=C(C=C1)C2=C3CCNC(=O)C4=C3C(=CC(=C4)F)N2"}, {"DRUG NAME": "Venetoclax", "CID": 49846579, "SMILES": "CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)CN3CCN(CC3)C4=CC(=C(C=C4)C(=O)NS(=O)(=O)C5=CC(=C(C=C5)NCC6CCOCC6)[N+](=O)[O-])OC7=CN=C8C(=C7)C=CN8)C"}, {"DRUG NAME": "Axitinib", "CID": 6450551, "SMILES": "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"}, {"DRUG NAME": "Afuresertib", "CID": 46843057, "SMILES": "CN1C(=C(C=N1)Cl)C2=C(SC(=C2)C(=O)NC(CC3=CC(=CC=C3)F)CN)Cl"}, {"DRUG NAME": "GSK2578215A", "CID": 68107965, "SMILES": "C1=CC=C(C=C1)COC2=C(C=C(C=C2)C3=CC(=NC=C3)F)C(=O)NC4=CN=CC=C4"}, {"DRUG NAME": "Crizotinib", "CID": 11626560, "SMILES": "CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=C(N=CC(=C2)C3=CN(N=C3)C4CCNCC4)N"}, {"DRUG NAME": "I-BRD9", "CID": 91668541, "SMILES": "CCN1C=C(C2=C(C1=O)C=C(S2)C(=NC3CCS(=O)(=O)CC3)N)C4=CC(=CC=C4)C(F)(F)F"}, {"DRUG NAME": "XAV939", "CID": 135418940, "SMILES": "C1CSCC2=C1N=C(NC2=O)C3=CC=C(C=C3)C(F)(F)F"}, {"DRUG NAME": "Entospletinib", "CID": 59473233, "SMILES": "C1COCCN1C2=CC=C(C=C2)NC3=NC(=CN4C3=NC=C4)C5=CC6=C(C=C5)C=NN6"}, {"DRUG NAME": "Wnt-C59", "CID": 57519544, "SMILES": "CC1=NC=CC(=C1)C2=CC=C(C=C2)CC(=O)NC3=CC=C(C=C3)C4=CN=CC=C4"}, {"DRUG NAME": "Serdemetan", "CID": 11609586, "SMILES": "C1=CC=C2C(=C1)C(=CN2)CCNC3=CC=C(C=C3)NC4=CC=NC=C4"}, {"DRUG NAME": "Tamoxifen", "CID": 2733526, "SMILES": "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3"}, {"DRUG NAME": "PD173074", "CID": 1401, "SMILES": "CCN(CC)CCCCNC1=NC2=NC(=C(C=C2C=N1)C3=CC(=CC(=C3)OC)OC)NC(=O)NC(C)(C)C"}, {"DRUG NAME": "Zoledronate", "CID": 68740, "SMILES": "C1=CN(C=N1)CC(O)(P(=O)(O)O)P(=O)(O)O"}, {"DRUG NAME": "AT13148", "CID": 24905401, "SMILES": "C1=CC(=CC=C1C2=CNN=C2)C(CN)(C3=CC=C(C=C3)Cl)O"}, {"DRUG NAME": "VE821", "CID": 51000408, "SMILES": "CS(=O)(=O)C1=CC=C(C=C1)C2=CN=C(C(=N2)C(=O)NC3=CC=CC=C3)N"}, {"DRUG NAME": "Palbociclib", "CID": 5330286, "SMILES": "CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C"}, {"DRUG NAME": "ML323", "CID": 60167849, "SMILES": "CC1=CN=C(N=C1NCC2=CC=C(C=C2)N3C=CN=N3)C4=CC=CC=C4C(C)C"}, {"DRUG NAME": "ICL-SIRT078", "CID": 45168044, "SMILES": "COC1=C(C2=CC=CC=C2C=C1)CN3C=NC4=C(C3=O)C5=C(S4)CC(CC5)NCC6=CN=CC=C6"}, {"DRUG NAME": "PLX-4720", "CID": 24180719, "SMILES": "CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)Cl)F"}, {"DRUG NAME": "GW441756", "CID": 73755109, "SMILES": "CN1C=C(C2=CC=CC=C21)C=C3C4=C(C=CC=N4)NC3=O.Cl"}, {"DRUG NAME": "GSK1904529A", "CID": 25124816, "SMILES": "CCC1=CC(=C(C=C1N2CCC(CC2)N3CCN(CC3)S(=O)(=O)C)OC)NC4=NC=CC(=N4)C5=C(N=C6N5C=CC=C6)C7=CC(=C(C=C7)OC)C(=O)NC8=C(C=CC=C8F)F"}, {"DRUG NAME": "Fludarabine", "CID": 657237, "SMILES": "C1=NC2=C(N=C(N=C2N1C3C(C(C(O3)CO)O)O)F)N"}, {"DRUG NAME": "Niraparib", "CID": 24958200, "SMILES": "C1CC(CNC1)C2=CC=C(C=C2)N3C=C4C=CC=C(C4=N3)C(=O)N"}, {"DRUG NAME": "Ribociclib", "CID": 44631912, "SMILES": "CN(C)C(=O)C1=CC2=CN=C(N=C2N1C3CCCC3)NC4=NC=C(C=C4)N5CCNCC5"}, {"DRUG NAME": "Leflunomide", "CID": 3899, "SMILES": "CC1=C(C=NO1)C(=O)NC2=CC=C(C=C2)C(F)(F)F"}, {"DRUG NAME": "AMG-319", "CID": 68947304, "SMILES": "CC(C1=C(N=C2C=C(C=CC2=C1)F)C3=CC=CC=N3)NC4=NC=NC5=C4NC=N5"}, {"DRUG NAME": "Ipatasertib", "CID": 24788740, "SMILES": "CC1CC(C2=C1C(=NC=N2)N3CCN(CC3)C(=O)C(CNC(C)C)C4=CC=C(C=C4)Cl)O"}, {"DRUG NAME": "CPI-637", "CID": 121271792, "SMILES": "CC1CC(=O)NC2=CC=CC(=C2N1)C3=CC4=C(C=C3)N(N=C4C5=CN(N=C5)C)C"}, {"DRUG NAME": "Oxaliplatin", "CID": 9887053, "SMILES": "C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]"}, {"DRUG NAME": "MN-64", "CID": 2802462, "SMILES": "CC(C)C1=CC=C(C=C1)C2=CC(=O)C3=CC=CC=C3O2"}, {"DRUG NAME": "Ruxolitinib", "CID": 25126798, "SMILES": "C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3"}, {"DRUG NAME": "Dabrafenib", "CID": 44462760, "SMILES": "CC(C)(C)C1=NC(=C(S1)C2=NC(=NC=C2)N)C3=C(C(=CC=C3)NS(=O)(=O)C4=C(C=CC=C4F)F)F"}, {"DRUG NAME": "RVX-208", "CID": 135564749, "SMILES": "CC1=CC(=CC(=C1OCCO)C)C2=NC3=C(C(=CC(=C3)OC)OC)C(=O)N2"}, {"DRUG NAME": "GSK2110183B", "CID": 46843056, "SMILES": "CN1C(=C(C=N1)Cl)C2=C(SC(=C2)C(=O)NC(CC3=CC(=CC=C3)F)CN)Cl.Cl"}, {"DRUG NAME": "AZD5363", "CID": 25227436, "SMILES": "C1CN(CCC1(C(=O)NC(CCO)C2=CC=C(C=C2)Cl)N)C3=NC=NC4=C3C=CN4"}, {"DRUG NAME": "Avagacestat", "CID": 46883536, "SMILES": "C1=CC(=CC=C1S(=O)(=O)N(CC2=C(C=C(C=C2)C3=NOC=N3)F)C(CCC(F)(F)F)C(=O)N)Cl"}, {"DRUG NAME": "Doramapimod", "CID": 156422, "SMILES": "CC1=CC=C(C=C1)N2C(=CC(=N2)C(C)(C)C)NC(=O)NC3=CC=C(C4=CC=CC=C43)OCCN5CCOCC5"}, {"DRUG NAME": "A-366", "CID": 76285486, "SMILES": "COC1=C(C=C2C(=C1)C3(CCC3)C(=N2)N)OCCCN4CCCC4"}, {"DRUG NAME": "AGI-5198", "CID": 56645356, "SMILES": "CC1=CC=CC=C1C(C(=O)NC2CCCCC2)N(C3=CC(=CC=C3)F)C(=O)CN4C=CN=C4C"}, {"DRUG NAME": "SB590885", "CID": 135564599, "SMILES": "CN(C)CCOC1=CC=C(C=C1)C2=NC(=C(N2)C3=CC=NC=C3)C4=CC5=C(C=C4)C(=NO)CC5"}, {"DRUG NAME": "Mirin", "CID": 1206243, "SMILES": "C1=CC(=CC=C1C=C2C(=O)N=C(S2)N)O"}, {"DRUG NAME": "Veliparib", "CID": 11960529, "SMILES": "CC1(CCCN1)C2=NC3=C(C=CC=C3N2)C(=O)N"}, {"DRUG NAME": "SB216763", "CID": 176158, "SMILES": "CN1C=C(C2=CC=CC=C21)C3=C(C(=O)NC3=O)C4=C(C=C(C=C4)Cl)Cl"}, {"DRUG NAME": "GSK591", "CID": 117072552, "SMILES": "C1CC(C1)NC2=NC=CC(=C2)C(=O)NCC(CN3CCC4=CC=CC=C4C3)O"}, {"DRUG NAME": "AZD5991", "CID": 131634760, "SMILES": "CC1=C2C(=NN1C)CSCC3=NN(C(=C3)CSC4=CC5=CC=CC=C5C(=C4)OCCCC6=C(N(C7=C6C=CC(=C27)Cl)C)C(=O)O)C"}, {"DRUG NAME": "PCI-34051", "CID": 24753719, "SMILES": "COC1=CC=C(C=C1)CN2C=CC3=C2C=C(C=C3)C(=O)NO"}, {"DRUG NAME": "SGC-CBP30", "CID": 72201027, "SMILES": "CC1=C(C(=NO1)C)C2=CC3=C(C=C2)N(C(=N3)CCC4=CC(=C(C=C4)OC)Cl)CC(C)N5CCOCC5"}, {"DRUG NAME": "KU-55933", "CID": 5278396, "SMILES": "C1COCCN1C2=CC(=O)C=C(O2)C3=C4C(=CC=C3)SC5=CC=CC=C5S4"}, {"DRUG NAME": "Olaparib", "CID": 23725625, "SMILES": "C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC(=C3)CC4=NNC(=O)C5=CC=CC=C54)F"}, {"DRUG NAME": "Ibrutinib", "CID": 24821094, "SMILES": "C=CC(=O)N1CCCC(C1)N2C3=NC=NC(=C3C(=N2)C4=CC=C(C=C4)OC5=CC=CC=C5)N"}, {"DRUG NAME": "GDC0810", "CID": 56941241, "SMILES": "CCC(=C(C1=CC=C(C=C1)C=CC(=O)O)C2=CC3=C(C=C2)NN=C3)C4=C(C=C(C=C4)F)Cl"}, {"DRUG NAME": "CZC24832", "CID": 42623951, "SMILES": "CC(C)(C)NS(=O)(=O)C1=CN=CC(=C1)C2=CN3C(=NC(=N3)N)C(=C2)F"}, {"DRUG NAME": "Vismodegib", "CID": 24776445, "SMILES": "CS(=O)(=O)C1=CC(=C(C=C1)C(=O)NC2=CC(=C(C=C2)Cl)C3=CC=CC=N3)Cl"}, {"DRUG NAME": "Nutlin-3a (-)", "CID": 11433190, "SMILES": "O=C(N1C(C2=C(C=C(C=C2)OC)OC(C)C)=N[C@H]([C@H]1C3=CC=C(C=C3)Cl)C4=CC=C(C=C4)Cl)N5CC(NCC5)=O"}, {"DRUG NAME": "Cyclophosphamide", "CID": 2907, "SMILES": "C1CNP(=O)(OC1)N(CCCl)CCCl"}, {"DRUG NAME": "LY2109761", "CID": 11655119, "SMILES": "C1CC2=C(C(=NN2C1)C3=CC=CC=N3)C4=C5C=CC(=CC5=NC=C4)OCCN6CCOCC6"}, {"DRUG NAME": "BIBR-1532", "CID": 9927531, "SMILES": "CC(=CC(=O)NC1=CC=CC=C1C(=O)O)C2=CC3=CC=CC=C3C=C2"}, {"DRUG NAME": "EPZ5676", "CID": 57345410, "SMILES": "CC(C)N(CC1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)O)C4CC(C4)CCC5=NC6=C(N5)C=C(C=C6)C(C)(C)C"}, {"DRUG NAME": "Elephantin", "CID": 442205, "SMILES": "CC(=CC(=O)OC1CC2=CC(CC3(C(O3)C4C1C(=C)C(=O)O4)C)OC2=O)C"}, {"DRUG NAME": "CCT007093", "CID": 2314623, "SMILES": "C1CC(=CC2=CC=CS2)C(=O)C1=CC3=CC=CS3"}, {"DRUG NAME": "LJI308", "CID": 118704762, "SMILES": "C1COCCN1C2=CC=C(C=C2)C3=C(C=NC=C3)C4=CC(=C(C(=C4)F)O)F"}, {"DRUG NAME": "Lenalidomide", "CID": 216326, "SMILES": "C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N"}, {"DRUG NAME": "JNK Inhibitor VIII", "CID": 11624601, "SMILES": "CCOC1=C(C(=CC(=N1)NC(=O)CC2=C(C=CC(=C2)OC)OC)N)C#N"}, {"DRUG NAME": "Nelarabine", "CID": 3011155, "SMILES": "COC1=NC(=NC2=C1N=CN2C3C(C(C(O3)CO)O)O)N"}, {"DRUG NAME": "EPZ004777", "CID": 56962336, "SMILES": "CC(C)N(CCCNC(=O)NC1=CC=C(C=C1)C(C)(C)C)CC2C(C(C(O2)N3C=CC4=C(N=CN=C43)N)O)O"}, {"DRUG NAME": "GSK2801", "CID": 73010930, "SMILES": "CCCOC1=CC2=C(C=C(N2C=C1)C(=O)C)C3=CC=CC=C3S(=O)(=O)C"}, {"DRUG NAME": "P22077", "CID": 46931953, "SMILES": "CC(=O)C1=CC(=C(S1)SC2=C(C=C(C=C2)F)F)[N+](=O)[O-]"}, {"DRUG NAME": "MIRA-1", "CID": 227681, "SMILES": "CCC(=O)OCN1C(=O)C=CC1=O"}, {"DRUG NAME": "Motesanib", "CID": 11667893, "SMILES": "CC1(CNC2=C1C=CC(=C2)NC(=O)C3=C(N=CC=C3)NCC4=CC=NC=C4)C"}, {"DRUG NAME": "AZD1208", "CID": 58423153, "SMILES": "C1CC(CN(C1)C2=C(C=CC=C2C3=CC=CC=C3)C=C4C(=O)NC(=O)S4)N"}, {"DRUG NAME": "GSK2830371", "CID": 70983932, "SMILES": "CC1=C(C=C(C=N1)Cl)NCC2=CC=C(S2)C(=O)NC(CC3CCCC3)C(=O)NC4CC4"}, {"DRUG NAME": "Dacarbazine", "CID": 135398738, "SMILES": "CN(C)N=NC1=C(NC=N1)C(=O)N"}, {"DRUG NAME": "Temozolomide", "CID": 5394, "SMILES": "CN1C(=O)N2C=NC(=C2N=N1)C(=O)N"}, {"DRUG NAME": "Carmustine", "CID": 2578, "SMILES": "C(CCl)NC(=O)N(CCCl)N=O"}, {"DRUG NAME": "alpha-lipoic acid", "CID": 864, "SMILES": "C1CSSC1CCCCC(=O)O"}, {"DRUG NAME": "glutathione", "CID": 124886, "SMILES": "C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N"}, {"DRUG NAME": "N-acetyl cysteine", "CID": 12035, "SMILES": "CC(=O)NC(CS)C(=O)O"}, {"DRUG NAME": "ascorbate (vitamin c)", "CID": 54670067, "SMILES": "C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O"}]')

def get_local_paad_identity_map():
    """Return normalized drug-name -> {CID, SMILES} mapping."""
    out = {}
    for rec in _LOCAL_PAAD_IDENTITY_RECORDS:
        name = str(rec.get("DRUG NAME", "")).strip().lower()
        name = re.sub(r"[^a-z0-9]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            out[name] = {
                "CID": rec.get("CID"),
                "SMILES": rec.get("SMILES")
            }
    return out

LOCAL_PAAD_IDENTITY_MAP = get_local_paad_identity_map()

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
    """Convert IC50 values to positive numeric µM values."""
    try:
        values = (
            ic50_series
            .astype(str)
            .str.strip()
            .str.replace(">", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        return pd.to_numeric(values, errors="coerce")
    except Exception as e:
        st.error(f"Error processing IC50 values: {str(e)}")
        return None

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PAAD QSAR App (pIC50 ≥ 5.522879)",
    page_icon="🧪",
    layout="wide"
)

# -------------------------------------------------
# APP TITLE
# -------------------------------------------------
st.title("🧪 PAAD QSAR App (pIC50 ≥ 5.522879)")

st.caption("Server-safe Pandas identity handling enabled for CID/SMILES fields.")
st.markdown("""
This app performs QSAR analysis with three main tasks:
1. **SMILES to Descriptors**: Convert SMILES to molecular descriptors
2. **Similarity + MCC**: Calculate similarity and Matthews Correlation Coefficient
3. **ML + pIC50 ≥ 5.522879**: Machine learning prediction of active compounds
""")


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1️⃣ SMILES to Descriptors",
    "2️⃣ IC50 → pIC50 + Top-10 Similarity",
    "3️⃣ ML (9 Models) + pIC50 ≥ 5.522879"
])

# =========================================================
# TASK 1: SMILES to Descriptors
# =========================================================
with tab1:
    st.subheader("🔬 TASK 1: Convert SMILES to Descriptors")

    smiles_file = st.file_uploader(
        "Upload SMILES file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="task1_smiles_uploader"
    )

    if smiles_file:
        try:
            if smiles_file.name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(smiles_file)
            else:
                df = pd.read_csv(smiles_file)

            if df.empty:
                st.error("The uploaded SMILES file is empty.")
                st.stop()

            smiles_col = next(
                (col for col in df.columns if "smiles" in str(col).lower()),
                None
            )

            if smiles_col is None:
                st.error(
                    "No column containing 'SMILES' was found."
                )
                st.write("Available columns:", df.columns.tolist())
                st.stop()

            df = df.copy()
            df = df.rename(columns={smiles_col: "SMILES"})

            # Preserve the original identity fields.  In particular, keep
            # CID, SMILES and the original drug/compound/IUPAC name so that
            # Task 2 can transfer CID + SMILES into the pIC50 output.
            name_source = next(
                (
                    c for c in df.columns
                    if str(c).strip().lower() in {
                        "drug name", "drug_name", "drugname",
                        "compound name", "compound_name",
                        "molecule name", "molecule_name",
                        "iupacname", "iupac name", "iupac_name",
                        "name"
                    }
                ),
                None
            )
            if name_source is not None and "DRUG NAME" not in df.columns:
                df["DRUG NAME"] = df[name_source].astype(str)

            # Preserve ALL original columns. This is important because
            # Task 3 may need the original drug/compound name, CID or SMILES
            # for identity matching. Do not discard the source identity fields.
            descriptor_records = []

            for smi in df["SMILES"]:
                try:
                    descriptor_records.append(
                        smiles_to_simple_descriptors(smi)
                    )
                except Exception:
                    descriptor_records.append(
                        smiles_to_simple_descriptors("C")
                    )

            df_desc = pd.concat(
                [
                    df.reset_index(drop=True),
                    pd.DataFrame(descriptor_records)
                ],
                axis=1
            )

            st.subheader("Generated Descriptors")
            st.dataframe(
                df_desc.head(20),
                use_container_width=True
            )

            st.info(
                f"Generated descriptors for {len(df_desc):,} compounds. "
                "Original identity columns are preserved for TASK 3 matching."
            )

            numeric_desc = df_desc.select_dtypes(
                include=[np.number]
            )
            if not numeric_desc.empty:
                st.subheader("Descriptor Statistics")
                st.dataframe(
                    numeric_desc.describe(),
                    use_container_width=True
                )

            output = BytesIO()
            with pd.ExcelWriter(
                output,
                engine="openpyxl"
            ) as writer:
                df_desc.to_excel(
                    writer,
                    index=False,
                    sheet_name="Descriptors"
                )

            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    "📥 Download as Excel",
                    data=output.getvalue(),
                    file_name="PAAD_Descriptors_RDKit.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="task1_download_xlsx"
                )

            with c2:
                st.download_button(
                    "📥 Download as CSV",
                    data=df_desc.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="PAAD_Descriptors_RDKit.csv",
                    mime="text/csv",
                    key="task1_download_csv"
                )

        except Exception as e:
            st.error(f"❌ TASK 1 error: {str(e)}")
            st.exception(e)

# =========================================================
# TASK 2: IC50 → pIC50 → Activity + Top-10 Similarity
# =========================================================
with tab2:
    st.subheader(
        "📊 TASK 2: IC50 → pIC50 → Activity + Top-10 Similarity"
    )

    PIC50_ACTIVE_THRESHOLD = 5.522879

    st.markdown(f"""
    ### Activity rule used throughout the application

    - **pIC50 ≥ {PIC50_ACTIVE_THRESHOLD} → Active**
    - **pIC50 < {PIC50_ACTIVE_THRESHOLD} → Inactive**

    pIC50 is calculated as:

    **pIC50 = 6 − log10(IC50 in µM)**
    """)

    st.markdown("""
    **Upload:**
    - Training IC50 file: `IC50 value PAAD celline specific.xlsx`
    - Query descriptors: the file generated by TASK 1

    TASK 2 generates **`PAAD_IC50_pIC50.xlsx`**, which is the file
    uploaded to TASK 3.
    """)

    train_file = st.file_uploader(
        "Upload Training IC50 File",
        type=["xlsx", "xls", "csv"],
        key="task2_train_file"
    )

    query_file = st.file_uploader(
        "Upload Query Descriptors (from TASK 1)",
        type=["csv", "xlsx", "xls"],
        key="task2_query_file"
    )

    if train_file and query_file:
        with st.spinner("Calculating pIC50 and Top-10 similarities..."):
            try:
                if train_file.name.lower().endswith((".xlsx", ".xls")):
                    df_train = pd.read_excel(train_file)
                else:
                    df_train = pd.read_csv(train_file)

                if query_file.name.lower().endswith((".xlsx", ".xls")):
                    df_query = pd.read_excel(query_file)
                else:
                    df_query = pd.read_csv(query_file)

                if df_train.empty or df_query.empty:
                    st.error("One or both uploaded files are empty.")
                    st.stop()

                df_train = standardize_columns(df_train)
                df_query = standardize_columns(df_query)

                # If a previously generated Task-2 file is uploaded again,
                # remove old derived columns before recalculating them.
                df_train = drop_derived_activity_columns(df_train)

                # -------------------------------------------------
                # LOCAL DRUG NAME -> CID + SMILES TRANSFER
                # -------------------------------------------------
                # The original IC50 file contains DRUG NAME + IC50 only.
                # The Task-1 library contains CID + SMILES but its DRUG NAME
                # field is an IUPAC/systematic name.  Therefore an exact
                # name join cannot solve the problem.  A local PAAD identity
                # map is used here; no PubChem/network request is made.
                source_name_col_local = next(
                    (
                        c for c in df_train.columns
                        if str(c).strip().lower() in {
                            "drug name", "drug_name", "drugname",
                            "compound name", "compound_name",
                            "molecule name", "molecule_name",
                            "name"
                        }
                    ),
                    None
                )

                if source_name_col_local is not None:
                    def _local_name_key(v):
                        if v is None:
                            return ""
                        try:
                            if pd.isna(v):
                                return ""
                        except Exception:
                            pass
                        s = str(v).strip().lower()
                        s = re.sub(r"[^a-z0-9]+", " ", s)
                        return re.sub(r"\s+", " ", s).strip()

                    # IMPORTANT:
                    # CID and SMILES are identity fields, not numeric ML
                    # features.  Explicitly use object dtype so newer
                    # Pandas versions on Streamlit Cloud do not reject
                    # string SMILES values after a column starts as NaN.
                    if "cid" not in df_train.columns:
                        df_train["cid"] = pd.Series(
                            [None] * len(df_train),
                            index=df_train.index,
                            dtype="object"
                        )
                    else:
                        df_train["cid"] = df_train["cid"].astype("object")

                    if "smiles" not in df_train.columns:
                        df_train["smiles"] = pd.Series(
                            [None] * len(df_train),
                            index=df_train.index,
                            dtype="object"
                        )
                    else:
                        df_train["smiles"] = df_train["smiles"].astype("object")

                    # Build lists first and assign complete columns once.
                    # This avoids cell-by-cell dtype coercion with .at[].
                    cid_values = df_train["cid"].tolist()
                    smiles_values = df_train["smiles"].tolist()

                    local_identity_hits = 0

                    for pos, drug_name in enumerate(
                        df_train[source_name_col_local].tolist()
                    ):
                        key = _local_name_key(drug_name)
                        rec = LOCAL_PAAD_IDENTITY_MAP.get(key)

                        if rec is None:
                            continue

                        if is_missing_value(cid_values[pos]):
                            cid_values[pos] = rec.get("CID")

                        if is_missing_value(smiles_values[pos]):
                            smiles_values[pos] = rec.get("SMILES")

                        local_identity_hits += 1

                    df_train["cid"] = pd.Series(
                        cid_values,
                        index=df_train.index,
                        dtype="object"
                    )

                    df_train["smiles"] = pd.Series(
                        smiles_values,
                        index=df_train.index,
                        dtype="object"
                    )

                    st.info(
                        f"Local PAAD identity map resolved "
                        f"**{local_identity_hits:,} / {len(df_train):,}** "
                        "IC50 drug names to CID + SMILES. "
                        "No PubChem lookup is used."
                    )

                ic50_col = next(
                    (
                        col for col in df_train.columns
                        if "ic50" in str(col).lower()
                        and "pic50" not in str(col).lower()
                    ),
                    None
                )

                if ic50_col is None:
                    st.error(
                        "Could not find an IC50 column in the training file."
                    )
                    st.write(
                        "Available columns:",
                        df_train.columns.tolist()
                    )
                    st.stop()

                ic50_values = process_ic50_values(
                    df_train[ic50_col]
                )

                if ic50_values is None:
                    st.stop()

                # IC50 is expected in µM.
                valid_mask = (
                    pd.to_numeric(
                        ic50_values,
                        errors="coerce"
                    ).notna()
                    &
                    (
                        pd.to_numeric(
                            ic50_values,
                            errors="coerce"
                        ) > 0
                    )
                )

                df_train = df_train.loc[
                    valid_mask
                ].copy()

                ic50_values = pd.to_numeric(
                    ic50_values.loc[
                        valid_mask
                    ],
                    errors="coerce"
                )

                df_train["ic50 value"] = ic50_values.values

                # Requested concept:
                # pIC50 >= 5.522879 = Active
                # pIC50 <  5.522879 = Inactive
                df_train["pIC50"] = (
                    6.0 - np.log10(
                        df_train["ic50 value"]
                    )
                )

                df_train["Activity"] = np.where(
                    df_train["pIC50"] >= PIC50_ACTIVE_THRESHOLD,
                    "Active",
                    "Inactive"
                )

                df_train["Label"] = (
                    df_train["pIC50"] >= PIC50_ACTIVE_THRESHOLD
                ).astype(int)

                # -------------------------------------------------
                # TRANSFER CID + SMILES FROM TASK 1
                # -------------------------------------------------
                # Task 3 must not depend on PubChem.  The Task-1
                # descriptor file is therefore used here to attach the
                # structural identity to every IC50 record whenever a
                # reliable local key is available.
                def _norm_id(value):
                    if pd.isna(value):
                        return ""
                    s = str(value).strip()
                    try:
                        f = float(s)
                        if f.is_integer():
                            return str(int(f))
                    except Exception:
                        pass
                    return re.sub(r"[^0-9]", "", s)

                def _norm_smiles_local(value):
                    if pd.isna(value):
                        return ""
                    return re.sub(r"\\s+", "", str(value).strip()).lower()

                def _norm_name_local(value):
                    if pd.isna(value):
                        return ""
                    s = str(value).strip().lower()
                    s = re.sub(r"[^a-z0-9]+", " ", s)
                    return re.sub(r"\\s+", " ", s).strip()

                task1_name_col = next(
                    (
                        c for c in df_query.columns
                        if str(c).strip().lower() in {
                            "drug name", "drug_name", "drugname",
                            "compound name", "compound_name",
                            "molecule name", "molecule_name",
                            "iupacname", "iupac name", "iupac_name",
                            "name"
                        }
                    ),
                    None
                )
                task1_cid_col = next(
                    (
                        c for c in df_query.columns
                        if str(c).strip().lower() in {
                            "cid", "pubchem cid", "pubchem_cid",
                            "pubchemcid"
                        }
                    ),
                    None
                )
                task1_smiles_col = next(
                    (
                        c for c in df_query.columns
                        if str(c).strip().lower() in {
                            "smiles", "canonical_smiles",
                            "isomeric_smiles"
                        }
                    ),
                    None
                )

                source_name_col = next(
                    (
                        c for c in df_train.columns
                        if str(c).strip().lower() in {
                            "drug name", "drug_name", "drugname",
                            "compound name", "compound_name",
                            "molecule name", "molecule_name",
                            "iupacname", "iupac name", "iupac_name",
                            "name"
                        }
                    ),
                    None
                )
                source_cid_col = next(
                    (
                        c for c in df_train.columns
                        if str(c).strip().lower() in {
                            "cid", "pubchem cid", "pubchem_cid",
                            "pubchemcid"
                        }
                    ),
                    None
                )
                source_smiles_col = next(
                    (
                        c for c in df_train.columns
                        if str(c).strip().lower() in {
                            "smiles", "canonical_smiles",
                            "isomeric_smiles"
                        }
                    ),
                    None
                )

                # Create clean lookup dictionaries from Task-1 output.
                q_cid_map = {}
                q_smiles_map = {}
                q_name_map = {}

                if task1_cid_col:
                    for idx, value in df_query[task1_cid_col].items():
                        key = _norm_id(value)
                        if key:
                            q_cid_map.setdefault(key, idx)

                if task1_smiles_col:
                    for idx, value in df_query[task1_smiles_col].items():
                        key = _norm_smiles_local(value)
                        if key:
                            q_smiles_map.setdefault(key, idx)

                if task1_name_col:
                    for idx, value in df_query[task1_name_col].items():
                        key = _norm_name_local(value)
                        if key:
                            q_name_map.setdefault(key, idx)

                transferred_cid = []
                transferred_smiles = []
                transferred_method = []

                for _, row in df_train.iterrows():
                    found = None
                    method = ""

                    if source_cid_col:
                        key = _norm_id(row[source_cid_col])
                        if key in q_cid_map:
                            found = q_cid_map[key]
                            method = "CID"

                    if found is None and source_smiles_col:
                        key = _norm_smiles_local(row[source_smiles_col])
                        if key in q_smiles_map:
                            found = q_smiles_map[key]
                            method = "SMILES"

                    if found is None and source_name_col:
                        key = _norm_name_local(row[source_name_col])
                        if key in q_name_map:
                            found = q_name_map[key]
                            method = "Name"

                    if found is None:
                        transferred_cid.append(np.nan)
                        transferred_smiles.append(np.nan)
                        transferred_method.append("")
                    else:
                        if task1_cid_col:
                            transferred_cid.append(
                                df_query.loc[found, task1_cid_col]
                            )
                        else:
                            transferred_cid.append(np.nan)

                        if task1_smiles_col:
                            transferred_smiles.append(
                                df_query.loc[found, task1_smiles_col]
                            )
                        else:
                            transferred_smiles.append(np.nan)

                        transferred_method.append(method)

                # Only add these columns if they do not already exist.
                # Existing source identity is preserved.
                if source_cid_col is None:
                    df_train["CID"] = transferred_cid

                if source_smiles_col is None:
                    df_train["SMILES"] = transferred_smiles

                # Mark local PAAD-map identities as successful even when
                # the compound is not present in the 65,482 screening
                # descriptor library.
                final_identity_methods = list(transferred_method)

                if source_name_col_local is not None:
                    for pos, (_, row_local) in enumerate(df_train.iterrows()):
                        if str(final_identity_methods[pos]).strip():
                            continue
                        key_local = _local_name_key(
                            row_local[source_name_col_local]
                        )
                        if key_local in LOCAL_PAAD_IDENTITY_MAP:
                            final_identity_methods[pos] = (
                                "Local PAAD name → CID/SMILES"
                            )

                df_train["Identity_Match"] = final_identity_methods

                st.subheader("🔗 Task 1 Identity Transfer")
                identity_n = int(
                    pd.Series(final_identity_methods).astype(str).ne("").sum()
                )
                st.info(
                    f"Transferred Task-1 identity for **{identity_n:,} / "
                    f"{len(df_train):,}** IC50 compounds using local "
                    "CID → SMILES → normalized-name matching. "
                    "No PubChem lookup is used."
                )

                st.subheader("📊 Activity Classification")

                total_n = len(df_train)
                active_n = int(
                    (df_train["Label"] == 1).sum()
                )
                inactive_n = int(
                    (df_train["Label"] == 0).sum()
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Total compounds", f"{total_n:,}")
                c2.metric("Active", f"{active_n:,}")
                c3.metric("Inactive", f"{inactive_n:,}")

                # Display available identity column, preferably drug name.
                preferred_name = next(
                    (
                        c for c in [
                            "drug name",
                            "drug_name",
                            "compound name",
                            "compound_name",
                            "name"
                        ]
                        if c in df_train.columns
                    ),
                    df_train.columns[0]
                )

                display_cols = [
                    c for c in [
                        preferred_name,
                        "CID",
                        "SMILES",
                        ic50_col,
                        "pIC50",
                        "Activity",
                        "Label",
                        "Identity_Match"
                    ]
                    if c in df_train.columns
                ]

                st.dataframe(
                    df_train[display_cols],
                    use_container_width=True,
                    height=400
                )

                # -------------------------------------------------
                # Top-10 Similarity
                # -------------------------------------------------
                st.subheader("🔝 Top-10 Similarity")

                # Generate the same descriptor representation used by
                # Task 1 from the Task-2 SMILES.  This makes similarity
                # independent of whether the IC50 input originally
                # contained descriptor columns.
                train_smiles_col = "smiles" if "smiles" in df_train.columns else None

                if train_smiles_col:
                    train_descriptor_records = []
                    for smi in df_train[train_smiles_col]:
                        if pd.isna(smi) or not str(smi).strip():
                            train_descriptor_records.append({})
                        else:
                            train_descriptor_records.append(
                                smiles_to_simple_descriptors(smi)
                            )

                    train_sim_desc = pd.DataFrame(
                        train_descriptor_records
                    )
                else:
                    train_sim_desc = pd.DataFrame()

                query_numeric = df_query.select_dtypes(
                    include=[np.number]
                ).copy()

                # Never use identity, activity or IC50 columns for similarity.
                excluded_similarity = {
                    "cid", "pubchem_cid", "pubchemcid",
                    "ic50", "ic50 value", "pic50",
                    "label", "activity", "id"
                }

                common_cols = [
                    c for c in train_sim_desc.columns
                    if c in query_numeric.columns
                    and str(c).lower() not in excluded_similarity
                ]

                if common_cols and not train_sim_desc.empty:
                    X_train_sim = train_sim_desc[
                        common_cols
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).fillna(0)

                    X_query_sim = query_numeric[
                        common_cols
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan
                    ).fillna(0)

                    similarity = cosine_similarity(
                        X_train_sim.values,
                        X_query_sim.values
                    )

                    k = min(10, len(X_query_sim))
                    results = []

                    preferred_name = next(
                        (
                            c for c in [
                                "drug name",
                                "drug_name",
                                "compound name",
                                "compound_name",
                                "name"
                            ]
                            if c in df_train.columns
                        ),
                        None
                    )

                    for train_pos in range(
                        similarity.shape[0]
                    ):
                        top_idx = np.argsort(
                            similarity[train_pos]
                        )[-k:][::-1]

                        for rank, query_pos in enumerate(
                            top_idx,
                            start=1
                        ):
                            result = {
                                "Training_Compound": (
                                    df_train.iloc[train_pos][
                                        preferred_name
                                    ]
                                    if preferred_name
                                    else train_pos + 1
                                ),
                                "Rank": rank,
                                "Library_Row": int(query_pos + 1),
                                "Similarity": float(
                                    similarity[
                                        train_pos,
                                        query_pos
                                    ]
                                ),
                                "IC50_Value": float(
                                    df_train.iloc[train_pos][
                                        "ic50 value"
                                    ]
                                ),
                                "pIC50": float(
                                    df_train.iloc[train_pos][
                                        "pIC50"
                                    ]
                                ),
                                "Activity": df_train.iloc[
                                    train_pos
                                ]["Activity"]
                            }

                            if "cid" in df_train.columns:
                                result["CID"] = (
                                    df_train.iloc[train_pos]["cid"]
                                )

                            results.append(result)

                    results_df = pd.DataFrame(results)

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        height=500
                    )

                    st.caption(
                        "Top-10 structurally similar compounds are shown "
                        "for each uploaded PAAD experimental compound. "
                        "Similarity uses the common Task-1 descriptor "
                        "representation and does not use IC50, pIC50, "
                        "Activity, Label or CID."
                    )
                else:
                    results_df = pd.DataFrame()
                    st.warning(
                        "Top-10 similarity could not be calculated because "
                        "Task-2 SMILES or common Task-1 descriptor columns "
                        "were unavailable."
                    )

                # -------------------------------------------------
                # TASK 2 output: only the pIC50/activity data needed
                # by Task 3. Preserve identity columns from source.
                # -------------------------------------------------
                output = BytesIO()

                with pd.ExcelWriter(
                    output,
                    engine="openpyxl"
                ) as writer:
                    df_train.to_excel(
                        writer,
                        sheet_name="pIC50_Activity",
                        index=False
                    )

                    if not results_df.empty:
                        results_df.to_excel(
                            writer,
                            sheet_name="Top10_Similarity",
                            index=False
                        )

                st.success(
                    f"✅ TASK 2 completed. "
                    f"{total_n:,} compounds processed. "
                    "CID/SMILES identity was carried forward where available. "
                    "The output is ready for TASK 3."
                )

                st.download_button(
                    "📥 Download PAAD_IC50_pIC50.xlsx",
                    data=output.getvalue(),
                    file_name="PAAD_IC50_pIC50.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="task2_download_pic50"
                )

            except Exception as e:
                st.error(f"❌ TASK 2 error: {str(e)}")
                st.exception(e)

# =========================================================
# TASK 3: ML + pIC50 ≥ 5.522879 + CV-MCC + Virtual Screening
# =========================================================
with tab3:
    st.subheader(
        "🤖 TASK 3: ML + Cross-Validation MCC + Virtual Screening"
    )

    PIC50_ACTIVE_THRESHOLD = 5.522879

    st.markdown(f"""
    ### Upload exactly TWO files

    **File 1:** `PAAD_Descriptors_RDKit.csv` generated by TASK 1.

    **File 2:** `PAAD_IC50_pIC50.xlsx` generated by TASK 2.

    **Activity rule:**
    - **pIC50 ≥ {PIC50_ACTIVE_THRESHOLD} → Active**
    - **pIC50 < {PIC50_ACTIVE_THRESHOLD} → Inactive**

    The number of pIC50 compounds is determined **only from the uploaded
    file**. No fixed number such as 203 is used.

    Descriptor columns are ML features.

    CID, SMILES and compound names are used only for identity matching
    and are never used as ML features.

    **No PubChem lookup is required.** Task 3 uses CID/SMILES carried
    in the Task-2 file. If an experimental compound is not present in
    the 65,482-compound screening library, the same Task-1 descriptor
    representation is generated locally from its supplied SMILES for
    ML training only.

    The complete Task-1 descriptor library is screened, but the GUI
    displays only the **Top 200 predicted Active candidates**.
    """)

    desc_file_ml = st.file_uploader(
        "File 1 — Descriptor file from TASK 1",
        type=["csv", "xlsx", "xls"],
        key="task3_desc_file"
    )

    pic50_file_ml = st.file_uploader(
        "File 2 — pIC50 file from TASK 2",
        type=["xlsx", "xls", "csv"],
        key="task3_pic50_file"
    )

    if desc_file_ml and pic50_file_ml:
        with st.spinner(
            "Matching compounds and preparing ML data..."
        ):
            try:
                # -------------------------------------------------
                # Read files
                # -------------------------------------------------
                if desc_file_ml.name.lower().endswith(
                    (".xlsx", ".xls")
                ):
                    df_desc = pd.read_excel(
                        desc_file_ml
                    )
                else:
                    df_desc = pd.read_csv(
                        desc_file_ml
                    )

                if pic50_file_ml.name.lower().endswith(
                    (".xlsx", ".xls")
                ):
                    xl = pd.ExcelFile(
                        pic50_file_ml
                    )

                    preferred_sheet = next(
                        (
                            s for s in xl.sheet_names
                            if str(s).strip().lower()
                            in {
                                "pic50_activity",
                                "pic50",
                                "activity"
                            }
                        ),
                        xl.sheet_names[0]
                    )

                    df_pic50 = pd.read_excel(
                        pic50_file_ml,
                        sheet_name=preferred_sheet
                    )
                else:
                    df_pic50 = pd.read_csv(
                        pic50_file_ml
                    )

                if df_desc.empty:
                    st.error(
                        "The descriptor file is empty."
                    )
                    st.stop()

                if df_pic50.empty:
                    st.error(
                        "The pIC50 file is empty."
                    )
                    st.stop()

                df_desc = standardize_columns(
                    df_desc
                )
                df_pic50 = standardize_columns(
                    df_pic50
                )

                st.subheader("📁 Input Data")

                c1, c2 = st.columns(2)
                c1.metric(
                    "Descriptor compounds",
                    f"{len(df_desc):,}"
                )
                c2.metric(
                    "pIC50 compounds",
                    f"{len(df_pic50):,}"
                )

                # -------------------------------------------------
                # Find pIC50 directly from Task 2.
                # -------------------------------------------------
                pic50_col = next(
                    (
                        c for c in df_pic50.columns
                        if str(c).strip().lower() == "pic50"
                    ),
                    None
                )

                if pic50_col is None:
                    pic50_col = next(
                        (
                            c for c in df_pic50.columns
                            if "pic50" in str(c).lower()
                        ),
                        None
                    )

                if pic50_col is None:
                    st.error(
                        "The uploaded Task 2 file does not contain "
                        "a pIC50 column."
                    )
                    st.write(
                        "Available columns:",
                        df_pic50.columns.tolist()
                    )
                    st.stop()

                df_pic50["pIC50"] = pd.to_numeric(
                    df_pic50[pic50_col],
                    errors="coerce"
                )

                df_pic50 = df_pic50[
                    df_pic50["pIC50"].notna()
                ].copy()

                # Re-create Activity/Label from pIC50.
                # This ensures the requested threshold is always used.
                df_pic50["Label"] = (
                    df_pic50["pIC50"]
                    >= PIC50_ACTIVE_THRESHOLD
                ).astype(int)

                df_pic50["Activity"] = np.where(
                    df_pic50["Label"] == 1,
                    "Active",
                    "Inactive"
                )

                # -------------------------------------------------
                # Identity helpers
                # -------------------------------------------------
                def norm_text(value):
                    if pd.isna(value):
                        return ""
                    value = str(value).strip().lower()
                    value = re.sub(
                        r"\s+",
                        " ",
                        value
                    )
                    return value

                def norm_cid(value):
                    if pd.isna(value):
                        return ""
                    text = str(value).strip()

                    try:
                        number = float(text)
                        if number.is_integer():
                            return str(
                                int(number)
                            )
                    except Exception:
                        pass

                    return re.sub(
                        r"[^0-9]",
                        "",
                        text
                    )

                def norm_smiles(value):
                    if pd.isna(value):
                        return ""
                    return re.sub(
                        r"\s+",
                        "",
                        str(value).strip()
                    ).lower()

                def find_col(
                    df,
                    candidates
                ):
                    for candidate in candidates:
                        if candidate in df.columns:
                            return candidate
                    return None

                cid_candidates = [
                    "cid",
                    "pubchem_cid",
                    "pubchemcid"
                ]

                smiles_candidates = [
                    "smiles",
                    "canonical_smiles",
                    "isomeric_smiles"
                ]

                name_candidates = [
                    "drug name",
                    "drug_name",
                    "drugname",
                    "compound name",
                    "compound_name",
                    "compoundname",
                    "molecule name",
                    "molecule_name",
                    "moleculename",
                    "iupacname",
                    "iupac_name",
                    "name"
                ]

                desc_cid = find_col(
                    df_desc,
                    cid_candidates
                )
                pic50_cid = find_col(
                    df_pic50,
                    cid_candidates
                )

                desc_smiles = find_col(
                    df_desc,
                    smiles_candidates
                )
                pic50_smiles = find_col(
                    df_pic50,
                    smiles_candidates
                )

                desc_names = [
                    c for c in name_candidates
                    if c in df_desc.columns
                ]

                pic50_names = [
                    c for c in name_candidates
                    if c in df_pic50.columns
                ]

                # -------------------------------------------------
                # ROBUST IDENTITY MATCHING + TRAINING-DESCRIPTOR FALLBACK
                # -------------------------------------------------
                # First use the uploaded Task-1 descriptor library:
                #   1) CID
                #   2) exact SMILES
                #   3) exact normalized name
                #
                # IMPORTANT:
                # The supplied 65,482-compound library does not contain
                # all 203 experimental PAAD compounds.  Therefore, when a
                # Task-2 compound has a valid SMILES but is absent from the
                # screening library, its descriptors are generated locally
                # from that SMILES for ML TRAINING ONLY.
                #
                # The 65,482 Task-1 compounds are still the complete
                # VIRTUAL-SCREENING library.
                # -------------------------------------------------

                def norm_text(value):
                    if pd.isna(value):
                        return ""
                    value = str(value).strip().lower()
                    value = re.sub(r"[^a-z0-9]+", " ", value)
                    return re.sub(r"\s+", " ", value).strip()

                def norm_cid(value):
                    if pd.isna(value):
                        return ""
                    s = str(value).strip()
                    try:
                        number = float(s)
                        if number.is_integer():
                            return str(int(number))
                    except Exception:
                        pass
                    return re.sub(r"[^0-9]", "", s)

                def norm_smiles(value):
                    if pd.isna(value):
                        return ""
                    return re.sub(r"\s+", "", str(value).strip()).lower()

                def find_col(df, candidates):
                    for candidate in candidates:
                        if candidate in df.columns:
                            return candidate
                    return None

                cid_candidates = [
                    "cid", "pubchem_cid", "pubchemcid"
                ]
                smiles_candidates = [
                    "smiles", "canonical_smiles", "isomeric_smiles"
                ]
                name_candidates = [
                    "drug name", "drug_name", "drugname",
                    "compound name", "compound_name", "compoundname",
                    "molecule name", "molecule_name", "moleculename",
                    "iupacname", "iupac_name", "name"
                ]

                desc_cid = find_col(df_desc, cid_candidates)
                pic50_cid = find_col(df_pic50, cid_candidates)

                desc_smiles = find_col(df_desc, smiles_candidates)
                pic50_smiles = find_col(df_pic50, smiles_candidates)

                desc_names = [
                    c for c in name_candidates if c in df_desc.columns
                ]
                pic50_names = [
                    c for c in name_candidates if c in df_pic50.columns
                ]

                # Descriptor-library lookup maps.
                cid_map = {}
                if desc_cid:
                    for idx, value in df_desc[desc_cid].items():
                        key = norm_cid(value)
                        if key:
                            cid_map.setdefault(key, idx)

                smiles_map = {}
                if desc_smiles:
                    for idx, value in df_desc[desc_smiles].items():
                        key = norm_smiles(value)
                        if key:
                            smiles_map.setdefault(key, idx)

                name_map = {}
                for col in desc_names:
                    for idx, value in df_desc[col].items():
                        key = norm_text(value)
                        if key:
                            name_map.setdefault(key, idx)

                # -------------------------------------------------
                # Build one training descriptor row for EVERY pIC50
                # compound for which a reliable SMILES is available.
                # -------------------------------------------------
                training_desc_rows = []
                training_pic_rows = []
                training_methods = []

                direct_cid_n = 0
                exact_smiles_n = 0
                exact_name_n = 0
                generated_desc_n = 0
                unresolved_n = 0

                for pic_idx, row in df_pic50.iterrows():
                    found_idx = None
                    method = None

                    # 1. Direct CID against Task-1 library.
                    if pic50_cid and desc_cid:
                        key = norm_cid(row[pic50_cid])
                        if key and key in cid_map:
                            found_idx = cid_map[key]
                            method = "Direct CID"

                    # 2. Exact SMILES against Task-1 library.
                    if found_idx is None and pic50_smiles and desc_smiles:
                        key = norm_smiles(row[pic50_smiles])
                        if key and key in smiles_map:
                            found_idx = smiles_map[key]
                            method = "Exact SMILES"

                    # 3. Exact name against Task-1 library.
                    if found_idx is None:
                        for col in pic50_names:
                            key = norm_text(row[col])
                            if key and key in name_map:
                                found_idx = name_map[key]
                                method = "Exact Name"
                                break

                    if found_idx is not None:
                        descriptor_row = df_desc.loc[found_idx].copy()

                        if method == "Direct CID":
                            direct_cid_n += 1
                        elif method == "Exact SMILES":
                            exact_smiles_n += 1
                        elif method == "Exact Name":
                            exact_name_n += 1

                    else:
                        # -------------------------------------------------
                        # Training-only fallback:
                        # generate the SAME descriptor representation used
                        # by Task 1 directly from the Task-2 SMILES.
                        # -------------------------------------------------
                        smi = (
                            row[pic50_smiles]
                            if pic50_smiles
                            else np.nan
                        )

                        if pd.isna(smi) or not str(smi).strip():
                            unresolved_n += 1
                            continue

                        descriptor_row = pd.Series(
                            index=df_desc.columns,
                            dtype=object
                        )

                        # Preserve identity where possible.
                        if desc_cid:
                            descriptor_row[desc_cid] = (
                                row[pic50_cid]
                                if pic50_cid
                                else np.nan
                            )

                        if desc_smiles:
                            descriptor_row[desc_smiles] = smi

                        # Generate the same simple descriptor columns as
                        # Task 1.
                        generated = smiles_to_simple_descriptors(smi)

                        for key, value in generated.items():
                            if key in descriptor_row.index:
                                descriptor_row[key] = value

                        generated_desc_n += 1
                        method = "Generated from Task-2 SMILES"

                    training_desc_rows.append(descriptor_row)
                    training_pic_rows.append(row.copy())
                    training_methods.append(method)

                if not training_desc_rows:
                    st.error(
                        "No usable training compounds were found. "
                        "Task 2 must contain valid SMILES/CID information "
                        "for the experimental compounds."
                    )
                    st.stop()

                matched_desc = pd.DataFrame(
                    training_desc_rows
                ).reset_index(drop=True)

                matched_pic = pd.DataFrame(
                    training_pic_rows
                ).reset_index(drop=True)

                matched_pic["Identity_Match"] = training_methods

                y = matched_pic["Label"].astype(int).to_numpy()

                active_n = int((y == 1).sum())
                inactive_n = int((y == 0).sum())

                total_training = len(matched_pic)

                # -------------------------------------------------
                # Identity Matching Summary
                # -------------------------------------------------
                st.subheader("🔗 Identity Matching & Training Summary")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Training compounds", f"{total_training:,}")
                m2.metric("Direct CID", f"{direct_cid_n:,}")
                m3.metric("Exact SMILES", f"{exact_smiles_n:,}")
                m4.metric("Exact Name", f"{exact_name_n:,}")
                m5.metric(
                    "Generated descriptors",
                    f"{generated_desc_n:,}"
                )

                if unresolved_n:
                    st.warning(
                        f"{unresolved_n:,} pIC50 compounds were excluded "
                        "because neither CID/SMILES matching nor a valid "
                        "Task-2 SMILES was available."
                    )

                if generated_desc_n:
                    st.info(
                        f"{generated_desc_n:,} experimental compounds were "
                        "not present in the 65,482-compound screening library. "
                        "Their descriptors were generated from the supplied "
                        "Task-2 SMILES for ML training only. The complete "
                        "Task-1 library is still used for virtual screening."
                    )

                st.success(
                    f"ML training set prepared: **{total_training:,} "
                    f"compounds** ({active_n:,} Active, "
                    f"{inactive_n:,} Inactive)."
                )

                # Show a compact training identity table.
                show_cols = [
                    c for c in [
                        "drug name", "cid", "smiles",
                        "ic50 value", "pIC50",
                        "Activity", "Label", "Identity_Match"
                    ]
                    if c in matched_pic.columns
                ]
                if show_cols:
                    st.dataframe(
                        matched_pic[show_cols].head(200),
                        use_container_width=True,
                        height=420
                    )

                # Use the prepared training matrices from above.
                # These include both compounds already present in Task 1
                # and compounds whose descriptors were generated from the
                # supplied Task-2 SMILES.

                # -------------------------------------------------
                # ML features
                # -------------------------------------------------
                excluded_features = {
                    "cid",
                    "pubchem_cid",
                    "pubchemcid",
                    "pic50",
                    "label",
                    "activity",
                    "id",
                    "smiles",
                    "canonical_smiles",
                    "isomeric_smiles",
                    "drug name",
                    "drug_name",
                    "drugname",
                    "compound name",
                    "compound_name",
                    "compoundname",
                    "molecule name",
                    "molecule_name",
                    "moleculename",
                    "iupacname",
                    "iupac_name",
                    "name"
                }

                numeric_cols = matched_desc.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

                feature_cols = [
                    c for c in numeric_cols
                    if str(c).lower()
                    not in excluded_features
                    and not str(c).startswith("_")
                ]

                X = matched_desc[
                    feature_cols
                ].replace(
                    [np.inf, -np.inf],
                    np.nan
                ).fillna(0)

                # Remove constant features.
                constant_cols = [
                    c for c in X.columns
                    if X[c].nunique(
                        dropna=False
                    ) <= 1
                ]

                if constant_cols:
                    X = X.drop(
                        columns=constant_cols
                    )

                if X.shape[1] == 0:
                    st.error(
                        "No usable numeric descriptor features "
                        "remain after removing constant columns."
                    )
                    st.stop()

                st.info(
                    f"ML features used: {X.shape[1]:,}. "
                    "CID/SMILES/name/pIC50/Label are excluded."
                )

                # -------------------------------------------------
                # Cross-validation setup
                # -------------------------------------------------
                min_class = min(
                    active_n,
                    inactive_n
                )

                if min_class < 2:
                    st.error(
                        "At least 2 Active and 2 Inactive compounds are "
                        "required for stratified cross-validation."
                    )
                    st.stop()

                n_splits = min(
                    5,
                    min_class
                )

                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42
                )

                smote_k = max(
                    1,
                    min(
                        5,
                        min_class - 1
                    )
                )

                models = {
                    "Logistic Regression": LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42
                    ),
                    "SVM": SVC(
                        class_weight="balanced",
                        probability=True,
                        random_state=42
                    ),
                    "Decision Tree": DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=42
                    ),
                    "Random Forest": RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1
                    ),
                    "Naive Bayes": GaussianNB(),
                    "MLP": MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        early_stopping=True,
                        random_state=42
                    ),
                    "Gradient Boosting": GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    ),
                    "XGBoost": XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1
                    ) if XGBOOST_OK else None,
                    "LightGBM": LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        num_leaves=31,
                        verbosity=-1,
                        random_state=42,
                        n_jobs=-1
                    ) if LIGHTGBM_OK else None,
                }

                unavailable_models = [
                    name for name, model in models.items()
                    if model is None
                ]
                models = {
                    name: model
                    for name, model in models.items()
                    if model is not None
                }

                if unavailable_models:
                    st.warning(
                        "Optional ML libraries are not installed for: "
                        + ", ".join(unavailable_models)
                        + ". Install xgboost and lightgbm "
                          "to run all 9 models."
                    )


                # -------------------------------------------------
                # ML + CV-MCC
                # -------------------------------------------------
                st.subheader(
                    f"🤖 {n_splits}-Fold Cross-Validation + MCC"
                )

                scoring = {
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                    "mcc": "matthews_corrcoef"
                }

                model_results = []
                fitted_models = {}

                for model_name, model in models.items():
                    try:
                        pipeline = ImbPipeline([
                            (
                                "smote",
                                SMOTE(
                                    random_state=42,
                                    k_neighbors=smote_k
                                )
                            ),
                            (
                                "scaler",
                                StandardScaler()
                            ),
                            (
                                "model",
                                model
                            )
                        ])

                        scores = cross_validate(
                            pipeline,
                            X,
                            y,
                            cv=cv,
                            scoring=scoring,
                            return_train_score=False,
                            n_jobs=-1,
                            error_score="raise"
                        )

                        model_results.append({
                            "Model": model_name,
                            "Accuracy":
                                np.mean(
                                    scores[
                                        "test_accuracy"
                                    ]
                                ),
                            "Precision":
                                np.mean(
                                    scores[
                                        "test_precision"
                                    ]
                                ),
                            "Recall":
                                np.mean(
                                    scores[
                                        "test_recall"
                                    ]
                                ),
                            "F1":
                                np.mean(
                                    scores[
                                        "test_f1"
                                    ]
                                ),
                            "ROC_AUC":
                                np.mean(
                                    scores[
                                        "test_roc_auc"
                                    ]
                                ),
                            "MCC":
                                np.mean(
                                    scores[
                                        "test_mcc"
                                    ]
                                )
                        })

                        fitted_models[
                            model_name
                        ] = pipeline

                    except Exception as model_error:
                        st.warning(
                            f"{model_name} failed: "
                            f"{model_error}"
                        )

                if not model_results:
                    st.error(
                        "All ML models failed."
                    )
                    st.stop()

                performance_df = pd.DataFrame(
                    model_results
                ).sort_values(
                    "MCC",
                    ascending=False
                ).reset_index(
                    drop=True
                )

                display_perf = performance_df.copy()

                for col in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC_AUC",
                    "MCC"
                ]:
                    display_perf[col] = (
                        display_perf[col]
                        .round(4)
                    )

                st.dataframe(
                    display_perf,
                    use_container_width=True
                )

                # -------------------------------------------------
                # Best model by CV-MCC
                # -------------------------------------------------
                best_model_name = (
                    performance_df.iloc[0]["Model"]
                )

                best_model = fitted_models[
                    best_model_name
                ]

                best_mcc = float(
                    performance_df.iloc[0]["MCC"]
                )

                st.success(
                    f"🏆 Best Model: {best_model_name} "
                    f"| CV-MCC = {best_mcc:.4f}"
                )

                # Fit best model using ALL matched training compounds.
                best_model.fit(
                    X,
                    y
                )

                # -------------------------------------------------
                # Virtual Screening
                # -------------------------------------------------
                st.subheader(
                    "🔎 Virtual Screening"
                )

                all_numeric = df_desc.select_dtypes(
                    include=[np.number]
                ).copy()

                for col in X.columns:
                    if col not in all_numeric.columns:
                        all_numeric[col] = 0

                X_screen = all_numeric.reindex(
                    columns=X.columns,
                    fill_value=0
                ).replace(
                    [np.inf, -np.inf],
                    np.nan
                ).fillna(0)

                screen_pred = best_model.predict(
                    X_screen
                )

                if hasattr(
                    best_model,
                    "predict_proba"
                ):
                    screen_prob = (
                        best_model.predict_proba(
                            X_screen
                        )[:, 1]
                    )
                elif hasattr(
                    best_model,
                    "decision_function"
                ):
                    decision = (
                        best_model.decision_function(
                            X_screen
                        )
                    )
                    screen_prob = (
                        1.0
                        / (
                            1.0
                            + np.exp(
                                -np.clip(
                                    decision,
                                    -50,
                                    50
                                )
                            )
                        )
                    )
                else:
                    screen_prob = np.asarray(
                        screen_pred,
                        dtype=float
                    )

                screening_df = df_desc.copy()

                screening_df[
                    "Predicted_Probability_Active"
                ] = screen_prob

                screening_df[
                    "Predicted_Activity"
                ] = np.where(
                    screen_pred == 1,
                    "Active",
                    "Inactive"
                )

                # -------------------------------------------------
                # Complete library is screened.
                # Only Top 200 Active candidates are displayed.
                # -------------------------------------------------
                active_candidates = screening_df[
                    screening_df[
                        "Predicted_Activity"
                    ] == "Active"
                ].copy()

                active_candidates = (
                    active_candidates.sort_values(
                        "Predicted_Probability_Active",
                        ascending=False
                    )
                )

                top200 = active_candidates.head(
                    200
                ).copy()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Descriptor Compounds",
                    f"{len(df_desc):,}"
                )

                c2.metric(
                    "Predicted Active",
                    f"{len(active_candidates):,}"
                )

                c3.metric(
                    "Displayed",
                    f"{len(top200):,}"
                )

                st.subheader(
                    "🏆 Top 200 Predicted Active Candidates"
                )

                st.dataframe(
                    top200,
                    use_container_width=True,
                    height=600
                )

                st.info(
                    f"The complete library of "
                    f"{len(df_desc):,} compounds was screened. "
                    f"Only the top {len(top200):,} predicted Active "
                    "candidates are displayed."
                )

                # -------------------------------------------------
                # Download report
                # -------------------------------------------------
                output = BytesIO()

                with pd.ExcelWriter(
                    output,
                    engine="openpyxl"
                ) as writer:

                    performance_df.to_excel(
                        writer,
                        sheet_name="CV_MCC_Performance",
                        index=False
                    )

                    matched_training = (
                        matched_desc.copy()
                    )

                    matched_training["pIC50"] = (
                        matched_pic[
                            "pIC50"
                        ].values
                    )

                    matched_training["Activity"] = (
                        matched_pic[
                            "Activity"
                        ].values
                    )

                    matched_training["Label"] = (
                        matched_pic[
                            "Label"
                        ].values
                    )

                    matched_training[
                        "Match_Method"
                    ] = matched_pic["Identity_Match"].tolist()

                    matched_training.to_excel(
                        writer,
                        sheet_name="Matched_Training",
                        index=False
                    )

                    top200.to_excel(
                        writer,
                        sheet_name="Top_200_Active",
                        index=False
                    )

                    pd.DataFrame({
                        "Metric": [
                            "Descriptor compounds",
                            "pIC50 compounds in uploaded file",
                            "Matched compounds",
                            "Unmatched pIC50 compounds",
                            "Active training compounds",
                            "Inactive training compounds",
                            "ML features",
                            "CV folds",
                            "Best model",
                            "Best CV MCC",
                            "Predicted Active candidates",
                            "Displayed candidates"
                        ],
                        "Value": [
                            len(df_desc),
                            len(df_pic50),
                            total_training,
                            unresolved_n,
                            active_n,
                            inactive_n,
                            X.shape[1],
                            n_splits,
                            best_model_name,
                            round(
                                best_mcc,
                                4
                            ),
                            len(
                                active_candidates
                            ),
                            len(top200)
                        ]
                    }).to_excel(
                        writer,
                        sheet_name="Summary",
                        index=False
                    )

                st.download_button(
                    "📥 Download TASK 3 Report",
                    data=output.getvalue(),
                    file_name=(
                        "PAAD_pIC50_ML_CV_MCC_Top200.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="task3_download_report"
                )

            except Exception as e:
                st.error(
                    f"❌ TASK 3 error: {str(e)}"
                )
                st.exception(e)
