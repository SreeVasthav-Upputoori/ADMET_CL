import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import io

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_DIR = "models"
N_BITS = 2048

MODEL_FILES = {
    "Ames Mutagenicity": "ames_lgbm_model.pkl",
    "Blood-Brain Barrier": "bbb_lgbm_model.pkl",
    "Carcinogenicity": "carcinogenicity_lgbm_model.pkl",
    "CYP2C9": "cyp2c9_lgbm_model.pkl",
    "CYP2D6": "cyp2d6_lgbm_model.pkl",
    "CYP3A4": "cyp3a4_lgbm_model.pkl",
    "Eye Irritation": "eye_irritation_lgbm_model.pkl",
    "hERG Blockade": "hERG_lgbm_model.pkl",
    "Human Intestinal Absorption": "hia_lgbm_model.pkl",
    "P-gp Inhibitor": "pgp_inhibitor_lgbm_model.pkl",
    "Oral Bioavailability": "ob_lgbm_model.pkl",
    "Nephrotoxicity": "Nephro_lgbm_model.pkl"
}

# ==========================================================
# EXACT TRAINING FEATURE PIPELINE (DO NOT MODIFY)
# ==========================================================

def aromatic_ring_count(mol):
    ri = mol.GetRingInfo()
    count = 0
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            count += 1
    return count

def saturated_ring_count(mol):
    ri = mol.GetRingInfo()
    count = 0
    for ring in ri.AtomRings():
        if all(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            count += 1
    return count

def aromatic_heterocycle_count(mol):
    ri = mol.GetRingInfo()
    count = 0
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6,) for i in ring):
                count += 1
    return count

def compute_physchem(mol):
    try:
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcLabuteASA(mol),
            Descriptors.MolMR(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            aromatic_ring_count(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            mol.GetRingInfo().NumRings(),
            aromatic_heterocycle_count(mol),
            saturated_ring_count(mol),
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),  # DUPLICATE (intentional – from training)
            Descriptors.qed(mol),
        ]
    except:
        desc = [0.0] * 18

    return np.array(desc, dtype=np.float32)

def ecfp4(mol, nBits=N_BITS):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = ecfp4(mol)
    phys = compute_physchem(mol)

    features = np.concatenate([fp, phys]).astype(np.float32)

    # Safety check: must be 2066
    if features.shape[0] != 2066:
        raise ValueError(f"Feature size mismatch: got {features.shape[0]}, expected 2066")

    return features

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(page_title="ADMET Multi-Predictor", layout="wide")
st.title("🧪 ADMET Prediction Dashboard")
st.markdown("LightGBM-based ADMET property prediction.")

# Sidebar
input_mode = st.sidebar.radio("Input Method", ["Manual SMILES", "Upload CSV"])

@st.cache_resource
def load_all_models():
    loaded = {}
    for name, file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, file)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded

models = load_all_models()

if not models:
    st.error(f"No models found inside '{MODEL_DIR}' directory.")
    st.stop()

# ==========================================================
# INPUT HANDLING
# ==========================================================

smiles_list = []

if input_mode == "Manual SMILES":
    smi = st.text_input("Enter SMILES:", "c1ccccc1")
    if smi:
        smiles_list = [smi]

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        cols = {c.upper(): c for c in df_upload.columns}
        if "SMILES" in cols:
            smiles_list = df_upload[cols["SMILES"]].astype(str).tolist()
        else:
            st.error("CSV must contain a 'SMILES' column.")

# ==========================================================
# PREDICTION
# ==========================================================

if smiles_list and st.button("Generate Predictions"):

    results = []

    with st.spinner("Generating features and running models..."):

        for smi in smiles_list:
            row = {"SMILES": smi}
            features = smiles_to_features(smi)

            if features is None:
                for name in models.keys():
                    row[name] = "Invalid SMILES"
            else:
                X = features.reshape(1, -1)

                for name, model in models.items():
                    pred = model.predict(X)[0]
                    row[name] = "Positive" if pred == 1 else "Negative"

            results.append(row)

    results_df = pd.DataFrame(results)

    st.success("Prediction Complete.")
    st.dataframe(results_df, use_container_width=True)

    # Export to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="ADMET_Predictions")

    st.download_button(
        label="Download Results (Excel)",
        data=output.getvalue(),
        file_name="admet_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )