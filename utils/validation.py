import pandas as pd

def validate_smiles_input(df):
    if "SMILES" not in df.columns:
        raise ValueError("Input must contain a 'SMILES' column")

    if df["SMILES"].isnull().any():
        raise ValueError("SMILES column contains missing values")

    return True
