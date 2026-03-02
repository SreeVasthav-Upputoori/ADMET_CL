import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def smiles_to_features(smiles_list, radius=2, n_bits=2048):
    rows = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            rows.append([np.nan] * (n_bits + 18))
            continue

        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits
        )
        fp = list(fp)

        # Physicochemical descriptors (18)
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NOCount(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol),
            Descriptors.BertzCT(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
        ]

        rows.append(fp + desc)

    columns = [f"fp_{i}" for i in range(n_bits)] + [f"desc_{i}" for i in range(18)]
    return pd.DataFrame(rows, columns=columns)
