import pandas as pd
from typing import List, Tuple, Union, Optional, Callable
from rdkit import Chem
from FPSim2 import FPSim2Engine
import sqlite3


class ChemblData:
    FP_PATH = "/home/rhjvanworkum/chembl_32.h5"
    DB_PATH = "/home/rhjvanworkum/chembl_32/chembl_32_sqlite/chembl_32.db"
    DF_PATH = "/home/rhjvanworkum/chembl_32_chemreps.txt"

    def __init__(self, n_workers: int = 4) -> None:
        self.fpe = FPSim2Engine(self.FP_PATH)
        self.n_workers = n_workers

        self.conn = sqlite3.connect(self.DB_PATH)
        # df = pd.read_table(self.DF_PATH)
        # self.chembl_compound_smiles = df['canonical_smiles'].values

    def _get_chembl_mol_from_id(self, mol_id: int):
        sql = f"""select smiles, name where id = {mol_id}"""
        return pd.read_sql(sql, self.conn)

    def get_similar_mols(
        self,
        smiles: str,
        n_compounds: int = 10,
        filter_fn: Optional[Callable] = None
    ) -> Tuple[List[Union[str, float]]]:
        results = self.fpe.similarity(smiles, 0.0, n_workers=self.n_workers)
        
        i = 0
        closest_smiles, similarities = [], []
        while len(closest_smiles) < n_compounds:
            smiles = self._get_chembl_mol_from_id([results[i][0]])
            print(smiles)
            # smiles = self.chembl_compound_smiles[results[i][0]]
            similarity_value = results[i][1]

            if filter_fn is None or filter_fn(smiles):
                closest_smiles.append(smiles)
                similarities.append(similarity_value)
        
        i += 1
        
        return closest_smiles, similarities
    

eas_reference_compounds = [
    smiles.split('.')[0] for smiles in 
    pd.read_csv('./data/eas/eas_dataset.csv')['substrates'].unique()
]

def filter_chembl_compounds_fn_eas(compound_smiles: str) -> bool:
    mol = Chem.MolFromSmiles(compound_smiles)
    match = len(mol.GetAromaticAtoms()) > 0 and \
            mol.GetNumHeavyAtoms() < 30 and \
            not Chem.CanonSmiles(compound_smiles) in eas_reference_compounds
    return match