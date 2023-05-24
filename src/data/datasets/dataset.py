import os
from typing import Any, List, Literal, Optional, Union
import pandas as pd
from rxnmapper import RXNMapper
import numpy as np
import ast
from tqdm import tqdm

import rdkit
from rdkit import Chem
import ast



SIMULATION_IDX_ATOM = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']

class Dataset:

    def __init__(
        self,
        csv_file_path: str,
    ) -> None:
        self.csv_file_path = os.path.join(os.environ['BASE_PATH'], f'data/{csv_file_path}')
        if os.path.exists(self.csv_file_path):
            self.generated = True
        else:
            self.generated = False

    @property
    def chemprop_csv_file_path(self) -> str:
        return self.csv_file_path.split('.')[0] + '_chemprop.csv'
    
    @property
    def chemprop_feature_file_path(self) -> str:
        return self.csv_file_path.split('.')[0] + '_chemprop_feat.csv'

    def load(
        self,
        uids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load the dataset as a pandas dataframe.
        """
        if self.generated:
            df = pd.read_csv(self.csv_file_path)
            if uids is not None:
                return df[df['uid'].isin(uids)]
            else:
                return df
        else:
            raise ValueError("Dataset is not generated yet!")

    def load_chemprop_dataset(
        self,
        uids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load the chemprop compatible dataset as a pandas dataframe.
        """
        if os.path.exists(self.chemprop_csv_file_path):
            df = pd.read_csv(self.chemprop_csv_file_path)
            if uids is not None:
                return df[df['uid'].isin(uids)]
            else:
                return df
        else:
            raise ValueError("Chemprop Dataset is not generated yet!")

    def load_chemprop_features(
        self,
        uids: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Load the atomic descriptor features for chemprop usage.
        """
        if os.path.exists(self.chemprop_feature_file_path) and  os.path.exists(self.chemprop_csv_file_path):
            df = pd.read_csv(self.chemprop_csv_file_path)
            if uids is not None:
                smiles = df[df['uid'].isin(uids)]['smiles']
            else:
                smiles = df['smiles']

            feature_df = pd.read_csv(self.chemprop_feature_file_path)
            descriptors = [
                np.array(ast.literal_eval(
                    feature_df[feature_df['smiles'] == smi]['descriptors'].values[0]
                )) for smi in smiles
            ]
            return descriptors
        else:
            raise ValueError("Chemprop Feature Dataset is not generated yet!")

    def generate_chemprop_dataset(
        self,
        force: bool = False,
        simulation_idx_as_features: bool = False
    ) -> None:
        """
        Generate chemprop compatible dataset.
        """
        if not os.path.exists(self.chemprop_csv_file_path) or force:
            dataframe = self.load()
            rxn_mapper = RXNMapper()

            # parse reaction smiles + simulation idx
            smiles = []
            simulation_idxs = []
            for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Mapping rxn smiles.."):
                if simulation_idx_as_features:
                    simulation_idxs.append(row["simulation_idx"])
                    reaction_smiles = [f'{row["substrates"]}>>{row["products"]}']
                else:
                    reaction_smiles = [f'{row["substrates"]}.[{SIMULATION_IDX_ATOM[row["simulation_idx"]]}]>>{row["products"]}']
                
                atom_mapped_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps(reaction_smiles)[0]['mapped_rxn']
                smiles.append(atom_mapped_reaction_smiles)

            dataframe['smiles'] = smiles

            # export dataframe
            dataframe.to_csv(self.chemprop_csv_file_path)

            if simulation_idx_as_features:
                feat_dataframe = pd.DataFrame()
                feat_dataframe['smiles'] = smiles
                descriptors = []
                for idx, smiles in enumerate(smiles):
                    smi_reac, smi_prod = smiles.split('>>')
                    mol_reac, mol_prod = Chem.MolFromSmiles(smi_reac), Chem.MolFromSmiles(smi_prod)
                    n_atoms = max(mol_reac.GetNumAtoms(), mol_prod.GetNumAtoms())
                    descriptors.append([simulation_idxs[idx] for _ in range(n_atoms)])
                feat_dataframe['descriptors'] = descriptors
                feat_dataframe.to_csv(self.chemprop_feature_file_path)
        else:
            pass