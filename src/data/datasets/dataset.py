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
        folder_path: str,
        simulation_type: Literal['smiles', 'index_feature', 'outcome_feature'] = 'index_feature',
    ) -> None:
        self.folder_path = os.path.join(os.environ['BASE_PATH'], f'data/{folder_path}')
        self.simulation_type = simulation_type

        self.__generated = False
        self.__chemprop_generated = False

        self.__generated = os.path.exists(self.dataset_path)

    @property
    def generated(self) -> bool:
        return self.__generated

    @property
    def chemprop_generated(self) -> bool:
        return self.__chemprop_generated

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.folder_path, 'dataset.csv')

    @property
    def chemprop_dataset_path(self) -> str:
        if self.simulation_type == 'smiles':
            return os.path.join(self.folder_path, 'chemprop_smiles_dataset.csv')
        elif self.simulation_type == 'index_feature':
            return os.path.join(self.folder_path, 'chemprop_index_dataset.csv')
        elif self.simulation_type == 'outcome_feature':
            return os.path.join(self.folder_path, 'chemprop_outcome_dataset.csv')
    
    @property
    def chemprop_feature_file_path(self) -> str:
        if self.simulation_type == 'smiles':
            return None
        elif self.simulation_type == 'index_feature':
            return os.path.join(self.folder_path, 'chemprop_index_feat.csv')
        elif self.simulation_type == 'outcome_feature':
            return os.path.join(self.folder_path, 'chemprop_outcome_feat.csv')

    def load(
        self,
        uids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load the dataset as a pandas dataframe.
        """
        if self.generated:
            df = pd.read_csv(self.dataset_path)
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
        if self.chemprop_generated:
            df = pd.read_csv(self.chemprop_dataset_path)
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
        if self.chemprop_generated:
            df = pd.read_csv(self.chemprop_dataset_path)
            if uids is not None:
                smiles = df[df['uid'].isin(uids)]['smiles']
            else:
                smiles = df['smiles']

            feature_df = pd.read_csv(self.chemprop_feature_file_path)
            descriptors = []
            for smi in smiles:
                desc = []
                for column in feature_df.columns:
                    if 'descriptor' in column:
                        descriptor = ast.literal_eval(feature_df[feature_df['smiles'] == smi][column].values[0])
                        desc.append(descriptor)
                desc = np.array(desc).T
                descriptors.append(desc)
            return descriptors
        else:
            raise ValueError("Chemprop Feature Dataset is not generated yet!")

    def generate_chemprop_dataset(
        self,
        force: bool = False,
    ) -> None:
        """
        Generate chemprop compatible dataset.
        """
        if (not os.path.exists(self.chemprop_dataset_path) or 
            (self.simulation_type != 'smiles' and not os.path.exists(self.chemprop_feature_file_path))
            ) or force:
            dataframe = self.load()
            rxn_mapper = RXNMapper()

            if self.simulation_type == "smiles":
                smiles = []
                for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Mapping rxn smiles.."):
                    reaction_smiles = [f'{row["substrates"]}.[{SIMULATION_IDX_ATOM[row["simulation_idx"]]}]>>{row["products"]}']
                    atom_mapped_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps(reaction_smiles)[0]['mapped_rxn']
                    smiles.append(atom_mapped_reaction_smiles)
                dataframe['smiles'] = smiles
                dataframe.to_csv(self.chemprop_dataset_path)

            elif self.simulation_type == "index_feature":
                smiles = []
                simulation_idxs = []
                for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Mapping rxn smiles.."):
                    simulation_idxs.append(row["simulation_idx"])
                    reaction_smiles = [f'{row["substrates"]}>>{row["products"]}']
                    atom_mapped_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps(reaction_smiles)[0]['mapped_rxn']
                    smiles.append(atom_mapped_reaction_smiles)
                dataframe['smiles'] = smiles
                dataframe.to_csv(self.chemprop_dataset_path)

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

            elif self.simulation_type == "outcome_feature":
                reaction_smiles = [
                    f'{row["substrates"]}>>{row["products"]}' for _, row in dataframe.iterrows()
                ]
                dataframe['_reaction_smiles'] = reaction_smiles                
                exp_dataframe = dataframe[dataframe['simulation_idx'] == 0]


                am_reaction_smiles_list = []
                reaction_smiles_list = []
                for reaction_smiles in tqdm(exp_dataframe['_reaction_smiles'], total=len(exp_dataframe), desc="Mapping rxn smiles.."):
                    am_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])[0]['mapped_rxn']
                    am_reaction_smiles_list.append(am_reaction_smiles)
                    reaction_smiles_list.append(reaction_smiles)
                exp_dataframe['smiles'] = am_reaction_smiles_list
                exp_dataframe.to_csv(self.chemprop_dataset_path)

                feat_dataframe = pd.DataFrame()
                feat_dataframe['smiles'] = am_reaction_smiles_list
                for sim_idx in range(1, int(max(dataframe['simulation_idx']) + 1)):
                    selection = dataframe[dataframe['simulation_idx'] == sim_idx]
                    descriptors = []
                    for reaction_smiles in reaction_smiles_list:
                        try:
                            label = selection[selection['_reaction_smiles'] == reaction_smiles]['label'].values[0]
                        except:
                            label = 2
                        smi_reac, smi_prod = reaction_smiles.split('>>')
                        mol_reac, mol_prod = Chem.MolFromSmiles(smi_reac), Chem.MolFromSmiles(smi_prod)
                        n_atoms = max(mol_reac.GetNumAtoms(), mol_prod.GetNumAtoms())
                        descriptors.append([label for _ in range(n_atoms)])
                    feat_dataframe[f'descriptors_{sim_idx}'] = descriptors
                feat_dataframe.to_csv(self.chemprop_feature_file_path)

        self.__chemprop_generated = True