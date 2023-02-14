from concurrent.futures import ProcessPoolExecutor
import os
from typing import Literal
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast
from rxnmapper import RXNMapper

from reactions.eas.eas_reaction import EASReaction


SIMULATION_IDX_ATOM = ['H', 'He', 'Li', 'Be']

class Dataset:

    def __init__(
        self,
        csv_file_path: str,
    ) -> None:
        self.csv_file_path = os.path.join(os.environ['BASE_PATH'], f'data/datasets/{csv_file_path}')
        if os.path.exists(self.csv_file_path):
            self.generated = True
        else:
            self.generated = False

    def load(self) -> pd.DataFrame:
        if self.generated:
            return pd.read_csv(self.csv_file_path)
        else:
            raise ValueError("Dataset is not generate yet!")

    def generate_chemprop_dataset(
        self,
        file_path: str,
        **kwargs
    ) -> None:
        dataframe = self.load(**kwargs)
        rxn_mapper = RXNMapper()

        # parse reaction smiles + simulation idx
        smiles = []
        for _, row in dataframe.iterrows():
            reaction_smiles = [f'{row["substrates"]}.[{SIMULATION_IDX_ATOM[row["simulation_idx"]]}]>>{row["products"]}']
            atom_mapped_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps(reaction_smiles)[0]['mapped_rxn']
            smiles.append(atom_mapped_reaction_smiles)

        dataframe['smiles'] = smiles

        # export dataframe
        dataframe.to_csv(file_path)