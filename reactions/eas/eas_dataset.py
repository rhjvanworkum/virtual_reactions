from concurrent.futures import ProcessPoolExecutor
import os
from typing import Literal
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast

from reactions.eas.eas_reaction import EASReaction

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
        file_path: str
    ) -> None:
        dataframe = self.load()

        # parse reaction smiles + simulation idx

        # export dataframe


def get_conformer_energies(args):
    substrate_smiles, product_smiles = args

    reaction = EASReaction(
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles
    )

    return reaction.compute_conformer_energies()


class XtbSimulatedEasDataset(Dataset):

    def __init__(
        self,
        csv_file_path: str
    ) -> None:
        super().__init__(csv_file_path=csv_file_path)

    def generate(
        self,
        source_dataset: Dataset,
        n_cpus: int
    ) -> None:
        source_data = source_dataset.load()

        arguments = [(row['substrates'].split('.')[0], row['products']) for _, row in source_data.iterrows()]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(get_conformer_energies, arguments), total=len(arguments)))

        substrates = []
        reaction_products = []
        labels = []
        conformer_energies = []
        simulation_idx = []

        assert len(results) == len(arguments)

        for result, (_, row) in zip(results, source_data.iterrows):
            substrates.append(row['substrates'])
            reaction_products.append(row['products'])
            labels.append(None)
            conformer_energies.append(result)
            simulation_idx.append(1)

        df = pd.DataFrame.from_dict({
            'substrates': substrates,
            'products': reaction_products,
            'label': labels,
            'conformer_energies': conformer_energies,
            'simulation_idx': simulation_idx
        })
        df.to_csv(self.csv_file_path)

        self.generated = True

    def load(
        self,
        aggregation_mode: Literal["avg", "low"] = "low"
    ) -> pd.DataFrame:
        if self.generated:
            dataframe = pd.read_csv(self.csv_file_path)
            source_dataframe = dataframe[dataframe['simulation_idx'] == 0]
            virtual_dataframe = dataframe[dataframe['simulation_idx'] != 0]

            # drop label column
            virtual_dataframe.drop('label')

            # aggregate across conformers
            barriers = []
            for _, row in virtual_dataframe.iterrows():
                sub_energies, ts_energies = ast.eval_literal(row['conformer_energies'])
                sub_energies = [e for e in sub_energies if e is not None]
                ts_energies = [e for e in ts_energies if e is not None]

                if aggregation_mode == 'avg':
                    barrier = np.mean(np.array(ts_energies)) - np.mean(np.array(sub_energies))
                elif aggregation_mode == 'low':
                    barrier = np.min(np.array(ts_energies)) - np.min(np.array(sub_energies))
                barriers.append(barrier)
            virtual_dataframe.insert(-1, 'barrier', barrier)

            # assing labels based on barriers across substrates
            labels = []
            for _, row in virtual_dataframe.iterrows():
                barrier = row['barrier']
                other_barriers = virtual_dataframe[virtual_dataframe['substrates'] == row['substrates']]['barrier']
                label = (barrier <= other_barriers)
                labels.append(label)
            virtual_dataframe.insert(-1, 'label', label)

            # drop conformer energy column
            virtual_dataframe.drop('conformer_energies')

            return pd.concat([source_dataframe, virtual_dataframe])
        else:
            raise ValueError("Dataset is not generate yet!")