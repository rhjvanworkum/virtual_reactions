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

        uids = []
        reaction_idxs = []
        substrates = []
        reaction_products = []
        labels = []
        conformer_energies = []
        simulation_idx = []

        assert len(results) == len(arguments)

        idx = 0
        for result, (_, row) in zip(results, source_data.iterrows()):
            uids.append(max(source_data['uid'].values) + idx)
            reaction_idxs.append(row['reaction_idx'])
            substrates.append(row['substrates'])
            reaction_products.append(row['products'])
            labels.append(None)
            conformer_energies.append(result)
            simulation_idx.append(1)
            idx += 1

        df = pd.DataFrame.from_dict({
            'uid': uids,
            'reaction_idx': reaction_idxs,
            'substrates': substrates,
            'products': reaction_products,
            'label': labels,
            'conformer_energies': conformer_energies,
            'simulation_idx': simulation_idx
        })
        df = pd.concat([source_data, df])
        df.to_csv(self.csv_file_path)

        self.generated = True

    def load(
        self,
        aggregation_mode: Literal["avg", "low"] = "low",
        margin: float = 0.0
    ) -> pd.DataFrame:
        if self.generated:
            dataframe = pd.read_csv(self.csv_file_path)
            source_dataframe = dataframe[dataframe['simulation_idx'] == 0]
            virtual_dataframe = dataframe[dataframe['simulation_idx'] != 0]

            # drop label column
            virtual_dataframe = virtual_dataframe.drop(columns=['label'])

            # aggregate across conformers
            barriers = []
            for _, row in virtual_dataframe.iterrows():
                sub_energies, ts_energies = ast.literal_eval(row['conformer_energies'])
                sub_energies = [e for e in sub_energies if e is not None]
                ts_energies = [e for e in ts_energies if e is not None]

                if aggregation_mode == 'avg':
                    barrier = np.mean(np.array(ts_energies)) - np.mean(np.array(sub_energies))
                elif aggregation_mode == 'low':
                    barrier = np.min(np.array(ts_energies)) - np.min(np.array(sub_energies))
                else:
                    raise ValueError("aggregation_mode {aggregation_mode} doesn't exists")
                barriers.append(barrier)
            virtual_dataframe['barrier'] = barriers

            # assing labels based on barriers across substrates
            labels = []
            for _, row in virtual_dataframe.iterrows():
                barrier = row['barrier']
                other_barriers = virtual_dataframe[virtual_dataframe['substrates'] == row['substrates']]['barrier']
                label = int((barrier - margin <= other_barriers).all())
                labels.append(label)
            virtual_dataframe['label'] = labels

            # drop conformer energy column
            dataframe = pd.concat([source_dataframe, virtual_dataframe])
            dataframe = dataframe.drop(columns=['conformer_energies'])
            return dataframe
        else:
            raise ValueError("Dataset is not generate yet!")