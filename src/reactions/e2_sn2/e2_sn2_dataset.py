from concurrent.futures import ProcessPoolExecutor
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Any, List, Tuple, Union

from src.methods.methods import NwchemMethod, XtbMethod
from src.dataset import Dataset, SimulatedDataset
from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from src.utils import Atom

def get_barriers_from_df(df, reaction_label) -> List[float]:
    df_label = df[df['label'] == reaction_label]
    df_label_ts = df_label[df_label['geometry'] == 'ts']
    df_label_rc = df_label[df_label['geometry'] == 'rcu']
    ts_energy = df_label_ts[df_label_ts['method'] == 'lccsd']['energy'].values[0]
    rc_energies = df_label_rc[df_label_rc['method'] == 'lccsd']['energy'].values
    return ts_energy - rc_energies

class E2Sn2Dataset(SimulatedDataset):
    JSON_FILE = './data/sn2_dataset.json'
    ENERGIES_FILE = '/home/ruard/Documents/datasets/qmrxn20/energies.txt'

    def __init__(
        self, 
        csv_file_path: str
    ) -> None:
        super().__init__(csv_file_path)

    def generate(
        self,
        n_processes: int
    ) -> None:
        with open(self.JSON_FILE) as json_file:
            dataset = json.load(json_file)

        energies_df = pd.read_csv(self.ENERGIES_FILE)

        uids = []
        df_reaction_idxs = []
        df_substrates = []
        df_products = []
        reaction_labels = []
        reaction_types = []
        labels = []
        conformer_energies = []
        simulation_idxs = []

        uid = 0
        for _, (reaction_label, items) in enumerate(dataset.items()):
            smiles = items['smiles'].replace('\n', '')
            reaction_type = 'sn2'

            if len(smiles) > 0:
                # add to df
                uids.append(uid)
                df_reaction_idxs.append(uid)
                df_substrates.append(smiles)
                df_products.append('')
                reaction_labels.append(reaction_label)
                reaction_types.append(reaction_type)
                conformer_energies.append(get_barriers_from_df(energies_df, reaction_label))
                labels.append(None)
                simulation_idxs.append(0)           

                uid += 1

        df = pd.DataFrame.from_dict({
            'uid': uids,
            'reaction_idx': df_reaction_idxs,
            'substrates': df_substrates,
            'products': df_products,
            'reaction_labels': reaction_labels,
            'reaction_types': reaction_types,
            'label': labels,
            'conformer_energies': conformer_energies,
            'simulation_idx': simulation_idxs
        })
        df.to_csv(self.csv_file_path)

        self.generated = True

def compute_e2_sn2_reaction_barriers(args) -> List[float]:
    reaction = E2Sn2Reaction(**args)
    energies = reaction.compute_activation_energies()
    return energies

def list_to_atom(geom):
    return [Atom(*atom) for atom in geom]

class SimulatedE2Sn2Dataset(SimulatedDataset):
    JSON_FILE = './data/sn2_dataset.json'

    def __init__(
        self, 
        csv_file_path: str, 
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path, 
            n_simulations=1
        )

    def _select_reaction_to_simulate(
        self,
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = pd.read_csv(source_dataset.csv_file_path)

        substrates = source_data['substrates'].values
        products = source_data['reaction_labels'].values
        compute_product_only_list = np.zeros(len(substrates))
        reaction_idxs = source_data['reaction_idx'].values

        return substrates, \
               products, \
               compute_product_only_list, \
               reaction_idxs

    def generate(
        self,
        source_dataset: Dataset,
        n_cpus: int,
    ) -> None:
        source_data = pd.read_csv(source_dataset.csv_file_path)
        substrates, products, compute_product_only_list, reaction_idxs = self._select_reaction_to_simulate(source_dataset)

        substrate_energies_list = {}

        uids = []
        df_reaction_idxs = []
        df_substrates = []
        df_products = []
        reaction_labels = []
        reaction_types = []
        labels = []
        conformer_energies = []
        simulation_idxs = []        
        for simulation_idx in range(self.n_simulations):
        
            simulation_results = self._simulate_reactions(substrates, products, compute_product_only_list, simulation_idx, n_cpus)
            assert len(simulation_results) == len(substrates) == len(compute_product_only_list)

            for idx, result in enumerate(simulation_results):
                # look up results for earlier calculated substrate
                if compute_product_only_list[idx]:
                    result = [
                        substrate_energies_list[substrates[idx]],
                        result[1]
                    ]
                else:
                    substrate_energies_list[substrates[idx]] = result[0]

                # add other properties
                uids.append(max(source_data['uid'].values) + len(uids) + 1)
                df_reaction_idxs.append(reaction_idxs[idx])
                df_substrates.append(substrates[idx])
                df_products.append('')
                reaction_labels.append(products[idx])
                reaction_types.append('sn2')
                labels.append(None)
                conformer_energies.append(result)
                simulation_idxs.append(simulation_idx + 1)
                idx += 1

        df = pd.DataFrame.from_dict({
            'uid': uids,
            'reaction_idx': df_reaction_idxs,
            'substrates': df_substrates,
            'products': df_products,
            'reaction_labels': reaction_labels,
            'reaction_types': reaction_types,
            'label': labels,
            'conformer_energies': conformer_energies,
            'simulation_idx': simulation_idxs
        })

        df = pd.concat([source_data, df])
        df.to_csv(self.csv_file_path)

        self.generated = True


class XtbSimulatedE2Sn2Dataset(SimulatedE2Sn2Dataset):

    def __init__(self, csv_file_path: str) -> None:
        super().__init__(csv_file_path)

    def _simulate_reactions(
        self,
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        
        with open(self.JSON_FILE) as json_file:
            dataset = json.load(json_file)

        arguments = [{
            'reactant_conformers': [list_to_atom(geom) for geom in dataset[reaction_label]['rc_conformers']],
            'ts': list_to_atom(dataset[reaction_label]['ts']),
            'product_conformers': [],
            'method': XtbMethod()
        } for reaction_label in products]

        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_e2_sn2_reaction_barriers, arguments), total=len(arguments)))
        return results
    
class DFTSimulatedE2Sn2Dataset(SimulatedE2Sn2Dataset):

    def __init__(self, csv_file_path: str) -> None:
        super().__init__(csv_file_path)

    def _simulate_reactions(
        self,
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        
        with open(self.JSON_FILE) as json_file:
            dataset = json.load(json_file)

        arguments = [{
            'reactant_conformers': [list_to_atom(geom) for geom in dataset[reaction_label]['rc_conformers']],
            'ts': list_to_atom(dataset[reaction_label]['ts']),
            'product_conformers': [],
            'method': NwchemMethod(
                functional='B3LYP',
                basis_set='6-311g',
                n_cores=2
            )
        } for reaction_label in products]

        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_e2_sn2_reaction_barriers, arguments), total=len(arguments)))
        return results
    