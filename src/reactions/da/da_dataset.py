from typing import Tuple, List, Union, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.compound import Compound

from src.dataset import SimulatedDataset, Dataset
from src.methods.methods import XtbMethod
from src.reactions.da.da_reaction import DAReaction


class SimulatedDADataset(SimulatedDataset):

    def __init__(
        self, 
        csv_file_path: str, 
        n_simulations: int = 1
    ) -> None:
        n_substrates = 2
        super().__init__(csv_file_path, n_simulations, n_substrates)

    def _select_reaction_to_simulate(
        self,
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = source_dataset.load()

        substrates = source_data['substrates'].values
        compute_substrate_only_list = np.zeros(len(substrates))

        #    source_data['solvent'].values, \

        return source_data['substrates'].values, \
               source_data['products'].values, \
               [None for _ in range(len(substrates))], \
               compute_substrate_only_list, \
               source_data['reaction_idx'].values
    

def compute_da_conformer_energies(args):
    reaction = DAReaction(**args)
    try:
        energies = reaction.compute_conformer_energies()
    except:
        energies = None
    return energies

class XtbSimulatedDADataset(SimulatedDADataset):

    def __init__(
        self, 
        csv_file_path: str, 
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path, 
            n_simulations=1
        )

    def _simulate_reactions(
        self,
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        solvents: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        arguments = [{
            'substrate_smiles': substrate, 
            'product_smiles': product,
            'solvent': solvent,
            'method': XtbMethod(),
            'has_openmm_compatability': False,
            'compute_product_only': compute_product_only

        } for substrate, product, solvent, compute_product_only in zip(substrates, products, solvents, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_da_conformer_energies, arguments), total=len(arguments)))
        return results
    

import rdkit
from rdkit import Chem
da_mol_pattern = Chem.MolFromSmiles("C1=CCCCC1")

def get_sum_square_diff(smiles: str):
    mol = Compound.from_smiles(smiles)
    mol.generate_conformers()
    mol.optimize_conformers()
    idxs = mol.compute_fukui_indices()
    
    elec_idxs = np.array([idx[0] for idx in idxs])
    nuc_idxs = np.array([idx[1] for idx in idxs])
    
    react_atom_idxs = mol.rdkit_mol.GetSubstructMatch(da_mol_pattern)
    
    diff = 0
    for atom_idx in react_atom_idxs:
        diff += (nuc_idxs[atom_idx] - elec_idxs[atom_idx])**2
    return diff

class FukuiSimulatedDADataset(Dataset):

    def __init__(
        self, 
        csv_file_path: str, 
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path, 
            n_simulations=1
        )

    def generate(
        self,
        source_dataset: Dataset,
        n_cpus: int,
    ) -> None:
        source_data = pd.read_csv(source_dataset.csv_file_path)

        uids = []
        df_reaction_idxs = []
        df_substrates = []
        df_products = []
        index = []
        simulation_idxs = [] 

        args = source_data['products'].values
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(get_sum_square_diff, args), total=len(args)))
        
        for idx, result in enumerate(results):
                uids.append(max(source_data['uid'].values) + len(uids) + 1)
                df_reaction_idxs.append(source_data['reaction_idx'].values[idx])
                df_substrates.append(source_data['substrates'].values[idx])
                df_products.append(source_data['products'].values[idx])
                index.append(result)
                simulation_idxs.append(1)
                idx += 1

        df = pd.DataFrame.from_dict({
            'uid': uids,
            'reaction_idx': df_reaction_idxs,
            'substrates': df_substrates,
            'products': df_products,
            'idx': idx,
            'simulation_idx': simulation_idxs
        })

        # determine labels
        labels = []
        for _, row in df.iterrows():
            index = row['index']
            other_indices = df[df['substrates'] == row['substrates']]['index']
            label = int((index <= other_indices).all())
            labels.append(label)
        df['label'] = labels

        df = pd.concat([source_data, df])
        df.to_csv(self.csv_file_path)

        self.generated = True