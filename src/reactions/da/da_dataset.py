from typing import Literal, Optional, Tuple, List, Union, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast
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
               source_data['solvent'].values, \
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
diene_mol_pattern = Chem.MolFromSmarts("[C,c][C,c]=[C,c][C,c]")
dienophile_mol_pattern = Chem.MolFromSmarts("[C,c][C,c]")

def get_idxs(args):
    smiles, solvent, functional, basis_set = args
    mol = Compound.from_smiles(smiles, solvent=None)
    mol.generate_conformers()
    mol.optimize_lowest_conformer()
    idxs = mol.compute_fukui_indices(functional, basis_set)
    return idxs
    # if idxs is None:
    #     return None
    # else:
    #     elec_idxs = np.array([idx[0] for idx in idxs])
    #     nuc_idxs = np.array([idx[1] for idx in idxs])
        
    #     react_atom_idxs = mol.rdkit_mol.GetSubstructMatch(da_mol_pattern)
        
    #     diff = 0
    #     for atom_idx in react_atom_idxs:
    #         diff += (nuc_idxs[atom_idx] - elec_idxs[atom_idx])**2
    #     return diff

def get_da_reacting_atoms(mol: Chem.Mol) -> Tuple[int]:
    return mol.GetSubstructMatch(da_mol_pattern)

def get_da_reacting_atoms_sub(mol: Chem.Mol) -> Tuple[Tuple[int]]:
    da_atoms = mol.GetSubstructMatch(da_mol_pattern)

    diene_atoms = None
    diene_matches = mol.GetSubstructMatches(diene_mol_pattern)
    for match in diene_matches:
        if set(match).issubset(set(da_atoms)):
            diene_atoms = match
            break

    dienophile_atoms = None
    dienophile_matches = mol.GetSubstructMatches(dienophile_mol_pattern)
    for match in dienophile_matches:
        if set(match).issubset(set(da_atoms)) and not set(match).intersection(set(diene_atoms)):
            dienophile_atoms = match
            break
    
    # try:
    #     print(diene_atoms, dienophile_atoms)
    # except:
    #     print(Chem.MolToSmiles(mol))
    return diene_atoms, dienophile_atoms

class FukuiSimulatedDADataset(Dataset):

    def __init__(
        self, 
        csv_file_path: str, 
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path, 
        )

    def generate(
        self,
        source_dataset: Dataset,
        n_cpus: int,
        functional: str,
        basis_set: str
    ) -> None:
        source_data = pd.read_csv(source_dataset.csv_file_path)

        uids = []
        df_reaction_idxs = []
        df_substrates = []
        df_products = []
        reaction_smiles = []
        solvents = []
        index = []
        simulation_idxs = [] 

        args = [(smi, solvent, functional, basis_set) for smi, solvent in zip(source_data['products'].values, source_data['solvent'].values)]
        # args = [(smi, solvent) for smi, solvent in zip(source_data['products'].values, ['Methanol' for _ in range(len(source_data))])]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(get_idxs, args), total=len(args)))
        
        for idx, result in enumerate(results):
            uids.append(max(source_data['uid'].values) + len(uids) + 1)
            df_reaction_idxs.append(source_data['reaction_idx'].values[idx])
            df_substrates.append(source_data['substrates'].values[idx])
            df_products.append(source_data['products'].values[idx])
            reaction_smiles.append(source_data['reaction_smiles'].values[idx])
            solvents.append(source_data['solvent'].values[idx])
            index.append(result)
            simulation_idxs.append(1)
            idx += 1

        df = pd.DataFrame.from_dict({
            'uid': uids,
            'reaction_idx': df_reaction_idxs,
            'substrates': df_substrates,
            'products': df_products,
            'index': index,
            'reaction_smiles': reaction_smiles,
            'solvent': solvents,
            'simulation_idx': simulation_idxs
        })

        df = pd.concat([source_data, df])
        df.to_csv(self.csv_file_path)

        self.generated = True

    def load_valid(
        self,
        uids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if self.generated:
            df = pd.read_csv(self.csv_file_path)

            source_df = df[df['simulation_idx'] == 0]
            virtual_df = df[df['simulation_idx'] == 1]

            # filter out failed ones
            filtered_virtual_df = virtual_df[~virtual_df['index'].isna()]
            valid_reaction_idxs = []
            for idx in filtered_virtual_df['reaction_idx'].unique():
                if len(filtered_virtual_df[filtered_virtual_df['reaction_idx'] == idx]) >= 2:
                    valid_reaction_idxs.append(idx)
            filtered_virtual_df = filtered_virtual_df[filtered_virtual_df['reaction_idx'].isin(valid_reaction_idxs)]
            filtered_source_df = source_df[source_df['reaction_idx'].isin(valid_reaction_idxs)] 

            # now add score here
            scores = []
            for idx, row in filtered_virtual_df.iterrows():
                elec_idxs, nuc_idxs = ast.literal_eval(row['index'])
                
                # mol = Compound.from_smiles(row['products']).rdkit_mol
                # react_atom_idxs = get_da_reacting_atoms(mol)
                # diff = 0
                # for atom_idx in react_atom_idxs:
                #     diff += (elec_idxs[atom_idx] - nuc_idxs[atom_idx])**2
                
                mol = Compound.from_smiles(row['products']).rdkit_mol
                diene_atoms, dienophile_atoms = get_da_reacting_atoms_sub(mol)
                diff = 0
                if diene_atoms is not None and dienophile_atoms is not None:
                    diff = sum([(elec_idxs[i] - nuc_idxs[i])**2 for i in diene_atoms]) / sum([(elec_idxs[i] - nuc_idxs[i])**2 for i in dienophile_atoms])

                scores.append(diff)
            filtered_virtual_df['score'] = scores

            # now add labels to virtual
            labels = []
            for _, row in filtered_virtual_df.iterrows():
                score = float(row['score'])
                other_scores = [
                    float(val) for val in filtered_virtual_df[filtered_virtual_df['substrates'] == row['substrates']]['score'].values
                ]
                label = int(score == min(other_scores))
                labels.append(label)
            filtered_virtual_df['label'] = labels

            print(len(source_df), len(filtered_source_df), len(filtered_virtual_df))
            df = pd.concat([filtered_source_df, filtered_virtual_df])

            if uids is not None:
                return df[df['uid'].isin(uids)]
            else:
                return df
        else:
            raise ValueError("Dataset is not generated yet!")

    def load(
        self,
        uids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if self.generated:
            df = pd.read_csv(self.csv_file_path)

            # add labels here
            source_df = df[df['simulation_idx'] == 0]
            virtual_df = df[df['simulation_idx'] == 1]

            labels = []
            for _, row in virtual_df.iterrows():
                print(row['index'])
                if row['index'] != "None":
                    index = float(row['index'])
                    other_indices = [
                        float(val) if val != "None" else 1000 for val in virtual_df[virtual_df['substrates'] == row['substrates']]['index'].values
                    ]
                    label = int(index == min(other_indices))
                else:
                    label = int(0)

                labels.append(label)
            virtual_df['label'] = labels

            df = pd.concat([source_df, virtual_df])

            if uids is not None:
                return df[df['uid'].isin(uids)]
            else:
                return df
        else:
            raise ValueError("Dataset is not generated yet!")