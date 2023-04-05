from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Any, List, Tuple, Union
from tqdm import tqdm

from src.reactions.eas.eas_reaction import EASReaction
from src.reactions.eas.eas_methods import eas_xtb_method, eas_ff_methods, eas_ff

from src.dataset import Dataset, SimulatedDataset

# class XtbSimulatedExtendedEasDataset(XtbSimulatedEasDataset):

#     def __init__(
#         self,
#         csv_file_path: str,
#         split_object: Split,
#         similarity_set: Literal["train", "test"] = "train"
#     ) -> None:
#         super().__init__(
#             csv_file_path=csv_file_path,
#         )

#         self.split_object = split_object
#         self.similarity_set = similarity_set

#     def _select_reaction_to_simulate(
#         self,
#         source_dataset: Dataset
#     ) -> Tuple[List[Union[str, int]]]:
#         source_data = source_dataset.load()

#         substrates = source_data['substrates'].values
#         products = source_data['products'].values
#         reaction_idxs = source_data['reaction_idx'].values

#         # write code here to query chembl compounds for new reactions
#         # train

# TODO: add extended dataset to this as well
class SimulatedEASDataset(SimulatedDataset):

    def __init__(
        self, 
        csv_file_path: str, 
        n_simulations: int = 1
    ) -> None:
        super().__init__(csv_file_path, n_simulations)

    def _select_reaction_to_simulate(
        self,
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = source_dataset.load()

        substrates = source_data['substrates'].values
        compute_substrate = (substrates != np.roll(substrates, 1))

        return source_data['substrates'].values, \
               source_data['products'].values, \
               compute_substrate, \
               source_data['reaction_idx'].values


def compute_eas_conformer_energies(args):
    reaction = EASReaction(**args)
    try:
        energies = reaction.compute_conformer_energies()
    except:
        energies = None
    return energies

class XtbSimulatedEasDataset(SimulatedEASDataset):

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
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        arguments = [{
            'substrate_smiles': substrate.split('.')[0], 
            'product_smiles': product,
            'method': eas_xtb_method,
            'has_openmm_compatability': False,
            'compute_product_only': compute_product_only

        } for substrate, product, compute_product_only in zip(substrates, products, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
        return results

class SingleFFSimulatedEasDataset(SimulatedEASDataset):

    def __init__(
        self,
        csv_file_path: str
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path,
            n_simulations=1
        )

    def _simulate_reactions(
        self,
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        arguments = [{
            'substrate_smiles': substrate.split('.')[0], 
            'product_smiles': product,
            'method': eas_ff,
            'has_openmm_compatability': True,
            'compute_product_only': compute_product_only

        } for substrate, product, compute_product_only in zip(substrates, products, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
        return results

class FFSimulatedEasDataset(SimulatedEASDataset):

    def __init__(
        self,
        csv_file_path: str
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path,
            n_simulations=4
        )

    def _simulate_reactions(
        self,
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        arguments = [{
            'substrate_smiles': substrate.split('.')[0], 
            'product_smiles': product,
            'method': eas_ff_methods[simulation_idx],
            'has_openmm_compatability': True,
            'compute_product_only': compute_product_only

        } for substrate, product, compute_product_only in zip(substrates, products, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
        return results

# class DFTSimulatedEasDataset(SimulatedEASDataset):

#     def __init__(
#         self,
#         csv_file_path: str
#     ) -> None:
#         super().__init__(
#             csv_file_path=csv_file_path,
#             n_simulations=1
#         )

#     def _simulate_reactions(
#         self,
#         substrates: Union[str, List[str]],
#         products: Union[str, List[str]],
#         compute_product_only_list: Union[bool, List[bool]],
#         simulation_idx: int,
#         n_cpus: int
#     ) -> List[Any]:
#         assert simulation_idx == 0
#         arguments = [{
#             'substrate_smiles': substrate.split('.')[0], 
#             'product_smiles': product,
#             'method':  EASDFT(
#                 functional='B3LYP',
#                 basis_set='6-311g'
#             ),
#             'has_openmm_compatability': False,
#             'compute_product_only': compute_product_only

#         } for substrate, product, compute_product_only in zip(substrates, products, compute_product_only_list)]
#         with ProcessPoolExecutor(max_workers=n_cpus) as executor:
#             results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
#         return results