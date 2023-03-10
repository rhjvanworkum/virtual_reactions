from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Tuple, Union
from tqdm import tqdm

from src.reactions.eas.eas_reaction import EASReaction
from src.reactions.eas.eas_methods import EASDFT, eas_ff_methods
from src.dataset import Dataset, SimulatedDataset

SIMULATION_IDX_ATOM = ['H', 'He', 'Li', 'Be']


def get_conformer_energies(args):
    substrate_smiles, product_smiles = args

    method = EASDFT(
        functional='B3LYP',
        basis_set='6-311g'
    )

    reaction = EASReaction(
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles,
        method=method
    )

    return reaction.compute_conformer_energies()

class DFTSimulatedEasDataset(SimulatedDataset):

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
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        arguments = [
            (substrate.split('.')[0], product) for substrate, product in zip(substrates, products)
        ]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(get_conformer_energies, arguments), total=len(arguments)))
        return results

    def _select_reaction_to_simulate(
        self,
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = source_dataset.load()
        return source_data['substrates'].values, \
               source_data['products'].values, \
               source_data['reaction_idx'].values