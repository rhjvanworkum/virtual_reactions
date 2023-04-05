from typing import Tuple, List, Union, Any
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.dataset import SimulatedDataset, Dataset
from src.methods.methods import XtbMethod
from src.reactions.da.da_reaction import DAReaction


class SimulatedDADataset(SimulatedDataset):

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
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        arguments = [{
            'substrate_smiles': substrate, 
            'product_smiles': product,
            'method': XtbMethod(),
            'has_openmm_compatability': False,
            'compute_product_only': compute_product_only

        } for substrate, product, compute_product_only in zip(substrates, products, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_da_conformer_energies, arguments), total=len(arguments)))
        return results