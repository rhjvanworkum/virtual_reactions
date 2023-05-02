from typing import Tuple, List, Union, Any
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.dataset import SimulatedDataset, Dataset
from src.methods.methods import XtbMethod
from src.reactions.da.da_reaction import DAReaction
from src.reactions.ma.ma_reaction import MAReaction


class SimulatedMADataset(SimulatedDataset):

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
    

def compute_ma_conformer_energies(args):
    reaction = MAReaction(**args)
    try:
        energies = reaction.compute_conformer_energies()
    except:
        energies = None
    return energies

class XtbSimulatedMADataset(SimulatedMADataset):

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
            results = list(tqdm(executor.map(compute_ma_conformer_energies, arguments), total=len(arguments)))
        return results