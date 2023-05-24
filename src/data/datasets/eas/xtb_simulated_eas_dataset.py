from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import List, Union, Any

from src.data.datasets.eas_datasets import SimulatedEASDataset
from src.methods import XtbMethod
from src.reactions.eas_reaction import EASReaction

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
        solvents: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        arguments = [{
            'substrate_smiles': substrate.split('.')[0], 
            'product_smiles': product,
            'solvent': solvent,
            'method': XtbMethod(),
            'has_openmm_compatability': False,
            'compute_product_only': compute_product_only

        } for substrate, product, solvent, compute_product_only in zip(substrates, products, solvents, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
        return results