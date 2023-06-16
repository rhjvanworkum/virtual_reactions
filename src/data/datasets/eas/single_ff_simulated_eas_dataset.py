try:
    from openff.toolkit.typing.engines.smirnoff import ForceField
except ImportError:
    ForceField = None

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import List, Literal, Union, Any

from src.data.datasets.eas import SimulatedEASDataset
from src.methods import ForceFieldMethod
from src.reactions.eas_reaction import EASReaction


def compute_eas_conformer_energies(args):
    reaction = EASReaction(**args)
    try:
        energies = reaction.compute_conformer_energies()
    except:
        energies = None
    return energies

class SingleFFSimulatedEasDataset(SimulatedEASDataset):

    def __init__(
        self,
        folder_path: str,
        simulation_type: Literal['smiles', 'index_feature', 'outcome_feature'] = 'index_feature',
    ) -> None:
        super().__init__(
            folder_path=folder_path,
            n_simulations=1,
            simulation_type=simulation_type
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
        arguments = [{
            'substrate_smiles': substrate.split('.')[0], 
            'product_smiles': product,
            'solvent': solvent,
            'method': ForceFieldMethod(ForceField("openff-2.0.0.offxml")),
            'has_openmm_compatability': True,
            'compute_product_only': compute_product_only

        } for substrate, product, solvent, compute_product_only in zip(substrates, products, solvents, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
        return results