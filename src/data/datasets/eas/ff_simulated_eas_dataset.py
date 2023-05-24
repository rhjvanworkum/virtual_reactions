try:
    import openmm
    from openmm import LangevinIntegrator, app, unit, Platform
    from openmm import *
    import openff.toolkit.typing.engines.smirnoff.parameters as offtk_parameters
    from openff.toolkit.typing.engines.smirnoff import ForceField
except ImportError:
    ForceField = None

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import List, Union, Any

from src.data.datasets.eas import SimulatedEASDataset
from src.methods import ForceFieldMethod
from src.reactions.eas_reaction import EASReaction

def construct_eas_force_field_parametrizations() -> List[ForceField]:
    """
    Function that construct 4 different force fields that can be used to simulated EAS reactions.
    Force Fields are parametrized as a grid of 2 parameters (aromatic C-C bond strength & aromatic C-N bond strenght)
    and 2 intervals (low strength: 100 kcal/mol & high strength: 1000 kcal/mol).

    All other force field parameters, besides the bond strength mentioned above, are taken from the openFF 2.0.0 Force field,
    see https://openforcefield.org/community/news/general/sage2.0.0-release/.
    """
    force_fields = []

    force_field_parameter_grid = [
        # aromatic C bond
        [
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X3:1]-[#6X3:2]",
                length=3 * unit.angstrom,
                k=1000 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X3:1]-[#6X3:2]",
                length=5 * unit.angstrom,
                k=100 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
        ],
        # aromatic C-N bond
        [
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X3:1]=[#7X2,#7X3+1:2]",
                length=3 * unit.angstrom,
                k=1000 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X3:1]=[#7X2,#7X3+1:2]",
                length=5 * unit.angstrom,
                k=100 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
        ],
    ]

    for parameter1 in force_field_parameter_grid[0]:
        for parameter2 in force_field_parameter_grid[1]:
            _force_field = ForceField("openff-2.0.0.offxml")
            _force_field["Bonds"].parameters[
                parameter1.smirks
            ].length = parameter1.length
            _force_field["Bonds"].parameters[parameter1.smirks].k = parameter1.k
            _force_field["Bonds"].parameters[
                parameter2.smirks
            ].length = parameter2.length
            _force_field["Bonds"].parameters[parameter2.smirks].k = parameter2.k
            force_fields.append(_force_field)

    return force_fields

eas_ff_methods = [ForceFieldMethod(forcefield) for forcefield in construct_eas_force_field_parametrizations()]


def compute_eas_conformer_energies(args):
    reaction = EASReaction(**args)
    try:
        energies = reaction.compute_conformer_energies()
    except:
        energies = None
    return energies

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
        solvents: Union[str, List[str]],
        compute_product_only_list: Union[bool, List[bool]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        arguments = [{
            'substrate_smiles': substrate.split('.')[0], 
            'product_smiles': product,
            'solvent': solvent,
            'method': eas_ff_methods[simulation_idx],
            'has_openmm_compatability': True,
            'compute_product_only': compute_product_only

        } for substrate, product, solvent, compute_product_only in zip(substrates, products, solvents, compute_product_only_list)]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(compute_eas_conformer_energies, arguments), total=len(arguments)))
        return results