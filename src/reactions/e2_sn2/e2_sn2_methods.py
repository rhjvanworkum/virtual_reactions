try:
    import openmm
    from openmm import LangevinIntegrator, app, unit, Platform
    from openmm import *
except ImportError:
    from simtk import openmm, unit
    from simtk.openmm import LangevinIntegrator, app, Platform
    from simtk.openmm import *
import openff.toolkit.typing.engines.smirnoff.parameters as offtk_parameters
from openff.toolkit.typing.engines.smirnoff import ForceField

from src.methods.methods import ForceFieldMethod, HuckelMethod

from typing import List

def construct_e2_sn2_force_field_parametrizations() -> List[ForceField]:
    """
    Function that construct 4 different force fields that can be used to simulated EAS reactions.
    Force Fields are parametrized as a grid of 2 parameters (aromatic C-C bond strength & aromatic C-N bond strenght)
    and 2 intervals (low strength: 100 kcal/mol & high strength: 1000 kcal/mol).

    All other force field parameters, besides the bond strength mentioned above, are taken from the openFF 2.0.0 Force field,
    see https://openforcefield.org/community/news/general/sage2.0.0-release/.
    """
    force_fields = []

    force_field_parameter_grid = [
        [
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#1:2]",
                length=3 * unit.angstrom,
                k=740 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#9:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#17:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#35:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
        ],
        [
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#1:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#9:2]",
                length=3 * unit.angstrom,
                k=680 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#17:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#35:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
        ],
        [
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#1:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#9:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#17:2]",
                length=3 * unit.angstrom,
                k=344 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#35:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
        ],
        [
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#1:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#9:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#17:2]",
                length=3 * unit.angstrom,
                k=10 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
            offtk_parameters.BondHandler.BondType(
                smirks="[#6X4:1]-[#35:2]",
                length=3 * unit.angstrom,
                k=366 * unit.kilocalorie / (unit.mole * unit.angstrom**2),
            ),
        ],
    ]

    for parameters in force_field_parameter_grid:
        _force_field = ForceField("openff-2.0.0.offxml")
        for parameter in parameters:
            _force_field["Bonds"].parameters[
                parameter.smirks
            ].length = parameter.length
            _force_field["Bonds"].parameters[parameter.smirks].k = parameter.k
        force_fields.append(_force_field)
    
    return force_fields

e2_sn2_ff_methods = [ForceFieldMethod(forcefield) for forcefield in construct_e2_sn2_force_field_parametrizations()]

e2_sn2_huckel_methods = [
    HuckelMethod('H_test'), 
    HuckelMethod('F_test'),
    HuckelMethod('Cl_test'),
    HuckelMethod('Br_test'),
]