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

from src.methods.methods import XtbMethod, ForceFieldMethod

from typing import List


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

eas_xtb_method = XtbMethod()

eas_ff_methods = [ForceFieldMethod(forcefield) for forcefield in construct_eas_force_field_parametrizations()]