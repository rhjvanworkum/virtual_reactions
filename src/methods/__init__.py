# try:
#     import openmm
#     from openmm import LangevinIntegrator, app, unit, Platform
#     from openmm import *
# except ImportError:
#     from simtk import openmm, unit
#     from simtk.openmm import LangevinIntegrator, app, Platform
#     from simtk.openmm import *
# import openff.toolkit.typing.engines.smirnoff.parameters as offtk_parameters
# from openff.toolkit.typing.engines.smirnoff import ForceField

import os
import numpy as np
from typing import Any, List, Optional

from src.methods.XTB import xtb

Compound = Any

class Method:
    def __init__(self) -> None:
        pass

    def single_point(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: str = 'Methanol'
    ) -> float:
        raise NotImplementedError
        
    def optimization(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: str = 'Methanol'
    ) -> float:
        raise NotImplementedError

class XtbMethod(Method):
    def __init__(self) -> None:
        pass

    def single_point(
        self, 
        molecule: Compound, 
        conformer_idx: int, 
        solvent: str = 'Methanol'
    ) -> float:
        energy, _ = xtb(
            molecule=molecule,
            conformer_idx=conformer_idx,
            keywords=[],
            method='2',
            solvent=solvent,
            xcontrol_file=None
        )
        return energy
    
    def optimization(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: str = 'Methanol'
    ) -> float:
        energy, _ = xtb(
            molecule=molecule,
            conformer_idx=conformer_idx,
            keywords=['--opt'],
            method='2',
            solvent=solvent,
            xcontrol_file=None
        )
        return energy

class ForceFieldMethod(Method):
    def __init__(self) -> None:
        super().__init__()

# class ForceFieldMethod(Method):
#     def __init__(
#         self,
#         force_field: ForceField
#     ) -> None:
#         self.force_field = force_field
#         if os.environ['OPENMM_CPU_THREADS'] != str(4):
#             raise ValueError("OPENMM_CPU_THREADS must be set to 4")

#     def single_point(
#         self, 
#         molecule: Compound, 
#         conformer_idx: int, 
#         solvent: str = 'Methanol'
#     ) -> float:
#         system = self.force_field.create_openmm_system(
#             molecule.mol_topology,
#             charge_from_molecules=[molecule.ff_mol],
#         )

#         platform = Platform.getPlatformByName("CPU")
#         integrator = LangevinIntegrator(
#             300 * unit.kelvin,
#             1 / unit.picosecond,
#             0.002 * unit.picoseconds,
#         )
#         simulation = app.Simulation(
#             molecule.mol_topology.to_openmm(), system, integrator, platform=platform
#         )

#         simulation.context.setPositions(molecule.openmm_conformers[conformer_idx])
#         minimized_state = simulation.context.getState(
#             getPositions=False, getEnergy=True
#         )
#         energy = minimized_state.getPotentialEnergy().value_in_unit(
#             unit.kilocalories_per_mole
#         )
#         return energy

#     def optimization(
#         self,
#         molecule: Compound,
#         conformer_idx: int,
#         solvent: Optional[str] = None
#     ) -> float:
#         system = self.force_field.create_openmm_system(
#             molecule.mol_topology,
#             charge_from_molecules=[molecule.ff_mol],
#         )

#         platform = Platform.getPlatformByName("CPU")
#         integrator = LangevinIntegrator(
#             300 * unit.kelvin,
#             1 / unit.picosecond,
#             0.002 * unit.picoseconds,
#         )
#         simulation = app.Simulation(
#             molecule.mol_topology.to_openmm(), system, integrator, platform=platform
#         )

#         simulation.context.setPositions(molecule.openmm_conformers[conformer_idx])
#         simulation.minimizeEnergy(tolerance=100)
#         minimized_state = simulation.context.getState(
#             getPositions=False, getEnergy=True
#         )
#         energy = minimized_state.getPotentialEnergy().value_in_unit(
#             unit.kilocalories_per_mole
#         )
#         return energy