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

import os
import numpy as np
from typing import List, Optional

from src.methods.NWCHEM import nwchem
from src.methods.pyscf import pyscf
from src.methods.XTB import xtb
from src.compound import Compound, Conformation

from huxel.main import compute_energy_of_rdkit_mol

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


class HuckelMethod(Method):
    def __init__(self, parameter_file = None) -> None:
        self.parameter_file = parameter_file

    def single_point(
        self, 
        molecule: Compound, 
        conformer_idx: int, 
        solvent: str = 'Methanol'
    ) -> float:
        return np.asarray(compute_energy_of_rdkit_mol(molecule, conformer_idx, self.parameter_file))
    
    def optimization(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: str = 'Methanol'
    ) -> float:
        raise NotImplementedError("Not implemented yet")


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


class NwchemMethod(Method):
    def __init__(
        self,
        functional: str,
        basis_set: str,
        n_cores: int = 4
    ) -> None:
        self.functional = functional
        self.basis_set = basis_set
        self.n_cores = n_cores

    def single_point(
        self, 
        molecule: Compound, 
        conformer_idx: int, 
        solvent: str = 'Methanol'
    ) -> float:
        energy, _ = nwchem(
            molecule=molecule,
            conformer_idx=conformer_idx,
            keywords=[],
            solvent=solvent,
            functional=self.functional,
            basis_set=self.basis_set,
            n_cores=self.n_cores
        )
        return energy

    def optimization(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: str = 'Methanol',
    ) -> float:
        energy, _ = nwchem(
            molecule=molecule,
            conformer_idx=conformer_idx,
            keywords=['--opt'],
            solvent=solvent,
            functional=self.functional,
            basis_set=self.basis_set,
            n_cores=self.n_cores
        )
        return energy


class PyscfMethod(Method):
    def __init__(
        self,
        functional: str,
        basis_set: str,
        n_cores: int = 4
    ) -> None:
        self.functional = functional
        self.basis_set = basis_set
        self.n_cores = n_cores

    def single_point(
        self, 
        molecule: Compound, 
        conformer_idx: int, 
        solvent: str = 'Methanol'
    ) -> float:
        energy, _ = pyscf(
            molecule=molecule,
            conformer_idx=conformer_idx,
            keywords=[],
            solvent=solvent,
            functional=self.functional,
            basis_set=self.basis_set,
            n_cores=self.n_cores
        )
        return energy

    def optimization(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: str = 'Methanol',
    ) -> float:
        energy, _ = pyscf(
            molecule=molecule,
            conformer_idx=conformer_idx,
            keywords=['--opt'],
            solvent=solvent,
            functional=self.functional,
            basis_set=self.basis_set,
            n_cores=self.n_cores
        )
        return energy

class OrcaMethod(Method):

    def __init__(self) -> None:
        super().__init__()

        

    def single_point(self, molecule: Compound, conformer_idx: int, solvent: str = 'Methanol') -> float:
        return super().single_point(molecule, conformer_idx, solvent)


class ForceFieldMethod(Method):
    def __init__(
        self,
        force_field: ForceField
    ) -> None:
        self.force_field = force_field
        if os.environ['OPENMM_CPU_THREADS'] != str(4):
            raise ValueError("OPENMM_CPU_THREADS must be set to 4")

    def single_point(
        self, 
        molecule: Compound, 
        conformer_idx: int, 
        solvent: str = 'Methanol'
    ) -> float:
        system = self.force_field.create_openmm_system(
            molecule.mol_topology,
            charge_from_molecules=[molecule.ff_mol],
        )

        platform = Platform.getPlatformByName("CPU")
        integrator = LangevinIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        simulation = app.Simulation(
            molecule.mol_topology.to_openmm(), system, integrator, platform=platform
        )

        simulation.context.setPositions(molecule.openmm_conformers[conformer_idx])
        minimized_state = simulation.context.getState(
            getPositions=False, getEnergy=True
        )
        energy = minimized_state.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )
        return energy

    def optimization(
        self,
        molecule: Compound,
        conformer_idx: int,
        solvent: Optional[str] = None
    ) -> float:
        system = self.force_field.create_openmm_system(
            molecule.mol_topology,
            charge_from_molecules=[molecule.ff_mol],
        )

        platform = Platform.getPlatformByName("CPU")
        integrator = LangevinIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        simulation = app.Simulation(
            molecule.mol_topology.to_openmm(), system, integrator, platform=platform
        )

        simulation.context.setPositions(molecule.openmm_conformers[conformer_idx])
        simulation.minimizeEnergy(tolerance=100)
        minimized_state = simulation.context.getState(
            getPositions=False, getEnergy=True
        )
        energy = minimized_state.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )
        return energy