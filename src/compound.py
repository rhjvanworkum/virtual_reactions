try:
    import openmm
    from openmm import LangevinIntegrator, app, unit
    from openmm import *
except ImportError:
    from simtk import openmm, unit
    from simtk.openmm import LangevinIntegrator, app
    from simtk.openmm import *

import time

from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from src.methods.ORCA import orca
from src.methods.XTB import run_xtb
import numpy as np
from operator import itemgetter

import autode as ade
from autode.atoms import Atom as AutodeAtom

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from src.methods.fukui import compute_fukui_indices

from src.utils import Atom, EditedRDKitToolkitWrapper, write_xyz_file

def _compute_rdkit_partial_charges(
    mol: Chem.Mol
) -> openmm.unit.Quantity:
    ComputeGasteigerCharges(mol)
    pc = openmm.unit.Quantity(
        value=np.array(
            [float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()]
        ),
        unit=openmm.unit.constants.elementary_charge,
    )
    return pc

def convert_rdkit_conformer(
    mol,
    conformer
):
    geometry = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = conformer.GetAtomPosition(i)
        geometry.append(Atom(atom.GetSymbol(), positions.x, positions.y, positions.z))
    return geometry

class Conformation:

    def __init__(
        self,
        geometry: List[Atom],
        charge: int,
        mult: int
    ) -> None:
        self.conformers = [geometry]
        self.charge = charge
        self.mult = mult

    def to_xyz(
        self, 
        xyz_file_name: str,
        conformer_idx : int
    ) -> None:
        conformer = self.conformers[conformer_idx]
        write_xyz_file(conformer, xyz_file_name)

class Compound:

    def __init__(
        self,
        rdkit_mol: Chem.Mol,
        has_openmm_compatability: bool = False,
        mult: int = 1,
        solvent: Optional[str] = None
    ) -> None:
        self.rdkit_mol = rdkit_mol
        self.charge = Chem.rdmolops.GetFormalCharge(self.rdkit_mol)
        self.mult = mult
        self.solvent = solvent

        self.conformers = []

        self.has_openmm_compatability = has_openmm_compatability
        if self.has_openmm_compatability:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(self.rdkit_mol))
            self.ff_mol = EditedRDKitToolkitWrapper().from_smiles(smiles, allow_undefined_stereo=True)
            self.ff_mol.partial_charges = _compute_rdkit_partial_charges(
                Chem.AddHs(self.rdkit_mol)
            )
            self.openmm_conformers = []
            self.mol_topology = self.ff_mol.to_topology()

    @classmethod
    def from_smiles(cls, smiles: str, **kwargs):
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        return cls(rdkit_mol, **kwargs)

    @property
    def n_atoms(self):
        return len(self.conformers[0])

    def to_xyz(
        self, 
        xyz_file_name: str,
        conformer_idx : int
    ) -> None:
        conformer = self.conformers[conformer_idx]
        write_xyz_file(conformer, xyz_file_name)

    def to_autode_mol(
        self,
        conformer_idx: int
    ) -> ade.Species:
        return ade.Species(
            name=str(time.time()),
            atoms=[
                AutodeAtom(atomic_symbol=atom.atomic_symbol, x=atom.x, y=atom.y, z=atom.z) for atom in self.conformers[conformer_idx]
            ],
            charge=self.charge,
            mult=self.mult,
            solvent_name=self.solvent
        )

    def _set_openmm_conformers(self):
        self.openmm_conformers = [
            openmm.unit.Quantity(
                value=np.array([[atom.x, atom.y, atom.z] for atom in conf]), unit=openmm.unit.constants.angstrom
            ) for conf in self.conformers
        ]

    def generate_conformers(
        self,
        rot_conf: int = 3,
        min_conf: int = 1,
        max_conf: int = 20,
    ):
        rot_bond = rdMolDescriptors.CalcNumRotatableBonds(self.rdkit_mol)
        n_confs = min(min_conf + rot_conf * rot_bond, max_conf)

        AllChem.EmbedMultipleConfs(
            self.rdkit_mol, 
            numConfs=n_confs,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True, 
            ETversion=2
        )

        if self.rdkit_mol.GetNumConformers() > 0:
            AllChem.UFFOptimizeMolecule(self.rdkit_mol)

        self.conformers = []
        for idx in range(self.rdkit_mol.GetNumConformers()):
            self.conformers.append(
                convert_rdkit_conformer(
                    self.rdkit_mol,
                    self.rdkit_mol.GetConformer(idx)
                )
            )
        if self.has_openmm_compatability:
            self._set_openmm_conformers()

    def optimize_conformers(
        self,
        num_cpu: int = 4,
        conf_cutoff: float = 3.0
    ):
        keywords = ['--opt']
        method = 'ff'
        solvent = self.solvent
        xcontrol_file = None

        xtb_arguments = [
            (self, keywords, idx, method, solvent, xcontrol_file) for idx in range(len(self.conformers))
        ]

        with ProcessPoolExecutor(max_workers=num_cpu) as executor:
            results = executor.map(run_xtb, xtb_arguments)

        energies, geometries = [], []
        for result in results:
            energy, geometry = result
            if energy is not None and geometry is not None:
                energies.append(energy)
                geometries.append(geometry)

        if len(energies) > 0:
            energies = np.array(energies)
            rel_energies = energies - np.min(energies) #covert to relative energies
            below_cutoff = (rel_energies <= conf_cutoff).sum() #get number of conf below cutoff
            conf_tuble = list(zip(geometries, rel_energies)) #make a tuble
            conf_tuble = sorted(conf_tuble, key=itemgetter(1))[:below_cutoff] #get only the best conf below cutoff
            best_conformers = [item[0] for item in conf_tuble]

            self.conformers = best_conformers
            if self.has_openmm_compatability:
                self._set_openmm_conformers()

    def optimize_lowest_conformer(
        self,
        num_cpu: int = 4
    ):
        keywords = ['--opt']
        method = '2'
        solvent = self.solvent
        xcontrol_file = None

        xtb_arguments = [
            (self, keywords, idx, method, solvent, xcontrol_file) for idx in range(len(self.conformers))
        ]

        with ProcessPoolExecutor(max_workers=num_cpu) as executor:
            results = executor.map(run_xtb, xtb_arguments)

        energies, geometries = [], []
        for result in results:
            energy, geometry = result
            if energy is not None and geometry is not None:
                energies.append(energy)
                geometries.append(geometry)

        if len(energies) > 0:
            best_conformer = geometries[np.argmin(energies)]
            self.conformers = [best_conformer]
        else:
            self.conformers = []

        if self.has_openmm_compatability:
            self._set_openmm_conformers()        


    def compute_fukui_indices(
        self,
        functional,
        basis_set
    ):
        if len(self.conformers) == 0:
            return None
        else:
            # 1. do geom opt
            try:
                coords = orca(
                    molecule=self,
                    job="opt",
                    conformer_idx=0,
                    functional=functional,
                    basis_set=basis_set
                )
                new_conf = [
                    Atom(atomic_symbol=a.atomic_symbol, 
                        x=coords[i, 0],
                        y=coords[i, 1],
                        z=coords[i, 2]) for i, a in enumerate(self.conformers[0])
                ]
                self.conformers = [new_conf]
            except:
                return None
            
            # 2. do sp on radical anion -> elec parr fn
            try:
                self.charge, self.mult = -1, 2
                _, elec_parr_idxs = orca(
                    molecule=self,
                    job="sp",
                    conformer_idx=0
                )
            except:
                return None

            # 3. do sp on radical cation -> nuc parr fn
            try:
                self.charge, self.mult = 1, 2
                _, nuc_parr_idxs = orca(
                    molecule=self,
                    job="sp",
                    conformer_idx=0
                )
            except:
                return None

            return (elec_parr_idxs, nuc_parr_idxs)