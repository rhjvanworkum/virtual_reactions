import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from XTB import run_xtb
import numpy as np
from operator import itemgetter

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

from utils import Atom, write_xyz_file

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
        rdkit_mol: Chem.Mol
    ) -> None:
        self.rdkit_mol = rdkit_mol

        self.charge = Chem.rdmolops.GetFormalCharge(self.rdkit_mol)
        self.mult = 0

        self.conformers = []

    @classmethod
    def from_smiles(cls, smiles: str):
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        return cls(rdkit_mol)

    def to_xyz(
        self, 
        xyz_file_name: str,
        conformer_idx : int
    ) -> None:
        conformer = self.conformers[conformer_idx]
        write_xyz_file(conformer, xyz_file_name)

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
        AllChem.UFFOptimizeMolecule(self.rdkit_mol)

        self.conformers = []
        for idx in range(self.rdkit_mol.GetNumConformers()):
            self.conformers.append(
                convert_rdkit_conformer(
                    self.rdkit_mol,
                    self.rdkit_mol.GetConformer(idx)
                )
            )

    def optimize_conformers(
        self,
        num_cpu: int = 4,
        conf_cutoff: float = 3.0
    ):
        keywords = ['--opt']
        method = 'ff'
        solvent = 'Methanol'
        xcontrol_file = None

        xtb_arguments = [
            (self, keywords, idx, method, solvent, xcontrol_file) for idx in range(len(self.conformers))
        ]

        results = [run_xtb(args) for args in xtb_arguments]
        # with ProcessPoolExecutor(max_workers=num_cpu) as executor:
        #     results = executor.map(run_xtb, xtb_arguments)

        energies, geometries = [], []
        for result in results:
            energy, geometry = result
            if energy is not None and geometry is not None:
                energies.append(energy)
                geometries.append(geometry)
        
        energies = np.array(energies)
        rel_energies = energies - np.min(energies) #covert to relative energies
        below_cutoff = (rel_energies <= conf_cutoff).sum() #get number of conf below cutoff
        conf_tuble = list(zip(geometries, rel_energies)) #make a tuble
        conf_tuble = sorted(conf_tuble, key=itemgetter(1))[:below_cutoff] #get only the best conf below cutoff
        best_conformers = [item[0] for item in conf_tuble]

        self.conformers = best_conformers