from typing import List, Any, Tuple
from rdkit import Chem

from src.utils import Atom
from src.compound import Compound

class E2Sn2Reaction:

    def __init__(
        self,
        reactant_conformers: List[List[Atom]],
        ts: List[Atom],
        product_conformers: List[List[Atom]],
        method: Any,
    ) -> None:
        self.reactant_conformers = reactant_conformers
        self.ts = ts
        self.product_conformers = product_conformers

        self.method = method

    def compute_activation_energies(self) -> List[float]:
        reactant_energies = []
        for conf in self.reactant_conformers:
            mol = Compound(Chem.MolFromSmiles('C'))
            mol.charge = -1
            mol.conformers = [conf]
            reactant_energies.append(self.method.single_point(mol, 0, None))
        
        mol = Compound(Chem.MolFromSmiles('C'))
        mol.charge = -1
        mol.conformers = [self.ts]
        ts_energy = self.method.single_point(mol, 0, None)

        return [ts_energy - reac_energy if (ts_energy is not None and reac_energy is not None) else None for reac_energy in reactant_energies]

    def compute_reaction_energies(self) -> Tuple[List[float]]:
        reactant_energies = []
        for conf in self.reactant_conformers:
            mol = Compound(Chem.MolFromSmiles('C'))
            mol.charge = -1
            mol.conformers = [conf]
            reactant_energies.append(self.method.single_point(mol, 0, None))
        
        product_energies = []
        for conf in self.product_conformers:
            mol = Compound(Chem.MolFromSmiles('C'))
            mol.charge = -1
            mol.conformers = [conf]
            product_energies.append(self.method.single_point(mol, 0, None))

        return reactant_energies, product_energies