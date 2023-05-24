from typing import List, Union
from rdkit import Chem
from rdkit.Chem import AllChem

from src.compound import Compound, Conformation
from src.methods import Method

class EASReaction:
    rxns = [
        AllChem.ReactionFromSmarts('[C,c:1](Br)=[C,c:2]>>[C,c:1]-[C+,c+:2]'),
        AllChem.ReactionFromSmarts('[C,c:1](Br)=[N,n:2]>>[C,c:1]-[NH0+,nH0+:2]')
    ]

    def __init__(
        self,
        substrate_smiles: str,
        product_smiles: str,
        solvent: str,
        method: Method,
        has_openmm_compatability: bool = False,
        compute_product_only: bool = False
    ) -> None:
        self.solvent = solvent
        self.method = method
        self.has_openmm_compatability = has_openmm_compatability
        self.compute_product_only = compute_product_only

        if not self.compute_product_only:
            self.substrate = Compound.from_smiles(
                substrate_smiles, 
                has_openmm_compatability=has_openmm_compatability
            )
            self.substrate.generate_conformers()
            self.substrate.optimize_conformers()

        self.transition_state = self._generate_protonated_ts(product_smiles)
        self.transition_state.generate_conformers()
        self.transition_state.optimize_conformers()

    def _generate_protonated_ts(
        self,
        product_smiles: str,
    ) -> Compound:
        product_mol = Chem.MolFromSmiles(product_smiles)
        Chem.Kekulize(product_mol, clearAromaticFlags=True)

        products = []
        for rxn in self.rxns:
            for p in rxn.RunReactants((product_mol,)):
                products.append(p[0])

        product = products[0]
        Chem.SanitizeMol(product)
        product = Chem.AddHs(product)
        return Compound(product, self.has_openmm_compatability)

    def compute_energy(
        self,
        molecule: Union[Compound, Conformation],
        conformer_idx: int
    ) -> float:
        if molecule.conformers[conformer_idx] is not None:
            energy = self.method.optimization(
                molecule, conformer_idx, self.solvent
            )
        else:
            energy = None

        return energy

    def compute_conformer_energies(
        self,
    ) -> List[List[float]]:
        energies = [
            [],
            []
        ]

        if not self.compute_product_only:
            for sub_conf_idx in range(len(self.substrate.conformers)):
                energies[0].append(self.compute_energy(self.substrate, sub_conf_idx))
        
        for ts_conf_idx in range(len(self.transition_state.conformers)):
            energies[1].append(self.compute_energy(self.transition_state, ts_conf_idx))

        return energies