


from typing import List, Union
from src.compound import Compound, Conformation
from src.methods.methods import Method


class DAReaction:

    def __init__(
        self,
        substrate_smiles: str,
        product_smiles: str,
        method: Method,
        has_openmm_compatability: bool = False,
        compute_product_only: bool = False
    ) -> None:
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

        self.product = Compound.from_smiles(
            product_smiles,
            has_openmm_compatability=has_openmm_compatability
        )
        self.product.generate_conformers()
        self.product.optimize_conformers()

    def compute_energy(
        self,
        molecule: Union[Compound, Conformation],
        conformer_idx: int
    ) -> float:
        if molecule.conformers[conformer_idx] is not None:
            energy = self.method.optimization(
                molecule, conformer_idx,  None
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
        
        for ts_conf_idx in range(len(self.product.conformers)):
            energies[1].append(self.compute_energy(self.product, ts_conf_idx))

        return energies