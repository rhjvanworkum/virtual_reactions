from typing import Callable, List, Literal, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem

from src.compound import Compound, Conformation
from src.methods.methods import Method

class DAReaction:

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
            self.substrates = [
                Compound.from_smiles(
                    substrate_smiles.split('.')[0], 
                    has_openmm_compatability=has_openmm_compatability
                ),
                Compound.from_smiles(
                    substrate_smiles.split('.')[1],
                    has_openmm_compatability=has_openmm_compatability
                ),
            ]
            for substrate in self.substrates:
                substrate.generate_conformers()
                substrate.optimize_conformers()

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
                molecule=molecule, 
                conformer_idx=conformer_idx, 
                solvent=self.solvent
            )
        else:
            energy = None

        return energy

    def compute_conformer_energies(
        self,
    ) -> List[List[float]]:
        energies = [
            [],
            [],
            []
        ]

        if not self.compute_product_only:
            for substrate_idx in range(2):
                for sub_conf_idx in range(len(self.substrates[substrate_idx].conformers)):
                    try:
                        energies[substrate_idx].append(self.compute_energy(self.substrates[substrate_idx], sub_conf_idx))
                    except:
                        continue

        for ts_conf_idx in range(len(self.product.conformers)):
            try:
                energies[-1].append(self.compute_energy(self.product, ts_conf_idx))
            except:
                continue

        return energies