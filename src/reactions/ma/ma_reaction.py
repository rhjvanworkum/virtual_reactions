from typing import Callable, List, Literal, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem

from src.compound import Compound, Conformation
from src.methods.methods import Method

def simulate_reaction(substrates, reaction_smarts, fix_oxygen: bool = False):
    products = []
    products += reaction_smarts.RunReactants(substrates)
    if len(substrates) == 2:
        substrates = [substrates[1], substrates[0]]
        products += reaction_smarts.RunReactants(substrates)
    products = [Chem.MolToSmiles(product[0]) for product in products]
    products = list(set(products))

    if fix_oxygen:
        products = [product.replace('=[O-]', '=[O]') for product in products]

    products = [Chem.MolFromSmiles(product) for product in products]
    return list(filter(lambda x: x is not None, products))

class MAReaction:
    INTERMEDIATE_RXN = AllChem.ReactionFromSmarts(
        "[#6:1]=[#6:2][#6:3](=[O:4]).[N,S,n,s:5]>>[N,S,n,s:5][#6:1][#6:2]=[#6:3](-[O-:4])"
    )

    INTERMEDIATE_TO_PRODUCT_RXN = AllChem.ReactionFromSmarts(
        "[N,S,n,s:5][#6:1][#6:2]=[#6:3](-[O-:4])>>[N,S,n,s:5][#6:1]-[#6:2][#6:3](=[O:4])"
    )

    def __init__(
        self,
        substrate_smiles: str,
        product_smiles: str,
        solvent: str,
        method: Method,
        has_openmm_compatability: bool = False,
        compute_product_only: bool = False
    ) -> None:
        self.substrate_smiles = substrate_smiles
        self.product_smiles = product_smiles

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

        self.ts_smiles = self._get_transition_state()
        self.transition_state = Compound(
            self.ts_smiles, 
            has_openmm_compatability=has_openmm_compatability
        )
        self.transition_state.generate_conformers()
        self.transition_state.optimize_conformers()

    def _get_transition_state(self):
        canonical_product_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.product_smiles), isomericSmiles=False)

        substrates = self.substrate_smiles.split('.')
        substrates = [Chem.MolFromSmiles(smi) for smi in substrates]
        products = simulate_reaction(substrates, self.INTERMEDIATE_RXN)
        if len(products) > 0:
            for product in products:
                outputs = simulate_reaction([product], self.INTERMEDIATE_TO_PRODUCT_RXN, fix_oxygen=True)
                for output in outputs:
                    output_smiles = Chem.MolToSmiles(output, isomericSmiles=False)
                    if output_smiles == canonical_product_smiles:
                        # ! found correct transition state
                        return product

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

        for ts_conf_idx in range(len(self.transition_state.conformers)):
            try:
                energies[-1].append(self.compute_energy(self.transition_state, ts_conf_idx))
            except:
                continue

        return energies