from typing import List, Union
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from XTB import xtb


from compound import Compound, Conformation


class EASReaction:
    rxns = [
        AllChem.ReactionFromSmarts('[C;R:1]([Br])=[C,N;R;H1:2]>>[C,R:1][*H+:2]'),
        AllChem.ReactionFromSmarts('[C,R:1](Br)=[C,N;R;H0:2]>>[C,R:1][*H+:2]')
    ]

    def __init__(
        self,
        substrate_smiles: str,
        product_smiles: str,
    ) -> None:
        self.substrate = Compound.from_smiles(substrate_smiles)
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
        return Compound(product)

    def compute_energy(
        self,
        molecule: Union[Compound, Conformation],
        conformer_idx: int
    ) -> float:
        if molecule.conformers[conformer_idx] is not None:
            energy, _ = xtb(
                molecule=molecule,
                conformer_idx=conformer_idx,
                keywords=[],
                method='2',
                solvent='Methanol',
                xcontrol_file=None
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

        for sub_conf_idx in range(len(self.substrate.conformers)):
            energies[0].append(self.compute_energy(self.substrate, sub_conf_idx))
        
        for ts_conf_idx in range(len(self.transition_state.conformers)):
            energies[1].append(self.compute_energy(self.transition_state, ts_conf_idx))

        return energies