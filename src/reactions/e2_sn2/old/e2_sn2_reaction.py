from typing import Any, List, Tuple, Union
from src.methods.XTB import xtb

from src.compound import Compound, Conformation
from src.reactions.e2_sn2.old.template import E2Sn2ReactionIndices, ReactionTemplate
from src.utils import Atom

FORCE_CONSTANT = 2

def construct_xcontrol_file(
    distance_constraints: List[Tuple[Tuple[int], float]]
) -> str:
    string = ""
    for (i, j), dist in distance_constraints:
        string += f"$constrain\n"
        string += f"force constant={FORCE_CONSTANT}\n"
        string += f"distance:{i+1}, {j+1}, {dist:.4f}\n$"
    return string


class E2Sn2Reaction:

    def __init__(
        self,
        substrate_smiles: str,
        nucleophile_smiles: str,
        indices: List[List[int]],
        reaction_complex_templates: List[ReactionTemplate],
        transition_state_templates: List[ReactionTemplate]
    ) -> None:
        self.substrate = Compound.from_smiles(substrate_smiles)
        self.substrate.generate_conformers()
        self.substrate.optimize_conformers()

        self.nucleophile_smiles = nucleophile_smiles
        self.nucleophile = Conformation(
            geometry=[Atom(''.join([c for c in nucleophile_smiles if c.isupper()]), 0.0, 0.0, 0.0)],
            charge=-1,
            mult=0
        )
 
        self.e2sn2_indices = E2Sn2ReactionIndices(*indices[len(indices[0]) != 4])

        self.rc_templates = reaction_complex_templates
        self.ts_templates = transition_state_templates

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

    def optimize_transition_state(
        self,
        molecule: Union[Compound, Conformation],
        conformer_idx: int,
        distance_constraints: List[Tuple[Tuple[int], float]]
    ):
        if molecule.conformers[conformer_idx] is not None:
            energy, _ = xtb(
                molecule=molecule,
                conformer_idx=conformer_idx,
                keywords=['--opt'],
                method='2',
                solvent='Methanol',
                xcontrol_file=construct_xcontrol_file(distance_constraints)
            )
        else:
            energy = None

        return energy

    def compute_reaction_barriers(
        self,
    ):
        reaction_barriers = []

        # iterate over all conformers of the substrate
        for conf_idx in range(len(self.substrate.conformers)):
            reaction_barriers.append([])

            for rc_template, ts_template in zip(self.rc_templates, self.ts_templates):

                # construct Reaction complex
                rc, distance_constraints = rc_template.generate_ts(
                    self.substrate.conformers[conf_idx], 
                    self.nucleophile_smiles,
                    self.e2sn2_indices
                )
                rc = Conformation(geometry=rc, charge=-1, mult=0)
                rc_energy = self.optimize_transition_state(
                    molecule=rc, 
                    conformer_idx=0,
                    distance_constraints=distance_constraints
                )

                # generate a transition state guess
                ts, distance_constraints = ts_template.generate_ts(
                    self.substrate.conformers[conf_idx], 
                    self.nucleophile_smiles,
                    self.e2sn2_indices
                )
                ts = Conformation(geometry=ts, charge=-1, mult=0)
                ts_energy = self.optimize_transition_state(
                    molecule=ts, 
                    conformer_idx=0,
                    distance_constraints=distance_constraints
                )

                if ts_energy is not None and rc_energy is not None:
                    barrier = (ts_energy - rc_energy)
                else:
                    barrier = 1e6
                    
                reaction_barriers[-1].append(barrier)

        return reaction_barriers

if __name__ == "__main__":
    substrate_smiles = "[N:1]#[C:2][C@@H:3]([NH2:4])[CH2:5][Br:6]"
    nucleophile_smiles = "[F-:7]"
    indices = [[5, 4, 6], [5, 4, 2, 6]]

    reaction = E2Sn2Reaction(
        substrate_smiles=substrate_smiles,
        nucleophile_smiles=nucleophile_smiles,
        indices=indices
    )

    print(reaction.compute_reaction_barriers())