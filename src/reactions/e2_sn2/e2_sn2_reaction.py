import autode as ade
from autode.conformers.conformer import Conformer
from autode.species.complex import Complex

from typing import List

from src.reactions.e2_sn2.template import E2Sn2ReactionIndices
from src.utils import write_xyz_file, translate_rotate_reactant


class E2Sn2Reaction:

    def __init__(
        self,
        substrate_smiles: str,
        nucleophile_smiles: str,
        indices: List[List[int]],
        sn2_reaction_complex_template, # : List[ReactionTemplate],
        e2_reaction_complex_template, # : List[ReactionTemplate],
        n_conformers: int = 100,
        max_iter: int = 100
    ) -> None:
        self.nucleophile = ade.Molecule(smiles=nucleophile_smiles)
        self.substrate = ade.Molecule(smiles=substrate_smiles)
        self.substrate._generate_conformers(n_conformers)

        indices = indices[len(indices[0]) != 4]
        indices[-1] = -1
        self.e2sn2_indices = E2Sn2ReactionIndices(*indices)

        self.sn2_reaction_complex_template = sn2_reaction_complex_template
        self.e2_reaction_complex_template = e2_reaction_complex_template

        self.max_iter = max_iter
        self.threshold = -30

    def _compute_sn2_barrier(
        self,
        method,
        ts_optimizer,
        random_seed: int = 42
    ):
        """ Construct Reaction Complexes """
        reaction_complexes = []
        for idx, conformer in enumerate(self.substrate.conformers):
            success, rc, bond_rearran = self.sn2_reaction_complex_template.generate_reaction_complex(
                conformer,
                self.nucleophile,
                self.e2sn2_indices,
                method
            )
            if success:
                reaction_complexes.append((rc, bond_rearran))

        if len(reaction_complexes) == 0:
            print("No sucesfull reaction complexes made")
            return None

        """ Construct TS initial guess """   
        ts_guesses = []
        for rc, bond_rearran in reaction_complexes:
            mol_1 = Conformer(atoms=rc.atoms[:-1])
            mol_2 = Conformer(atoms=rc.atoms[-1:], charge=-1)
            ts_guess = Complex(mol_1, mol_2)
            translate_rotate_reactant(
                ts_guess,
                bond_rearrangement=bond_rearran,
                shift_factor=1.5 if ts_guess.charge == 0 else 2.5,
                random_seed=random_seed
            )
            ts_guesses.append((rc, ts_guess))

        """ TS optimization """
        barriers = []
        for (rc, ts_guess) in ts_guesses:
            try:
                ts_optimizer.optimise(
                    species=ts_guess,
                    method=method,
                    maxiter=self.max_iter
                )
            except Exception as e:
                print(e)
                continue

            try:
                ts_guess.calc_hessian(method=method)
                if ts_guess.imaginary_frequencies is not None:
                    for im_freq in ts_guess.imaginary_frequencies:
                        if im_freq < self.threshold:
                            barriers.append(ts_guess.energy - rc.energy)
                            break
            except Exception as e:
                print(e)
                continue
        
        return barriers
