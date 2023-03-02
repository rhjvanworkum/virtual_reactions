import autode as ade
from autode.species.complex import Complex
from autode.opt.optimisers import PRFOptimiser
from autode.wrappers.XTB import XTB
from autode.values import Frequency, Distance, Allocation

from typing import Dict
import os
import shutil
import numpy as np

from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from src.reactions.e2_sn2.template import E2ReactionComplexTemplate, E2Sn2ReactionIndices, Sn2ReactionComplexTemplate
from src.utils import angle_between


def check_C_H_Y_angle(
    reaction_complex: Complex,
    h_idx: int,
    indices: E2Sn2ReactionIndices,
    dist_dict: Dict[str, float],
    nuc_symbol: str
) -> bool:
    vec_1 = reaction_complex.atoms[indices.attacked_atom_idx].coord - reaction_complex.atoms[h_idx].coord
    vec_2 = reaction_complex.atoms[indices.nucleophile_idx].coord - reaction_complex.atoms[h_idx].coord
    angle = angle_between(vec_1, vec_2)
    print(angle)
    return angle > 170


if __name__ == "__main__":
    BASE_DIR = '/home/ruard/code/virtual_reactions/calculations/'
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
    ade.Config.max_atom_displacement = Distance(7.0, units="Å")
    ade.Config.rmsd_threshold = Distance(0.1, units="Å")

    substrate = "[CH3:1][C@@H:2]([NH2:3])[CH2:4][Cl:5]"
    nucleophile = "[F-:6]"
    indices = [[4, 3, 5], [4, 3, 1, 5]]
    # substrate = "[N:1]#[C:2][CH2:3][C@:4]([NH2:5])([Cl:6])[N+:7](=[O:8])[O-:9]"
    # nucleophile = "[H-:10]"
    # indices = [[5, 3, 9], [5, 3, 2, 9]]

    # create temp dir here & checkout
    dir = os.path.join(BASE_DIR, f'{1}_calc/')
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    os.chdir(dir)

    e2_reaction_complex_template = E2ReactionComplexTemplate(
        checks=[check_C_H_Y_angle]
    )

    reaction = E2Sn2Reaction(
        substrate_smiles=substrate,
        nucleophile_smiles=nucleophile,
        indices=indices,
        sn2_reaction_complex_template=None,
        e2_reaction_complex_template=e2_reaction_complex_template,
        n_conformers=300,
        random_seed=42
    )

    method = XTB()

    ts_optimizer = PRFOptimiser(
        maxiter=100,
        gtol=1e-2,
        etol=1e-2,
        init_alpha=0.03,
        recalc_hessian_every=5
    )

    output = reaction._compute_e2_barrier(
        method=method,
        ts_optimizer=ts_optimizer
    )
    print(output)