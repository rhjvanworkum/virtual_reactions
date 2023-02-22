import autode as ade
from autode.species.complex import Complex
from autode.opt.optimisers import PRFOptimiser
from autode.wrappers.XTB import XTB

from typing import Dict
import os
import shutil
import numpy as np

from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from src.reactions.e2_sn2.template import E2Sn2ReactionIndices, Sn2ReactionComplexTemplate


def check_C_nuc_distance(
    reaction_complex: Complex,
    indices: E2Sn2ReactionIndices,
    dist_dict: Dict[str, float],
    hydrogen_dist_dict: Dict[str, float],
    nuc_symbol: str
) -> bool:
    C_nuc_dist = np.linalg.norm(reaction_complex.atoms[indices.nucleophile_idx].coord - reaction_complex.atoms[indices.central_atom_idx].coord)
    return C_nuc_dist > dist_dict[nuc_symbol]

def check_nuc_H_distances(
    reaction_complex: Complex,
    indices: E2Sn2ReactionIndices,
    dist_dict: Dict[str, float],
    hydrogen_dist_dict: Dict[str, float],
    nuc_symbol: str
) -> bool:
    h_distances = []
    for atom in reaction_complex.atoms[:-1]:
        if atom.atomic_symbol == 'H':
            h_distances.append(np.linalg.norm(reaction_complex.atoms[indices.nucleophile_idx].coord - atom.coord))
    return min(h_distances) > hydrogen_dist_dict[nuc_symbol]

# def check_angle()

# def check_isomorphism()

if __name__ == "__main__":
    BASE_DIR = '/home/ruard/code/virtual_reactions/calculations/'
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'

    substrate = "[CH3:1][C@@H:2]([NH2:3])[CH2:4][Cl:5]"
    nucleophile = "[F-:6]"
    indices = [[4, 3, 5], [4, 3, 1, 5]]

    # create temp dir here & checkout
    dir = os.path.join(BASE_DIR, f'{1}_calc/')
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    os.chdir(dir)

    sn2_reaction_complex_template = Sn2ReactionComplexTemplate(
        checks=[check_C_nuc_distance, check_nuc_H_distances]
    )

    reaction = E2Sn2Reaction(
        substrate_smiles=substrate,
        nucleophile_smiles=nucleophile,
        indices=indices,
        sn2_reaction_complex_template=sn2_reaction_complex_template,
        e2_reaction_complex_template=None
    )

    method = XTB()

    ts_optimizer = PRFOptimiser(
        maxiter=100,
        gtol=1e-4,
        etol=1e-3
    )

    output = reaction._compute_sn2_barrier(
        method=method,
        ts_optimizer=ts_optimizer
    )
    print(output)