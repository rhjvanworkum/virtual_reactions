from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from typing import Dict
import ast
import numpy as np
import os
from tqdm import tqdm

import autode as ade
from autode.species.complex import Complex
from autode.opt.optimisers import PRFOptimiser
from autode.wrappers.XTB import XTB

from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from src.reactions.e2_sn2.template import E2Sn2ReactionIndices, Sn2ReactionComplexTemplate


BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
N_CPUS = 90
N_CPUS_CONFORMERS = 4

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

def get_reaction_barriers(args):
    substrate_smiles, nucleophile_smiles, indices, idx = args

    # create temp dir here & checkout
    dir = os.path.join(BASE_DIR, f'{idx}_calc/')
    if os.path.exists(dir):
        os.removedirs(dir)
    os.makedirs(dir)
    os.chdir(dir)

    sn2_reaction_complex_template = Sn2ReactionComplexTemplate(
        checks=[check_C_nuc_distance, check_nuc_H_distances]
    )

    reaction = E2Sn2Reaction(
        substrate_smiles=substrate_smiles,
        nucleophile_smiles=nucleophile_smiles,
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
    
    return output

if __name__ == "__main__":
    class_data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')

    arguments = []
    labels = []

    for idx, row in class_data.iterrows():
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['products_run'])

        arguments.append((substrate_smiles, nucleophile_smiles, indices, idx))

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

    energies = []
    for result in results:
        energies.append(result)

    data = {
        "sn2_energies": energies,
    }
    df = pd.DataFrame(data)
    df.to_csv('test.csv')



