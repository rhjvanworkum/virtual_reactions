from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from typing import Dict
import ast
import numpy as np
import shutil
import os
from tqdm import tqdm

import autode as ade
from autode.species.complex import Complex
from autode.opt.optimisers import PRFOptimiser
from autode.wrappers.XTB import XTB
from autode.values import Distance

from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from src.reactions.e2_sn2.template import E2ReactionComplexTemplate, E2Sn2ReactionIndices, Sn2ReactionComplexTemplate
from src.utils import angle_between


BASE_DIR = '/home/ruard/code/virtual_reactions/calculations/'
N_PROCESSES = 2
RANDOM_SEED = 42

ade.Config.n_cores = 4
ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
ade.Config.max_atom_displacement = Distance(7.0, units="Å")
ade.Config.rmsd_threshold = Distance(0.05, units="Å")

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
    return angle > 160

def get_reaction_barriers(args):
    substrate_smiles, nucleophile_smiles, indices, label, idx = args

    # create temp dir here & checkout
    dir = os.path.join(BASE_DIR, f'{idx}_calc/')
    if os.path.exists(dir):
        os.removedirs(dir)
    os.makedirs(dir)
    os.chdir(dir)

    sn2_reaction_complex_template = Sn2ReactionComplexTemplate(
        checks=[check_C_nuc_distance, check_nuc_H_distances]
    )
    e2_reaction_complex_template = E2ReactionComplexTemplate(
        checks=[check_C_H_Y_angle]
    )

    reaction = E2Sn2Reaction(
        substrate_smiles=substrate_smiles,
        nucleophile_smiles=nucleophile_smiles,
        indices=indices,
        sn2_reaction_complex_template=sn2_reaction_complex_template,
        e2_reaction_complex_template=e2_reaction_complex_template,
        n_conformers=300,
        random_seed=RANDOM_SEED
    )

    method = XTB()

    ts_optimizer = PRFOptimiser(
        maxiter=100,
        gtol=1e-3,
        etol=1e-3,
        init_alpha=0.03,
        recalc_hessian_every=1
    )

    sn2_output = reaction._compute_sn2_barrier(
        method=method,
        ts_optimizer=ts_optimizer
    )
    e2_output = reaction._compute_e2_barrier(
        method=method,
        ts_optimizer=ts_optimizer
    )
    
    return f"{substrate_smiles}.{nucleophile_smiles}", sn2_output, e2_output, label

if __name__ == "__main__":
    class_data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')

    arguments = []

    for idx, row in class_data[:4].iterrows():
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['products_run'])
        if len(indices[0]) == 4:
            label = 'e2' # 'e2'
        else:
            label = 'sn2' # 'sn2'

        arguments.append((substrate_smiles, nucleophile_smiles, indices, label, idx))

    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

    sn2_states, e2_states, smiless, labels = [], [], [], []
    for result in results:
        smiles, sn2_output, e2_output, label = result
        sn2_states.append(sn2_output)
        e2_states.append(e2_output)
        smiless.append(smiles)
        labels.append(label)
  
    data = {
        "smiles": smiless,
        "label": labels,
        "sn2_states": sn2_states,
        "e2_states": e2_states
    }
    df = pd.DataFrame(data)
    df.to_csv('test_2.csv')

    shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR)