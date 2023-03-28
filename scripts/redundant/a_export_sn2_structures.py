from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from typing import Dict, List
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

from src.reactions.e2_sn2.old.e2_sn2_reaction import E2Sn2Reaction
from src.reactions.e2_sn2.old.template import E2ReactionComplexTemplate, E2Sn2ReactionIndices, Sn2ReactionComplexTemplate
from src.utils import Atom, angle_between

def write_xyz_file_a(atoms: List[Atom], filename: str):
  with open(filename, 'a') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
        f.write(atom.atomic_symbol)
        for cartesian in ['x', 'y', 'z']:
            if getattr(atom.coord, cartesian) < 0:
                f.write('         ')
            else:
                f.write('          ')
            f.write("%.5f" % getattr(atom.coord, cartesian))
        f.write('\n')

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

def check_leav_C_Y_angle(
    reaction_complex: Complex,
    indices: E2Sn2ReactionIndices,
    dist_dict: Dict[str, float],
    hydrogen_dist_dict: Dict[str, float],
    nuc_symbol: str
) -> bool:
    vec_1 = reaction_complex.atoms[indices.central_atom_idx].coord - reaction_complex.atoms[indices.leaving_group_idx].coord
    vec_2 = reaction_complex.atoms[indices.central_atom_idx].coord - reaction_complex.atoms[indices.nucleophile_idx].coord
    angle = angle_between(vec_1, vec_2)
    return angle > 150

sn2_reaction_complex_template = Sn2ReactionComplexTemplate(
    checks=[check_C_nuc_distance, check_nuc_H_distances, check_leav_C_Y_angle]
)

def construct_reaction_complex(
    nucleophile_smiles,
    substrate_smiles,
    indices,
    method
):
    nucleophile = ade.Molecule(smiles=nucleophile_smiles)
    substrate = ade.Molecule(smiles=substrate_smiles)
    substrate._generate_conformers(1000, 42)
    for conformer in substrate.conformers:
        conformer.optimise(method=method)

    indices = indices[len(indices[0]) != 4]
    indices[-1] = -1
    indices = E2Sn2ReactionIndices(*indices)

    reaction_complexes = []
    product_complexes = []
    for idx, conformer in enumerate(substrate.conformers):
        try:
            success, rc, pc, bond_rearran = sn2_reaction_complex_template.generate_reaction_complex(
                conformer,
                nucleophile,
                indices,
                method
            )
            if success:
                reaction_complexes.append(rc)
                product_complexes.append(pc)
        except Exception as e:
            continue

    return reaction_complexes, product_complexes


if __name__ == "__main__":
    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
    ade.Config.rmsd_threshold = Distance(0.03, units="Å")

    # read data
    data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')

    # prepare files
    if os.path.exists('/home/ruard/code/virtual_reactions/rcs.xyz'):
        os.remove('/home/ruard/code/virtual_reactions/rcs.xyz')
    if os.path.exists('/home/ruard/code/virtual_reactions/pcs.xyz'):
        os.remove('/home/ruard/code/virtual_reactions/pcs.xyz')

    # draw 10 random reactions
    os.chdir('./calculations/1_calc/')

    for idx in np.random.choice(np.arange(len(data)), size=10):
        rcs, pcs = construct_reaction_complex(
            data['smiles'].values[idx].split('.')[1],
            data['smiles'].values[idx].split('.')[0],
            ast.literal_eval(data['products_run'].values[idx]),
            XTB()
        )
        print(f'Created {len(rcs)} RCs and {len(pcs)} PCs at reaction {idx}')

        for rc in rcs:
            write_xyz_file_a(rc.atoms, '/home/ruard/code/virtual_reactions/rcs.xyz')

        for pc in pcs:
            write_xyz_file_a(pc.atoms, '/home/ruard/code/virtual_reactions/pcs.xyz')