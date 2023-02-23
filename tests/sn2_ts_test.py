
import ast
import os
from typing import List

import pandas as pd

from src.compound import Conformation
from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from src.reactions.e2_sn2.template import E2ReactionTemplate
from src.utils import read_xyz_file

BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 90
N_CPUS_CONFORMERS = 4

os.environ["BASE_DIR"] = BASE_DIR
os.environ["XTB_PATH"] = XTB_PATH

def test_consisten_atoms(
    substrate_smiles: str,
    nucleophile_smiles: str,
    indices: List[List[int]],
    idx
):
    template = E2ReactionTemplate(
        d_nucleophile=1.8,
        d_leaving_group=1.5,
        d_H=1.22
    )

    reaction = E2Sn2Reaction(
        substrate_smiles=substrate_smiles,
        nucleophile_smiles=nucleophile_smiles,
        indices=indices,
        reaction_complex_templates=[template],
        transition_state_templates=[template]
    )

    atom_count = {}
    for atom in reaction.substrate.rdkit_mol.GetAtoms():
        if atom.GetSymbol() not in atom_count.keys():
            atom_count[atom.GetSymbol()] = 1
        else:
            atom_count[atom.GetSymbol()] += 1
    nuc_atom = reaction.nucleophile.conformers[0][0].type
    if nuc_atom not in atom_count.keys():
        atom_count[nuc_atom] = 1
    else:
        atom_count[nuc_atom] += 1


    for conf_idx in range(len(reaction.substrate.conformers)):
        rc, distance_constraints = template.generate_ts(
            reaction.substrate.conformers[conf_idx], 
            reaction.nucleophile_smiles,
            reaction.e2sn2_indices
        )
        rc = Conformation(geometry=rc, charge=-1, mult=0)
        
        atom_count_after = {}
        for atom in rc.conformers[0]:
            if atom.type not in atom_count_after.keys():
                atom_count_after[atom.type] = 1
            else:
                atom_count_after[atom.type] += 1
        
        for key in atom_count.keys():
            assert atom_count[key] == atom_count_after[key]


        rc.to_xyz(f'test_{idx}.xyz', 0)
        geom = read_xyz_file(f'test_{idx}.xyz')

        atom_count_after = {}
        for atom in geom:
            if atom.type not in atom_count_after.keys():
                atom_count_after[atom.type] = 1
            else:
                atom_count_after[atom.type] += 1
        
        for key in atom_count.keys():
            assert atom_count[key] == atom_count_after[key]


    
    
if __name__ == "__main__":
    # substrate_smiles = '[CH3:1][C@@H:2]([NH2:3])[C@H:4]([Br:5])[N+:6](=[O:7])[O-:8]'
    # nucleophile_smiles = '[Cl-:9]'
    # indices = [[4, 3, 1, 8], [4, 3, 8]]


    # substrate_smiles = '[CH3:1][C@H:2]([Br:3])[N+:4](=[O:5])[O-:6]'
    # nucleophile_smiles = '[Cl-:7]'
    # indices = [[2, 1, 0, 6], [2, 1, 6]]

    # substrate_smiles = '[O:1]=[N+:2]([O-:3])[CH2:4][CH2:5][Cl:6]'
    # nucleophile_smiles = '[Cl-:7]'
    # indices = [[5, 4, 6], [5, 4, 3, 6]]

    substrate_smiles = '[CH3:1][C@H:2]([Br:3])[C:4]#[N:5]'
    nucleophile_smiles = '[Br-:6]'
    indices = [[2, 1, 0, 5], [2, 1, 5]]
    idx = 0



    # template = E2ReactionTemplate(
    #     d_nucleophile=1.8,
    #     d_leaving_group=1.8,
    #     d_H=1.22
    # )

    # reaction = E2Sn2Reaction(
    #     substrate_smiles=substrate_smiles,
    #     nucleophile_smiles=nucleophile_smiles,
    #     indices=indices,
    #     reaction_complex_templates=[template],
    #     transition_state_templates=[template]
    # )

    # conf_idx = 0
    # rc, distance_constraints = template.generate_ts(
    #     reaction.substrate.conformers[conf_idx], 
    #     reaction.nucleophile_smiles,
    #     reaction.e2sn2_indices
    # )
    # rc = Conformation(geometry=rc, charge=-1, mult=0)
    # rc.to_xyz('test.xyz', 0)





    # class_data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')[:90]

    # for idx, row in class_data.iterrows():
        # substrate_smiles = row['smiles'].split('.')[0]
        # nucleophile_smiles = row['smiles'].split('.')[1]
        # indices = ast.literal_eval(row['products_run'])

        # test_consisten_atoms(substrate_smiles, nucleophile_smiles, indices, idx)

