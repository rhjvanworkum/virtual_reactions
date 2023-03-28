import os
import json
from typing import List
import subprocess

from src.utils import Atom, read_xyz_file

def get_all_labels(base_path: str) -> List[str]:
    """
    Returns all labels for which both an E2 & Sn2 transition state has been found
    """
    sn2_labels, e2_labels = [], []
    for _, _, files in os.walk(os.path.join(base_path, 'transition-states/e2/')):
        for file in files:
            e2_labels.append(file.split('.')[0])
    for _, _, files in os.walk(os.path.join(base_path, 'transition-states/sn2/')):
        for file in files:
            sn2_labels.append(file.split('.')[0])
    
    sn2_labels_filtered = []
    for label in sn2_labels:
        path = os.path.join(base_path, f'reactant-complex-unconstrained-conformers/sn2/{label}/')
        if os.path.exists(path):
            if len(os.listdir(path)) > 0:
                sn2_labels_filtered.append(label)

    e2_labels_filtered = []
    for label in e2_labels:
        path = os.path.join(base_path, f'reactant-complex-unconstrained-conformers/e2/{label}/')
        if os.path.exists(path):
            if len(os.listdir(path)) > 0:
                e2_labels_filtered.append(label)

    return sn2_labels_filtered , e2_labels_filtered

def get_geometries(geometry_path: str) -> List[List[Atom]]:
    geometries = []
    for _, _, files in os.walk(geometry_path):
        for file in files:
            if file.split('.')[-1] == 'xyz':
                geometry = read_xyz_file(os.path.join(geometry_path, file))
                geometry = [atom.to_list() for atom in geometry]
                geometries.append(geometry)
    return geometries

def get_smiles_from_xyz(xyz_file_path) -> str:
    output = subprocess.check_output(f'python xyz2mol.py {xyz_file_path} --charge -1', shell=True)
    return output.decode("utf-8")

if __name__ == "__main__":
    path = '/home/ruard/Documents/datasets/qmrxn20/'
    sn2_reaction_labels, e2_reaction_labels = get_all_labels(path)

    print(len(sn2_reaction_labels))

    dataset = {}
    for idx, reaction_label in enumerate(sn2_reaction_labels):
        dataset[reaction_label] = {}

        # get RC's
        sn2_rc_path = os.path.join(path, f'reactant-complex-unconstrained-conformers/sn2/{reaction_label}/')
        dataset[reaction_label]['rc_conformers'] = get_geometries(sn2_rc_path)

        # get ts
        geometry = read_xyz_file(os.path.join(path, f'transition-states/sn2/{reaction_label}.xyz'))
        geometry = [atom.to_list() for atom in geometry]
        dataset[reaction_label]['ts'] = geometry

        # get PC's
        sn2_pc_path = os.path.join(path, f'product-conformers/sn2/{reaction_label}/')
        dataset[reaction_label]['sn2_conformers'] = get_geometries(sn2_pc_path)

        # extract reaction smiles
        sn2_rc_path = os.path.join(path, f'reactant-complex-constrained-conformers/sn2/{reaction_label}/')
        for _, _, files in os.walk(sn2_rc_path):
            for file in files:
                smiles = get_smiles_from_xyz(os.path.join(sn2_rc_path, file))
                dataset[reaction_label]['smiles'] = smiles
                break

    with open('./data/sn2_dataset.json', 'w') as f:
        json.dump(dataset, f)