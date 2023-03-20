import os
import json

from src.utils import read_xyz_file

def get_all_labels(base_path: str):
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
    
    labels = []
    for label in sn2_labels + e2_labels:
        if (label in sn2_labels) and (label in e2_labels):
            labels.append(label)
    labels = list(set(labels))

    return labels
            

if __name__ == "__main__":
    path = '/home/ruard/Documents/datasets/qmrxn20/'
    reaction_labels = get_all_labels(path)

    dataset = {}
    for reaction_label in reaction_labels:
        dataset[reaction_label] = {
            'e2_rc_conformers': [],
            'e2_ts': [],
            'e2_pc_conformers': [],
            'sn2_rc_conformers': [],
            'sn2_ts': [],
            'sn2_pc_conformers': []
        }

        # get RC's
        e2_rc_path = os.path.join(path, f'reactant-complex-unconstrained-conformers/e2/{reaction_label}/')
        for _, _, files in os.walk(e2_rc_path):
            for file in files:
                if file.split('.')[-1] == 'xyz':
                    geometry = read_xyz_file(os.path.join(e2_rc_path, file))
                    geometry = [atom.to_list() for atom in geometry]
                    dataset[reaction_label]['e2_rc_conformers'].append(geometry)

        sn2_rc_path = os.path.join(path, f'reactant-complex-unconstrained-conformers/sn2/{reaction_label}/')
        for _, _, files in os.walk(sn2_rc_path):
            for file in files:
                if file.split('.')[-1] == 'xyz':
                    geometry = read_xyz_file(os.path.join(sn2_rc_path, file))
                    geometry = [atom.to_list() for atom in geometry]
                    dataset[reaction_label]['sn2_rc_conformers'].append(geometry)

        # get ts
        geometry = read_xyz_file(os.path.join(path, f'transition-states/e2/{reaction_label}.xyz'))
        geometry = [atom.to_list() for atom in geometry]
        dataset[reaction_label]['e2_ts'].append(geometry)
        geometry = read_xyz_file(os.path.join(path, f'transition-states/sn2/{reaction_label}.xyz'))
        geometry = [atom.to_list() for atom in geometry]
        dataset[reaction_label]['sn2_ts'].append(geometry)

        # get PC's
        e2_rc_path = os.path.join(path, f'product-conformers/e2/{reaction_label}/')
        for _, _, files in os.walk(e2_rc_path):
            for file in files:
                if file.split('.')[-1] == 'xyz':
                    geometry = read_xyz_file(os.path.join(e2_rc_path, file))
                    geometry = [atom.to_list() for atom in geometry]
                    dataset[reaction_label]['e2_pc_conformers'].append(geometry)

        sn2_rc_path = os.path.join(path, f'product-conformers/sn2/{reaction_label}/')
        for _, _, files in os.walk(sn2_rc_path):
            for file in files:
                if file.split('.')[-1] == 'xyz':
                    geometry = read_xyz_file(os.path.join(sn2_rc_path, file))
                    geometry = [atom.to_list() for atom in geometry]
                    dataset[reaction_label]['sn2_pc_conformers'].append(geometry)

    with open('./data/dataset.json', 'w') as f:
        json.dump(dataset, f)