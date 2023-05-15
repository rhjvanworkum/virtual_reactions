import os
import h5py as h5
import numpy as np
from typing import List, Dict
from ase import Atoms
from schnetpack.data import ASEAtomsData
import yaml


ANI_DATASET_FILE_PATH = "/home/ruard/Documents/datasets/ani1x-release.h5"
CC_ENERGY_KEY = "ccsd(t)_cbs.energy"
DFT_ENERGY_KEY = "wb97x_dz.energy"
DFT_FORCES_KEY = "wb97x_dz.forces"
Z_KEY = "atomic_numbers"
R_KEY = "coordinates"
KCAL_MOL_HARTREE = 627.5096

# ani dataset
ani_dataset = h5.File(ANI_DATASET_FILE_PATH)

def get_cc_available_data() -> Dict[str, List[int]]:
    # geom idxs of every mol for which CC data is available
    cc_mol_geom_idx = {}
    for key in ani_dataset.keys():
        cc_energies = ani_dataset[key][CC_ENERGY_KEY][:]
        nan_idxs = np.argwhere(np.isnan(cc_energies)).flatten()
        all_idxs = np.arange(len(cc_energies))
        cc_idxs = np.array(list(set(all_idxs) - set(nan_idxs)))
        cc_idxs.sort()
        cc_mol_geom_idx[key] = cc_idxs

    return cc_mol_geom_idx

def get_cc_atoms(molecule_name, geom_idx):
        atoms = Atoms(
            numbers=ani_dataset[molecule_name][Z_KEY][:],
            positions=ani_dataset[molecule_name][R_KEY][:][geom_idx, ...]
        )
        # CC - "experiment"
        properties = {
            'energy': np.array([ani_dataset[molecule_name][CC_ENERGY_KEY][:][geom_idx]]) / KCAL_MOL_HARTREE,
            'simulation_idx': np.array([0])
        }
        return atoms, properties

def get_dft_atoms(molecule_name, geom_idx):
    atoms = Atoms(
        numbers=ani_dataset[molecule_name][Z_KEY][:],
        positions=ani_dataset[molecule_name][R_KEY][:][geom_idx, ...]
    )
    # DFT - "simulation"
    properties = {
        'energy': np.array([ani_dataset[molecule_name][DFT_ENERGY_KEY][:][geom_idx]]) / KCAL_MOL_HARTREE,
        'simulation_idx': np.array([1])
    }
    return atoms, properties

if __name__ == "__main__":
    name = "experiment_2"
    N_train_molecules = 5
    N_test_molecules = 3

    save_folder = os.path.join('./data/', name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get Coupled Cluster data
    coupled_cluster_data_dict = get_cc_available_data()

    # select roughly the largest available geometry data mols
    cc_mol_len = {k: len(v) for k, v in coupled_cluster_data_dict.items()}
    sorted_keys = sorted(cc_mol_len, key=cc_mol_len.get)
    ood_test_mols = sorted_keys[-N_test_molecules:]
    train_mols = sorted_keys[-(N_train_molecules + N_test_molecules):-N_test_molecules]    
    
    atoms = []
    properties = []
    i = 0
    idxs = {
        'train': {i: {'cc': [], 'dft': []} for i in range(len(train_mols))},
        'test': {i: [] for i in range(len(ood_test_mols))},
    }

    # add CC_data
    for mol_idx, mol in enumerate(train_mols):
        idxs = coupled_cluster_data_dict[mol]
        for idx in idxs:
            atom, property = get_cc_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)

            idxs['train'][mol_idx]['cc'].append(i)
            i += 1
    n_cc_datapoints = len(atoms)

    # get DFT data
    for mol in train_mols:
        idxs = coupled_cluster_data_dict[mol]
        for idx in idxs:
            atom, property = get_dft_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)

            idxs['train'][mol_idx]['dft'].append(i)
            i += 1
    n_dft_datapoints = len(atoms) - n_cc_datapoints

    # add the test data
    for mol in ood_test_mols:
        idxs = coupled_cluster_data_dict[mol]
        for idx in idxs:
            atom, property = get_cc_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)

            idxs['test'][mol_idx].append(i)
            i += 1
    n_test_datapoints = len(atoms) - n_dft_datapoints - n_cc_datapoints

    assert n_cc_datapoints + n_dft_datapoints + n_test_datapoints == len(atoms)
    
    # save db
    dataset = ASEAtomsData.create(
        os.path.join(save_folder, 'dataset.db'),
        distance_unit='Ang',
        property_unit_dict={'energy':'kcal/mol', 'simulation_idx': ''}
    )
    dataset.add_systems(properties, atoms)

    # save splits
    if not os.path.exists(os.path.join(save_folder, 'splits/')):
        os.makedirs(os.path.join(save_folder, 'splits/'))
    if not os.path.exists(os.path.join(save_folder, 'splits/mol_splits/')):
        os.makedirs(os.path.join(save_folder, 'splits/mol_splits/'))
    with open(os.path.join(save_folder, 'splits/split.yaml'), 'w') as yaml_file:
        yaml.dump(idxs, yaml_file, default_flow_style=False)