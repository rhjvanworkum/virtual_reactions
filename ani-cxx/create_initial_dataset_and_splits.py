import os
import h5py as h5
import numpy as np
from typing import List, Dict
from ase import Atoms
from schnetpack.data import ASEAtomsData


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
    name = "experiment_1"
    N_train_molecules = 5
    N_test_molecules = 3
    percentage_real_data = 0.05
    percentage_simulated_data = 0.10
    train_split, val_split = 0.9, 0.1

    save_folder = os.path.join('./data/', name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get Coupled Cluster data
    cc_mol_geom_idx = get_cc_available_data()

    # select roughly the largest available geometry data mols
    cc_mol_len = {k: len(v) for k, v in cc_mol_geom_idx.items()}
    sorted_keys = sorted(cc_mol_len, key=cc_mol_len.get)
    test_mols = sorted_keys[-N_test_molecules:]
    train_mols = sorted_keys[-(N_train_molecules + N_test_molecules):-N_test_molecules]

    
    
    atoms = []
    properties = []

    # get 5% CC data
    for mol in train_mols:
        idxs = cc_mol_geom_idx[mol]
        np.random.shuffle(idxs)
        cc_idxs = idxs[:len(int(percentage_real_data * len(idxs)))]
        for idx in cc_idxs:
            atom, property = get_cc_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)
    n_cc_datapoints = len(atoms)

    # get 100% DFT data
    dft_10_indices = {}
    idx = 0
    for mol_idx, mol in enumerate(train_mols):
        idxs = cc_mol_geom_idx[mol]
        np.random.shuffle(idxs)

        i = 0
        for idx in cc_idxs:
            atom, property = get_dft_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)
            i += 1

        dft_10_indices[mol_idx] = np.arange(idx, idx + i)[:int(percentage_simulated_data * i)]
        idx += i

    dft_10_indices = {k: [v + n_cc_datapoints for v in values] for k, values in dft_10_indices.items()}
    n_dft_datapoints = len(atoms) - n_cc_datapoints

    # add the test data
    for mol in test_mols:
        idxs = cc_mol_geom_idx[mol]
        np.random.shuffle(idxs)
        for idx in idxs:
            atom, property = get_cc_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)
    n_test_datapoints = len(atoms) - n_dft_datapoints - n_cc_datapoints

    assert n_cc_datapoints + n_dft_datapoints + n_test_datapoints == len(atoms)


    # save db
    dataset = ASEAtomsData.create(
        os.path.join(save_folder, 'cc_dft_dataset.db'),
        distance_unit='Ang',
        property_unit_dict={'energy':'kcal/mol', 'simulation_idx': ''}
    )
    dataset.add_systems(properties, atoms)
    
    
    # save splits
    ood_test_idxs = np.arange(len(atoms) - n_test_datapoints, n_test_datapoints)

    # 1) CC_5 split
    training_idxs = np.arange(n_cc_datapoints)
    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]
    np.savez(os.path.join(save_folder, 'splits/cc_5.npz'), train_idx=train_idxs, val_idx=val_idxs, test_idx=ood_test_idxs)

    # 2) CC_5_DFT_100 split
    training_idxs = np.arange(n_cc_datapoints + n_dft_datapoints)
    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]
    np.savez(os.path.join(save_folder, 'splits/cc_5_dft_100.npz'), train_idx=train_idxs, val_idx=val_idxs, test_idx=ood_test_idxs)

    # 3) CC_5_DFT_10 split
    training_idxs = np.arange(n_cc_datapoints).tolist()
    for k, v in dft_10_indices.items():
        training_idxs += v
    training_idxs = np.array(training_idxs)
    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]
    np.savez(os.path.join(save_folder, 'splits/cc_5_dft_10.npz'), train_idx=train_idxs, val_idx=val_idxs, test_idx=ood_test_idxs)

    # 4) MOLECULES split
    for key, value in dft_10_indices.items():
        training_idxs = np.array(value)
        np.random.shuffle(training_idxs)
        train_idxs = training_idxs[:int(train_split * len(training_idxs))]
        val_idxs = training_idxs[int(train_split * len(training_idxs)):]
        np.savez(os.path.join(save_folder, f'splits/mol_splits/mol{key}.npz'), train_idx=train_idxs, val_idx=val_idxs, test_idx=ood_test_idxs)