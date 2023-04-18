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
    iid_test_split = 0.05
    virtual_test_split = 0.05

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
        cc_idxs = idxs[:int(percentage_real_data * len(idxs))]
        for idx in cc_idxs:
            atom, property = get_cc_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)
    n_cc_datapoints = len(atoms)

    # get 100% DFT data
    dft_10_indices = {}
    tot_i = 0
    for mol_idx, mol in enumerate(train_mols):
        idxs = cc_mol_geom_idx[mol]
        np.random.shuffle(idxs)

        i = 0
        for idx in idxs:
            atom, property = get_dft_atoms(mol, idx)
            atoms.append(atom)
            properties.append(property)
            i += 1

        dft_10_indices[mol_idx] = np.arange(tot_i, tot_i + i)[:int(percentage_simulated_data * i)]
        tot_i += i
    
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
    
    cc_data_idxs = np.arange(n_cc_datapoints)
    dft_data_idxs = np.arange(n_cc_datapoints, n_cc_datapoints + n_dft_datapoints)
    dft10_data_idxs = []
    for k, values in dft_10_indices.items():
        dft10_data_idxs += values
    dft10_data_idxs = np.array(dft10_data_idxs)
    ood_test_idxs = np.arange(len(atoms) - n_test_datapoints, len(atoms))

    np.random.shuffle(cc_data_idxs)
    np.random.shuffle(dft_data_idxs)
    np.random.shuffle(dft10_data_idxs)
    np.random.shuffle(ood_test_idxs)

    assert set(cc_data_idxs).isdisjoint(dft_data_idxs)
    assert set(cc_data_idxs).isdisjoint(ood_test_idxs)
    assert set(dft_data_idxs).isdisjoint(ood_test_idxs)
    assert set(dft10_data_idxs).issubset(dft_data_idxs)
    assert not set(dft_data_idxs).issubset(dft10_data_idxs)

    # save db
    # dataset = ASEAtomsData.create(
    #     os.path.join(save_folder, 'cc_dft_dataset.db'),
    #     distance_unit='Ang',
    #     property_unit_dict={'energy':'kcal/mol', 'simulation_idx': ''}
    # )
    # dataset.add_systems(properties, atoms)
    
    
    # save splits
    if not os.path.exists(os.path.join(save_folder, 'splits/')):
        os.makedirs(os.path.join(save_folder, 'splits/'))
    if not os.path.exists(os.path.join(save_folder, 'splits/mol_splits/')):
        os.makedirs(os.path.join(save_folder, 'splits/mol_splits/'))

    # 1) CC_5 split
    iid_test_idxs = cc_data_idxs[:int(iid_test_split * len(cc_data_idxs))]
    leftover_cc_idxs = list(filter(lambda x: x not in iid_test_idxs, cc_data_idxs))
    train_idxs = leftover_cc_idxs[:int(train_split * len(leftover_cc_idxs))]
    val_idxs = leftover_cc_idxs[int(train_split * len(leftover_cc_idxs)):]
    
    np.savez(
        os.path.join(save_folder, 'splits/cc_5.npz'), 
        train_idx=train_idxs, 
        val_idx=val_idxs, 
        ood_test_idx=ood_test_idxs,
        iid_test_idx=iid_test_idxs,
        virtual_test_idx=[]
    )

    # 2) CC_5_DFT_100 split
    virtual_test_idxs = dft_data_idxs[:int(virtual_test_split * len(dft_data_idxs))]
    leftover_dft_idxs = list(filter(lambda x: x not in virtual_test_idxs, dft_data_idxs))
    iid_test_idxs = cc_data_idxs[:int(iid_test_split * len(cc_data_idxs))]
    leftover_cc_idxs = list(filter(lambda x: x not in iid_test_idxs, cc_data_idxs))

    training_idxs = np.array(leftover_dft_idxs + leftover_cc_idxs)
    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]
    
    np.savez(
        os.path.join(save_folder, 'splits/cc_5_dft_100.npz'), 
        train_idx=train_idxs, 
        val_idx=val_idxs, 
        ood_test_idx=ood_test_idxs,
        iid_test_idx=iid_test_idxs,
        virtual_test_idx=virtual_test_idxs
    )

    # 3) CC_5_DFT_10 split
    virtual_test_idxs = dft10_data_idxs[:int(virtual_test_split * len(dft10_data_idxs))]
    leftover_dft_idxs = list(filter(lambda x: x not in virtual_test_idxs, dft_data_idxs))
    iid_test_idxs = cc_data_idxs[:int(iid_test_split * len(cc_data_idxs))]
    leftover_cc_idxs = list(filter(lambda x: x not in iid_test_idxs, cc_data_idxs))

    training_idxs = np.array(leftover_dft_idxs + leftover_cc_idxs)
    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]

    np.savez(
        os.path.join(save_folder, 'splits/cc_5_dft_10.npz'), 
        train_idx=train_idxs, 
        val_idx=val_idxs, 
        ood_test_idx=ood_test_idxs,
        iid_test_idx=iid_test_idxs,
        virtual_test_idx=virtual_test_idxs
    )

    # 4) MOLECULES split
    for key, value in dft_10_indices.items():
        training_idxs = np.array(value)
        np.random.shuffle(training_idxs)
        train_idxs = training_idxs[:int(train_split * len(training_idxs))]
        val_idxs = training_idxs[int(train_split * len(training_idxs)):]
        np.savez(
            os.path.join(save_folder, f'splits/mol_splits/mol{key}.npz'), 
            train_idx=train_idxs,
            val_idx=val_idxs, 
            ood_test_idx=[],
            iid_test_idx=[],
            virtual_test_idx=[]
        )