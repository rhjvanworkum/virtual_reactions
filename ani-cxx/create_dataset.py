import h5py as h5
import numpy as np
import math
from ase import Atoms
from schnetpack.data import ASEAtomsData


ANI_DATASET_FILE_PATH = "/home/ruard/Documents/datasets/ani1x-release.h5"
CC_ENERGY_KEY = "ccsd(t)_cbs.energy"
DFT_ENERGY_KEY = "wb97x_dz.energy"
DFT_FORCES_KEY = "wb97x_dz.forces"
Z_KEY = "atomic_numbers"
R_KEY = "coordinates"

KCAL_MOL_HARTREE = 627.5096


if __name__ == "__main__":
    train_split, val_split = 0.9, 0.1

    # ani dataset
    ani_dataset = h5.File(ANI_DATASET_FILE_PATH)

    # geom idxs of every mol for which CC data is available
    cc_mol_geom_idx = {}
    for key in ani_dataset.keys():
        cc_energies = ani_dataset[key][CC_ENERGY_KEY][:]
        nan_idxs = np.argwhere(np.isnan(cc_energies)).flatten()
        all_idxs = np.arange(len(cc_energies))
        cc_idxs = np.array(list(set(all_idxs) - set(nan_idxs)))
        cc_idxs.sort()
        cc_mol_geom_idx[key] = cc_idxs

    # let's select 10k datasamples for our custom dataset
    n_datapoints = 0
    custom_dataset = {}
    while n_datapoints < 1e4:
        key = list(cc_mol_geom_idx.keys())[int(np.random.random() * len(cc_mol_geom_idx.keys()))]
        if key not in custom_dataset.keys():
            custom_dataset[key] = cc_mol_geom_idx[key]
            n_datapoints += len(cc_mol_geom_idx[key])

    print(f"No. of Molecules: {len(custom_dataset.keys())}")
    print(f"No. of geometries: {sum([len(custom_dataset[k]) for k in custom_dataset.keys()])}")

    # let's select 10% of the molecules as OOD test set
    ood_molecules = []
    while len(ood_molecules) <= math.floor(0.1 * len(custom_dataset.keys())):
        key = list(custom_dataset.keys())[int(np.random.random() * len(custom_dataset.keys()))]
        if key not in ood_molecules:
            ood_molecules.append(key)

    train_dataset = {k: v for k, v in custom_dataset.items() if k not in ood_molecules}
    test_dataset = {k: v for k, v in custom_dataset.items() if k in ood_molecules}
    print(f"No. of train Molecules: {len(train_dataset.keys())}")
    print(f"No. of train geometries: {sum([len(train_dataset[k]) for k in train_dataset.keys()])}")
    print(f"No. of test Molecules: {len(test_dataset.keys())}")
    print(f"No. of test geometries: {sum([len(test_dataset[k]) for k in test_dataset.keys()])}")

    def get_data(key, idx):
        atoms_list, property_list = [], []
        atoms = Atoms(
            numbers=ani_dataset[key][Z_KEY][:],
            positions=ani_dataset[key][R_KEY][:][idx, ...]
        )
        # CC - "experiment"
        properties = {
            'energy': np.array([ani_dataset[key][CC_ENERGY_KEY][:][idx]]) / KCAL_MOL_HARTREE,
            'simulation_idx': np.array([0])
        }
        atoms_list.append(atoms)
        property_list.append(properties)

        # DFT - "simulation"
        properties = {
            'energy': np.array([ani_dataset[key][DFT_ENERGY_KEY][:][idx]]) / KCAL_MOL_HARTREE,
            'simulation_idx': np.array([1])
        }
        atoms_list.append(atoms)
        property_list.append(properties)
        return atoms_list, property_list

    # create ASE DB
    atoms_list = []
    property_list = []
    for key in train_dataset.keys():
        for idx in train_dataset[key]:
            atoms, properties = get_data(key, idx)
            atoms_list += atoms
            property_list += properties
    for key in test_dataset.keys():
        for idx in test_dataset[key]:
            atoms, properties = get_data(key, idx)
            atoms_list += atoms
            property_list += properties

    dataset = ASEAtomsData.create(
        './10k_dataset.db',
        distance_unit='Ang',
        property_unit_dict={'energy':'kcal/mol', 'simulation_idx': ''}
    )
    dataset.add_systems(property_list, atoms_list)

    # get split idxs
    training_idxs = np.arange(sum([len(train_dataset[k]) for k in train_dataset.keys()]))
    test_idxs = np.arange(len(training_idxs), len(training_idxs) + sum([len(test_dataset[k]) for k in test_dataset.keys()]))
    assert len(training_idxs) + len(test_idxs) == sum([len(custom_dataset[k]) for k in custom_dataset.keys()])

    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]

    np.savez('split.npz', train_idx=train_idxs, val_idx=val_idxs, test_idx=test_idxs)