import shutil
import yaml
import numpy as np
import h5py as h5
from typing import Dict, List
from schnetpack.data import ASEAtomsData
from ase import Atoms

ANI_DATASET_FILE_PATH = "/home/rhjvanworkum/ani1x-release.h5"
CC_ENERGY_KEY = "ccsd(t)_cbs.energy"
DFT_ENERGY_KEY = "wb97x_dz.energy"
DFT_FORCES_KEY = "wb97x_dz.forces"
Z_KEY = "atomic_numbers"
R_KEY = "coordinates"
KCAL_MOL_HARTREE = 627.5096

# ani dataset
ani_dataset = h5.File(ANI_DATASET_FILE_PATH)

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

if __name__ == "__main__":
    prev_dataset_path = './data/ani-cxx/experiment_2/surrogate_dataset_small.db'
    new_dataset_path = './data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db'

    prev_split_path = './data/ani-cxx/experiment_2/splits/surrogate_split_small.yaml'
    new_split_path = './data/ani-cxx/experiment_2/splits/surrogate_split_small_ood.yaml'

    # copy new dataset & load it in 
    shutil.copy2(prev_dataset_path, new_dataset_path)
    db = ASEAtomsData(new_dataset_path)

    # get split
    with open(prev_split_path, 'r') as f:
        data_idxs = yaml.load(f, Loader=yaml.Loader)
    data_idxs['ood_test'] = {}

    # now get cc data
    coupled_cluster_data_dict = get_cc_available_data()
    cc_mol_len = {k: len(v) for k, v in coupled_cluster_data_dict.items()}
    sorted_keys = sorted(cc_mol_len, key=cc_mol_len.get)
    ood_test_mols = sorted_keys[:-8]
    atoms, properties = [], []
    for mol_idx, mol in enumerate(ood_test_mols):
        data_idxs['ood_test'][mol_idx] = []
        for idx in np.array(coupled_cluster_data_dict[mol]):
            atoms, properties = get_cc_atoms(mol, idx)
            db.add_system(atoms, **properties)
            data_idxs['ood_test'][mol_idx].append(len(db) - 1)
        print(mol_idx / len(ood_test_mols) * 100, '%')

    # save split
    with open(new_split_path, 'w') as yaml_file:
        yaml.dump(data_idxs, yaml_file, default_flow_style=False)

