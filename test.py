import os
import h5py as h5
import numpy as np
from typing import List, Dict
from ase import Atoms
from schnetpack.data import ASEAtomsData
import yaml


ANI_DATASET_FILE_PATH = "/home/rhjvanworkum/ani1x-release.h5"
CC_ENERGY_KEY = "ccsd(t)_cbs.energy"
DFT_ENERGY_KEY = "wb97x_dz.energy"
DFT_FORCES_KEY = "wb97x_dz.forces"
Z_KEY = "atomic_numbers"
R_KEY = "coordinates"
KCAL_MOL_HARTREE = 627.5096

# ani dataset
ani_dataset = h5.File(ANI_DATASET_FILE_PATH)

a = np.array(ani_dataset['O2'][DFT_FORCES_KEY][:][0])
print(a, a.shape)



# import numpy as np

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_0.npz')
# base = len(split['train_idx']) + len(split['val_idx'])

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_10.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_20.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_30.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_10.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_20.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_30.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_10.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_20.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_30.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)