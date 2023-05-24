import yaml
from ase.db import connect
import numpy as np
from schnetpack.data import ASEAtomsData

# db_path = './data/experiment_2/surrogate_dataset.db'

# data_idxs = np.load('./data/experiment_2/splits/cc_5_mol_sim_30.npz')['train_idx']

# i = 0
# for idx in data_idxs:
#     try:
#         with connect(db_path) as conn:
#             numbers = conn.get(int(idx) + 1).numbers
#     except:
#         i += 1
#         print(idx)

# print(i)

# print(len(ASEAtomsData(db_path)))

for i in range(5):
    data = np.load(f'./data/experiment_2/splits/mol_splits/mol_{i}.npz')
    print(len(data['train_idx']), len(data['val_idx']))


# with open(f'data/experiment_2/splits/split.yaml', "r") as f:
#     split = yaml.load(f, Loader=yaml.Loader)


# for i in split['train'].keys():
#     idx = split['train'][i]['dft'][0]

#     with connect(db_path) as conn:
#         print(conn.get(idx + 1).numbers)
