from sklearn.metrics import mean_absolute_error
import yaml
import numpy as np
from ase.db import connect

def get_mol_with_index(db_idx, mol_idx):
    for mol in db_idx.keys():
        if db_idx[mol]['idx'] == mol_idx:
            return mol


if __name__ == "__main__":
    with open(f'data/ani-cxx/experiment_3/db.yaml', "r") as f:
        db_idxs = yaml.load(f, Loader=yaml.Loader)

    with open(f'data/ani-cxx/experiment_3/mol.yaml', "r") as f:
        mol_idxs = yaml.load(f, Loader=yaml.Loader)['train_mol_idxs']
    mol_idxs = sorted(mol_idxs)

    for idx, mol_idx in enumerate(mol_idxs):
        mol = get_mol_with_index(db_idxs, mol_idx)
        data_idxs_1 = db_idxs[mol]['cc'][0]
        data_idxs_2 = db_idxs[mol]['dft'][1]

        true_energies, pred_energies = [], []

        with connect('data/ani-cxx/experiment_3/dataset.db') as conn:
            for idx in data_idxs_1:
                atoms = conn.get_atoms(int(idx + 1))
                cc_energy = conn.get(int(idx + 1)).data['energy'][0]
                true_energies.append(cc_energy)

        with connect('data/ani-cxx/experiment_3/dataset.db') as conn:
            for idx in data_idxs_2:
                atoms = conn.get_atoms(int(idx + 1))
                dft_energy = conn.get(int(idx + 1)).data['energy'][0]
                pred_energies.append(dft_energy)

        print(mean_absolute_error(true_energies, pred_energies))
