import torch
import yaml
import os
import numpy as np
from sklearn.metrics import mean_absolute_error

from ase.db import connect


def get_mol_with_index(db_idx, mol_idx):
    for mol in db_idx.keys():
        if db_idx[mol]['idx'] == mol_idx:
            return mol

if __name__ == "__main__":
    save_folder = "./data/ani-cxx/experiment_3/"
    
    # torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    with open(f'data/ani-cxx/experiment_3/db.yaml', "r") as f:
        db_idxs = yaml.load(f, Loader=yaml.Loader)

    with open('data/ani-cxx/experiment_3/mol.yaml', "r") as f:
        mol_idx = yaml.load(f, Loader=yaml.Loader)
    train_mol_idx = sorted(mol_idx['train_mol_idxs'])
    ood_mol_idx = sorted(mol_idx['ood_mol_idxs'])
    ood_mol_idx = [i for i in ood_mol_idx if i not in train_mol_idx]

    for idx, mol_idx in enumerate(ood_mol_idx):
        mol = get_mol_with_index(db_idxs, mol_idx)

        true_energies, pred_energies = [], []
        with connect(os.path.join(save_folder, 'dataset.db')) as conn:
            for idx in db_idxs[mol]['cc'][0]:
                energy = conn.get(int(idx + 1)).data['energy'][0]
                true_energies.append(energy)
            for idx in db_idxs[mol]['dft'][1]:
                energy = conn.get(int(idx + 1)).data['energy'][0]
                pred_energies.append(energy)
        
        true_energies = np.array(true_energies)
        pred_energies = np.array(pred_energies)
        score = mean_absolute_error(true_energies, pred_energies)
        print(score)