import os
import yaml
import numpy as np
from ase.db import connect
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def pearson(x, y):
    corr, _ = pearsonr(x,y)
    return corr


if __name__ == "__main__":
    experiment_name = "experiment_2"
    name = "mol_sim"
    
    save_folder = os.path.join("data", experiment_name)

    database = os.path.join(save_folder, "surrogate_dataset.db")
    with open(os.path.join(save_folder, "splits/surrogate_split.yaml"), 'r') as f:
        split_idxs = yaml.load(f, Loader=yaml.Loader)

    for i in split_idxs['train'].keys():
        true_idx = split_idxs['train'][i]['cc']
        pred_idx = split_idxs['train'][i][name]
        with connect(database) as conn:
            true_energies = np.array([conn.get(int(i) + 1).data['energy'][0] for i in true_idx])
            pred_energies = np.array([conn.get(int(i) + 1).data['energy'][0] for i in pred_idx])

        print(f'Scores for mol {i}: {mean_absolute_error(true_energies, pred_energies)}, {mean_squared_error(true_energies, pred_energies)}')