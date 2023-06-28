import os
import yaml
import numpy as np
from ase.db import connect
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def pearson(x, y):
    corr, _ = pearsonr(x,y)
    return corr


if __name__ == "__main__":
    name = "mol_sim"
    
    save_folder = os.path.join("data/ani-cxx", "experiment_2")

    database = os.path.join(save_folder, "surrogate_dataset_whole.db")
    with open(os.path.join(save_folder, "splits/surrogate_split_whole.yaml"), 'r') as f:
        split_idxs = yaml.load(f, Loader=yaml.Loader)


    data = []

    for sim_idx in split_idxs['train'].keys():
        data.append([])
        for mol_idx in split_idxs['train'].keys():
            true_idx = split_idxs['train'][mol_idx]['cc']
            pred_idx = split_idxs['train'][mol_idx][f"{name}_{sim_idx}"]
            with connect(database) as conn:
                true_energies = np.array([conn.get(int(i) + 1).data['energy'][0] for i in true_idx])
                pred_energies = np.array([conn.get(int(i) + 1).data['energy'][0] for i in pred_idx])

            data[sim_idx].append(mean_absolute_error(true_energies, pred_energies))


    for idx, d in enumerate(data):
        plt.bar(
            np.arange(len(d)) + idx * 0.15, 
            d, 
            width=0.15,
            label=f"sim_{idx}"
        )

    plt.xlabel('Mol x')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()