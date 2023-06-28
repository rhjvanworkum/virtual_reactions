import yaml
import numpy as np
import torch
import os
from ase.db import connect
import ase
import schnetpack as spk
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
from src.atomwise_simulation import SimulationIdxPerAtom
import schnetpack.transform as trn
from src.simulated_atoms_datamodule import SimulatedAtomsDataModule



def pearson(x, y):
    corr, _ = pearsonr(x,y)
    return corr

def get_energy_from_model(
    atoms: ase.Atoms,
    model: torch.nn,
    device: torch.device
) -> float:
    converter = spk.interfaces.AtomsConverter(
        additional_inputs={'simulation_idx': torch.Tensor([0 for _ in range(len(atoms))])},
        neighbor_list=spk.transform.ASENeighborList(cutoff=5.0), 
        dtype=torch.float32, 
        device=device
    )
    input = converter(atoms)
    output = model(input)
    return output['energy'].detach().cpu().numpy()[0]


# torch device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

save_folder = os.path.join('./data/ani-cxx/', "experiment_2")


mol_idx = 1

dataset = SimulatedAtomsDataModule(
    datapath='./data/ani-cxx/experiment_3/dataset.db',
    split_file=f'./data/ani-cxx/experiment_3/splits/mol_splits/mol_{mol_idx}_dft_10.npz',
    batch_size=32,
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.CastTo32(),
        SimulationIdxPerAtom()
    ],
    num_workers=2,
    pin_memory=True, # set to false, when not using a GPU
    load_properties=['energy', 'simulation_idx'], #only load U0 property
)

transform = trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)
transform.datamodule(dataset)



model = torch.load(os.path.join(save_folder, f'models/mol{mol_idx}.pt'), map_location=device).to(device)
model.eval() 

data_idxs = np.load(f'data/ani-cxx/experiment_2/splits/mol_splits/mol_{mol_idx}.npz')
data_idxs = data_idxs['train_idx'].tolist() + data_idxs['val_idx'].tolist()

with open(f'data/ani-cxx/experiment_2/splits/split.yaml', "r") as f:
    split = yaml.load(f, Loader=yaml.Loader)
# data_idxs = [split['train'][mol_idx]['cc'][split['train'][mol_idx]['dft'].index(idx)] for idx in data_idxs]


true_energies, pred_energies = [], []
with connect(os.path.join(save_folder, 'dataset.db')) as conn:
    for idx in data_idxs:
        atoms = conn.get_atoms(int(idx + 1))
        cc_energy = conn.get(int(idx + 1)).data['energy'][0]
        energy = get_energy_from_model(atoms, model, device) + 0.6

        true_energies.append(cc_energy)
        pred_energies.append(energy)

true_energies = np.array(true_energies)
# true_energies = true_energies - np.mean(true_energies)
# true_energies = true_energies / np.std(true_energies)
pred_energies = np.array(pred_energies)
# pred_energies = pred_energies - np.mean(pred_energies)
# pred_energies = pred_energies / np.std(pred_energies)
print(true_energies)
print(pred_energies)

# print(mean_absolute_error(true_energies, pred_energies))
# print(mean_squared_error(true_energies, pred_energies))
# print(pearson(true_energies, pred_energies))


# data_idxs = split['train'][mol_idx]['cc']

# true_energies, pred_energies = [], []
# with connect(os.path.join(save_folder, 'dataset.db')) as conn:
#     for idx in data_idxs:
#         atoms = conn.get_atoms(int(idx + 1))
#         cc_energy = conn.get(int(idx + 1)).data['energy'][0]
#         energy = get_energy_from_model(atoms, model, device)

#         true_energies.append(cc_energy)
#         pred_energies.append(energy)

# true_energies = np.array(true_energies)
# true_energies = true_energies - np.mean(true_energies)
# true_energies = true_energies / np.std(true_energies)
# pred_energies = np.array(pred_energies)
# pred_energies = pred_energies - np.mean(pred_energies)
# pred_energies = pred_energies / np.std(pred_energies)

# print(mean_absolute_error(true_energies, pred_energies))
# print(mean_squared_error(true_energies, pred_energies))
# print(pearson(true_energies, pred_energies))