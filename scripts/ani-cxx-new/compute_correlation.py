import yaml
import numpy as np
import torch
import os
from ase.db import connect
import ase
import schnetpack as spk
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
from src.atomwise_simulation import SimulationIdxPerAtom
import schnetpack.transform as trn
from src.simulated_atoms_datamodule import SimulatedAtomsDataModule


def get_energy_from_model(
    atoms: ase.Atoms,
    model: torch.nn,
    transform,
    device: torch.device
) -> float:
    converter = spk.interfaces.AtomsConverter(
        additional_inputs={
            'simulation_idx': torch.Tensor([0 for _ in range(len(atoms))]),
        },
        neighbor_list=spk.transform.ASENeighborList(cutoff=5.0), 
        dtype=torch.float32, 
        device=device
    )
    input = converter(atoms)
    for key, val in model(input).items():
        input[key] = val
    input = transform(input)
    return input['energy'].detach().cpu().numpy()[0]

def get_mol_with_index(db_idx, mol_idx):
    for mol in db_idx.keys():
        if db_idx[mol]['idx'] == mol_idx:
            return mol

def get_model_transformations(mol_idx):
    dataset = SimulatedAtomsDataModule(
        datapath='./data/ani-cxx/experiment_3/dataset.db',
        split_file=f'./data/ani-cxx/experiment_3/splits/mol_splits/mol_{mol_idx}_dft_10.npz',
        batch_size=32,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.CastTo32(),
            trn.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
            trn.ScaleProperty(input_key='energy', target_key='energy', output_key='energy', scale=torch.Tensor([1e4])),
            SimulationIdxPerAtom()
        ],
        num_workers=2,
        pin_memory=True, # set to false, when not using a GPU
        load_properties=['energy', 'simulation_idx'], #only load U0 property
    )
    dataset.prepare_data()
    dataset.setup()
    transform = trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)
    transform.datamodule(dataset)

    return transform

def compute_score(message, data_idxs, model, transform):
    true_energies, pred_energies = [], []
    with connect(os.path.join(save_folder, 'dataset.db')) as conn:
        for idx in data_idxs:
            atoms = conn.get_atoms(int(idx + 1))
            cc_energy = conn.get(int(idx + 1)).data['energy'][0]
            energy = get_energy_from_model(atoms, model, transform, device)

            true_energies.append(cc_energy)
            pred_energies.append(energy)

    true_energies = np.array(true_energies)
    pred_energies = np.array(pred_energies)
    return mean_absolute_error(true_energies, pred_energies)


if __name__ == "__main__":
    save_folder = "./data/ani-cxx/experiment_3/"
    
    # torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    with open(f'data/ani-cxx/experiment_3/db.yaml', "r") as f:
        db_idxs = yaml.load(f, Loader=yaml.Loader)

    with open(f'data/ani-cxx/experiment_3/mol.yaml', "r") as f:
        mol_idxs = yaml.load(f, Loader=yaml.Loader)['train_mol_idxs']
    mol_idxs = sorted(mol_idxs)

    data = []

    mol_idx = 0
    for idx, mol_idx in enumerate(mol_idxs):
        data.append([])
        # model
        transform = get_model_transformations(mol_idx)
        model = torch.load(os.path.join(save_folder, f'models/mol_{idx}_10%_surrogate.pt'), map_location=device).to(device)
        model.eval() 

        # data
        data_idxs = np.load(f'data/ani-cxx/experiment_3/splits/mol_splits/mol_{mol_idx}_dft_10.npz')
        data_idxs = data_idxs['train_idx'].tolist() + data_idxs['val_idx'].tolist()

        # mol = get_mol_with_index(db_idxs, mol_idx)
        # data_idxs = [db_idxs[mol]['cc'][0][db_idxs[mol]['dft'][1].index(idx)] for idx in data_idxs]
        # compute_score(f'MAE on subset {idx}: ', data_idxs, model, transform)

        for _mol_idx in mol_idxs:
            mol = get_mol_with_index(db_idxs, _mol_idx)
            data_idxs = db_idxs[mol]['cc'][0]
            scores = compute_score(f'MAE on mol {_mol_idx} with model {idx}: ', data_idxs, model, get_model_transformations(_mol_idx))
            data[-1].append(scores)

    print(np.array(data))