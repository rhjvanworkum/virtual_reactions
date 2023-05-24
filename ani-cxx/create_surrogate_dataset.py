import os
import yaml
import shutil
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import torch
from ase.db import connect
import ase
import numpy as np

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


if __name__ == "__main__":
    name = "experiment_2"
    N_train_molecules = 5
    N_test_molecules = 3

    iid_test_split = 0.05
    virtual_test_split = 0.05

    save_folder = os.path.join('./data/', name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load split and add mol sim to train set
    with open(os.path.join(save_folder, 'splits/split.yaml'), 'r') as f:
        data_idxs = yaml.load(f, Loader=yaml.Loader)

    for i in data_idxs['train'].keys():
        data_idxs['train'][i]['mol_sim'] = []

    # load dataset and add mol sim to dataset
    shutil.copy2(
        os.path.join(save_folder, 'dataset.db'),
        os.path.join(save_folder, 'surrogate_dataset_small.db')
    )

    db = ASEAtomsData(os.path.join(save_folder, 'surrogate_dataset_small.db'))

    for i in data_idxs['train'].keys():
        model = torch.load(os.path.join(save_folder, f'models/mol{i}_small.pt'), map_location=device).to(device)
        model.eval()

        with connect(os.path.join(save_folder, 'dataset.db')) as conn:
            for idx in data_idxs['train'][i]['cc']:
                atoms = conn.get_atoms(int(idx + 1))
                energy = get_energy_from_model(atoms, model, device)
                db.add_system(atoms, **{
                    'energy': np.array([energy]),
                    'simulation_idx': np.array([i + 1])
                })
                data_idxs['train'][i]['mol_sim'].append(len(db) - 1)

        print(f'Finished mol {i}')

    # save split
    with open(os.path.join(save_folder, 'splits/surrogate_split_small.yaml'), 'w') as yaml_file:
        yaml.dump(data_idxs, yaml_file, default_flow_style=False)
