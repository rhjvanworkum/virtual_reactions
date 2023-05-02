import os
import ase
from ase.db import connect
import numpy as np
import torch
import schnetpack as spk
from schnetpack.data import ASEAtomsData

PROPERTY_LIST = ['energy', 'simulation_idx']

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
    name = "experiment_1"
    n_mols = 5
    train_split, val_split = 0.9, 0.1
    virtual_test_split = 0.05

    save_folder = os.path.join('./data/', name)
    idxs = np.load(os.path.join(save_folder, 'splits/cc_5.npz'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    atoms = []
    properties = []
    
    # 1. get CC train data & OOD test data from standard DB
    with connect(os.path.join(save_folder, 'cc_dft_dataset.db')) as conn:
        # add cc data
        for idx in np.concatenate([idxs['train_idx'], idxs['val_idx']]):
            atoms.append(conn.get_atoms(int(idx + 1)))
            properties.append({
                key: conn.get(int(idx + 1)).data[key] for key in PROPERTY_LIST
            })
    n_cc_datapoints = len(atoms)

    # 2. add data from surrogate models
    simulated_atoms = []
    simulated_properties = []
    for n in range(n_mols):
        model = torch.load(os.path.join(save_folder, f'models/mol{n}.pt'), map_location=device).to(device)
        model.eval()

        for atom in atoms:
            energy = get_energy_from_model(atom, model, device)
            simulated_atoms.append(atom)
            simulated_properties.append({
                'energy': np.array([energy]),
                'simulation_idx': np.array([n + 1])
            })

    n_simulated_datapoints = len(simulated_atoms)

    atoms += simulated_atoms
    properties += simulated_properties

    # 3. add OOD test data
    with connect(os.path.join(save_folder, 'cc_dft_dataset.db')) as conn:
        # add cc data
        for idx in idxs['ood_test_idx']:
            atoms.append(conn.get_atoms(int(idx + 1)))
            properties.append({
                key: conn.get(int(idx + 1)).data[key] for key in PROPERTY_LIST
            })
    n_ood_test_datapoints = len(idxs['ood_test_idx'])

    # 4. add IID test data
    with connect(os.path.join(save_folder, 'cc_dft_dataset.db')) as conn:
        # add cc data
        for idx in idxs['iid_test_idx']:
            atoms.append(conn.get_atoms(int(idx + 1)))
            properties.append({
                key: conn.get(int(idx + 1)).data[key] for key in PROPERTY_LIST
            })
    n_iid_test_datapoints = len(idxs['iid_test_idx'])

    # save db
    dataset = ASEAtomsData.create(
        os.path.join(save_folder, 'cc_surrogate_dataset.db'),
        distance_unit='Ang',
        property_unit_dict={'energy':'kcal/mol', 'simulation_idx': ''}
    )
    dataset.add_systems(properties, atoms)

    # save splits
    cc_idxs = np.arange(n_cc_datapoints)
    sim_idxs = np.arange(n_cc_datapoints, n_cc_datapoints + n_simulated_datapoints)
    ood_test_idxs = np.arange(len(atoms) - (n_ood_test_datapoints + n_iid_test_datapoints), len(atoms) - n_iid_test_datapoints)
    iid_test_idxs = np.arange(len(atoms) - n_iid_test_datapoints, len(atoms))

    np.random.shuffle(cc_idxs)
    np.random.shuffle(sim_idxs)
    np.random.shuffle(ood_test_idxs)
    np.random.shuffle(iid_test_idxs)

    virtual_test_idxs = sim_idxs[:int(virtual_test_split * len(sim_idxs))]
    leftover_sim_idxs = list(filter(lambda x: x not in virtual_test_idxs, sim_idxs))

    training_idxs = np.array(cc_idxs.tolist() + leftover_sim_idxs)
    np.random.shuffle(training_idxs)
    train_idxs = training_idxs[:int(train_split * len(training_idxs))]
    val_idxs = training_idxs[int(train_split * len(training_idxs)):]
    np.savez(
        os.path.join(save_folder, 'splits/cc_5_surrogate.npz'), 
        train_idx=train_idxs, 
        val_idx=val_idxs, 
        ood_test_idx=ood_test_idxs,
        iid_test_idx=iid_test_idxs,
        virtual_test_idx=virtual_test_idxs
    )