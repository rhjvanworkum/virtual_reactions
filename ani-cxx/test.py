import logging
import pytorch_lightning

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data.datamodule import AtomsDataModule
from pytorch_lightning.loggers import WandbLogger

import torch
import torchmetrics

from src.atomwise_simulation import AtomwiseSimulation, SimulationIdxPerAtom
from src.simulated_atoms_datamodule import SimulatedAtomsDataModule
from src.task import SimulatedAtomisticTask, SimulatedModelOutput


name = 'cc_5_dft_100_big'
data_path = './data/experiment_1/cc_dft_dataset.db'
save_path = f"./data/experiment_1/models/{name}.pt"
split_file = './data/experiment_1/splits/cc_5_dft_100.npz'
has_virtual_reactions = True

name = 'cc_5_surrogate'
data_path = './data/experiment_1/cc_surrogate_dataset.db'
save_path = f"./data/experiment_1/models/{name}.pt"
split_file = './data/experiment_1/splits/cc_5_surrogate.npz'
has_virtual_reactions = True
    
lr = 1e-4
batch_size = 32
cutoff = 5.0
n_radial = 64
n_atom_basis = 128
n_interactions = 3

### dataset
dataset = SimulatedAtomsDataModule(
    datapath=data_path,
    split_file=split_file,
    batch_size=batch_size,
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.CastTo32(),
        SimulationIdxPerAtom()
    ],
    num_workers=2,
    pin_memory=True, # set to false, when not using a GPU
    load_properties=['energy', 'simulation_idx'], #only load U0 property
)
dataset.prepare_data()
dataset.setup()


print(dataset.num_train, dataset.num_val, dataset.num_test)

# for idx, sample in enumerate(dataset.val_dataloader()):
#     # print()
#     # print([(k, v.shape) for k, v in sample.items()])
#     print(sample['simulation_idx'].shape)
    