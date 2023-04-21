"""
I guess we wanna be able to do 2 approaches again: 
    1) add simulation index as input feature
    2) add just multiple labels for different simulation outcomes

"""

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


if __name__ == "__main__":
    name = 'mol0'
    data_path = './data/experiment_1/cc_dft_dataset.db'
    save_path = f"./data/experiment_1/models/{name}.pt"
    split_file = './data/experiment_1/splits/mol_splits/mol0.npz'
    has_virtual_reactions = True
     
    lr = 1e-4
    batch_size = 32
    cutoff = 5.0
    n_radial = 16
    n_atom_basis = 32
    n_interactions = 3

    use_wandb = True
    epochs = 200
    n_devices = 1


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


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

    ### model
    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_radial, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, 
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_energy = AtomwiseSimulation(n_in=n_atom_basis, output_key='energy')
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)
        ]
    )

    output = SimulatedModelOutput(
        device=device,
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    model = SimulatedAtomisticTask(
        model=nnpot,
        outputs=[output],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr}
    )


    """ Testing purposes only """
    # for idx, sample in enumerate(dataset.train_dataloader()):
    #     print(sample['simulation_idx'])
    #     # # output = model(sample)
    #     # loss = model.training_step(sample, 0)
    #     # # loss = model.training_step(sample, 1)
    #     # print(loss)
    #     break


    logging.info("Setup trainer")
    callbacks = [
        spk.train.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath="checkpoints",
            filename="{epoch:02d}",
            model_path=save_path
        ),
        pytorch_lightning.callbacks.LearningRateMonitor(
        logging_interval="epoch"
        ),
        pytorch_lightning.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-6, 
        patience=10, 
        verbose=False, 
        mode="min"
        )
    ]

    args = {
        'callbacks': callbacks,
        'default_root_dir': './test/',
        'max_epochs': epochs,
        'devices': n_devices
    }

    if torch.cuda.is_available():
        args['accelerator'] = 'gpu'

    if use_wandb:
        wandb_project = 'ani-cxx'
        logger = WandbLogger(project=wandb_project, name=name)
        args['logger'] = logger

    trainer = pytorch_lightning.Trainer(**args)

    logging.info("Start training")
    trainer.fit(model, datamodule=dataset)