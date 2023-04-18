"""
I guess we wanna be able to do 2 approaches again: 
    1) add simulation index as input feature
    2) add just multiple labels for different simulation outcomes

"""

import logging
import os
import pytorch_lightning

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data.datamodule import AtomsDataModule
from pytorch_lightning.loggers import WandbLogger

import torch
import torchmetrics

from atomwise_simulation import AtomwiseSimulation, SimulationIdxPerAtom


if __name__ == "__main__":
    name = 'cc_5_dft_100'
    data_path = './data/experiment_1/cc_dft_dataset.db'
    save_path = f"./data/experiment_1/models/{name}.pt"
    split_file = './data/experiment_1/splits/cc_5_dft_100.npz'
     
    cutoff = 5.0
    n_atom_basis = 32
    lr = 1e-4
    batch_size = 32


    use_wandb = True
    epochs = 200
    n_devices = 1


    ### dataset
    dataset = AtomsDataModule(
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
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, 
        n_interactions=3,
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

    output = spk.task.ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    model = spk.task.AtomisticTask(
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
        min_delta=1e-4, 
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
    trainer.test(model, datamodule=dataset)
