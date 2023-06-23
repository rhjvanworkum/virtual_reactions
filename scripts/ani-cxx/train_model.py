"""
I guess we wanna be able to do 2 approaches again: 
    1) add simulation index as input feature
    2) add just multiple labels for different simulation outcomes

"""
import os
import logging
import pytorch_lightning

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data.datamodule import AtomsDataModule
from pytorch_lightning.loggers import WandbLogger

import torch
import torchmetrics
from src.args import parse_schnet_config_from_command_line

from src.schnetpack.atomwise_simulation import AtomwiseSimulation, SimulationIdxPerAtom
from src.schnetpack.simulated_atoms_datamodule import SimulatedAtomsDataModule
from src.schnetpack.task import SimulatedAtomisticTask, SimulatedModelOutput


if __name__ == "__main__":
    config = parse_schnet_config_from_command_line()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    ### dataset
    dataset = SimulatedAtomsDataModule(
        datapath=config.get('dataset_path'),
        split_file=config.get('split_file_path'),
        batch_size=config.get('batch_size'),
        transforms=[
            trn.ASENeighborList(cutoff=config.get('cutoff')),
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
    radial_basis = spk.nn.GaussianRBF(n_rbf=config.get('n_radial'), cutoff=config.get('cutoff'))
    schnet = spk.representation.SchNet(
        n_atom_basis=config.get('n_atom_basis'), 
        n_interactions=config.get('n_interactions'),
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(config.get('cutoff'))
    )
    pred_energy = AtomwiseSimulation(
        n_in=config.get('n_atom_basis'), 
        n_simulations=config.get('n_simulations'),
        sim_embedding_dim=config.get('sim_embedding_dim'), 
        output_key='energy'
    )
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
        optimizer_args={"lr": config.get('lr')},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={'threshold': 1e-4, 'patience': 5, 'factor': 0.5},
        scheduler_monitor='val_loss'
    )

    logging.info("Setup trainer")
    callbacks = [
        spk.train.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath="checkpoints",
            filename="{epoch:02d}",
            model_path=os.path.join(config.get('models_path'), f'{config.get("name")}.pt')
        ),
        pytorch_lightning.callbacks.LearningRateMonitor(
            logging_interval="epoch"
        ),
        pytorch_lightning.callbacks.EarlyStopping(
            monitor="val_loss", 
            min_delta=1e-5, 
            patience=25, 
            verbose=False, 
            mode="min"
        )
    ]

    args = {
        'callbacks': callbacks,
        'default_root_dir': './test/',
        'max_epochs': config.get('epochs'),
        'devices': config.get('n_devices')
    }
    if torch.cuda.is_available():
        args["accelerator"] = "cuda"
        args["num_nodes"] = 1
    if config.get('use_wandb'):
        args['logger'] = WandbLogger(project=config.get('project'), name=config.get('name'))

    trainer = pytorch_lightning.Trainer(**args)

    logging.info("Start training")
    trainer.fit(model, datamodule=dataset)


    logging.info("Start testing")
    dataloaders = [
        dataset.ood_test_dataloader(),
        dataset.iid_test_dataloader(),
    ]
    if config.get('has_virtual_reactions'):
        dataloaders.append(dataset.virtual_test_dataloader())
    trainer.test(model, dataloaders=dataloaders)
