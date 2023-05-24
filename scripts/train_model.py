import os
from typing import Callable, Dict, List, Optional
import pandas as pd
from sklearn.metrics import roc_auc_score
from random import randint
import numpy as np
import wandb
from src.chemprop.train_vr import train_and_evaluate_chemprop_vr_model

from src.chemprop.train_utils import get_predictions, make_chemprop_training_args, prepare_csv_files_for_chemprop

from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training import run_training
from src.data.datasets.dataset import Dataset

from src.reactions.eas.eas_dataset import FFSimulatedEasDataset, SingleFFSimulatedEasDataset, XtbSimulatedEasDataset

from chemprop.args import TrainArgs, PredictArgs

from src.splits.hetero_cycle_split import HeteroCycleSplit


if __name__ == "__main__":
    n_replications = 1
    name = 'fp_sim_test'
    project = 'vr'
    use_features = True
    use_wandb = True

    training_args = {
        # 'hidden_size': 512,
        # 'ffn_hidden_size': 64,
        # 'depth': 3,
        # 'ffn_num_layers': 3,
        'epochs': 15,
        # 'init_lr': 1e-3,
        # 'batch_size': 50,
    }

    """ FP simulated EAS dataset """
    dataset = Dataset(
        csv_file_path="eas/eas_dataset_fingerprint_simulated.csv"
    )
    source_data = dataset.load()

    """ Original EAS dataset"""
    # dataset = Dataset(
    #     csv_file_path="eas/eas_dataset.csv"
    # )
    # source_data = dataset.load()
    
    """ Simulated EAS datasets"""
    # dataset = FFSimulatedEasDataset(d
    #     csv_file_path="eas/ff_simulated_eas.csv"
    # )
    # dataset = SingleFFSimulatedEasDataset(
    #     csv_file_path="eas/single_ff_simulated_eas.csv",
    # )
    # dataset = XtbSimulatedEasDataset(
    #     csv_file_path="eas/xtb_simulated_eas.csv",
    # )
    # source_data = dataset.load(
    #     aggregation_mode='low',
    #     margin=3 / 627.5
    # )

    dataset_split = HeteroCycleSplit(
        train_split=0.9,
        val_split=0.1,
        transductive=True
    )


    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    train_and_evaluate_chemprop_vr_model(
        use_wandb=use_wandb,
        wandb_name=name,
        wandb_project_name=project,
        n_replications=n_replications,
        use_features=use_features,
        dataset=dataset,
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args={}
    )