import os
from typing import Callable, Dict, List, Literal, Optional
import pandas as pd
from sklearn.metrics import roc_auc_score
from random import randint
import numpy as np
import wandb

from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training import run_training

from src.data.datasets.dataset import Dataset
from src.data.splits import Split
from src.chemprop.train_utils import get_predictions, make_chemprop_training_args, prepare_csv_files_for_chemprop


def train_and_evaluate_chemprop_model(
    use_wandb: bool,
    wandb_project_name: str,
    wandb_name: str,
    n_replications: int,
    use_features: bool,
    metrics,
    task: Literal["classification", "regression"],
    dataset: Dataset,
    source_data: pd.DataFrame,
    dataset_split: Split,
    base_dir: os.path,
    other_training_args: Dict[str, str],
    other_prediction_args: Dict[str, str],
    force_dataset_generation: bool = False,
    scheduler_fn: Optional[Callable] = None
) -> None:
    # generate dataset if needed
    dataset.generate_chemprop_dataset(force=force_dataset_generation, simulation_idx_as_features=use_features)

    # train and evaluate model
    for i in range(n_replications):
        if use_wandb:
            wandb.init(
                project=wandb_project_name,
                name=f'{wandb_name}_{i}'
            )

        random_seed = randint(1, 1000)

        # 1. Prepare CSV files
        train_uids, val_uids, ood_test_uids = dataset_split.generate_splits(source_data, random_seed)
        for uids, data_file_path in zip(
            [train_uids, val_uids, ood_test_uids],
            [os.path.join(base_dir, f'{split}_data.csv') for split in ['train', 'val', 'ood_test']]
        ):
            dataframe = dataset.load_chemprop_dataset(uids=uids)
            dataframe.to_csv(data_file_path)

        if use_features:
            for uids, data_file_path in zip(
                [train_uids, val_uids, ood_test_uids],
                [os.path.join(base_dir, f'{split}_feat.npz') for split in ['train', 'val', 'ood_test']]
            ):
                features = dataset.load_chemprop_features(uids=uids)
                np.savez(data_file_path, *features)

        # 2. Perform training
        training_args = make_chemprop_training_args(
            use_wandb=use_wandb,
            base_dir=base_dir,
            metric_names=metrics,
            task=task,
            random_seed=random_seed,
            other_args=other_training_args,
            use_features=use_features,
        )
        cross_validate(args=training_args, train_func=run_training, scheduler_fn=scheduler_fn)

        if use_wandb:
            wandb.finish()

    # clean up
    files = [
        'train_data.csv',
        'train_data_preds.csv',
        'train_feat.npz',
        'val_data.csv',
        'val_data_preds.csv',
        'val_feat.npz',
        'ood_test_data.csv',
        'ood_test_data_preds.csv',
        'ood_test_feat.npz',
    ]
    for file in files:
        if os.path.exists(os.path.join(base_dir, file)):
            os.remove(os.path.join(base_dir, file))