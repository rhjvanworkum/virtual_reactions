import os
from typing import Callable, Dict, List, Optional
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
    dataset: Dataset,
    source_data: pd.DataFrame,
    dataset_split: Split,
    base_dir: os.path,
    other_training_args: Dict[str, str],
    other_prediction_args: Dict[str, str],
    force_dataset_generation: bool = False,
    scheduler_fn: Optional[Callable] = None
) -> None:
    if use_wandb:
        wandb.init(
            project=wandb_project_name,
            name=wandb_name
        )

    max_simulation_idx = max(source_data['simulation_idx'].values) + 1

    # generate dataset if needed
    dataset.generate_chemprop_dataset(force=force_dataset_generation, simulation_idx_as_features=use_features)

    # construct evaluation metrics
    tot_train_auc, tot_val_auc, tot_ood_test_auc = [], [], []
    train_auc, val_auc, ood_test_auc =  [[] for _ in range(max_simulation_idx)], \
                                        [[] for _ in range(max_simulation_idx)], \
                                        [[] for _ in range(max_simulation_idx)], \

    # train and evaluate model
    for _ in range(n_replications):
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
            metric_names=[],
            random_seed=random_seed,
            other_args=other_training_args,
            use_features=use_features,
        )
        cross_validate(args=training_args, train_func=run_training, scheduler_fn=scheduler_fn)
        

        # 3. Evaluate model
        for split, tot_auc_list, auc_list in zip(
            ['train', 'val', 'ood_test'],
            [tot_train_auc, tot_val_auc, tot_ood_test_auc],
            [train_auc, val_auc, ood_test_auc]
        ):
            _ = get_predictions(
                label=split,
                base_dir=base_dir,
                other_args=other_prediction_args,
                use_features=use_features,
            )
            
            true_df = pd.read_csv(os.path.join(base_dir, f'{split}_data.csv'))
            if os.path.exists(os.path.join(base_dir, f'{split}_data_preds.csv')):
                pred_df = pd.read_csv(os.path.join(base_dir, f'{split}_data_preds.csv'))

                # total roc auc
                if len(true_df['label'].values) > 0 and len(pred_df['label'].values) > 0:
                    tot_auc_list.append(
                        roc_auc_score(
                            true_df['label'].values, 
                            pred_df['label'].values
                        )
                    )

                # virtual reaction specific roc auc
                for simulation_idx in range(max_simulation_idx):
                    labels = true_df[true_df['simulation_idx'] == simulation_idx]['label'].values
                    preds = pred_df[pred_df['simulation_idx'] == simulation_idx]['label'].values
                    if len(labels) > 0 and len(preds) > 0:
                        auc_list[simulation_idx].append(
                            roc_auc_score(
                                labels,
                                preds
                            )
                        )
                    else:
                        auc_list[simulation_idx].append(0)

    # evaluate simulation
    simulation_auc = [1.0]
    for simulation_idx in range(1, max_simulation_idx):
        targets, preds = [], []
        for idx in source_data['reaction_idx'].unique():
            target = source_data[(source_data['reaction_idx'] == idx) & (source_data['simulation_idx'] == 0)]['label']
            pred = source_data[(source_data['reaction_idx'] == idx) & (source_data['simulation_idx'] == simulation_idx)]['label']
            
            if len(pred) > 0 and len(target) > 0:
                targets.append(target.values[0])
                preds.append(pred.values[0])

        simulation_auc.append(roc_auc_score(targets, preds))

    # save raw metrics and log important ones
    simulation_auc = np.array(simulation_auc)
    tot_train_auc, tot_val_auc, tot_ood_test_auc = np.array(tot_train_auc), \
                                                   np.array(tot_val_auc), \
                                                   np.array(tot_ood_test_auc)
    train_auc, val_auc, ood_test_auc = np.array(train_auc), \
                                       np.array(val_auc), \
                                       np.array(ood_test_auc)

    print(f'Simulation AUROC:')
    print(f'Sim idx: {", ".join([str(i) for i in range(max_simulation_idx)])}')
    print(f'AUROC  : {", ".join([str(round(i, 3)) for i in simulation_auc])} \n\n')

    print(f'Mean Model AUROC:')
    print(f'Train       : {round(np.mean(tot_train_auc), 3)} ({", ".join([str(round(np.mean(train_auc[i]), 3)) for i in range(max_simulation_idx)])})')
    print(f'Val         : {round(np.mean(tot_val_auc), 3)} ({", ".join([str(round(np.mean(val_auc[i]), 3)) for i in range(max_simulation_idx)])})')
    print(f'OOD Test    : {round(np.mean(tot_ood_test_auc), 3)} ({", ".join([str(round(np.mean(ood_test_auc[i]), 3)) for i in range(max_simulation_idx)])})')

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
            os.remove(base_dir, file)