import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from random import randint
import numpy as np

from src.reactions.eas.xtb_eas_dataset import XtbSimulatedEasDataset
from src.dataset import Dataset
from src.split import HeteroCycleSplit

BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
BASE_PATH = '/home/rhjvanworkum/virtual_reactions/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 30
N_CPUS_CONFORMERS = 4

os.environ["BASE_DIR"] = BASE_DIR
os.environ["BASE_PATH"] = BASE_PATH
os.environ["XTB_PATH"] = XTB_PATH


if __name__ == "__main__":
    n_iterations = 3
    name = 'testje_2'

    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dataset = XtbSimulatedEasDataset(
    #     csv_file_path="xtb_simulated_eas.csv",
    # )
    dataset = Dataset(
        csv_file_path="eas_dataset.csv"
    )

    source_data = dataset.load(
        # aggregation_mode='avg',
        # margin=3 / 627.5
    )
    max_simulation_idx = max(source_data['simulation_idx'].values) + 1

    dataset_split = HeteroCycleSplit(
        train_split=0.9,
        val_split=0.1,
        transductive=True
    )

    # construct evaluation metrics
    tot_train_auc, tot_val_auc, tot_test_auc = [], [], []
    train_auc, val_auc, test_auc = [[] for _ in range(max_simulation_idx)], \
                                   [[] for _ in range(max_simulation_idx)], \
                                   [[] for _ in range(max_simulation_idx)]

    for n in range(n_iterations):
        random_seed = randint(1, 1000)

        # 1. Generate splits
        train_uids, val_uids, test_uids = dataset_split.generate_splits(source_data, random_seed)
        for uids, data_file_path in zip(
            [train_uids, val_uids, test_uids],
            [os.path.join(base_dir, f'{split}_data.csv') for split in ['train', 'val', 'test']]
        ):
            dataset.generate_chemprop_dataset(
                file_path=data_file_path,
                uids=uids,
                # aggregation_mode='avg',
                # margin=3 / 627.5
            )

        # 2. Perform training
        os.system(
            f"chemprop_train \
            --reaction --reaction_mode reac_prod \
            --smiles_columns smiles --target_columns label \
            --data_path {os.path.join(base_dir, 'train_data.csv')} \
            --separate_val_path {os.path.join(base_dir, 'val_data.csv')} \
            --separate_test_path {os.path.join(base_dir, 'test_data.csv')} \
            --dataset_type classification \
            --pytorch_seed {random_seed} \
            --save_dir {base_dir}"
        )

        # 3. Evaluate model
        for split, tot_auc_list, auc_list in zip(
            ['train', 'val', 'test'],
            [tot_train_auc, tot_val_auc, tot_test_auc],
            [train_auc, val_auc, test_auc]
        ):
            os.system(
                f"chemprop_predict \
                --smiles_columns smiles \
                --test_path {os.path.join(base_dir, f'{split}_data.csv')} \
                --checkpoint_dir {base_dir} \
                --preds_path {os.path.join(base_dir, f'{split}_data_preds.csv')}"
            )
            true_df = pd.read_csv(os.path.join(base_dir, f'{split}_data.csv'))
            pred_df = pd.read_csv(os.path.join(base_dir, f'{split}_data_preds.csv'))

            # total roc auc
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
            pred = source_data[(source_data['reaction_idx'] == idx) & (source_data['simulation_idx'] == 1)]['label']
            
            if len(pred) > 0 and len(target) > 0:
                targets.append(target.values[0])
                preds.append(pred.values[0])

        simulation_auc.append(roc_auc_score(targets, preds))

    # save raw metrics and log important ones
    simulation_auc = np.array(simulation_auc)
    tot_train_auc, tot_val_auc, tot_test_auc = np.array(tot_train_auc), np.array(tot_val_auc), np.array(tot_test_auc)
    train_auc, val_auc, test_auc = np.array(train_auc), np.array(val_auc), np.array(test_auc)

    np.save(os.path.join(base_dir, 'simulation_auc.npy'), simulation_auc)
    np.savez(os.path.join(base_dir, 'tot_auc.npz'), tot_train_auc=tot_train_auc, tot_val_auc=tot_val_auc, tot_test_auc=tot_test_auc)
    np.savez(os.path.join(base_dir, 'auc.npz'), train_auc=train_auc, val_auc=val_auc, test_auc=test_auc)

    print(f'Simulation AUROC:')
    print(f'Sim idx {", ".join([str(i) for i in range(max_simulation_idx)])}')
    print(f'AUROC: {", ".join([str(round(i, 3)) for i in simulation_auc])} \n\n')

    print(f'Mean Model AUROC:')
    print(f'Split: train, val, test')
    print(f'AUROC: {round(np.mean(tot_train_auc), 3)} ({", ".join([str(round(np.mean(train_auc[i]), 3)) for i in range(max_simulation_idx)])}), \
                   {round(np.mean(tot_val_auc), 3)} ({", ".join([str(round(np.mean(val_auc[i]), 3)) for i in range(max_simulation_idx)])}), \
                   {round(np.mean(tot_test_auc), 3)} ({", ".join([str(round(np.mean(test_auc[i]), 3)) for i in range(max_simulation_idx)])}), \n\n')

