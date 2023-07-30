import random
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset


def print_sim_score(df, sim_idx):
    right, targets, preds = 0, [], []
    for idx in df['reaction_idx'].unique():
        target = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 0)]['label'].values
        pred = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == sim_idx)]['label'].values
        if len(pred) > 0 and len(target) > 0:

            if (target == pred).all():
                right += 1
                
            for val in target:
                targets.append(val)
            for val in pred:
                preds.append(val)

    print(roc_auc_score(targets, preds) * 100, 'AUROC')
    print(right/ len(df['reaction_idx'].unique()) * 100, "%", "accuracy")


if __name__ == "__main__":
    # random_alphas = [0.76, 0.75, 0.76, 0.76]

    # dataset = Dataset(
    #     folder_path="eas/eas_dataset/"
    # )
    # source_df = dataset.load()

    # sim_dfs = [source_df]
    # for idx, r in enumerate(random_alphas):
    #     sim_df = source_df.copy()
    #     sim_df['simulation_idx'] = idx + 1

    #     sim_labels = []
    #     for label in source_df['label'].values:
    #         if random.random() > r:
    #             sim_labels.append(1 - label)
    #         else:
    #             sim_labels.append(label)

    #     sim_df['label'] = sim_labels
    #     sim_dfs.append(sim_df)
    
    # df = pd.concat(sim_dfs)

    # for i in range(1, 5):
    #     print_sim_score(df, i)

    # df['uid'] = np.arange(len(df))
    # df.to_csv('./data/eas/xtb_match_random_simulated_dataset.csv')





    random_alphas = [0.90, 0.90, 0.90, 0.90]

    dataset = XtbSimulatedEasDataset(
        folder_path="eas/xtb_simulated_eas/"
    )
    df = dataset.load(
        aggregation_mode='low',
        margin=3 / 627.5
    )

    source_df = df[df['simulation_idx'] == 0]
    init_sim_df = df[df['simulation_idx'] == 1]

    sim_dfs = [source_df]
    for idx, r in enumerate(random_alphas):
        sim_df = init_sim_df.copy()
        sim_df['simulation_idx'] = idx + 1

        size = int((1 - r) * len(sim_df['substrates'].unique()))
        flip_r_subs = np.random.choice(sim_df['substrates'].unique(), size)

        sim_labels = []
        for sub in sim_df['substrates'].unique():
            if sub not in flip_r_subs:
                sim_labels.extend(
                    sim_df[(sim_df['substrates'] == sub)]['label'].values.tolist()
                )
            else:
                labels = sim_df[(sim_df['substrates'] == sub)]['label'].values.tolist()
                true_idxs = []
                for idx, label in enumerate(labels):
                    if label == 1: true_idxs.append(idx) 
                flip_idxs = []
                while len(flip_idxs) > len(true_idxs):
                    idx = random.randint(0, len(labels) - 1)
                    if idx not in true_idxs and idx not in flip_idxs:
                        flip_idxs.append(idx)

                for idx in flip_idxs:
                    labels[idx] = 1
                for idx in true_idxs:
                    labels[idx] = 0

                sim_labels.extend(labels)

        sim_df['label'] = sim_labels
        sim_dfs.append(sim_df)
    
    df = pd.concat(sim_dfs)

    for i in range(1, 5):
        print_sim_score(df, i)

    df['uid'] = np.arange(len(df))
    df.to_csv('./data/eas/xtb_random_simulated_dataset.csv')