import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc(
    source_distribution: pd.DataFrame,
    sim_distribution: pd.DataFrame
):
    targets, preds = [], []
    for reaction_idx in source_distribution['reaction_idx'].unique():
        if reaction_idx not in sim_distribution['reaction_idx'].unique():
            continue
        else:
            target = source_distribution[source_distribution['reaction_idx'] == reaction_idx]['label'].values[0]
            pred = sim_distribution[sim_distribution['reaction_idx'] == reaction_idx]['label'].values[0]
            targets.append(target)
            preds.append(pred)
    sim_acc = roc_auc_score(targets, preds)
    return sim_acc


def modify_source_distribution(source_df, target_auc):
    new_sim_acc = 0

    while np.abs(new_sim_acc - target_auc) > 0.1:
        new_df = source_df.copy()

        # sample a number of reactions to change
        total_subs = len(source_df['substrates'].unique())
        n_subs = np.random.randint(1, total_subs)
        subs = np.random.choice(source_df['substrates'].unique(), n_subs, replace=False)
        for sub in subs:
            selected_reactions = new_df[new_df['substrates'] == sub]
            neg_uid = selected_reactions[selected_reactions['label'] == 1]['uid']
            pos_uid = np.random.choice(selected_reactions['uid'])

            if len(neg_uid) > 0:
                new_df.loc[new_df['uid'] == neg_uid.values[0], 'label'] = 0
                new_df.loc[new_df['uid'] == pos_uid, 'label'] = 1

        new_sim_acc = compute_roc_auc(source_df, new_df)
    
    return new_df


if __name__ == "__main__":
    source_dataset_path = './data/eas/xtb_simulated_eas/chemprop_index_dataset.csv'
    target_dataset_dir = './data/eas/random_simulated_eas/'
    
    df = pd.read_csv(source_dataset_path)
    assert len(df['simulation_idx'].unique()) == 2

    source_distribution = df[df['simulation_idx'] == 0]
    sim_distribution = df[df['simulation_idx'] == 1]

    # compute ROC AUC
    sim_acc = compute_roc_auc(source_distribution, sim_distribution)

    new_df = modify_source_distribution(source_distribution, sim_acc)
    new_df['simulation_idx'] = 1
    new_df = pd.concat(source_distribution, new_df)
    new_df.to_csv(target_dataset_dir + 'dataset.csv', index=False)
