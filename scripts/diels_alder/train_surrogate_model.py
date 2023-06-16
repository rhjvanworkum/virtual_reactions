import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.chemprop.prediction_utils import create_simulated_df
from src.chemprop.train import train_and_evaluate_chemprop_model
from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.random_split import RandomSplit
from src.chemprop.metrics import mean_absolute_error, mean_squared_error, pearson

if __name__ == "__main__":
    n_replications = 1
    name = 'reaxys_rps_regression'
    project = 'vr-da-surrogate'
    use_features = False # turn this off when using the pretrained Grambow model
    use_wandb = True
    mode = "regression"
    evaluate_on_dataset = True
    save_simulated_dataset = True

    training_args = {
        # 'hidden_size': 512,
        # 'ffn_hidden_size': 512,
        # 'depth': 3,
        # 'ffn_num_layers': 3,
        'epochs': 100,
        # 'init_lr': 1e-4,
        # 'batch_size': 16,
    }

    prediction_args = {
    }

    dataset = Dataset(
        folder_path="da/reaxys_rps/"
    )
    source_data = dataset.load()

    dataset_split = RandomSplit(
        train_split=0.9,
        val_split=0.1,
    )

    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


    if mode == "regression":
        metrics = [
            'mae', 'mse', 'pearson'
        ]
    else:
        metrics = []


    train_and_evaluate_chemprop_model(
        use_wandb=use_wandb,
        wandb_name=name,
        wandb_project_name=project,
        n_replications=n_replications,
        use_features=use_features,
        metrics=metrics,
        task=mode,
        dataset=dataset,
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args=prediction_args
    )

    if evaluate_on_dataset:
        dataset = Dataset(
            folder_path="da/DA_literature/"
        )
        source_data = dataset.load()

        new_data = create_simulated_df(
            sim_idx=1,
            model_path=os.path.join('./experiments', name, 'fold_0', 'model_0', 'model.pt'),
            dataset=dataset,
            use_features=use_features,
            mode=mode
        )
        
        df = pd.concat([source_data, new_data])
        df['uid'] = np.arange(len(df))
        if save_simulated_dataset:
            df.to_csv('./data/da/global_simulated_DA_literature.csv')

        source_df = df[df['simulation_idx'] == 0]
        for idx in [1]:
            sim_df = df[df['simulation_idx'] == idx]
            
            preds, targets = [], []
            for reaction_idx in sim_df['reaction_idx']:
                pred = sim_df[sim_df['reaction_idx'] == reaction_idx]['label'].values[0]
                target = source_df[source_df['reaction_idx'] == reaction_idx]['label'].values[0]
                preds.append(pred)
                targets.append(target)
            
            print('EVALUATION ON DATASET SCORE (DA literature)')
            print(len(sim_df))
            print(roc_auc_score(targets, preds))


        dataset = Dataset(
            folder_path="da/DA_literature_tight/"
        )
        source_data = dataset.load()

        new_data = create_simulated_df(
            sim_idx=1,
            model_path=os.path.join('./experiments', name, 'fold_0', 'model_0', 'model.pt'),
            dataset=dataset,
            use_features=use_features,
            mode=mode
        )
        
        df = pd.concat([source_data, new_data])
        df['uid'] = np.arange(len(df))
        if save_simulated_dataset:
            df.to_csv('./data/da/global_simulated_DA_literature_tight.csv')
        
        source_df = df[df['simulation_idx'] == 0]
        for idx in [1]:
            sim_df = df[df['simulation_idx'] == idx]
            
            preds, targets = [], []
            for reaction_idx in sim_df['reaction_idx']:
                pred = sim_df[sim_df['reaction_idx'] == reaction_idx]['label'].values[0]
                target = source_df[source_df['reaction_idx'] == reaction_idx]['label'].values[0]
                preds.append(pred)
                targets.append(target)
            
            print('EVALUATION ON DATASET SCORE (DA literature tight)')
            print(len(sim_df))
            print(roc_auc_score(targets, preds))