import os

from src.chemprop.train import train_and_evaluate_chemprop_model
from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.random_split import RandomSplit
from src.chemprop.metrics import mean_absolute_error, mean_squared_error, pearson

if __name__ == "__main__":
    n_replications = 1
    name = 'xtb_eas_regression_full_class'
    project = 'vr-fingerprints'
    use_features = False # turn this off when using the pretrained Grambow model
    use_wandb = True
    # metrics = [
    #     'mae', 'mse', 'pearson'
    # ]
    metrics = []

    training_args = {
        # 'hidden_size': 600,
        # 'ffn_hidden_size': 512,
        # 'depth': 3,
        # 'ffn_num_layers': 3,
        'epochs': 200,
        # 'init_lr': 1e-4,
        # 'batch_size': 50,

        # 'checkpoint_paths': ['./data/models/grambow_pre_100.pt'],
        # 'exclude_parameters': ['readout'],

        # 'features_generator': ['rdkit_2d_normalized'],
        # 'no_features_scaling': '',
    }

    prediction_args = {
        # 'features_generator': ['rdkit_2d_normalized'],
        # 'no_features_scaling': '',
    }

    dataset = Dataset(
        csv_file_path="eas/xtb_simulated_eas_20pct_class.csv"
    )
    source_data = dataset.load()

    dataset_split = RandomSplit(
        train_split=0.9,
        val_split=0.1,
    )

    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    train_and_evaluate_chemprop_model(
        use_wandb=use_wandb,
        wandb_name=name,
        wandb_project_name=project,
        n_replications=n_replications,
        use_features=use_features,
        metrics=metrics,
        task="classification",
        dataset=dataset,
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args=prediction_args
    )