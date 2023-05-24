import os

from src.chemprop.train_vr import train_and_evaluate_chemprop_vr_model
from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.random_split import RandomSplit


if __name__ == "__main__":
    n_replications = 1
    name = 'xtb_eas_regression_full'
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

    dataset = XtbSimulatedEasDataset(
        csv_file_path="eas/xtb_simulated_eas_20pct.csv"
    )
    source_data = dataset.load(mode='regression')

    dataset_split = RandomSplit(
        train_split=0.9,
        val_split=0.1,
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