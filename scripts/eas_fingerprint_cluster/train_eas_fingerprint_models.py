import os
import shutil

from src.chemprop.train import train_and_evaluate_chemprop_model
from src.data.datasets.dataset import Dataset
from src.data.splits.random_split import RandomSplit

if __name__ == "__main__":
    split_idx = 4
    name = f'fingeprint_split_{split_idx}'
    project = 'vr-fingerprints'

    n_replications = 1
    use_features = True
    use_wandb = True

    training_args = {
        'hidden_size': 150,
        'depth': 4,
        'epochs': 100,
        'init_lr': 1e-4,
        # 'ffn_hidden_size': 64,
        # 'ffn_num_layers': 3,
    }

    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    else:
        shutil.rmtree(base_dir)
        os.makedirs(base_dir)


    dataset = Dataset(
        csv_file_path=f"eas/fingerprint_splits/split_{split_idx}.csv"
    )
    source_data = dataset.load()

    dataset_split = RandomSplit(
        train_split=0.9,
        val_split=0.1,
    )

    train_and_evaluate_chemprop_model(
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


    