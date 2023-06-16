import os
import shutil

from src.chemprop.train import train_and_evaluate_chemprop_model
from src.data.datasets.dataset import Dataset
from src.data.splits.random_split import RandomSplit
from src.chemprop.metrics import mean_absolute_error, mean_squared_error, pearson

if __name__ == "__main__":
    split_idx = 0
    name = f'20_pct_fingerprint_split_{split_idx}_class'
    project = 'vr-fingerprints'

    n_replications = 3
    use_features = False # note turn use features off when using the pretrained grambow model
    use_wandb = True
    # metrics = [
    #     'mae', 'mse', 'pearson'
    # ]
    metrics = []

    training_args = {
        # 'hidden_size': 300,
        # 'depth': 3
        'epochs': 200,
        # 'init_lr': 1e-3,
        # 'ffn_hidden_size': 300,
        # 'ffn_num_layers': 3,
        'checkpoint_paths': ['./data/models/grambow_pre_10.pt'],
        'exclude_parameters': ['readout'],
    }

    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    else:
        shutil.rmtree(base_dir)
        os.makedirs(base_dir)


    dataset = Dataset(
        folder_path=f"eas/20_pct_fingerprint_splits_class/split_{split_idx}/"
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
        task="classification",
        metrics=metrics,
        dataset=dataset,
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args={}
    )


    