import os
import pandas as pd

from src.chemprop.train_vr import train_and_evaluate_chemprop_vr_model
from src.data.datasets.dataset import Dataset
from src.data.splits.hetero_cycle_split import HeteroCycleSplit
from src.data.splits.manual_vr_split import ManualVirtualReactionSplit


if __name__ == "__main__":
    n_replications = 1
    name = 'baseline_model_fp_split_3'
    project = 'vr-new'
    use_features = True
    use_wandb = True

    training_args = {
        # 'hidden_size': 512,
        # 'ffn_hidden_size': 64,
        # 'depth': 3,
        # 'ffn_num_layers': 3,
        'epochs': 100,
        # 'init_lr': 1e-3,
        # 'batch_size': 50,
    }

    dataset = Dataset(
        csv_file_path="eas/eas_dataset.csv"
    )
    source_data = dataset.load()

    # dataset_split = HeteroCycleSplit(
    #     train_split=0.9,
    #     val_split=0.1,
    #     transductive=True
    # )
    dataset_split = ManualVirtualReactionSplit(
        train_split=0.9,
        val_split=0.1,
        transductive=True,
        ood_test_substrates=pd.read_csv('./data/eas/fingerprint_splits/split_2.csv')['substrates'].values
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
        task='classification',
        dataset=dataset,
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args={}
    )