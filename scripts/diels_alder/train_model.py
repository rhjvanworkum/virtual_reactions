import os

from src.chemprop.train_vr import train_and_evaluate_chemprop_vr_model
from src.chemprop.train import train_and_evaluate_chemprop_model
from src.data.datasets.dataset import Dataset
from src.data.splits.da_split import DASplit
from src.data.splits.hetero_cycle_split import HeteroCycleSplit
from src.data.splits.random_split import RandomSplit


if __name__ == "__main__":
    n_replications = 3
    name = 'global_simulated_DA_literature_Butina=2_nontrans'
    project = 'vr-da'
    use_features = True
    use_wandb = True

    training_args = {
        # 'hidden_size': 512,
        # 'ffn_hidden_size': 64,
        # 'depth': 3,
        # 'ffn_num_layers': 3,
        'epochs': 50,
        # 'init_lr': 1e-3,
        # 'batch_size': 50,
    }

    dataset = Dataset(
        folder_path="da/global_simulated_DA_literature/"
    )
    source_data = dataset.load()
    
    dataset_split = DASplit(
        train_split=0.9,
        val_split=0.1,
        clustering_method='Butina',
        min_cluster_size=2,
        transductive=False
    )
    # dataset_split = DASplit(
    #     train_split=0.9,
    #     val_split=0.1,
    #     clustering_method='Birch',
    #     n_clusters=4,
    #     n_ood_test_clusters=2,
    #     exclude_simulations=True
    # )

    base_dir = os.path.join('./experiments', name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    train_and_evaluate_chemprop_model(
        use_wandb=use_wandb,
        wandb_name=name,
        wandb_project_name=project,
        n_replications=n_replications,
        use_features=use_features,
        dataset=dataset,
        metrics=[],
        task="classification",
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args={}
    )