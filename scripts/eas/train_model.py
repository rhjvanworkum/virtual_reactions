import os
from src.args import parse_chemprop_config_from_command_line

from src.chemprop.train_vr import train_and_evaluate_chemprop_vr_model
from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.ff_simulated_eas_dataset import FFSimulatedEasDataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.fingerprint_vr_split import FingerprintVirtualReactionSplit
from src.data.splits.hetero_cycle_split import HeteroCycleSplit
from src.data.splits.random_split import RandomSplit


dataset_class_dict = {
    'xtb': XtbSimulatedEasDataset,
    'ff': FFSimulatedEasDataset,
    'dataset': Dataset
}


if __name__ == "__main__":
    config = parse_chemprop_config_from_command_line()
    
    training_args = {
        'epochs': config.get('epochs'),
    }

    # create and load dataset class
    dataset_class = dataset_class_dict[config.get('dataset_type')]
    dataset = dataset_class(
        folder_path=config.get('folder_path'),
        simulation_type=config.get('simulation_type')
    )
    if config.get('dataset_type') == 'ff' or config.get('dataset_type') == 'xtb':
        source_data = dataset.load(
            aggregation_mode='low',
            margin=3 / 627.5
        )
    else:
        source_data = dataset.load()


    if config.get('split_type') == 'fp_vr':
        dataset_split = FingerprintVirtualReactionSplit(
            train_split=0.9,
            val_split=0.1,
            clustering_method=config.get('clustering_method'),
            n_ood_test_compounds=config.get('n_ood_test_compounds'),
            transductive=config.get('transductive')
        )
    elif config.get('split_type') == 'hetero_cycle':
        dataset_split = HeteroCycleSplit(
            train_split=0.9,
            val_split=0.1,
            transductive=config.get('transductive')
        )
    elif config.get('split_type') == 'random':
        dataset_split = RandomSplit(
            train_split=0.9,
            val_split=0.1
        )


    base_dir = os.path.join('./experiments', config.get('name'))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


    train_and_evaluate_chemprop_vr_model(
        use_wandb=config.get('use_wandb'),
        wandb_name=config.get('name'),
        wandb_project_name=config.get('project'),
        n_replications=config.get('n_replications'),
        use_features=config.get('use_features'),
        dataset=dataset,
        source_data=source_data,
        dataset_split=dataset_split,
        base_dir=base_dir,
        other_training_args=training_args,
        other_prediction_args={}
    )