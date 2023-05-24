import os
import pandas as pd
from src.dataset import Dataset
from src.split import Split
import numpy as np
from typing import Dict, List, Callable

from chemprop.args import TrainArgs, PredictArgs
from chemprop.train.make_predictions import make_predictions


def prepare_csv_files_for_chemprop(
    use_features: bool,
    source_data: pd.DataFrame,
    base_dir: str,
    dataset: Dataset,
    dataset_split: Split,
    random_seed: int = 420,
):
    # splits
    train_uids, val_uids, ood_test_uids, iid_test_uids, virtual_test_uids = dataset_split.generate_splits(source_data, random_seed)
    for uids, data_file_path in zip(
        [train_uids, val_uids, ood_test_uids, iid_test_uids, virtual_test_uids],
        [os.path.join(base_dir, f'{split}_data.csv') for split in ['train', 'val', 'ood_test', 'iid_test', 'virtual_test']]
    ):
        dataframe = dataset.load_chemprop_dataset(uids=uids)
        dataframe.to_csv(data_file_path)

    # features
    if use_features:
        for uids, data_file_path in zip(
            [train_uids, val_uids, ood_test_uids, iid_test_uids, virtual_test_uids],
            [os.path.join(base_dir, f'{split}_feat.npz') for split in ['train', 'val', 'ood_test', 'iid_test', 'virtual_test']]
        ):
            features = dataset.load_chemprop_features(uids=uids)
            np.savez(data_file_path, *features)



def make_chemprop_training_args(
    use_wandb: bool,
    base_dir: str,
    metric_names: List[str],
    random_seed: int = 420,
    other_args: Dict = {},
    use_features: bool = False,
):
    args = {
        # reaction graph
        'reaction': True,
        'reaction_mode': "reac_diff",
        # smiles/target columns
        "smiles_columns": "smiles",
        "target_columns": "label",
        # data paths
        "data_path": os.path.join(base_dir, 'train_data.csv'),
        "separate_val_path": os.path.join(base_dir, 'val_data.csv'),
        "separate_test_path": os.path.join(base_dir, 'ood_test_data.csv'),
        # other stuff
        "dataset_type": "classification",
        "pytorch_seed": random_seed,
        "save_dir": base_dir,
    }

    args_list = []
    for k, v in args.items():
        args_list.append(f'--{k}')
        if isinstance(v, bool):
            pass
        else:
            args_list.append(str(v))
    args = TrainArgs().parse_args(args=args_list)

    args.extra_metrics = metric_names
    args.use_wandb_logger = use_wandb
    for key, value in other_args.items():
        setattr(args, key, value)

    if use_features:
        args.atom_descriptors = "descriptor"
        args.atom_descriptors_path = os.path.join(base_dir, 'train_feat.npz')
        args.separate_val_atom_descriptors_path = os.path.join(base_dir, 'val_feat.npz')
        args.separate_test_atom_descriptors_path = os.path.join(base_dir, 'ood_test_feat.npz')

    return args


def make_chemprop_predict_args(
    label:str,
    base_dir: str,
    other_args: Dict = {},
    use_features: bool = False,
):
    """
    Make ChemProp PredArgs object
    """
    args = {
        'smiles_columns': "smiles",
        'test_path': f"{os.path.join(base_dir, f'{label}_data.csv')}",
        'preds_path': f"{os.path.join(base_dir, f'{label}_data_preds.csv')}",
        'checkpoint_path': os.path.join(base_dir, "fold_0/model_0/model.pt"),
        # 'checkpoint_paths': [os.path.join(base_dir, "fold_0/model_0/model.pt")]
    }

    args_list = []
    for k, v in args.items():
        args_list.append(f'--{k}')
        args_list.append(str(v))
    args = PredictArgs().parse_args(args=args_list)

    for key, value in other_args.items():
        setattr(args, key, value)

    if use_features:
        args.atom_descriptors = "descriptor"
        args.atom_descriptors_path = os.path.join(base_dir, f'{label}_feat.npz')
    
    return args

def get_predictions(
    label:str,
    base_dir: str,
    other_args: Dict = {},
    use_features: bool = False,
):
    pred_args = make_chemprop_predict_args(
        label,
        base_dir,
        other_args,
        use_features,
    )
    preds = make_predictions(args=pred_args)
    preds = [p[0] for p in preds]
    return preds


def get_scores_from_preds(
    label:str,
    source_df: pd.DataFrame,
    metrics: List[Callable],
    base_dir: str,
    other_args: Dict = {},
    use_features: bool = False,
) -> List[float]:
    """
    Function to compute scores after training ChemProp model
    more easily
    """
    pred_args = make_chemprop_predict_args(
        label,
        base_dir,
        other_args,
        use_features,
    )
    true = source_df['energies'].values
    preds = make_predictions(args=pred_args)
    preds = [p[0] for p in preds]
    metrics = [metric(true, preds) for metric in metrics]
    return metrics