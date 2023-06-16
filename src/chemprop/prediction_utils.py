from typing import Literal
import numpy as np
import os

from src.chemprop.train_utils import get_predictions
from src.data.datasets.dataset import Dataset


def create_simulated_df(
    sim_idx: int,
    model_path: str,
    dataset: Dataset,
    use_features: bool = False,
    mode: Literal["classification", "regression"] = "classification"
):
    """
    Create a simulated version of a reaction dataset using the predictions of a trained
    ChemProp model.
    """
    dataset.generate_chemprop_dataset(simulation_idx_as_features=True)
    source_data = dataset.load()

    pred_path = './test.csv'

    atom_descriptor_path = f'./test.npz'
    feat = dataset.load_chemprop_features()
    np.savez(atom_descriptor_path, *feat)

    preds = get_predictions(
        data_path=dataset.chemprop_csv_file_path,
        pred_path=pred_path,
        model_path=model_path,
        other_args={},
        use_features=use_features,
        atom_descriptor_path=atom_descriptor_path
    )

    if os.path.exists(pred_path):
        os.remove(pred_path)
    if os.path.exists(atom_descriptor_path):
        os.remove(atom_descriptor_path)

    new_data = source_data.copy()
    new_data['simulation_idx'] = [sim_idx for _ in range(len(new_data))]

    if mode == "classification":
        new_data['label'] = [round(pred) for pred in preds]
    else:
        new_data['barrier'] = preds

        labels = []
        for _, row in new_data.iterrows():
            barrier = row['barrier']
            other_barriers = new_data[new_data['substrates'] == row['substrates']]['barrier']
            label = int((barrier <= other_barriers).all())
            labels.append(label)
        new_data['label'] = labels

    return new_data