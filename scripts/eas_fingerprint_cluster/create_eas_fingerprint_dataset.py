import os
import numpy as np
import pandas as pd

from src.chemprop.train_utils import get_predictions
from src.data.datasets.dataset import Dataset


if __name__ == "__main__":
    output_dataset_name = './data/eas/eas_dataset_fingerprint_simulated_whole.csv'

    dataset = Dataset(
        folder_path="eas/eas_dataset/"
    )
    source_data = dataset.load()

    for i in range(4):
        # data_path = f'./data/eas/fingerprint_splits/split_{i}_chemprop.csv'
        # dataset_path = f'eas/fingerprint_splits/split_{i}.csv'
        data_path = f'./data/eas/eas_dataset_chemprop.csv'
        dataset_path = f'eas/eas_dataset/'

        pred_path = f'./test.csv'
        model_path = f'./experiments/fingeprint_split_{i}/fold_0/model_0/model.pt'

        atom_descriptor_path = f'./test.npz'
        _dataset = Dataset(folder_path=dataset_path)
        feat = _dataset.load_chemprop_features()
        np.savez(atom_descriptor_path, *feat)

        preds = get_predictions(
            data_path=data_path,
            pred_path=pred_path,
            model_path=model_path,
            other_args={},
            use_features=True,
            atom_descriptor_path=atom_descriptor_path
        )
        preds = [round(pred) for pred in preds]

        os.remove(pred_path)
        os.remove(atom_descriptor_path)

        df = _dataset.load()
        df['simulation_idx'] = i + 1
        df['label'] = preds
        df['uid'] = [max(source_data['uid'].values) + i + 1 for i in range(len(df))]
        source_data = pd.concat([source_data, df])

    source_data.to_csv(output_dataset_name)