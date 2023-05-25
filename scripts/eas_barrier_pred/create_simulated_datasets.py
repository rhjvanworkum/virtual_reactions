import numpy as np
import os
import pandas as pd

from src.chemprop.train_utils import get_predictions
from src.data.datasets.dataset import Dataset


def create_simulated_df(
    sim_idx: int,
    data_path: str,
    model_path: str,
    source_data: pd.DataFrame,
    feat: np.ndarray
):
    pred_path = './test.csv'

    atom_descriptor_path = f'./test.npz'
    feat = dataset.load_chemprop_features()
    np.savez(atom_descriptor_path, *feat)

    preds = get_predictions(
        data_path=data_path,
        pred_path=pred_path,
        model_path=model_path,
        other_args={},
        use_features=True,
        atom_descriptor_path=atom_descriptor_path
    )

    os.remove(pred_path)
    os.remove(atom_descriptor_path)

    new_data = source_data.copy()
    new_data['simulation_idx'] = [sim_idx for _ in range(len(new_data))]
    new_data['barrier'] = preds

    labels = []
    for _, row in new_data.iterrows():
        barrier = row['barrier']
        other_barriers = new_data[new_data['substrates'] == row['substrates']]['barrier']
        label = int((barrier <= other_barriers).all())
        labels.append(label)
    new_data['label'] = labels

    return new_data
    
    


if __name__ == "__main__":
    # make global simulated dataset
    dataset = Dataset(
        csv_file_path="eas/eas_dataset.csv"
    )
    dataset.generate_chemprop_dataset(simulation_idx_as_features=True)
    source_data = dataset.load()
    feat = dataset.load_chemprop_features()

    data_path = f'./data/eas/eas_dataset_chemprop.csv'
    model_path = f'./experiments/xtb_eas_regression_full_biggest/fold_0/model_0/model.pt'

    new_data = create_simulated_df(
        sim_idx=1,
        data_path=dataset.chemprop_csv_file_path,
        model_path=model_path,
        source_data=source_data,
        feat=feat
    )
    
    new_dataset = pd.concat([source_data, new_data])
    new_dataset['uid'] = np.arange(len(new_dataset))
    new_dataset.to_csv('./data/eas/global_simulated_eas_dataset.csv')


    # make local simulated dataset
    dataset = Dataset(
        csv_file_path="eas/eas_dataset.csv"
    )
    dataframes = [dataset.load()]
    for i in range(5):
        dataset = Dataset(
            csv_file_path=f"eas/fingerprint_splits/split_{i}.csv"
        )
        dataset.generate_chemprop_dataset(simulation_idx_as_features=True)
        source_data = dataset.load()
        feat = dataset.load_chemprop_features()

        model_path = f'./experiments/20_pct_fingerprint_split_{i}/fold_0/model_0/model.pt'

        new_data = create_simulated_df(
            sim_idx=i + 1,
            data_path=dataset.chemprop_csv_file_path,
            model_path=model_path,
            source_data=source_data,
            feat=feat
        )
        dataframes.append(new_data)
    
    new_dataset = pd.concat(dataframes)
    new_dataset['uid'] = np.arange(len(new_dataset))
    new_dataset.to_csv('./data/eas/local_simulated_eas_dataset.csv')