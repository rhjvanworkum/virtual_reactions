import os
import numpy as np
import pandas as pd
from src.chemprop.prediction_utils import create_simulated_df

from src.chemprop.train_utils import get_predictions
from src.data.datasets.dataset import Dataset


if __name__ == "__main__":
    mode = "classification"
    use_features = False

    dataset = Dataset(
        folder_path="eas/eas_dataset/"
    )
    source_data = dataset.load()

    output_dataset_name = './data/eas/eas_dataset_fingerprint_xtb_simulated_whole.csv'
    for i in range(4):
        name = f'fingeprint_split_xtb_{i}'
        new_data = create_simulated_df(
            sim_idx=i + 1,
            model_path=os.path.join('./experiments', name, 'fold_0', 'model_0', 'model.pt'),
            dataset=dataset,
            use_features=use_features,
            mode=mode
        )
        source_data = pd.concat([source_data, new_data])
    source_data['uid'] = np.arange(len(source_data))
    source_data.to_csv(output_dataset_name)

    dataset = Dataset(
        folder_path="eas/eas_dataset/"
    )
    source_data = dataset.load()

    output_dataset_name = './data/eas/eas_dataset_fingerprint_xtb_simulated.csv'
    for i in range(4):
        name = f'fingeprint_split_xtb_{i}'
        dataset = Dataset(
            folder_path=f"eas/fingerprint_splits_xtb/split_{i}/",
            simulation_type="index_feature"
        )
        new_data = create_simulated_df(
            sim_idx=i + 1,
            model_path=os.path.join('./experiments', name, 'fold_0', 'model_0', 'model.pt'),
            dataset=dataset,
            use_features=use_features,
            mode=mode
        )
        source_data = pd.concat([source_data, new_data])
    source_data['uid'] = np.arange(len(source_data))
    source_data.to_csv(output_dataset_name)