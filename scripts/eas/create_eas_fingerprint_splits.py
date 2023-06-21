"""
This script takes the Jensen EAS dataset, splits it into 5 different DataFrames based on fingerprint
similarity
"""
import os
from src.data.datasets.dataset import Dataset

from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.fingerprint_similarity_split import FingerprintSimilaritySplit


if __name__ == "__main__":
    random_seed = 420
    base_path = './data/eas/fingerprint_splits_xtb/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)


    dataset = XtbSimulatedEasDataset(
        folder_path="eas/xtb_simulated_eas/"
    )
    source_data = dataset.load(
        aggregation_mode='low',
        margin=3 / 627.5
    )
    # source_data = dataset.load()
    source_data = source_data[source_data['simulation_idx'] == 1]

    split = FingerprintSimilaritySplit(n_clusters=5)
    dataframes = split.generate_splits(source_data, random_seed)

    for idx, dataframe in enumerate(dataframes):
        dataset_path = os.path.join(base_path, f'split_{idx}/')
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        dataframe.to_csv(os.path.join(dataset_path, 'dataset.csv'))
