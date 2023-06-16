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
    base_path = './data/eas/fingerprint_splits/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)


    dataset = Dataset(
        folder_path="eas/eas_dataset/"
    )
    source_data = dataset.load()

    split = FingerprintSimilaritySplit(n_clusters=5)
    dataframes = split.generate_splits(source_data, random_seed)

    for idx, dataframe in enumerate(dataframes):
        dataframe.to_csv(os.path.join(base_path, f'split_{idx}.csv'))
