import os
from src.data.datasets.dataset import Dataset

from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.fingerprint_similarity_split import FingerprintSimilaritySplit


if __name__ == "__main__":
    random_seed = 420
    base_path = './data/da/fingerprint_splits_3/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)


    dataset = Dataset(
        folder_path="da/reaxys_rps/"
    )
    source_data = dataset.load()

    split = FingerprintSimilaritySplit(
        key='products',
        n_clusters=5,
        clustering_method='Birch',
    )
    dataframes = split.generate_splits(source_data, random_seed)

    for idx, dataframe in enumerate(dataframes):
        dataset_path = os.path.join(base_path, f'split_{idx}/')
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        dataframe.to_csv(os.path.join(dataset_path, 'dataset.csv'))
