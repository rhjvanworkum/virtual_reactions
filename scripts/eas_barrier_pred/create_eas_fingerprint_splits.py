"""
This script takes the Jensen EAS dataset, splits it into 5 different DataFrames based on fingerprint
similarity
"""
import os
import pandas as pd

from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset
from src.data.splits.fingerprint_similarity_split import FingerprintSimilaritySplit


if __name__ == "__main__":
    random_seed = 420
    base_path = './data/eas/20_pct_fingerprint_splits/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    n_clusters = 5


    dataset = XtbSimulatedEasDataset(
        csv_file_path="eas/xtb_simulated_eas.csv"
    )
    df = dataset.load(mode='regression')
    max_data_points = int(0.2 * len(df) / n_clusters)

    split = FingerprintSimilaritySplit(n_clusters=n_clusters)
    dataframes = split.generate_splits(df, 420)
    dataframes = [dataframe[:max_data_points] for dataframe in dataframes]

    for idx, dataframe in enumerate(dataframes):
        dataframe.to_csv(os.path.join(base_path, f'split_{idx}.csv'))

    df = pd.concat(dataframes)
    df.to_csv('./data/eas/xtb_simulated_eas_20pct.csv')