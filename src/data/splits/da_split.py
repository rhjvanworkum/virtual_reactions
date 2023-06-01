from typing import List, Literal, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
import random

from src.data.splits import Split
from src.data.splits.fingerprint_similarity_split import FingerprintSimilaritySplit


class DASplit(Split):

    def __init__(
        self,
        train_split: float = 0.9,
        val_split: float = 0.1,
        clustering_method: Literal["KMeans", "Birch"] = "Birch",
        n_clusters: int = 6,
        n_ood_test_clusters: int = 2
    ) -> None:
        self.train_split = train_split
        self.val_split = val_split

        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.n_ood_test_clusters = n_ood_test_clusters
        assert self.n_ood_test_clusters < self.n_clusters

        self.fingerprint_split = FingerprintSimilaritySplit(
            clustering_method=self.clustering_method,
            n_clusters=self.n_clusters
        )

    def generate_splits(
        self,
        data: pd.DataFrame,
        random_seed: int = 42
    ) -> Tuple[List[str]]:
        np.random.seed(random_seed)

        clusters = self.fingerprint_split.generate_splits(data)
        test_cluster_idxs = random.sample(np.arange(len(clusters)).tolist(), 2)

        test_df = pd.concat([clusters[i] for i in test_cluster_idxs])
        other_df = pd.concat([clusters[i] for i in np.arange(len(clusters)) if i not in test_cluster_idxs])

        test_idxs = test_df['uid'].values

        uids = other_df['uid'].values
        np.random.shuffle(uids)
        train_idxs = uids[:int(self.train_split * len(uids))]
        val_idxs = uids[int(self.train_split * len(uids)):]

        print(
            f'n train: {len(train_idxs)}, \
              n val: {len(val_idxs)}, \
              n test: {len(test_idxs)} \n'
        )

        assert set(train_idxs).isdisjoint(val_idxs)
        assert set(train_idxs).isdisjoint(test_idxs)
        assert set(val_idxs).isdisjoint(test_idxs)

        return train_idxs, val_idxs, test_idxs