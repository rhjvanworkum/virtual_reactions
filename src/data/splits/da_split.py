from typing import List, Literal, Optional, Tuple, Union
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
        clustering_method: Literal["KMeans", "Birch", "Butina"] = "Birch",
        n_clusters: int = 6,
        n_ood_test_clusters: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        transductive: bool = True
    ) -> None:
        self.train_split = train_split
        self.val_split = val_split

        self.n_ood_test_clusters = n_ood_test_clusters
        self.min_cluster_size = min_cluster_size
        if self.n_ood_test_clusters is not None and self.min_cluster_size is not None:
            raise ValueError("Can Only set either n ood clusters or min cluster size")
        if self.n_ood_test_clusters is None and self.min_cluster_size is None:
            raise ValueError("Must set either n ood clusters or min cluster size") 

        self.fingerprint_split = FingerprintSimilaritySplit(
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            key="products"
        )

        self.transductive = transductive

    def generate_splits(
        self,
        data: pd.DataFrame,
        random_seed: int = 42
    ) -> Tuple[List[str]]:
        np.random.seed(random_seed)

        # separate simulated data
        tmp_data = data
        data = tmp_data[tmp_data['simulation_idx'] == 0]
        sim_data = tmp_data[tmp_data['simulation_idx'] != 0]

        clusters = self.fingerprint_split.generate_splits(data)

        if self.min_cluster_size is not None:
            test_cluster_idxs = [i for i, cluster in enumerate(clusters) if len(cluster) < self.min_cluster_size]
        elif self.n_ood_test_clusters is not None:
            test_cluster_idxs = random.sample(np.arange(len(clusters)).tolist(), self.n_ood_test_clusters)

        test_df = pd.concat([clusters[i] for i in test_cluster_idxs])
        other_df = pd.concat([clusters[i] for i in np.arange(len(clusters)) if i not in test_cluster_idxs])
        
        # add simulated data properly
        if self.transductive:
            other_df = pd.concat([other_df, sim_data])
        else:
            test_products = test_df['products'].values
            other_sim_df = sim_data[~sim_data['products'].isin(test_products)]
            other_df = pd.concat([other_df, other_sim_df])

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