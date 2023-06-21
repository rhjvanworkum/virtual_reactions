import gin
import pandas as pd
import random
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from src.data.splits.fingerprint_similarity_split import FingerprintSimilaritySplit
from src.data.splits.virtual_reaction_split import VirtualReactionSplit


def select_clusters(
    clusters: List[List[str]],
    n_compounds: int,
) -> List[List[str]]:
    current_clusters = list(clusters).copy()
    selected_clusters = []

    while sum(len(clu) for clu in selected_clusters) < n_compounds:
        idx = random.sample(np.arange(len(current_clusters)).tolist(), 1)[0]

        if sum(len(clu) for clu in selected_clusters) + len(current_clusters[idx]) <= n_compounds:
            selected_clusters.append(current_clusters.pop(idx))

    return selected_clusters


class FingerprintVirtualReactionSplit(VirtualReactionSplit):

    def __init__(
        self,
        train_split: Union[float, int] = 0.9,
        val_split: Union[float, int] = 0.1,
        iid_test_split: float = 0.05,
        virtual_test_split: float = 0.05,
        transductive: bool = True,
        clustering_method: Literal["KMeans", "Birch", "Butina"] = "Birch",
        n_clusters: int = 6,
        n_ood_test_clusters: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        n_ood_test_compounds: Optional[int] = None
    ) -> None:
        super().__init__(transductive)
        self.train_split = train_split
        self.val_split = val_split
        assert self.train_split + self.val_split == 1

        self.iid_test_split = iid_test_split
        self.virtual_test_split = virtual_test_split

        self.n_ood_test_clusters = n_ood_test_clusters
        self.min_cluster_size = min_cluster_size
        self.n_ood_test_compounds = n_ood_test_compounds
        # if self.n_ood_test_clusters is not None and self.min_cluster_size is not None:
        #     raise ValueError("Can Only set either n ood clusters or min cluster size")
        # if self.n_ood_test_clusters is None and self.min_cluster_size is None:
        #     raise ValueError("Must set either n ood clusters or min cluster size") 

        self.fingerprint_split = FingerprintSimilaritySplit(
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            key="products"
        )

        self._test_products = None

    def _set_ood_test_compounds_from_clusters(
        self,
        data: pd.DataFrame
    ) -> List[str]:
        tmp_data = data
        data = tmp_data[tmp_data['simulation_idx'] == 0]
        sim_data = tmp_data[tmp_data['simulation_idx'] != 0]

        clusters = self.fingerprint_split.generate_splits(data)
        
        if self.min_cluster_size is not None:
            test_cluster_idxs = [i for i, cluster in enumerate(clusters) if len(cluster) < self.min_cluster_size]
            test_clusters = [clusters[i] for i in test_cluster_idxs]
        elif self.n_ood_test_clusters is not None:
            test_cluster_idxs = random.sample(np.arange(len(clusters)).tolist(), self.n_ood_test_clusters)
            test_clusters = [clusters[i] for i in test_cluster_idxs]
        elif self.n_ood_test_compounds is not None:
            test_clusters = select_clusters(clusters, self.n_ood_test_compounds)

        self._test_products = pd.concat(test_clusters)['products'].values.tolist()

    def _get_ood_test_set_uids(
        self,
        products: List[str],
        uids: List[int],
        simulation_idxs: List[int]
    ) -> List[int]:
        """
        Returns the OOD test set, experimental data that is consider out-of-distribution
        """
        ood_test_set_uids = []
        for product, uid, simulation_idx in zip(products, uids, simulation_idxs):
            if product in self._test_products and simulation_idx == 0:
                ood_test_set_uids.append(uid)
        return np.array(ood_test_set_uids)
    
    def _get_iid_test_set_uids(
        self,
        products: List[str],
        uids: List[int],
        simulation_idxs: List[int]
    ) -> List[int]:
        """
        Returns the IID test set, experimental data that is consider in-distribution
        """
        potential_iid_test_set_uids = []
        for product, uid, simulation_idx in zip(products, uids, simulation_idxs):
            if not product in self._test_products and simulation_idx == 0:
                potential_iid_test_set_uids.append(uid)

        potential_iid_test_set_uids = np.array(potential_iid_test_set_uids)
        np.random.shuffle(potential_iid_test_set_uids)
        return potential_iid_test_set_uids[:int(self.iid_test_split * len(potential_iid_test_set_uids))]
    
    def _get_virtual_test_set_uids(
        self,
        products: List[str],
        uids: List[int],
        simulation_idxs: List[int]
    ) -> List[int]:
        """
        Returns the virtual test set, just some simulated test set
        """
        potential_virtual_test_set_uid = []
        for _, uid, simulation_idx in zip(products, uids, simulation_idxs):
            if simulation_idx != 0:
                potential_virtual_test_set_uid.append(uid)

        potential_virtual_test_set_uid = np.array(potential_virtual_test_set_uid)
        np.random.shuffle(potential_virtual_test_set_uid)
        return potential_virtual_test_set_uid[:int(self.virtual_test_split * len(potential_virtual_test_set_uid))]
    
    def _get_excluded_set_uids(
        self,
        products: List[str],
        uids: List[int],
        simulation_idxs: List[int],
        data: pd.DataFrame
    ) -> List[int]:
        """
        Returns the excluded set, certain simulated reactions might be excluded from training at all,
        due to the split being non-transductive
        """
        excluded_set_uids = []
        if not self.transductive:
            for product, uid, simulation_idx in zip(products, uids, simulation_idxs):
                if product in self._test_products and simulation_idx != 0:
                    r_idx = data[data['uid'] == uid]['reaction_idx'].values[0]
                    if r_idx in data[data['simulation_idx'] == 0]['reaction_idx'].values:
                        excluded_set_uids.append(uid)
        return np.array(excluded_set_uids)


    def generate_splits(
        self,
        data: pd.DataFrame,
        random_seed: int = 42
    ) -> Tuple[List[str]]:
        """
        Takes in a dataset & returns list of uids for corresponding
        train, val & test set.
        """
        np.random.seed(random_seed)
        self._set_ood_test_compounds_from_clusters(data)

        # get test / excluded IDs
        products, uids, simulation_idxs = data['products'].values, data['uid'].values, data['simulation_idx'].values
        ood_test_set_uids = self._get_ood_test_set_uids(products, uids, simulation_idxs)
        iid_test_set_uids = self._get_iid_test_set_uids(products, uids, simulation_idxs)
        virtual_test_set_uids = self._get_virtual_test_set_uids(products, uids, simulation_idxs)
        excluded_set_uids = self._get_excluded_set_uids(products, list(filter(lambda x: x not in virtual_test_set_uids, uids)), simulation_idxs, data)

        left_over_uids = np.array([
            id for id in uids if (
                (id not in ood_test_set_uids) and 
                (id not in iid_test_set_uids) and 
                (id not in virtual_test_set_uids) and 
                (id not in excluded_set_uids)
            )
        ])
        data_idxs = np.arange(len(left_over_uids))
        np.random.shuffle(data_idxs)
        if isinstance(self.train_split, float) and isinstance(self.val_split, float):
            train_idxs = data_idxs[:int(self.train_split * len(left_over_uids))]
            val_idxs = data_idxs[int(self.train_split * len(left_over_uids)):]
        elif isinstance(self.train_split, int) and isinstance(self.val_split, int):
            train_idxs = data_idxs[:self.train_split]
            val_idxs = data_idxs[self.train_split:(self.train_split + self.val_split)]
        else:
            raise ValueError("train split & val split must be of same instance int or float")

        train_set_uids = left_over_uids[train_idxs]
        val_set_uids = left_over_uids[val_idxs]

        print(f'n train: {len(train_set_uids)}, \
                n val: {len(val_set_uids)}, \
                n OOD test: {len(ood_test_set_uids)}, \
                n IID test: {len(iid_test_set_uids)}, \
                n virtual test: {len(virtual_test_set_uids)}, \
                n excluded: {len(excluded_set_uids)}, \
               \n')

        assert set(train_set_uids).isdisjoint(val_set_uids)
        assert set(train_set_uids).isdisjoint(ood_test_set_uids)
        assert set(train_set_uids).isdisjoint(iid_test_set_uids)
        assert set(train_set_uids).isdisjoint(virtual_test_set_uids)
        assert set(train_set_uids).isdisjoint(excluded_set_uids)
        
        assert set(val_set_uids).isdisjoint(ood_test_set_uids)
        assert set(val_set_uids).isdisjoint(iid_test_set_uids)
        assert set(val_set_uids).isdisjoint(virtual_test_set_uids)
        assert set(val_set_uids).isdisjoint(excluded_set_uids)

        assert set(ood_test_set_uids).isdisjoint(iid_test_set_uids)
        assert set(ood_test_set_uids).isdisjoint(virtual_test_set_uids)
        assert set(ood_test_set_uids).isdisjoint(excluded_set_uids)

        assert set(iid_test_set_uids).isdisjoint(virtual_test_set_uids)
        assert set(iid_test_set_uids).isdisjoint(excluded_set_uids)

        assert set(virtual_test_set_uids).isdisjoint(excluded_set_uids)

        return train_set_uids, val_set_uids, ood_test_set_uids, iid_test_set_uids, virtual_test_set_uids