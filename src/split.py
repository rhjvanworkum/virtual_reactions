from typing import List, Tuple
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem

from src.dataset import Dataset


class Split:

    def __init__(
        self,
    ) -> None:
        pass



class HeteroCycleSplit:

    excluded_hetero_cycles = [
        'c1cOcc1',
        'c1cOnc1',
        'c1cOcn1',
        'c1cScc1',
        'c1cSnc1',
        'c1cScn1',
    ]

    def __init__(
        self,
        train_split: float = 0.9,
        val_split: float = 0.1,
        transductive: bool = False,
    ) -> None:
        self.train_split = train_split
        self.val_split = val_split
        self.transductive = transductive

        assert self.train_split + self.val_split == 1
       
    def _contains_heterocycle(
        self,
        smiles: str
    ) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        for heterocycle in self.excluded_hetero_cycles:
                if mol.HasSubstructMatch(Chem.MolFromSmiles(heterocycle)):
                    return True
        return False

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

        substrates, uids, simulation_idxs = data['substrates'].values, data['uid'].values, data['simulation_idx'].values

        test_set_uids = []
        for substrate, uid, simulation_idx in zip(substrates, uids, simulation_idxs):
            if self._contains_heterocycle(substrate) and simulation_idx == 0:
                test_set_uids.append(uid)
        test_set_uids = np.array(test_set_uids)

        # in case the split is non-transductive we should exclude simulated reactions
        # on the test set (except when they were added through chembl querying)
        excluded_set_uids = []
        if not self.transductive:
            for substrate, uid, simulation_idx in zip(substrates, uids, simulation_idxs):
                if self._contains_heterocycle(substrate) and simulation_idx != 0:
                    r_idx = data[data['uid'] == uid][0]['reaction_idx']
                    if r_idx in data[data['simulation_idx'] == 0]['reaction_idx'].values:
                        excluded_set_uids.append(uid)

        left_over_uids = np.array(
            [id for id in uids if ((id not in test_set_uids) and (id not in excluded_set_uids))]
        )
        data_idxs = np.arange(len(left_over_uids))
        np.random.shuffle(data_idxs)
        train_idxs = data_idxs[:int(self.train_split * len(left_over_uids))]
        val_idxs = data_idxs[int(self.train_split * len(left_over_uids)):]

        train_set_uids = left_over_uids[train_idxs]
        val_set_uids = left_over_uids[val_idxs]

        print(f'n train: {len(train_set_uids)}, n val: {len(val_set_uids)}, n test: {len(test_set_uids)} \n')

        assert set(train_set_uids).isdisjoint(val_set_uids)
        assert set(train_set_uids).isdisjoint(test_set_uids)
        assert set(val_set_uids).isdisjoint(test_set_uids)

        return train_set_uids, val_set_uids, test_set_uids