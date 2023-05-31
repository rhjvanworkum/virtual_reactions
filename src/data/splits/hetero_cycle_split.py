from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem

from src.data.splits.virtual_reaction_split import VirtualReactionSplit


class HeteroCycleSplit(VirtualReactionSplit):

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
        train_split: Union[float, int] = 0.9,
        val_split: Union[float, int] = 0.1,
        iid_test_split: float = 0.05,
        virtual_test_split: float = 0.05,
        transductive: bool = True,
    ) -> None:
        super().__init__(transductive=transductive)
        self.train_split = train_split
        self.val_split = val_split
        assert self.train_split + self.val_split == 1

        self.iid_test_split = iid_test_split
        self.virtual_test_split = virtual_test_split
       
    def _contains_heterocycle(
        self,
        smiles: str
    ) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        for heterocycle in self.excluded_hetero_cycles:
                if mol.HasSubstructMatch(Chem.MolFromSmiles(heterocycle)):
                    return True
        return False

    def _get_ood_test_set_uids(
        self,
        substrates: List[str],
        uids: List[int],
        simulation_idxs: List[int]
    ) -> List[int]:
        """
        Returns the OOD test set, experimental data that is consider out-of-distribution
        """
        ood_test_set_uids = []
        for substrate, uid, simulation_idx in zip(substrates, uids, simulation_idxs):
            if self._contains_heterocycle(substrate) and simulation_idx == 0:
                ood_test_set_uids.append(uid)
        return np.array(ood_test_set_uids)

    def _get_iid_test_set_uids(
        self,
        substrates: List[str],
        uids: List[int],
        simulation_idxs: List[int]
    ) -> List[int]:
        """
        Returns the IID test set, experimental data that is consider in-distribution
        """
        potential_iid_test_set_uids = []
        for substrate, uid, simulation_idx in zip(substrates, uids, simulation_idxs):
            if not self._contains_heterocycle(substrate) and simulation_idx == 0:
                potential_iid_test_set_uids.append(uid)

        potential_iid_test_set_uids = np.array(potential_iid_test_set_uids)
        np.random.shuffle(potential_iid_test_set_uids)
        return potential_iid_test_set_uids[:int(self.iid_test_split * len(potential_iid_test_set_uids))]
    
    def _get_virtual_test_set_uids(
        self,
        substrates: List[str],
        uids: List[int],
        simulation_idxs: List[int]
    ) -> List[int]:
        """
        Returns the virtual test set, just some simulated test set
        """
        potential_virtual_test_set_uid = []
        for _, uid, simulation_idx in zip(substrates, uids, simulation_idxs):
            if simulation_idx != 0:
                potential_virtual_test_set_uid.append(uid)

        potential_virtual_test_set_uid = np.array(potential_virtual_test_set_uid)
        np.random.shuffle(potential_virtual_test_set_uid)
        return potential_virtual_test_set_uid[:int(self.virtual_test_split * len(potential_virtual_test_set_uid))]
    
    def _get_excluded_set_uids(
        self,
        substrates: List[str],
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
            for substrate, uid, simulation_idx in zip(substrates, uids, simulation_idxs):
                if self._contains_heterocycle(substrate) and simulation_idx != 0:
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

        # get test / excluded IDs
        substrates, uids, simulation_idxs = data['substrates'].values, data['uid'].values, data['simulation_idx'].values
        ood_test_set_uids = self._get_ood_test_set_uids(substrates, uids, simulation_idxs)
        iid_test_set_uids = self._get_iid_test_set_uids(substrates, uids, simulation_idxs)
        virtual_test_set_uids = self._get_virtual_test_set_uids(substrates, uids, simulation_idxs)
        excluded_set_uids = self._get_excluded_set_uids(substrates, list(filter(lambda x: x not in virtual_test_set_uids, uids)), simulation_idxs, data)

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