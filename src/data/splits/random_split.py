from typing import Tuple, List  
import pandas as pd
import numpy as np

from src.data.splits import Split


class RandomSplit(Split):

    def __init__(
        self,
        train_split: float = 0.9,
        val_split: float = 0.1,
        test_split: float = 0.0,
    ) -> None:
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def generate_splits(
        self,
        data: pd.DataFrame,
        random_seed: int = 42
    ) -> Tuple[List[str]]:
        np.random.seed(random_seed)

        uids = data['uid'].values
        np.random.shuffle(uids)
        train_idxs = uids[:int(self.train_split * len(uids))]
        val_idxs = uids[int(self.train_split * len(uids)):int((self.train_split + self.val_split) * len(uids))]
        test_idxs = uids[int((self.train_split + self.val_split) * len(uids)):]

        print(
            f'n train: {len(train_idxs)}, \
              n val: {len(val_idxs)}, \
              n test: {len(test_idxs)} \n'
        )

        assert set(train_idxs).isdisjoint(val_idxs)
        assert set(train_idxs).isdisjoint(test_idxs)
        assert set(val_idxs).isdisjoint(test_idxs)

        return train_idxs, val_idxs, test_idxs