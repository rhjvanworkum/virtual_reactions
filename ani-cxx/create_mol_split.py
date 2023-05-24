import yaml
import numpy as np
from typing import Tuple

def get_train_val_test_split(
    data_idxs: np.ndarray,
    train_split: float,
    val_split: float,
) -> Tuple[np.ndarray]:
    train_idxs = data_idxs[:int(train_split * len(data_idxs))]
    val_idxs = data_idxs[int(train_split * len(data_idxs)):int((train_split + val_split) * len(data_idxs))]
    test_idxs = data_idxs[int((train_split + val_split) * len(data_idxs)):]
    return train_idxs, val_idxs, test_idxs


if __name__ == "__main__":
    name = 'experiment_2'
    num_dft = 0.02
    train_split, val_split = 0.9, 0.1

    with open(f'data/{name}/splits/split.yaml', "r") as f:
        split = yaml.load(f, Loader=yaml.Loader)

    # dft data
    for mol_idx in split['train'].keys():
        dft_idxs = np.array(split['train'][mol_idx]['dft'])
        np.random.shuffle(dft_idxs)
        dft_idxs = dft_idxs[:int(num_dft * len(dft_idxs))].tolist()
        dft_train_idxs, dft_val_idxs, _ = get_train_val_test_split(dft_idxs, train_split, val_split)

        assert len(set(dft_train_idxs).intersection(set(dft_val_idxs))) == 0

        np.savez(
            f'data/{name}/splits/mol_splits/mol_{mol_idx}_small.npz', 
            train_idx=dft_train_idxs, 
            val_idx=dft_val_idxs, 
            ood_test_idx=[],
            iid_test_idx=[],
            virtual_test_idx=[],
        )
