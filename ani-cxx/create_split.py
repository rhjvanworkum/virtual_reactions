from typing import Tuple
import yaml
import numpy as np
import os

def get_train_val_test_split(
    data_idxs: np.ndarray,
    train_split: float,
    val_split: float,
) -> Tuple[np.ndarray]:
    train_idxs = data_idxs[:int(train_split * len(data_idxs))]
    val_idxs = data_idxs[int(train_split * len(data_idxs)):int((train_split + val_split) * len(data_idxs))]
    test_idxs = data_idxs[int((train_split + val_split) * len(data_idxs)):]
    return train_idxs, val_idxs, test_idxs

def get_train_val_split(
    data_idxs: np.ndarray,
    train_split: float,
) -> Tuple[np.ndarray]:
    train_idxs = data_idxs[:int(train_split * len(data_idxs))]
    val_idxs = data_idxs[int(train_split * len(data_idxs)):]
    return train_idxs, val_idxs


if __name__ == "__main__":
    name = 'experiment_2'

    num_cc = 0.25
    num_dft = 0
    train_split, val_split = 0.9, 0.1
    split_name = f'cc_{int(num_cc * 100)}_dft_{int(num_dft * 100)}'

    with open(f'data/{name}/splits/split.yaml', "r") as f:
        split = yaml.load(f, Loader=yaml.Loader)

    # coupled cluster data
    cc_idxs = []
    for mol_idx in split['train'].keys():
        cc_data_idxs = np.array(split['train'][mol_idx]['cc'])
        np.random.shuffle(cc_data_idxs)
        cc_idxs += cc_data_idxs[:int(num_cc * len(cc_data_idxs))].tolist()
    cc_idxs = np.array(cc_idxs)
    np.random.shuffle(cc_idxs)
    cc_train_idxs, cc_val_idxs = get_train_val_split(cc_idxs, train_split)

    # dft data
    dft_idxs = []
    for mol_idx in split['train'].keys():
        dft_data_idxs = np.array(split['train'][mol_idx]['dft'])
        np.random.shuffle(dft_data_idxs)
        dft_idxs += dft_data_idxs[:int(num_dft * len(dft_data_idxs))].tolist()
    dft_idxs = np.array(dft_idxs)
    np.random.shuffle(dft_idxs)
    dft_train_idxs, dft_val_idxs = get_train_val_split(dft_idxs, train_split)
    
    # ood test data
    ood_test_idxs = []
    for mol_idx in split['ood_test'].keys():
        ood_test_idxs += split['ood_test'][mol_idx]
    ood_test_idxs = np.array(ood_test_idxs)
    np.random.shuffle(ood_test_idxs)

    # iid test data
    iid_test_idxs = []
    for mol_idx in split['iid_test'].keys():
        iid_test_idxs += split['iid_test'][mol_idx]
    iid_test_idxs = np.array(iid_test_idxs)
    np.random.shuffle(iid_test_idxs)

    # virtual test data
    virtual_test_idxs = []
    for mol_idx in split['virtual_test'].keys():
        virtual_test_idxs += split['virtual_test'][mol_idx]
    virtual_test_idxs = np.array(virtual_test_idxs)
    np.random.shuffle(virtual_test_idxs)

    # final train, val
    train_idxs = np.concatenate([cc_train_idxs, dft_train_idxs])
    val_idxs = np.concatenate([cc_val_idxs, dft_val_idxs])

    assert len(set(train_idxs).intersection(set(val_idxs))) == 0
    assert len(set(train_idxs).intersection(set(iid_test_idxs))) == 0
    assert len(set(train_idxs).intersection(set(ood_test_idxs))) == 0
    assert len(set(train_idxs).intersection(set(virtual_test_idxs))) == 0
    assert len(set(val_idxs).intersection(set(iid_test_idxs))) == 0
    assert len(set(val_idxs).intersection(set(ood_test_idxs))) == 0
    assert len(set(val_idxs).intersection(set(virtual_test_idxs))) == 0
    assert len(set(iid_test_idxs).intersection(set(ood_test_idxs))) == 0
    assert len(set(iid_test_idxs).intersection(set(virtual_test_idxs))) == 0
    assert len(set(ood_test_idxs).intersection(set(virtual_test_idxs))) == 0

    train_idxs = [int(i) for i in train_idxs]
    val_idxs = [int(i) for i in val_idxs]
    ood_test_idxs = [int(i) for i in ood_test_idxs]
    iid_test_idxs = [int(i) for i in iid_test_idxs]
    virtual_test_idxs = [int(i) for i in virtual_test_idxs]

    # save split file
    np.savez(
        f'data/{name}/splits/{split_name}.npz', 
        train_idx=train_idxs, 
        val_idx=val_idxs, 
        ood_test_idx=ood_test_idxs,
        iid_test_idx=iid_test_idxs,
        virtual_test_idx=virtual_test_idxs,
    )
    