from typing import Literal
import numpy as np
import os
import pandas as pd
import time
from src.b2r2_parallel import get_b2r2_a_parallel

from src.chemprop.train_utils import get_predictions
from src.data.datasets.dataset import Dataset
from src.data.splits.random_split import RandomSplit
from src.qml import featurize, get_qml_compound_from_smiles, predict_KRR
from src.b2r2 import get_b2r2_l




def create_simulated_df(
    sim_idx: int,
    data_path: str,
    model_path: str,
    source_data: pd.DataFrame,
    feat: np.ndarray,
    use_features: bool = False,
    mode: Literal["classification", "regression"] = "classification"
):
    pred_path = './test.csv'

    atom_descriptor_path = f'./test.npz'
    feat = dataset.load_chemprop_features()
    np.savez(atom_descriptor_path, *feat)

    preds = get_predictions(
        data_path=data_path,
        pred_path=pred_path,
        model_path=model_path,
        other_args={},
        use_features=use_features,
        atom_descriptor_path=atom_descriptor_path
    )

    os.remove(pred_path)
    os.remove(atom_descriptor_path)

    new_data = source_data.copy()
    new_data['simulation_idx'] = [sim_idx for _ in range(len(new_data))]

    if mode == "classification":
        new_data['label'] = [round(pred) for pred in preds]
    else:
        new_data['barrier'] = preds

        labels = []
        for _, row in new_data.iterrows():
            barrier = row['barrier']
            other_barriers = new_data[new_data['substrates'] == row['substrates']]['barrier']
            label = int((barrier <= other_barriers).all())
            labels.append(label)
        new_data['label'] = labels

    return new_data
    
    


if __name__ == "__main__":
    # # make B2R2 KRR simulated dataset
    # t = time.time()
    # dataset = Dataset(
    #     folder_path="eas/eas_dataset/"
    # )
    # source_data = dataset.load()
    # test_features, elements = featurize(
    #     substrate_smiles=source_data['substrates'].to_numpy(),
    #     product_smiles=source_data['products'].to_numpy()
    # )
    # print(f'Finished featurizing test set in {time.time() - t}')

    # t = time.time()
    # dataset = Dataset(
    #     folder_path="eas/xtb_simulated_eas_20pct/"
    # )
    # source_data = dataset.load()

    # targets = source_data['label'].to_numpy()
    # features, _ = featurize(
    #     substrate_smiles=source_data['substrates'].to_numpy(),
    #     product_smiles=source_data['products'].to_numpy(),
    #     elements=elements
    # )
    # print(f'Finsihed featurizing training set in {time.time() - t}')
    
    # t = time.time()
    # y_preds = predict_KRR(
    #     X_train=features,
    #     y_train=targets,
    #     X_test=test_features
    # )
    # print(f'KRR fitted in {time.time() - t}')

    # dataset = Dataset(
    #     folder_path="eas/eas_dataset/"
    # )
    # source_data = dataset.load()
    # new_data = source_data.copy()
    # new_data['simulation_idx'] = [1 for _ in range(len(new_data))]
    # new_data['barrier'] = y_preds

    # labels = []
    # for _, row in new_data.iterrows():
    #     barrier = row['barrier']
    #     other_barriers = new_data[new_data['substrates'] == row['substrates']]['barrier']
    #     label = int((barrier <= other_barriers).all())
    #     labels.append(label)
    # new_data['label'] = labels

    # new_data = pd.concat([source_data, new_data])
    # new_data.to_csv('./data/eas/global_simulated_eas_dataset_krr.csv')




    # # make global simulated dataset
    # dataset = Dataset(
    #     folder_path="eas/eas_dataset/"
    # )
    # source_data = dataset.load()
    # feat = dataset.load_chemprop_features()

    # data_path = f'./data/eas/eas_dataset_chemprop.csv'
    # model_path = f'./experiments/xtb_eas_regression_full_class/fold_0/model_0/model.pt'

    # new_data = create_simulated_df(
    #     sim_idx=1,
    #     data_path=dataset.chemprop_csv_file_path,
    #     model_path=model_path,
    #     source_data=source_data,
    #     feat=feat,
    #     use_features=False
    # )
    
    # new_dataset = pd.concat([source_data, new_data])
    # new_dataset['uid'] = np.arange(len(new_dataset))
    # new_dataset.to_csv('./data/eas/global_simulated_eas_dataset.csv')



    # make local simulated dataset
    dataset = Dataset(
        folder_path="eas/eas_dataset/"
    )
    dataframes = [dataset.load()]
    for i in range(1):
        dataset = Dataset(
            folder_path=f"eas/fingerprint_splits/split_{i}/"
        )
        source_data = dataset.load()
        feat = dataset.load_chemprop_features()

        model_path = f'./experiments/20_pct_fingerprint_split_{i}_class/fold_0/model_0/model.pt'

        new_data = create_simulated_df(
            sim_idx=i + 1,
            data_path=dataset.chemprop_csv_file_path,
            model_path=model_path,
            source_data=source_data,
            feat=feat,
            use_features=False
        )
        dataframes.append(new_data)
    
    new_dataset = pd.concat(dataframes)
    new_dataset['uid'] = np.arange(len(new_dataset))
    new_dataset.to_csv('./data/eas/local_simulated_eas_dataset.csv')