"""
The idea here is to create an artificial experimental setting

We need to create 

/data
    /experiment1
        cc_dft_dataset.db
        cc_surrogate_dataset.db
        /models
            /surrogate
                mol1.pt
                mol2.pt
                mol3.pt
                mol4.pt
                mol5.pt
        /splits
            cc_5.npz
            cc_5_dft_100.npz
            cc_5_dft_10.npz
            cc_5_surrogate.npz
            /mol_splits
                mol1.npz
                mol2.npz
                mol3.npz
                mol4.npz
                mol5.npz

create_initial_dataset_and_splits.py
    -> creates cc_dft_dataset.db + splits (except cc_5_surrogate.npz)

train_models.py
train_surrogate_models.py
create_surrogate_dataset_and_split.py
train_final_model.py

"""