#!/bin/bash

source env.sh

# python scripts/eas/train_model.py --name ma_testt --project ma_test_small --dataset_type dataset \
#     --folder_path ma/ma_dataset/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 20 --n_replications 3

python scripts/snar/train_surrogate_model.py

# python scripts/eas/train_eas_fingerprint_models.py

## DA dataset
# python scripts/diels_alder/train_model.py
# python scripts/diels_alder/train_surrogate_model.py

## FCA dataset
# python scripts/fca/train_surrogate_model.py

## EAS barrier pred
# python scripts/eas_barrier_pred/train_global_surrogate_model.py
# python scripts/eas_barrier_pred/train_local_surrogate_model.py
# python scripts/eas_barrier_pred/train_local_simulated_model.py
# python scripts/eas_barrier_pred/train_global_simulated_model.py
# python scripts/eas_barrier_pred/train_baseline_model.py


## FP split models
# python scripts/eas_fingerprint_cluster/train_model.py


## Train baseline model(s)
# python scripts/train_baseline_model.py
# python scripts/train_xtb_model.py
# python scripts/train_ff_model.py