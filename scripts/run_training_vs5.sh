#!/bin/bash

#SBATCH -N 1
#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

source env.sh

## DA dataset
python scripts/diels_alder/train_model.py

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