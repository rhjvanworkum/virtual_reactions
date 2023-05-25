#!/bin/bash

source env.sh

# python scripts/eas_barrier_pred/train_global_surrogate_model.py
# python scripts/eas_barrier_pred/train_local_surrogate_model.py
python scripts/eas_barrier_pred/train_local_simulated_model.py
# python scripts/eas_barrier_pred/train_global_simulated_model.py
# python scripts/eas_barrier_pred/train_baseline_model.py

# python scripts/train_model.py
# python scripts/train_eas_fingerprint_models.py