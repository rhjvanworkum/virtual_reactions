#!/bin/bash

source env.sh

export AUTODE_LOG_LEVEL=""

# generation
python scripts/datasets/generation/generate_da_dataset.py 
# --functional B3LYP --basis_set 6-31G  --name fukui_simulated_DA_regio_orca_solvent_B3LYP_6-31G

# python scripts/train_chemprop_model.py

# python scripts/generate_xtb_eas_dataset.py
# python scripts/run_e2_sn2_simulation.py

# python scripts/test_dft.py

# export OMP_NUM_THREADS=4
# # python scripts/datasets/generation/generate_da_dataset.py
# python scripts/datasets/generation/generate_xtb_sn2_dataset.py
# # python scripts/datasets/generation/generate_ff_eas_dataset.py
