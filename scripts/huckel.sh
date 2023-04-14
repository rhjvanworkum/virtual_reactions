#!/bin/bash

source env.sh

# generation
# python scripts/datasets/generation/generate_dft_eas_dataset.py

# python scripts/train_chemprop_model.py
# python scripts/generate_xtb_eas_dataset.py
# python scripts/run_e2_sn2_simulation.py

# python scripts/test_dft.py

export OMP_NUM_THREADS=4

python scripts/datasets/generate_huckel_parameters.py

# python scripts/datasets/generation/generate_xtb_sn2_dataset.py
# python scripts/datasets/generation/generate_ff_eas_dataset.py


# python test_bo.py
# python run_regression.py
# python run_classification.py

# python run_bo_regression.py
# python run_bo_classification.py