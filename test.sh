#!/bin/bash

source env.sh

# generation
python scripts/datasets/generation/generate_dft_eas_dataset.py

# python scripts/train_chemprop_model.py
# python scripts/generate_xtb_eas_dataset.py
# python scripts/run_e2_sn2_simulation.py

# python test_bo.py
# python run_regression.py
# python run_classification.py

# python run_bo_regression.py
# python run_bo_classification.py
