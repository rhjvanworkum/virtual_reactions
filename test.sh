#!/bin/bash

source env.sh

# python scripts/train_chemprop_model.py
# python scripts/generate_xtb_eas_dataset.py
python scripts/run_sn2_simulation.py

# python test_bo.py
# python run_regression.py
# python run_classification.py

# python run_bo_regression.py
# python run_bo_classification.py
