#!/bin/bash

source env.sh



# 1. vanilla model
python scripts/train_reaction_model.py --name eas_vanilla --project eas_exp_2 --dataset_type dataset \
    --folder_path snar/snar_literature/ --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 1500

# 2. ensembled model
python scripts/train_reaction_model.py --name eas_ensembled --project eas_exp_2 --dataset_type dataset \
    --folder_path snar/snar_literature/ --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 1500 --ensemble_size 2

# 3. Model with simulations as features
python scripts/train_reaction_model.py --name eas_qm_features --project eas_exp_2 --dataset_type dataset \
    --folder_path snar/global_simulated_snar_literature/ --simulation_type outcome_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 1500 

# 4. Model with QM simulations added
python scripts/train_reaction_model.py --name eas_qm_sim --project eas_exp_2 --dataset_type dataset \
    --folder_path snar/global_simulated_snar_literature/ --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 1500 

# 5. Model with random exp simulations added
python scripts/train_reaction_model.py --name eas_random_sim --project eas_exp_2 --dataset_type dataset \
    --folder_path snar/random_simulated_snar_literature/ --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 1500