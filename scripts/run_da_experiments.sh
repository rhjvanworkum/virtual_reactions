#!/bin/bash

source env.sh

###
### 1. Baseline
###
# python scripts/train_reaction_model.py --name DA_baseline_butina_100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/DA_literature --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/train_reaction_model.py --name DA_baseline_butina_800 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/DA_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/train_reaction_model.py --name DA_baseline_butina_1700 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/DA_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1700
# python scripts/train_reaction_model.py --name DA_baseline_butina_2100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/DA_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2100


###
### 2. With RPS simulations features
###
# python scripts/train_reaction_model.py --name DA_baseline_sim_feat_butina_100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature --simulation_type outcome_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/train_reaction_model.py --name DA_baseline_sim_feat_butina_800 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type outcome_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python -u scripts/train_reaction_model.py --name DA_baseline_sim_feat_butina_1700 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type outcome_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1700
# python scripts/train_reaction_model.py --name DA_baseline_sim_feat_butina_2100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type outcome_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2100


###
### 3. With RPS simulations (Transductive)
###
# python scripts/train_reaction_model.py --name DA_baseline_sim_trans_butina_100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/train_reaction_model.py --name DA_baseline_sim_trans_butina_800 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python -u scripts/train_reaction_model.py --name DA_baseline_sim_trans_butina_1700 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1700
# python scripts/train_reaction_model.py --name DA_baseline_sim_trans_butina_2100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2100


###
### 4. With RPS simulations (Non-Transductive)
##
# python scripts/train_reaction_model.py --name DA_baseline_sim_nontrans_butina_100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/train_reaction_model.py --name DA_baseline_sim_nontrans_butina_800 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/train_reaction_model.py --name DA_baseline_sim_nontrans_butina_1700 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1700
# python scripts/train_reaction_model.py --name DA_baseline_sim_nontrans_butina_2100 --project da_exp_2 --dataset_type dataset \
#     --folder_path da/global_simulated_DA_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2100