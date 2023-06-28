#!/bin/bash

source env.sh

###
### 1. Baseline
###
# python scripts/train_reaction_model.py --name snar_baseline_butina_200 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/snar_literature --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 200
# python scripts/train_reaction_model.py --name snar_baseline_butina_1400 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1400
# python scripts/train_reaction_model.py --name snar_baseline_butina_2800 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2800
# python scripts/train_reaction_model.py --name snar_baseline_butina_3800 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 3800
# python scripts/train_reaction_model.py --name snar_baseline_butina_3900 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 3900


###
### 2. With RPS simulations (Transductive)
###
# python scripts/train_reaction_model.py --name snar_baseline_sim_trans_butina_200 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 200
# python scripts/train_reaction_model.py --name snar_baseline_sim_trans_butina_1400 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1400
# python scripts/train_reaction_model.py --name snar_baseline_sim_trans_butina_2800 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2800
# python scripts/train_reaction_model.py --name snar_baseline_sim_trans_butina_3800 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 3800
# python scripts/train_reaction_model.py --name snar_baseline_sim_trans_butina_3900 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 3900


###
### 3. With RPS simulations (Non-Transductive)
###
# python scripts/train_reaction_model.py --name snar_baseline_sim_nontrans_butina_200 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 200
# python scripts/train_reaction_model.py --name snar_baseline_sim_nontrans_butina_1400 --project snar_exp_2 --dataset_type dataset \
#     --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1400
python scripts/train_reaction_model.py --name snar_baseline_sim_nontrans_butina_2800 --project snar_exp_2 --dataset_type dataset \
    --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 2800
python scripts/train_reaction_model.py --name snar_baseline_sim_nontrans_butina_3800 --project snar_exp_2 --dataset_type dataset \
    --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 3800
python scripts/train_reaction_model.py --name snar_baseline_sim_nontrans_butina_3900 --project snar_exp_2 --dataset_type dataset \
    --folder_path snar/global_simulated_snar_literature  --simulation_type index_feature --split_type fp_vr --transductive False \
    --clustering_method Butina --n_ood_test_compounds 3900