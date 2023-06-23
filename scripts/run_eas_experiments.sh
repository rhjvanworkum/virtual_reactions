#!/bin/bash

source env.sh

###
### 1. run baseline
###
# python scripts/eas/train_model.py --name baseline_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name baseline_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name baseline_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name baseline_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 2. run baseline with XTB features added
###
# python scripts/eas/train_model.py --name baseline_xtbfeat_butina_100 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type outcome_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name baseline_xtbfeat_butina_800 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type outcome_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name baseline_xtbfeat_butina_1800 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type outcome_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name baseline_xtbfeat_butina_2400 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type outcome_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 3. Train model on exp + XTB simulations Transductive
###
# python scripts/eas/train_model.py --name xtb_trans_butina_100 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name xtb_trans_butina_800 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name xtb_trans_butina_1800 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name xtb_trans_butina_2400 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 4. Train model on exp + FF simulations Transductive
###
# python scripts/eas/train_model.py --name ff_trans_butina_100 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name ff_trans_butina_800 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name ff_trans_butina_1800 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name ff_trans_butina_2400 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 5. Train model on exp + local simulations (subset) Transductive
###
# python scripts/eas/train_model.py --name local_sim_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_sim_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_sim_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_sim_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400

###
### 5.5 Train model on exp + local (XTB) simulations (subset) Transductive
###
# python scripts/eas/train_model.py --name local_xtb_sim_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_xtb_sim_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_xtb_sim_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_xtb_sim_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 6. Train model on exp + local simulations (whole) Transductive
###
# python scripts/eas/train_model.py --name local_sim_whole_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_sim_whole_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_sim_whole_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_sim_whole_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 6.5 Train model on exp + local (XTB) simulations (whole) Transductive
###
# python scripts/eas/train_model.py --name local_xtb_sim_whole_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_xtb_sim_whole_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_xtb_sim_whole_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_xtb_sim_whole_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/ --simulation_type index_feature --split_type fp_vr --transductive True \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 7. Train model on exp + XTB simulations Non-Transductive
###
# python scripts/eas/train_model.py --name xtb_non_trans_butina_100 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name xtb_non_trans_butina_800 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name xtb_non_trans_butina_1800 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name xtb_non_trans_butina_2400 --project eas_exp_2 --dataset_type xtb \
#     --folder_path eas/xtb_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 8. Train model on exp + FF simulations Non-Transductive
###
# python scripts/eas/train_model.py --name ff_non_trans_butina_100 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name ff_non_trans_butina_800 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name ff_non_trans_butina_1800 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name ff_non_trans_butina_2400 --project eas_exp_2 --dataset_type ff \
#     --folder_path eas/ff_simulated_eas/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 9. Train model on exp + local simulations (subset) Non-Transductive
###
# python scripts/eas/train_model.py --name local_sim_non_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_sim_non_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_sim_non_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_sim_non_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2400

###
### 9.5 Train model on exp + local (XTB) simulations (subset) Non-Transductive
###
# python scripts/eas/train_model.py --name local_xtb_sim_non_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_xtb_sim_non_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_xtb_sim_non_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_xtb_sim_non_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2400



###
### 10. Train model on exp + local simulations (whole) Non-Transductive
###
# python scripts/eas/train_model.py --name local_sim_whole_non_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_sim_whole_non_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_sim_whole_non_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_sim_whole_non_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_simulated_whole/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2400


###
### 10.5 Train model on exp + local (XTB) simulations (whole) Non-Transductive
###
# python scripts/eas/train_model.py --name local_xtb_sim_whole_non_trans_butina_100 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 100
# python scripts/eas/train_model.py --name local_xtb_sim_whole_non_trans_butina_800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 800
# python scripts/eas/train_model.py --name local_xtb_sim_whole_non_trans_butina_1800 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/  --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 1800
# python scripts/eas/train_model.py --name local_xtb_sim_whole_non_trans_butina_2400 --project eas_exp_2 --dataset_type dataset \
#     --folder_path eas/eas_dataset_fingerprint_xtb_simulated_whole/ --simulation_type index_feature --split_type fp_vr --transductive False \
#     --clustering_method Butina --n_ood_test_compounds 2400