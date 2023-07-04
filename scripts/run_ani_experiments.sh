#!/bin/bash

source env.sh


###
### Train surrogate models
###

# python scripts/ani-cxx-new/train_model.py --name "mol_0_10%_surrogate" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/mol_splits/mol_1_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/train_model.py --name "mol_1_10%_surrogate" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/mol_splits/mol_3_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/train_model.py --name "mol_2_10%_surrogate" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/mol_splits/mol_4_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/train_model.py --name "mol_3_10%_surrogate" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/mol_splits/mol_5_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/train_model.py --name "mol_4_10%_surrogate" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/mol_splits/mol_6_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16






# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.0
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_0.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.0
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_0(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_0.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.0
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_0(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_0.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16



# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.15 --num_dft 0.0
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_15_dft_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_15_dft_0.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.15 --num_dft 0.0
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_15_dft_0(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_15_dft_0.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.15 --num_dft 0.0
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_15_dft_0(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_15_dft_0.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16





# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.1
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_10" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.1
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_10(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.1
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_10(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_10.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16



# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.2
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_20" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_20.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.2
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_20(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_20.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.2
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_20(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_20.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16



# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.3
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_30" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_30.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.3
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_30(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_30.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_split.py --num_cc 0.05 --num_dft 0.3
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_dft_30(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/dataset.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_dft_30.npz --has_virtual_reactions False --n_simulations 2 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16






# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.1
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_10" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_10.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.1
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_10(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_10.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.1
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_10(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_10.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16



# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.2
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_20" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_20.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.2
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_20(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_20.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.2
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_20(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_20.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16



# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.3
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_30" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_30.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.3
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_30(1)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_30.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16

# python scripts/ani-cxx-new/data/create_surrogate_split.py --num_cc 0.05 --num_mol_sim 0.3
# python scripts/ani-cxx-new/train_model.py --name "new_test_cc_5_surrgate_10%_30(2)" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_3/surrogate_dataset_10%.db \
#     --split_file_path ./data/ani-cxx/experiment_3/splits/cc_5_surrogate_10%_30.npz --has_virtual_reactions False --n_simulations 6 \
#     --lr 0.001 --n_radial 16 --n_atom_basis 16









###
### Run Baseline OOD experiment
###
# python scripts/ani-cxx/train_model.py --name "cc_5_baseline(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_0_ood.npz --has_virtual_reactions False --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_baseline(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_0_ood.npz --has_virtual_reactions False --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_baseline(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_0_ood.npz --has_virtual_reactions False --n_simulations 2

# # this is an extra experiment
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_2(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_2_ood.npz --has_virtual_reactions False --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_2(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_2_ood.npz --has_virtual_reactions False --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_2(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_2_ood.npz --has_virtual_reactions False --n_simulations 2



###
### Run DFT OOD experiment
###
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_10(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_10_ood.npz --has_virtual_reactions True --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_10(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_10_ood.npz --has_virtual_reactions True --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_10(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_10_ood.npz --has_virtual_reactions True --n_simulations 2

# python scripts/ani-cxx/train_model.py --name "cc_5_dft_20(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_20_ood.npz --has_virtual_reactions True --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_20(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_20_ood.npz --has_virtual_reactions True --n_simulations 2
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_20(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_20_ood.npz --has_virtual_reactions True --n_simulations 2

# python scripts/ani-cxx/train_model.py --name "cc_5_dft_30(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_30_ood.npz --has_virtual_reactions True --n_simulations 2 --n_radial 64 --n_atom_basis 64
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_30(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_30_ood.npz --has_virtual_reactions True --n_simulations 2 --n_radial 64 --n_atom_basis 64
# python scripts/ani-cxx/train_model.py --name "cc_5_dft_30(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_30_ood.npz --has_virtual_reactions True --n_simulations 2 --n_radial 64 --n_atom_basis 64



###
### Run Surrogate (10%) OOD experiment
###
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_10(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_10_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_10(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_10_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_10(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_10_ood.npz --has_virtual_reactions True --n_simulations 6

# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_20(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_20_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_20(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_20_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_20(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_20_ood.npz --has_virtual_reactions True --n_simulations 6

# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_30(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_30_ood.npz --has_virtual_reactions True --n_simulations 6 --n_radial 64 --n_atom_basis 64
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_30(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_30_ood.npz --has_virtual_reactions True --n_simulations 6 --n_radial 64 --n_atom_basis 64
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_30(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_30_ood.npz --has_virtual_reactions True --n_simulations 6 --n_radial 64 --n_atom_basis 64



###
### Run Surrogate (2%) OOD experiment
###
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_10(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_10_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_10(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_10_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_10(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_10_ood.npz --has_virtual_reactions True --n_simulations 6

# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_20(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_20_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_20(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_20_ood.npz --has_virtual_reactions True --n_simulations 6
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_20(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_20_ood.npz --has_virtual_reactions True --n_simulations 6

# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_30(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_30_ood.npz --has_virtual_reactions True --n_simulations 6 --n_radial 64 --n_atom_basis 64
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_30(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_30_ood.npz --has_virtual_reactions True --n_simulations 6 --n_radial 64 --n_atom_basis 64
# python scripts/ani-cxx/train_model.py --name "cc_5_mol_sim_small_30(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/surrogate_dataset_small_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_30_ood.npz --has_virtual_reactions True --n_simulations 6 --n_radial 64 --n_atom_basis 64