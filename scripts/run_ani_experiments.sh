#!/bin/bash

source env.sh


###
### Run Baseline OOD experiment
###
# python scripts/ani-cxx/train_model.py --name "cc_5_baseline(ood)_0" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_0_ood.npz --has_virtual_reactions False --n_simulations 1
# python scripts/ani-cxx/train_model.py --name "cc_5_baseline(ood)_1" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_0_ood.npz --has_virtual_reactions False --n_simulations 1
# python scripts/ani-cxx/train_model.py --name "cc_5_baseline(ood)_2" --project ani-cxx --dataset_path ./data/ani-cxx/experiment_2/dataset_ood.db \
#     --split_file_path ./data/ani-cxx/experiment_2/splits/cc_5_dft_0_ood.npz --has_virtual_reactions False --n_simulations 1



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