#!/bin/bash -l

#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --output=job_%A.out

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

conda activate schnetpack

python create_surrogate_dataset_and_split.py