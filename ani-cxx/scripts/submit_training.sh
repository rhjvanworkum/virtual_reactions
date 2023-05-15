#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=job_%A.out

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

python train_model.py