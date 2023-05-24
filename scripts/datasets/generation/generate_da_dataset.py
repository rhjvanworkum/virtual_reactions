import os
from src.data.datasets.dataset import Dataset
from src.reactions.da.da_dataset import XtbSimulatedDADataset, FukuiSimulatedDADataset
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--functional', type=str)
    parser.add_argument('--basis_set', type=str)
    parser.add_argument('--name', type=str) 
    args = parser.parse_args()

    n_processes = 25

    # source_dataset = Dataset(
    #     csv_file_path="da/da_no_solvent_dataset.csv"
    # )
    # dataset = FukuiSimulatedDADataset(
    #     csv_file_path="da/xtb_simulated_da_solvent_without.csv"
    # )

    # source_dataset = Dataset(
    #     csv_file_path="da/DA_regio_orca_solvent_tight.csv"
    # )
    # dataset = FukuiSimulatedDADataset(
    #     csv_file_path=f"da/{args.name}.csv"
    # )
    # dataset.generate(source_dataset, n_cpus=n_processes, functional=args.functional, basis_set=args.basis_set)

    source_dataset = Dataset(
        csv_file_path="da/DA_regio_orca_solvent_tight.csv"
    )
    dataset = XtbSimulatedDADataset(
        csv_file_path="da/xtb_DA_regio_orca_solvent_tight.csv"
    )
    dataset.generate(source_dataset, n_cpus=n_processes)
