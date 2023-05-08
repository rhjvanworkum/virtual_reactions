import os
from src.dataset import Dataset
from src.reactions.da.da_dataset import XtbSimulatedDADataset, FukuiSimulatedDADataset

n_processes = 25

# source_dataset = Dataset(
#     csv_file_path="da/da_no_solvent_dataset.csv"
# )
# dataset = FukuiSimulatedDADataset(
#     csv_file_path="da/xtb_simulated_da_solvent_without.csv"
# )

source_dataset = Dataset(
    csv_file_path="da/DA_regio_orca_solvent.csv"
)

dataset = FukuiSimulatedDADataset(
    csv_file_path="da/fukui_simulated_DA_regio_orca_solvent.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)