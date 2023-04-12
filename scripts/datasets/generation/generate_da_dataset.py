import os
from src.dataset import Dataset
from src.reactions.da.da_dataset import XtbSimulatedDADataset

n_processes = 14

source_dataset = Dataset(
    csv_file_path="da/da_solvent_dataset.csv"
)

dataset = XtbSimulatedDADataset(
    csv_file_path="da/xtb_simulated_da_solvent_without.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)