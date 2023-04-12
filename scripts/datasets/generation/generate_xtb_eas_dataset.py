import os
from src.dataset import Dataset
from src.reactions.eas.xtb_eas_dataset import XtbSimulatedEasDataset

n_processes = 14

source_dataset = Dataset(
    csv_file_path="eas_dataset.csv"
)

dataset = XtbSimulatedEasDataset(
    csv_file_path="xtb_simulated_eas.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)