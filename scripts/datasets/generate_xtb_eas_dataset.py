"""
Script to generate the Xtb simulated EAS dataset
"""


import os
from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset

n_processes = 14

source_dataset = Dataset(
    csv_file_path="eas_dataset.csv"
)

dataset = XtbSimulatedEasDataset(
    csv_file_path="xtb_simulated_eas.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)