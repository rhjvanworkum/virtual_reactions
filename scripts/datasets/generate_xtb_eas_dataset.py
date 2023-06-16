"""
Script to generate the Xtb simulated EAS dataset
"""


import os
from src.data.datasets.dataset import Dataset
from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset

n_processes = 14

source_dataset = Dataset(
    folder_path="eas/eas_dataset/"
)

dataset = XtbSimulatedEasDataset(
    folder_path="eas/xtb_simulated_eas/"
)

dataset.generate(source_dataset, n_cpus=n_processes)