import os
from src.dataset import Dataset
from src.reactions.ma.ma_dataset import XtbSimulatedMADataset


n_processes = 35

source_dataset = Dataset(
    csv_file_path="ma/ma_dataset.csv"
)

dataset = XtbSimulatedMADataset(
    csv_file_path="ma/xtb_simulated_ma_dataset.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)