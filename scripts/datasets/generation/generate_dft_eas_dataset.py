import os
from src.dataset import Dataset
from src.reactions.eas.dft_eas_dataset import DFTSimulatedEasDataset

n_processes = 30

source_dataset = Dataset(
    csv_file_path="eas_dataset.csv"
)

dataset = DFTSimulatedEasDataset(
    csv_file_path="dft_simulated_eas_B3LYP_6-311G.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)