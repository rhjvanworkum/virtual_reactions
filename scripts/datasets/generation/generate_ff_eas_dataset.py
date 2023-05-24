import os
from src.data.datasets.dataset import Dataset
from src.reactions.eas.eas_dataset import SingleFFSimulatedEasDataset, FFSimulatedEasDataset

n_processes = 25

source_dataset = Dataset(
    csv_file_path="eas/eas_dataset.csv"
)

# dataset = FFSimulatedEasDataset(
#     csv_file_path="ff_simulated_eas.csv"
# )

# dataset.generate(source_dataset, n_cpus=N_CPUS)

dataset = SingleFFSimulatedEasDataset(
    csv_file_path="eas/single_ff_simulated_eas.csv"
)

dataset.generate(source_dataset, n_cpus=n_processes)