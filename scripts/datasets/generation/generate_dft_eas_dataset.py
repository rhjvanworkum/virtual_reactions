import os
from src.dataset import Dataset
from src.reactions.eas.dft_eas_dataset import DFTSimulatedEasDataset

BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
BASE_PATH = '/home/rhjvanworkum/virtual_reactions/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 30

os.environ["BASE_DIR"] = BASE_DIR
os.environ["BASE_PATH"] = BASE_PATH
os.environ["XTB_PATH"] = XTB_PATH


source_dataset = Dataset(
    csv_file_path="eas_dataset.csv"
)

dataset = DFTSimulatedEasDataset(
    csv_file_path="dft_simulated_eas_B3LYP_6-311G.csv"
)

dataset.generate(source_dataset, n_cpus=N_CPUS)