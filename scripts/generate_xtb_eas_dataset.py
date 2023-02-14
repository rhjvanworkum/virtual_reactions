import os
from reactions.eas.eas_dataset import Dataset, XtbSimulatedEasDataset

BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
BASE_PATH = '/home/rhjvanworkum/virtual_reactions/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 30
N_CPUS_CONFORMERS = 4

os.environ["BASE_DIR"] = BASE_DIR
os.environ["BASE_PATH"] = BASE_PATH
os.environ["XTB_PATH"] = XTB_PATH


source_dataset = Dataset(
    csv_file_path="eas_dataset_2.csv"
)

dataset = XtbSimulatedEasDataset(
    csv_file_path="xtb_simulated_eas_2_opt_2.csv"
)

dataset.generate(source_dataset, n_cpus=N_CPUS)