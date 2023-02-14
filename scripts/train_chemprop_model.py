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


if __name__ == "__main__":
    # dataset = XtbSimulatedEasDataset(
    #     csv_file_path="xtb_simulated_eas_2.csv"
    # )

    temp_dataset_path = 'test.csv'

    # dataset.generate_chemprop_dataset(temp_dataset_path)

    os.system(f"chemprop_train --reaction --reaction_mode reac_prod --smiles_columns smiles --target_columns label --data_path {temp_dataset_path} --dataset_type classification --save_dir test")

