# sbatch --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out submit_training.sh

from src.dataset import Dataset
from src.reactions.eas.eas_dataset import XtbSimulatedEasDataset
from src.split import HeteroCycleSplit, Split

random_seed = 420

# dataset = Dataset(
#     csv_file_path="eas/eas_dataset.csv"
# )
# source_data = dataset.load()
dataset = XtbSimulatedEasDataset(
    csv_file_path="eas/xtb_simulated_eas.csv",
)
source_data = dataset.load(
    aggregation_mode='low',
    margin=3 / 627.5
)


dataset_split = HeteroCycleSplit(
    train_split=0.9,
    val_split=0.1,
    transductive=True
)

_, _, _, _, _ = dataset_split.generate_splits(source_data, random_seed)