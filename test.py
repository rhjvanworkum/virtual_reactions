# sbatch -N 1 --ntasks-per-node=8 --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out scripts/submit_vr_training.sh

# sbatch -N 1 --ntasks-per-node=16 --gres=gpu:2 --partition zen2_0256_a40x2 --qos zen2_0256_a40x2 --output=job_%A.out scripts/submit_vr_training.sh


from src.data.datasets.dataset import Dataset
from src.data.splits.hetero_cycle_split import HeteroCycleSplit
from src.data.splits.random_split import RandomSplit


dataset = Dataset(
    csv_file_path="da/DA_literature.csv"
)
source_data = dataset.load()

dataset_split = HeteroCycleSplit(
    train_split=0.9,
    val_split=0.1,
    transductive=True
)

train_uids, val_uids, ood_test_uids, iid_test_uids, virtual_test_uids = dataset_split.generate_splits(source_data, 420)