# sbatch -N 1 --ntasks-per-node=8 --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out scripts/submit_vr_training.sh

# sbatch -N 1 --ntasks-per-node=16 --gres=gpu:2 --partition zen2_0256_a40x2 --qos zen2_0256_a40x2 --output=job_%A.out scripts/submit_vr_training.sh

from src.data.datasets.dataset import Dataset
from src.reactions.eas.eas_dataset import XtbSimulatedEasDataset
from src.splits.fingerprint_similarity_split import FingerprintSimilaritySplit

random_seed = 420

dataset = Dataset(
    csv_file_path="eas/eas_dataset.csv"
)
source_data = dataset.load()

split = FingerprintSimilaritySplit()
df_1, df_2, df_3, df_4, df_5 = split.generate_splits(source_data, random_seed)
print(len(df_1))
print(len(df_2))
print(len(df_3))
print(len(df_4))
print(len(df_5))