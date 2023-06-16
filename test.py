# # sbatch -N 1 --ntasks-per-node=8 --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out scripts/submit_vr_training.sh

# # sbatch -N 1 --ntasks-per-node=16 --gres=gpu:2 --partition zen2_0256_a40x2 --qos zen2_0256_a40x2 --output=job_%A.out scripts/submit_vr_training.sh

# from src.data.datasets.dataset import Dataset
# from src.data.splits.da_split import DASplit

# dataset = Dataset(
#     csv_file_path="da/DA_literature.csv"
# )
# source_data = dataset.load()

# dataset_split = DASplit(
#     train_split=0.9,
#     val_split=0.1,
#     clustering_method='Butina',
#     min_cluster_size=4,
#     exclude_simulations=True
# )

# dataset_split.generate_splits(source_data)