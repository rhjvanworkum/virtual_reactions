# sbatch -N 1 --ntasks-per-node=8 --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out scripts/submit_vr_training.sh

# sbatch -N 1 --ntasks-per-node=16 --gres=gpu:2 --partition zen2_0256_a40x2 --qos zen2_0256_a40x2 --output=job_%A.out scripts/submit_vr_training.sh


from transformers import AlbertModel


# from src.chembl import ChemblData, filter_chembl_compounds_fn_eas

# chembl_data = ChemblData(n_workers=8)
# closest_smiles, similarity = chembl_data.get_similar_mols(
#     smiles='COc1nc2nc(C(=O)c3ccccc3)cn2c2c1CSCC2',
#     n_compounds=5,
#     filter_fn=filter_chembl_compounds_fn_eas
# )
# print(closest_smiles, similarity)


# from src.data.datasets.eas.xtb_simulated_eas_dataset import XtbSimulatedEasDataset

# dataset = XtbSimulatedEasDataset(
#     csv_file_path="eas/xtb_simulated_eas.csv"
# )
# ds = dataset.load(mode='regression')
# print(ds[ds['simulation_idx'] == 0])




# from src.data.datasets.dataset import Dataset
# from src.reactions.eas.eas_dataset import XtbSimulatedEasDataset
# from src.splits.fingerprint_similarity_split import FingerprintSimilaritySplit

# random_seed = 420

# dataset = Dataset(
#     csv_file_path="eas/eas_dataset.csv"
# )
# source_data = dataset.load()

# split = FingerprintSimilaritySplit()
# df_1, df_2, df_3, df_4, df_5 = split.generate_splits(source_data, random_seed)
# print(len(df_1))
# print(len(df_2))
# print(len(df_3))
# print(len(df_4))
# print(len(df_5))