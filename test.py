# sbatch -N 1 --ntasks-per-node=8 --gres=gpu:2 --partition zen3_0512_a100x2 --qos zen3_0512_a100x2 --output=job_%A.out scripts/submit_vr_training.sh

# sbatch -N 1 --ntasks-per-node=16 --gres=gpu:2 --partition zen2_0256_a40x2 --qos zen2_0256_a40x2 --output=job_%A.out scripts/submit_vr_training.sh

from src.reactions.da.da_dataset import FukuiSimulatedDADataset
from sklearn.metrics import roc_auc_score
import pandas as pd



ds = FukuiSimulatedDADataset(
    csv_file_path='da/fukui_simulated_DA_regio_orca_solvent.csv'
)

df = ds.load_valid()

right = 0
targets, preds = [], []
for idx in df['reaction_idx'].unique():
    target = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 0)]['label']
    pred = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 1)]['label']

    if len(pred) > 0 and 1 in pred.values and (len(pred) == len(target)):
        if (target.values == pred.values).all():
            right += 1
            
        for val in target:
            targets.append(val)
        for val in pred:
            preds.append(val)

print(roc_auc_score(targets, preds) * 100, 'AUROC')
print(right/ len(df['reaction_idx'].unique()) * 100, "%", "accuracy")












# from src.dataset import Dataset
# from src.reactions.eas.eas_dataset import XtbSimulatedEasDataset
# from src.split import HeteroCycleSplit, Split

# random_seed = 420

# # dataset = Dataset(
# #     csv_file_path="eas/eas_dataset.csv"
# # )
# # source_data = dataset.load()
# dataset = XtbSimulatedEasDataset(
#     csv_file_path="eas/xtb_simulated_eas.csv",
# )
# source_data = dataset.load(
#     aggregation_mode='low',
#     margin=3 / 627.5
# )


# dataset_split = HeteroCycleSplit(
#     train_split=0.9,
#     val_split=0.1,
#     transductive=True
# )

# _, _, _, _, _ = dataset_split.generate_splits(source_data, random_seed)
