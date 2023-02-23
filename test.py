# import os
# from src.reactions.eas.eas_dataset import Dataset, XtbSimulatedEasDataset
# from sklearn.metrics import roc_auc_score

# BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
# BASE_PATH = '/home/rhjvanworkum/virtual_reactions/'
# XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
# N_CPUS = 30
# N_CPUS_CONFORMERS = 4

# os.environ["BASE_DIR"] = BASE_DIR
# os.environ["BASE_PATH"] = BASE_PATH
# os.environ["XTB_PATH"] = XTB_PATH

# dataset = XtbSimulatedEasDataset(
#     csv_file_path="xtb_simulated_eas_2_opt_2.csv"
# )

# df = dataset.load(
#     aggregation_mode='avg',
#     margin=3 / 627.5
# )

# targets, preds = [], []
# for idx in df['reaction_idx'].unique():
#     target = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 0)]['label']
#     pred = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 1)]['label']
    
#     if len(pred) > 0 and len(target) > 0:
#         targets.append(target.values[0])
#         preds.append(pred.values[0])

# print(len(targets))

# print(roc_auc_score(targets, preds))