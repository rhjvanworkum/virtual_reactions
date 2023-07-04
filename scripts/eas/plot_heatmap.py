import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

df = pd.read_csv('./data/eas/eas_dataset_fingerprint_xtb_simulated_whole/dataset.csv')

split_products = [
    pd.read_csv(f'./data/eas/fingerprint_splits_xtb/split_{i}/dataset.csv') 
    for i in range(4)
]
ex_products = df[~df['products'].isin(pd.concat(split_products)['products'].unique())]['products'].values
split_products = [df['products'].values for df in split_products] + [ex_products]

print([len(ps) for ps in split_products])

data = []

for i in range(1,5):
    data.append([])
    new_df = df[df['simulation_idx'] == i]

    for products in split_products:
        temp_df = new_df[new_df['products'].isin(products)]
        preds = temp_df['label'].values
        targets = df[df['simulation_idx'] == 0]
        targets = targets[targets['products'].isin(products)]['label'].values
        data[-1].append(roc_auc_score(targets, preds))

    print('\n')

sn.heatmap(data, annot=True, cmap='Blues')
plt.show()