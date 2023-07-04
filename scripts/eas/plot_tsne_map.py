import numpy as np
from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
import seaborn as sns

def compute_ecfp_descriptors(smiles_list: List[str]):
    keep_idx = []
    descriptors = []
    for i, smiles in enumerate(smiles_list):
        ecfp = _compute_single_ecfp_descriptor(smiles)
        if ecfp is not None:
            keep_idx.append(i)
            descriptors.append(ecfp)
    return np.vstack(descriptors), keep_idx

def _compute_single_ecfp_descriptor(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as E:
        return None

    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)
    
    return None

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))

df = pd.read_csv('./data/eas/eas_dataset_fingerprint_simulated_whole/dataset.csv')

source_data = df[df['simulation_idx'] == 0]['substrates'].unique()
ecfp_descriptors, keep_idx = compute_ecfp_descriptors(source_data)
# assert np.array_equal(np.array(keep_idx), np.arange(535))
# umap_model = umap.UMAP(metric = "jaccard",
#                       n_neighbors = 25,
#                       n_components = 2,
#                       low_memory = False,
#                       min_dist = 0.001)
# X_umap = umap_model.fit_transform(ecfp_descriptors)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(ecfp_descriptors)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(ecfp_descriptors)

axis = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
for i in range(1,5):
    source_df = df[df['simulation_idx'] == 0]
    sim_df = df[df['simulation_idx'] == i]

    assert np.array_equal(source_df['reaction_idx'].values, sim_df['reaction_idx'].values)

    data = []
    for r_idx in source_df['substrates'].unique():
        targets = source_df[source_df['substrates'] == r_idx]['label'].values
        preds = sim_df[sim_df['substrates'] == r_idx]['label'].values
        score = np.sum([t == p for t, p in zip(targets, preds)])
        data.append(score / len(targets))

    print(len(data))

    g = sns.scatterplot(
        ax=axis[i-1],
        x=X_tsne[:,0], 
        y=X_tsne[:,1],
        hue=data,
        size=(1 - np.array(data)) * 25,
        alpha=1.0,
        size_norm=(0, 25),
        palette="coolwarm"
    )
    g.legend([],[], frameon=False)

plt.savefig(f'map.png')

# print(X_tsne.shape)
# bbbp["TNSE_0"], bbbp["TNSE_1"] = X_tsne[:,0], X_tsne[:,1]