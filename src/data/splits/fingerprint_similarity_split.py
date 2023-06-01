from typing import List, Literal, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator


from src.data.splits import Split

def fingerprint_to_numpy(fp):
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

class FingerprintSimilaritySplit(Split):

    def __init__(
        self,
        n_clusters: int = 5,
        clustering_method: Literal["KMeans", "Birch"] = "Birch",
        pad_with_chembl_reactions: bool = True
    ) -> None:
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.pad_with_chembl_reactions = pad_with_chembl_reactions

    def generate_splits(
        self,
        data: pd.DataFrame,
        random_seed: int = 42
    ) -> Tuple[List[str]]:
        """
        Takes in a dataset & returns list of uids for corresponding
        train, val & test set.
        """
        np.random.seed(random_seed)

        unique_reactants = data['substrates'].unique()
        unique_mols = [Chem.MolFromSmiles(smi) for smi in unique_reactants]
        
        rdkit_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in unique_mols]
        fingerprints = [fingerprint_to_numpy(fp) for fp in fingerprints]

        if self.clustering_method == "KMeans":
            clustering = KMeans(n_clusters=self.n_clusters).fit(fingerprints)
        elif self.clustering_method == "Birch":
            clustering = Birch(n_clusters=self.n_clusters).fit(fingerprints)
        else:
            raise ValueError(f"Clustering method {self.clustering_method} not supported.")

        clustered_unique_reactants = {i: [] for i in range(self.n_clusters)}
        for reactants, label in zip(unique_reactants, clustering.labels_):
            clustered_unique_reactants[label].append(reactants)

        clustered_data = ()
        for label in range(self.n_clusters):
            new_df = data[data['substrates'].isin(clustered_unique_reactants[label])]
            clustered_data += (new_df,)

        print(f'Cluster sizes: {", ".join([str(len(df)) for df in clustered_data])}')

        return clustered_data