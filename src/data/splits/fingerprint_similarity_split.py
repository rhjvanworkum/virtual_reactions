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

def get_butina_clusters(
    fingerprints: List[Any],
    cutoff: float = 0.6
):
    dissimilarity_matrix = []
    for i in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])

    clusters = Butina.ClusterData(dissimilarity_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class FingerprintSimilaritySplit(Split):

    def __init__(
        self,
        n_clusters: int = 5,
        key: Literal["substrates", "products"] = "substrates",
        clustering_method: Literal["KMeans", "Birch", "Butina"] = "Birch",
        pad_with_chembl_reactions: bool = True
    ) -> None:
        self.n_clusters = n_clusters
        self.key = key
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

        unique_reactants = data[self.key].unique()
        unique_mols = [Chem.MolFromSmiles(smi) for smi in unique_reactants]
        
        rdkit_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in unique_mols]

        if self.clustering_method == "KMeans":
            fingerprints = [fingerprint_to_numpy(fp) for fp in fingerprints]
            clustering = KMeans(n_clusters=self.n_clusters).fit(fingerprints)
            clustered_unique_reactants = {i: [] for i in range(self.n_clusters)}
            for reactants, label in zip(unique_reactants, clustering.labels_):
                clustered_unique_reactants[label].append(reactants)
        elif self.clustering_method == "Birch":
            fingerprints = [fingerprint_to_numpy(fp) for fp in fingerprints]
            clustering = Birch(n_clusters=self.n_clusters).fit(fingerprints)
            clustered_unique_reactants = {i: [] for i in range(self.n_clusters)}
            for reactants, label in zip(unique_reactants, clustering.labels_):
                clustered_unique_reactants[label].append(reactants)
        elif self.clustering_method == "Butina":
            clustering = get_butina_clusters(fingerprints)
            self.n_clusters = len(clustering)
            clustered_unique_reactants = {
                i: [unique_reactants[idx] for idx in clustering[i]] for i in range(len(clustering))
            }
        else:
            raise ValueError(f"Clustering method {self.clustering_method} not supported.")

        clustered_data = ()
        for label in range(self.n_clusters):
            new_df = data[data[self.key].isin(clustered_unique_reactants[label])]
            clustered_data += (new_df,)

        print(f'Cluster sizes: {", ".join([str(len(df)) for df in clustered_data])}')

        return clustered_data