import torch
import schnetpack as spk
from ase import io
from ase.db import connect
from typing import Any
import numpy as np

cutoff = 5.0

def get_vector_representations_of_mol(
    atoms: Any,
    model: Any,
) -> np.ndarray:
    converter = spk.interfaces.AtomsConverter(
        additional_inputs={'simulation_idx': torch.Tensor([simulation_idx for _ in range(len(atoms))])},
        neighbor_list=spk.transform.ASENeighborList(cutoff=cutoff), 
        dtype=torch.float32, 
        device=device
    )
    input = converter(atoms)

    model_outputs = {}
    def save_forward_outputs(name):
        def hook(model, input, output):
            model_outputs[name] = output
        return hook
    model.representation.register_forward_hook(save_forward_outputs('representation'))

    output = model(input)
    return np.mean(model_outputs['representation']['scalar_representation'].detach().cpu().numpy(), axis=0)

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def mbkmeans_clusters(
	X, 
    k, 
    mb, 
    print_silhouette_values: bool = False, 
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

if __name__ == "__main__":
    output_key = 'energy'
    model_path = './models/cc_10.pt'
    geometry_path = 'test.xyz'
    simulation_idx = 0

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    # model = torch.load(model_path, map_location=device).to(device)
    # model.eval()

    # vectors = []

    mols = []
    with connect('10k_dataset.db') as conn:
        for idx in np.load('split_cc_10.npz')['train_idx']:
            mols.append(str(conn.get_atoms(int(idx) + 1).symbols))
            # vec = get_vector_representations_of_mol(atoms, model)
            # vectors.append(vec.tolist())

    # np.save('vectors.npy', np.array(vectors))

    print(set(mols))


    # vectors = np.load('vectors.npy')
    # km, labels = mbkmeans_clusters(vectors, 3, 16)
    # print(labels)


   
