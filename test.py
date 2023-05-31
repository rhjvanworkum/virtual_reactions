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

import numpy as np

from src.b2r2 import get_bags

def get_mu_sigma(R):
    mu = R / 2 + 1e-8
    sigma = R / 8 + 1e-8
    return mu, sigma

def get_gaussian(x, R, mask):
    mu, sigma = get_mu_sigma(R)
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    norm *= mask

    norm, mu, sigma = norm[..., None], mu[..., None], sigma[..., None]

    return norm * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def get_b2r2_a_molecular(
    ncharges: Union[np.ndarray, List[int]],
    coords: np.ndarray,
    elements: List[int],
    r_cut: float = 3.5,
    gridspace: float = 0.03,
) -> np.ndarray:
    ncharges = [x for x in ncharges if x in elements]
    bags = get_bags(elements)

    grid = np.arange(0, r_cut, gridspace)
    size = len(grid)

    twobodyrep = np.zeros((len(bags), size))

    R = coords[None, :, :] - coords[:, None, :]
    R = np.linalg.norm(R, axis=-1)

    mask = R.copy()
    mask[mask < r_cut] = 1
    mask[mask > r_cut] = 0

    gaussian = get_gaussian(grid, R, mask)
    assert gaussian.shape == (len(ncharges), len(ncharges), size)
    
    for i in range(len(ncharges)):
        ncharge_a = ncharges[i]
        for j in range(i+1, len(ncharges)):
            ncharge_b = ncharges[j]

            try:
                k = bags.index([ncharge_a, ncharge_b]) 
            except ValueError:
                k = bags.index([ncharge_b, ncharge_a])
            
            twobodyrep[k] += gaussian[i, j]

    twobodyrep = np.concatenate(twobodyrep)
    return twobodyrep

# coords = np.array([
#     [1, 0, 0],
#     [0, 2, 0],
#     [0, 0, 3],
# ])

# ncharges = [1, 4, 5]

# get_b2r2_a_molecular(ncharges, coords, elements=[1, 4, 5])

# # elements = [1, 6, 7, 8, 9, 17]
# # bags = get_bags(elements)
# # print(bags)
for i in range(5):
    for j in range(i+1, 5):
        print(i, j)