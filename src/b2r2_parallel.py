from typing import Union, List
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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


def get_b2r2_a_molecular_parallel(
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

def get_b2r2_a_parallel(
    reactants_ncharges,
    products_ncharges,
    reactants_coords,
    products_coords,
    elements=[1, 6, 7, 8, 9, 17],
    r_cut=3.5,
    gridspace=0.03,
    n_processes = 4
):
    """
    Reactants_ncharges is a list of lists where the outer list is the total number
    of reactions and the inner list is the number of reactants in each reaction
    Same for coords, and for products
    """
    all_ncharges_reactants = [np.concatenate(x) for x in reactants_ncharges]
    u_ncharges_reactants = np.unique(np.concatenate(all_ncharges_reactants))
    all_ncharges_products = [np.concatenate(x) for x in products_ncharges]
    u_ncharges_products = np.unique(np.concatenate(all_ncharges_products))
    u_ncharges = np.unique(np.concatenate((u_ncharges_reactants, u_ncharges_products)))

    for ncharge in u_ncharges:
        if ncharge not in elements:
            print("warning!", ncharge, "not included in rep")

    b2r2_l_reactants = np.sum(
        [
            [
                get_b2r2_a_molecular_parallel(
                    reactants_ncharges[i][j],
                    reactants_coords[i][j],
                    r_cut=r_cut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(reactants_ncharges[i]))
            ]
            for i in range(len(reactants_ncharges))
        ],
        axis=1,
    )

    b2r2_l_products = np.sum(
        [
            [
                get_b2r2_a_molecular_parallel(
                    products_ncharges[i][j],
                    products_coords[i][j],
                    r_cut=r_cut,
                    gridspace=gridspace,
                    elements=elements,
                )
                for j in range(len(products_ncharges[i]))
            ]
            for i in range(len(products_ncharges))
        ],
        axis=1,
    )

    b2r2_l = b2r2_l_products - b2r2_l_reactants
    return b2r2_l