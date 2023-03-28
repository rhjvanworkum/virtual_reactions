from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any, Callable, Iterable, Mapping, Optional
import pandas as pd
import time
from tqdm import tqdm

from src.reactions.eas.eas_methods import EASDFT
from src.reactions.eas.eas_reaction import EASReaction

import multiprocessing
import multiprocessing.pool
import time

class NoDaemonProcess(multiprocessing.Process):
    def __init__(
            self, 
            group: None = None, 
            target: Callable[..., object] = None, 
            name: Optional[str] = None, 
            args: Iterable[Any] = ..., 
            kwargs: Mapping[str, Any] = ..., *, 
            daemon: Optional[bool] = None
        ) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.daemon = property(self._get_daemon, self._set_daemon)

    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    
    def _set_daemon(self, value):
        pass
    
class MyPool(multiprocessing.pool.Pool):
    def __init__(
            self, 
            processes: Optional[int] = None, 
            initializer: Callable[..., object] = None, 
            initargs: Iterable[Any] = ..., 
            maxtasksperchild: Optional[int] = None, 
            context: Any = None
        ) -> None:
        super().__init__(processes, initializer, initargs, maxtasksperchild, context)

        self.process = NoDaemonProcess

data = pd.read_csv('./data/datasets/eas_dataset.csv')
substrate_smiles = data['substrates'].values[0].split('.')[0]
product_smiles = data['products'].values[0]

def compute_barriers(args):
    method = EASDFT(
        functional='B3LYP',
        basis_set='6-311g',
        n_cores=args
    )

    reaction = EASReaction(
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles,
        method=method
    )

    results = reaction.compute_conformer_energies()
    return results

# def main():
#     n_cpus = 8
#     arguments = ['lala' for _ in range(n_cpus)]
#     pool = ProcessPoolExecutor(max_workers=32)

#     results = []
#     for result in pool.map(compute_barriers, arguments):
#         results.append(result)
#     print(results)

if __name__ == "__main__":
    # data = pd.read_csv('./data/datasets/eas_dataset.csv')
    # idx = 656
    # substrate_smiles = data['substrates'].values[idx].split('.')[0]
    # product_smiles = data['products'].values[idx]

    # print(product_smiles)

    substrate_smiles = "n1ccc(nc1N)c1[nH]c(c2ccccc2(OC))c(C(=O)OCC)c1"
    product_smiles = "CCOC(=O)c1cc(-c2ccnc(N)n2)[nH]c1-c1ccc(Br)cc1(OC)"
    # substrate_smiles = "n1ccc[nH]1"
    # product_smiles = "Brc1ccn[nH]1"

    n_processes = 2
    n_cpus_per_process = 16
    print(f'SLURM command should be: sbatch --cpus-per-task=1 --ntasks={n_cpus_per_process * n_processes} test.sh')

    tstart = time.time()
    arguments = [n_cpus_per_process for _ in range(n_processes)]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(tqdm(executor.map(compute_barriers, arguments), total=len(arguments)))
    print(results)
    print(time.time() - tstart)