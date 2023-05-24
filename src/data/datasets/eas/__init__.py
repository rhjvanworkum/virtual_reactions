import numpy as np
from typing import List, Tuple, Union

from src.data.datasets.dataset import Dataset
from src.data.datasets.simulated_dataset import SimulatedDataset

class SimulatedEASDataset(SimulatedDataset):

    def __init__(
        self, 
        csv_file_path: str, 
        n_simulations: int = 1
    ) -> None:
        super().__init__(csv_file_path, n_simulations)

    def _select_reaction_to_simulate(
        self,
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = source_dataset.load()

        substrates = source_data['substrates'].values
        # compute_substrate = (substrates != np.roll(substrates, 1))
        compute_substrate_only_list = np.zeros(len(substrates))

        return source_data['substrates'].values, \
               source_data['products'].values, \
               ['Methanol' for _ in range(len(substrates))], \
               compute_substrate_only_list, \
               source_data['reaction_idx'].values