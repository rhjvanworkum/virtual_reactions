from concurrent.futures import ProcessPoolExecutor
import os
from typing import Any, List, Literal, Tuple, Union
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast

from src.reactions.eas.eas_reaction import EASReaction
from src.dataset import Dataset, SimulatedDataset
from src.split import Split


SIMULATION_IDX_ATOM = ['H', 'He', 'Li', 'Be']


def get_conformer_energies(args):
    substrate_smiles, product_smiles = args

    reaction = EASReaction(
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles
    )

    return reaction.compute_conformer_energies()

class XtbSimulatedEasDataset(SimulatedDataset):

    def __init__(
        self,
        csv_file_path: str
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path,
            n_simulations=1
        )

    def _simulate_reactions(
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        assert simulation_idx == 0
        arguments = [
            (substrate.split('.')[0], product) for substrate, product in zip(substrates, products)
        ]
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            results = list(tqdm(executor.map(get_conformer_energies, arguments), total=len(arguments)))
        return results

    def _select_reaction_to_simulate(
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = source_dataset.load()
        return source_data['substrates'].values, \
               source_data['products'].values, \
               source_data['reaction_idx'].values


class XtbSimulatedExtendedEasDataset(XtbSimulatedEasDataset):

    def __init__(
        self,
        csv_file_path: str,
        split_object: Split,
        similarity_set: Literal["train", "test"] = "train"
    ) -> None:
        super().__init__(
            csv_file_path=csv_file_path,
        )

        self.split_object = split_object
        self.similarity_set = similarity_set

    def _select_reaction_to_simulate(
        source_dataset: Dataset
    ) -> Tuple[List[Union[str, int]]]:
        source_data = source_dataset.load()

        substrates = source_data['substrates'].values
        products = source_data['products'].values
        reaction_idxs = source_data['reaction_idx'].values

        # write code here to query chembl compounds for new reactions
        # train