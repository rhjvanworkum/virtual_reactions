import pandas as pd

from src.reactions.eas.eas_methods import eas_ff_methods
from src.reactions.eas_reaction import EASReaction


def test_eas_reaction_ff():
    df = pd.read_csv('./data/datasets/eas_dataset.csv')

    substrate = df['substrates'].values[2].split('.')[0]
    product = df['products'].values[2]

    for ff_method in eas_ff_methods:
        reaction = EASReaction(
            substrate, 
            product, 
            method=ff_method,
            has_openmm_compatability=True
        )

        conformer_energies = reaction.compute_conformer_energies()
        assert len(conformer_energies) == 2
        assert len(conformer_energies[0]) > 0 and len(conformer_energies[1]) > 0


if __name__ == "__main__":
    test_eas_reaction_ff()
