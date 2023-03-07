import pandas as pd

from src.reactions.eas.eas_methods import eas_xtb_method
from src.reactions.eas.eas_reaction import EASReaction


def test_eas_reaction_xtb():
    df = pd.read_csv('./data/datasets/eas_dataset.csv')

    substrate = df['substrates'].values[2].split('.')[0]
    product = df['products'].values[2]
    reaction = EASReaction(
        substrate, 
        product, 
        method=eas_xtb_method
    )

    conformer_energies = reaction.compute_conformer_energies()
    assert len(conformer_energies) == 2
    assert len(conformer_energies[0]) > 0 and len(conformer_energies[1]) > 0


if __name__ == "__main__":
    test_eas_reaction_xtb()
