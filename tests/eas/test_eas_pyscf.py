import pandas as pd
import time
from rdkit import Chem

from src.reactions.eas.eas_methods import EASDFTPySCF
from src.reactions.eas.eas_reaction import EASReaction


def test_eas_reaction_dft():
    t = time.time()
    df = pd.read_csv('./data/datasets/eas_dataset.csv')

    substrate = df['substrates'].values[516].split('.')[0]
    product = df['products'].values[516]

    reaction = EASReaction(
        substrate, 
        product, 
        method=EASDFTPySCF(
            functional='B3LYP',
            basis_set='6-311g'
        ),
        compute_product_only=False
    )

    conformer_energies = reaction.compute_conformer_energies()
    # assert len(conformer_energies) == 2
    # assert len(conformer_energies[0]) > 0 and len(conformer_energies[1]) > 0

    print(time.time() - t)


if __name__ == "__main__":
    test_eas_reaction_dft()
