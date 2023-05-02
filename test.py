from src.reactions.ma.ma_reaction import MAReaction
from src.methods.methods import XtbMethod

import pandas as pd

df = pd.read_csv('./data/datasets/ma/ma_dataset.csv')

reaction = MAReaction(
    substrate_smiles=df['substrates'].values[0],
    product_smiles=df['products'].values[0],
    solvent=None,
    method=XtbMethod(),
    has_openmm_compatability=False,
    compute_product_only=False
)

print(reaction._get_transition_state())