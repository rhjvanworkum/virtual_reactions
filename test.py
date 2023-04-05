import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
from src.reactions.e2_sn2 import e2_sn2_methods

from src.reactions.e2_sn2.e2_sn2_dataset import XtbSimulatedE2Sn2Dataset, compute_e2_sn2_reaction_barriers, list_to_atom
from src.reactions.e2_sn2.e2_sn2_dataset import E2Sn2Dataset

import json


JSON_FILE = './data/sn2_dataset.json'
with open(JSON_FILE) as json_file:
    dataset = json.load(json_file)

reaction_label = 'D_C_D_E_B_B'
substrate = 'C[C@H](N)[C@](C)(Cl)C#N.[F-]'

args = {
    'reactant_smiles': substrate,
    'reactant_conformers': [list_to_atom(geom) for geom in dataset[reaction_label]['rc_conformers']],
    'ts': list_to_atom(dataset[reaction_label]['ts']),
    'product_conformers': [],
    'method': e2_sn2_methods.e2_sn2_ff_methods[0],
    'has_openmm_compatability': True
}

compute_e2_sn2_reaction_barriers(args)