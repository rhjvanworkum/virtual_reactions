import pandas as pd
import json
from src.methods.methods import HuckelMethod
from src.reactions.e2_sn2.e2_sn2_dataset import list_to_atom
from src.reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction

from src.reactions.eas.eas_methods import eas_ff_methods
from src.reactions.eas.eas_reaction import EASReaction


def test_sn2_reaction_huckel():
    with open('./data/sn2_dataset.json') as json_file:
        dataset = json.load(json_file)

    reaction_label = 'D_C_D_E_B_B'
    args = {
        'reactant_smiles': dataset[reaction_label]['smiles'],
        'reactant_conformers': [list_to_atom(geom) for geom in dataset[reaction_label]['rc_conformers']],
        'ts': list_to_atom(dataset[reaction_label]['ts']),
        'product_conformers': [],
        'method': HuckelMethod(),
        'has_openmm_compatability': False
    }

    reaction = E2Sn2Reaction(**args)

    output = reaction.compute_activation_energies()
    print(output)

if __name__ == "__main__":
    test_sn2_reaction_huckel()
