from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import ast
import numpy as np
import time

from e2_sn2_reaction.e2_sn2_reaction import E2Sn2Reaction

def get_reaction_barriers(args):
    substrate_smiles, nucleophile_smiles, indices = args

    reaction = E2Sn2Reaction(
            substrate_smiles=substrate_smiles,
            nucleophile_smiles=nucleophile_smiles,
            indices=indices
        )

    energies = np.array(reaction.compute_reaction_barriers())
    return energies


if __name__ == "__main__":
    class_data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')

    arguments = []

    for idx, row in class_data.iterrows():
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['products_run'])

        if len(indices[0]) == 4:
            label = 'e2'
        else:
            label= 'sn2'

        arguments.append((substrate_smiles, nucleophile_smiles, indices))

        if idx == 20:
            break

        # reaction = E2Sn2Reaction(
        #     substrate_smiles=substrate_smiles,
        #     nucleophile_smiles=nucleophile_smiles,
        #     indices=indices
        # )

        # energies = np.array(reaction.compute_reaction_barriers())

    tstart = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(get_reaction_barriers, arguments)    

    print(time.time() - tstart)

    d = results[0]
    # sn2_energy = np.min(energies[:, 0])
    # e2_energy = np.min(energies[:, 1])

    # print(f'{label}, sn2_energy: {sn2_energy}, e2_energy: {e2_energy}')