from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import ast
from src.reactions.e2_sn2.old.template import E2ReactionTemplate, Sn2ReactionTemplate
from src.reactions.e2_sn2.old.e2_sn2_reaction import E2Sn2Reaction
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 90
N_CPUS_CONFORMERS = 4

os.environ["BASE_DIR"] = BASE_DIR
os.environ["XTB_PATH"] = XTB_PATH

def construct_templates(
    rc_args,
    ts_args
):

    sn2_d_nuc, sn2_d_leav, e2_d_nuc, e2_d_leav = rc_args

    rc_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=sn2_d_nuc,
            d_leaving_group=sn2_d_leav,
            angle=180
        ),
        E2ReactionTemplate(
            d_nucleophile=e2_d_nuc,
            d_leaving_group=e2_d_leav,
            d_H=1
        )
    ]

    sn2_d_nuc, sn2_d_leav, e2_d_nuc, e2_d_leav = ts_args

    ts_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=sn2_d_nuc,
            d_leaving_group=sn2_d_leav,
            angle=180
        ),
        E2ReactionTemplate(
            d_nucleophile=e2_d_nuc,
            d_leaving_group=e2_d_leav,
            d_H=1
        )
    ]

    return rc_templates, ts_templates

def get_reaction_barriers(args):
    substrate_smiles, nucleophile_smiles, indices, rc_templates, ts_templates = args

    reaction = E2Sn2Reaction(
            substrate_smiles=substrate_smiles,
            nucleophile_smiles=nucleophile_smiles,
            indices=indices,
            reaction_complex_templates=rc_templates,
            transition_state_templates=ts_templates
        )

    energies = np.array(reaction.compute_reaction_barriers())
    return energies

def run_experiment(
    name,
    rc_templates,
    ts_templates,
):
    class_data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')

    arguments = []
    labels = []

    for idx, row in class_data.iterrows():
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['products_run'])

        if len(indices[0]) == 4:
            label = 0 # 'e2'
        else:
            label = 1 # 'sn2'

        arguments.append((substrate_smiles, nucleophile_smiles, indices, rc_templates, ts_templates))
        labels.append(label)

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

    sn2_energies, e2_energies = [], []
    for energies in results:
        sn2_energies.append(np.min(energies[:, 0]))
        e2_energies.append(np.min(energies[:, 1]))


    preds = []
    for idx in range(len(labels)):
        if e2_energies[idx] < sn2_energies[idx]:
            pred = 0 # e2
        else:
            pred = 1 # sn2
        preds.append(pred)

    print(preds)

    print(f'Name: {name}')
    print('Accuracy: ', accuracy_score(labels, preds))
    print('Roc AUC: ', roc_auc_score(labels, preds))
    print('\n\n')

    labels = np.array(labels)
    preds = np.array(preds)
    image = np.ones(900) * 2
    values = (labels == preds).astype(int)
    for idx in range(len(values)):
        image[idx] = values[idx]
    image = image.reshape((30, 30))

    plt.imshow(image)
    plt.savefig(f'{name}.png')
    plt.clf()

if __name__ == "__main__":
    # this had a pcc of 0.58
    rc_sn2_d_nucs = [1.5]
    rc_sn2_d_leavs = [1.2]
    ts_sn2_d_nucs = [1.5]
    ts_sn2_d_leavs = [1.8]

    # this had a pcc of 0.22
    rc_e2_d_nucs = [1]
    rc_e2_d_leavs = [1]
    ts_e2_d_nucs = [1]
    ts_e2_d_leavs = [1.97]

    idx = 0

    for rc_sn2_d_nuc in rc_sn2_d_nucs:
        for rc_sn2_d_leav in rc_sn2_d_leavs:
            for rc_e2_d_nuc in rc_e2_d_nucs:
                for rc_e2_d_leav in rc_e2_d_leavs:

                    for ts_sn2_d_nuc in ts_sn2_d_nucs:
                        for ts_sn2_d_leav in ts_sn2_d_leavs:
                            for ts_e2_d_nuc in ts_e2_d_nucs:
                                for ts_e2_d_leav in ts_e2_d_leavs:

                                    rc_templates, ts_templates = construct_templates(
                                        rc_args=(rc_sn2_d_nuc, rc_sn2_d_leav, rc_e2_d_nuc, rc_e2_d_leav),
                                        ts_args=(ts_sn2_d_nuc, ts_sn2_d_leav, ts_e2_d_nuc, ts_e2_d_leav)
                                    )

                                    run_experiment(
                                        name=f'{idx}',
                                        rc_templates=rc_templates,
                                        ts_templates=ts_templates
                                    )
