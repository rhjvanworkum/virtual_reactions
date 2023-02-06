import pandas as pd
import ast
import os
from e2_sn2_reaction.template import E2ReactionTemplate, Sn2ReactionTemplate
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from run_classification import get_reaction_barriers


BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 30
N_CPUS_CONFORMERS = 4

os.environ["BASE_DIR"] = BASE_DIR
os.environ["XTB_PATH"] = XTB_PATH

def get_templates_sn2(
    sn2_d_nuc: float = 2.0,
    sn2_d_leav: float = 1.5
):
    rc_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=sn2_d_nuc,
            d_leaving_group=sn2_d_leav,
            angle=180
        )
    ]

    ts_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=sn2_d_nuc,
            d_leaving_group=sn2_d_leav,
            angle=180
        )
    ]

    return rc_templates, ts_templates

def get_templates_e2(
    e2_d_nuc: float = 1.2,
    e2_d_leav: float = 1.5
):
    rc_templates = [
        E2ReactionTemplate(
            d_nucleophile=e2_d_nuc,
            d_leaving_group=e2_d_leav,
            d_H=1.22
        )
    ]

    ts_templates = [
        E2ReactionTemplate(
            d_nucleophile=e2_d_nuc,
            d_leaving_group=e2_d_leav,
            d_H=1.22
        )
    ]

    return rc_templates, ts_templates


def run_experiment():
    reg_data = pd.read_csv('./data/regression_dataset.csv').sample(30)

    arguments = []
    labels = []
    e2_idxs, sn2_idxs = [], []

    for idx, row in reg_data.iterrows():
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['reaction_core'])
        label = float(row['activation_energy'])
        
        reaction_type = str(row['reaction_type'])
        if reaction_type == 'sn2':
            rc_templates, ts_templates = get_templates_sn2()
            sn2_idxs.append(idx)
        elif reaction_type == 'e2':
            rc_templates, ts_templates = get_templates_e2()
            e2_idxs.append(idx)

        arguments.append((substrate_smiles, nucleophile_smiles, indices, rc_templates, ts_templates))
        labels.append(label)

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

    preds = []
    for idx, energies in enumerate(results):
        preds.append(np.min(energies[:, 0]))

    print(preds)
    print(labels)
    print(e2_idxs)
    print(sn2_idxs)

    sn2_preds, sn2_targets = [], []
    for idx in sn2_idxs:
        sn2_preds.append(preds[idx])
        sn2_targets.append(labels[idx])
    e2_preds, e2_targets = [], []
    for idx in e2_idxs:
        e2_preds.append(preds[idx])
        e2_targets.append(labels[idx])

    print('Sn2 MSE: ', mean_squared_error(sn2_targets, sn2_preds))
    print('E2 MSE: ', mean_squared_error(e2_targets, e2_preds))

    sns.lmplot(x=sn2_targets, y=sn2_preds)
    plt.savefig('sn2.png')

    sns.lmplot(x=sn2_targets, y=sn2_preds)
    plt.savefig('e2.png')

if __name__ == "__main__":
    run_experiment()











# """
# Script used to convert the dataframe
# """
# data = pd.read_csv('./data/e2_sn2_regression_dataset.csv')
# new_rows = []
# for idx, row in data.iterrows():
#     if len(ast.literal_eval(row['reaction_core'])[0]) == 3:
#         similar = data[data['smiles'] == row['smiles']]
#         if len(similar) > 1:
#             for idx in range(len(similar)):
#                 if len(ast.literal_eval(similar.iloc[idx]['reaction_core'])[0]) == 4:
#                     reaction_core = similar.iloc[idx]['reaction_core']
#                     break
#             _row = row.copy()
#             _row['reaction_core'] = reaction_core
#             _row = _row.tolist()
#             _row.insert(-1, 'sn2')
#             new_rows.append(_row)
#     else:
#         _row = row.tolist()
#         _row.insert(-1, 'e2')
#         new_rows.append(_row)

# new_data = pd.DataFrame(new_rows, columns=data.columns.insert(-1, 'reaction_type'))
# new_data.to_csv('./data/regression_dataset.csv')