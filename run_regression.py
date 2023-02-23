import pandas as pd
import ast
import os
from src.reactions.e2_sn2.old.template import E2ReactionTemplate, Sn2ReactionTemplate
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from run_classification import get_reaction_barriers


BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 90
N_CPUS_CONFORMERS = 4

HARTREE_TO_KCAL = 627.5

os.environ["BASE_DIR"] = BASE_DIR
os.environ["XTB_PATH"] = XTB_PATH

def get_templates_sn2(
    rc_sn2_d_nuc: float = 2.0,
    rc_sn2_d_leav: float = 1.5,
    ts_sn2_d_nuc: float = 2.0,
    ts_sn2_d_leav: float = 1.5
):
    rc_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=rc_sn2_d_nuc,
            d_leaving_group=rc_sn2_d_leav,
            angle=180
        )
    ]

    ts_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=ts_sn2_d_nuc,
            d_leaving_group=ts_sn2_d_leav,
            angle=180
        )
    ]

    return rc_templates, ts_templates

def get_templates_e2(
    rc_e2_d_nuc: float = 1.2,
    rc_e2_d_leav: float = 1.5,
    ts_e2_d_nuc: float = 1.2,
    ts_e2_d_leav: float = 1.5
):
    rc_templates = [
        E2ReactionTemplate(
            d_nucleophile=rc_e2_d_nuc,
            d_leaving_group=rc_e2_d_leav,
            d_H=1.22
        )
    ]

    ts_templates = [
        E2ReactionTemplate(
            d_nucleophile=ts_e2_d_nuc,
            d_leaving_group=ts_e2_d_leav,
            d_H=1.22
        )
    ]

    return rc_templates, ts_templates

def compute_sn2_correlation(
    rc_sn2_d_nuc: float = 2.0,
    rc_sn2_d_leav: float = 1.5,
    ts_sn2_d_nuc: float = 2.0,
    ts_sn2_d_leav: float = 1.5,
):
    reg_data = pd.read_csv('./data/regression_dataset.csv')

    arguments = []
    labels = []

    for idx, (_, row) in enumerate(reg_data.iterrows()):
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['reaction_core'])
        label = float(row['activation_energy'])
        
        reaction_type = str(row['reaction_type'])
        if reaction_type == 'sn2':
            rc_templates, ts_templates = get_templates_sn2(
                rc_sn2_d_nuc=rc_sn2_d_nuc,
                rc_sn2_d_leav=rc_sn2_d_leav,
                ts_sn2_d_nuc=ts_sn2_d_nuc,
                ts_sn2_d_leav=ts_sn2_d_leav,
            )

            arguments.append((substrate_smiles, nucleophile_smiles, indices, rc_templates, ts_templates))
            labels.append(label)

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

    preds = [np.min(energies[:, 0]) * HARTREE_TO_KCAL for energies in results]

    sn2_preds, sn2_targets = [], []
    for idx in range(len(preds)):
        if preds[idx] < 1e5:
            sn2_preds.append(preds[idx])
            sn2_targets.append(labels[idx])

    print('Sn2 MSE: ', mean_squared_error(sn2_targets, sn2_preds), \
        'Corr: ', np.corrcoef(np.array(sn2_targets), np.array(sn2_preds)))

    df = pd.DataFrame(np.stack([
        np.array(sn2_targets),
        np.array(sn2_preds)
    ], axis=1), columns=['targets', 'preds'])
    sns.lmplot(x='targets', y='preds', data=df)
    plt.savefig('sn2_plot.png')

def compute_e2_correlation(
    rc_e2_d_nuc: float = 1.2,
    rc_e2_d_leav: float = 1.5,
    ts_e2_d_nuc: float = 1.2,
    ts_e2_d_leav: float = 1.5,
):
    reg_data = pd.read_csv('./data/regression_dataset.csv')

    arguments = []
    labels = []

    for idx, (_, row) in enumerate(reg_data.iterrows()):
        substrate_smiles = row['smiles'].split('.')[0]
        nucleophile_smiles = row['smiles'].split('.')[1]
        indices = ast.literal_eval(row['reaction_core'])
        label = float(row['activation_energy'])
        
        reaction_type = str(row['reaction_type'])
        if reaction_type == 'sn2':
            rc_templates, ts_templates = get_templates_e2(
                rc_e2_d_nuc=rc_e2_d_nuc,
                rc_e2_d_leav=rc_e2_d_leav,
                ts_e2_d_nuc=ts_e2_d_nuc,
                ts_e2_d_leav=ts_e2_d_leav,
            )

            arguments.append((substrate_smiles, nucleophile_smiles, indices, rc_templates, ts_templates))
            labels.append(label)

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

    preds = [np.min(energies[:, 0]) * HARTREE_TO_KCAL for energies in results]

    e2_preds, e2_targets = [], []
    for idx in range(len(preds)):
        if preds[idx] < 1e5:
            e2_preds.append(preds[idx])
            e2_targets.append(labels[idx])

    print('E2 MSE: ', mean_squared_error(e2_targets, e2_preds), \
        'Corr: ', np.corrcoef(np.array(e2_targets), np.array(e2_preds)))

if __name__ == "__main__":
    # rc_sn2_d_nucs = [1.5]
    # rc_sn2_d_leavs = [1.2]
    # ts_sn2_d_nucs = [1.5]
    # ts_sn2_d_leavs = [1.8]

    # for rc_sn2_d_nuc in rc_sn2_d_nucs:
    #     for rc_sn2_d_leav in rc_sn2_d_leavs:
    #         for ts_sn2_d_nuc in ts_sn2_d_nucs:
    #             for ts_sn2_d_leav in ts_sn2_d_leavs:
    #                 print('\n\n\n')
    #                 compute_sn2_correlation(
    #                     rc_sn2_d_nuc=rc_sn2_d_nuc,
    #                     rc_sn2_d_leav=rc_sn2_d_leav,
    #                     ts_sn2_d_nuc=ts_sn2_d_nuc,
    #                     ts_sn2_d_leav=ts_sn2_d_leav,
    #                 )
    #                 print(f'rc_sn2_d_nuc {rc_sn2_d_nuc} rc_sn2_d_leav {rc_sn2_d_leav} \
    #                      ts_sn2_d_nuc {ts_sn2_d_nuc} ts_sn2_d_leav {ts_sn2_d_leav}')

    rc_e2_d_nucs = [3]
    rc_e2_d_leavs = [1.2]
    ts_e2_d_nucs = [1.5]
    ts_e2_d_leavs = [1.8]
    # rc_e2_d_nucs = [1.5, 1,3]
    # rc_e2_d_leavs = [1.2, 1.4]
    # ts_e2_d_nucs = [1.5, 1.2]
    # ts_e2_d_leavs = [1.4, 1.8]

    for rc_e2_d_nuc in rc_e2_d_nucs:
        for rc_e2_d_leav in rc_e2_d_leavs:
            for ts_e2_d_nuc in ts_e2_d_nucs:
                for ts_e2_d_leav in ts_e2_d_leavs:
                    print('\n\n\n')
                    compute_e2_correlation(
                        rc_e2_d_nuc=rc_e2_d_nuc,
                        rc_e2_d_leav=rc_e2_d_leav,
                        ts_e2_d_nuc=ts_e2_d_nuc,
                        ts_e2_d_leav=ts_e2_d_leav,
                    )
                    print(f'rc_e2_d_nuc {rc_e2_d_nuc} rc_e2_d_leav {rc_e2_d_leav} \
                         ts_e2_d_nuc {ts_e2_d_nuc} ts_e2_d_leav {ts_e2_d_leav}')







# df = pd.DataFrame(np.stack([
#     np.array(sn2_targets),
#     np.array(sn2_preds)
# ], axis=1), columns=['targets', 'preds'])
# sns.lmplot(x='targets', y='preds', data=df)
# plt.savefig('sn2.png')

# df = pd.DataFrame(np.stack([
#     np.array(e2_targets),
#     np.array(e2_preds)
# ], axis=1), columns=['targets', 'preds'])
# sns.lmplot(x='targets', y='preds', data=df)
# plt.savefig('e2.png')



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