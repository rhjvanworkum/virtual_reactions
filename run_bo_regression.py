from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
import os

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

from skopt import BayesSearchCV
from skopt.space import Real

from reactions.e2_sn2.e2_sn2_reaction import E2Sn2Reaction
from reactions.e2_sn2.template import E2ReactionTemplate, Sn2ReactionTemplate
from run_regression import HARTREE_TO_KCAL

BASE_DIR = '/home/rhjvanworkum/virtual_reactions/calculations/'
XTB_PATH = "/home/rhjvanworkum/xtb-6.5.1/bin/xtb"
N_CPUS = 100
N_CPUS_CONFORMERS = 4

os.environ["BASE_DIR"] = BASE_DIR
os.environ["XTB_PATH"] = XTB_PATH

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

def construct_templates(
    rc_args,
    ts_args
):

    sn2_d_nuc, sn2_d_leav, sn2_angle, e2_d_nuc, e2_d_leav, e2_d_H = rc_args

    rc_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=sn2_d_nuc,
            d_leaving_group=sn2_d_leav,
            angle=sn2_angle
        ),
        E2ReactionTemplate(
            d_nucleophile=e2_d_nuc,
            d_leaving_group=e2_d_leav,
            d_H=e2_d_H
        )
    ]

    sn2_d_nuc, sn2_d_leav, sn2_angle, e2_d_nuc, e2_d_leav, e2_d_H = ts_args

    ts_templates = [
        Sn2ReactionTemplate(
            d_nucleophile=sn2_d_nuc,
            d_leaving_group=sn2_d_leav,
            angle=sn2_angle
        ),
        E2ReactionTemplate(
            d_nucleophile=e2_d_nuc,
            d_leaving_group=e2_d_leav,
            d_H=e2_d_H
        )
    ]

    return rc_templates, ts_templates

def get_templates_e2(
    rc_e2_d_nuc: float = 1.2,
    rc_e2_d_leav: float = 1.5,
    rc_e2_d_H: float = 1.22,
    ts_e2_d_nuc: float = 1.2,
    ts_e2_d_leav: float = 1.5,
    ts_e2_d_H: float = 1.22,
):
    rc_templates = [
        E2ReactionTemplate(
            d_nucleophile=rc_e2_d_nuc,
            d_leaving_group=rc_e2_d_leav,
            d_H=rc_e2_d_H
        )
    ]

    ts_templates = [
        E2ReactionTemplate(
            d_nucleophile=ts_e2_d_nuc,
            d_leaving_group=ts_e2_d_leav,
            d_H=ts_e2_d_H
        )
    ]

    return rc_templates, ts_templates

class XtbSimulationEstimator(BaseEstimator):

    def __init__(
        self,
        # rc_sn2_d_nuc: float = 1.0,
        # rc_sn2_d_leav: float = 1.0,
        # rc_sn2_angle: int = 180,
        # ts_sn2_d_nuc: float = 1.0,
        # ts_sn2_d_leav: float = 1.0,
        # ts_sn2_angle: int = 180,
        rc_e2_d_nuc: float = 1.0,
        rc_e2_d_leav: float = 1.0,
        rc_e2_d_H: float = 1.22,
        ts_e2_d_nuc: float = 1.0,
        ts_e2_d_leav: float = 1.0,
        ts_e2_d_H: float = 1.22,
    ) -> None:
        # self.rc_sn2_d_nuc = rc_sn2_d_nuc
        # self.rc_sn2_d_leav = rc_sn2_d_leav
        # self.rc_sn2_angle = rc_sn2_angle
        # self.ts_sn2_d_nuc = ts_sn2_d_nuc
        # self.ts_sn2_d_leav = ts_sn2_d_leav
        # self.ts_sn2_angle = ts_sn2_angle

        self.rc_e2_d_nuc = rc_e2_d_nuc
        self.rc_e2_d_leav = rc_e2_d_leav
        self.rc_e2_d_H = rc_e2_d_H
        self.ts_e2_d_nuc = ts_e2_d_nuc
        self.ts_e2_d_leav= ts_e2_d_leav
        self.ts_e2_d_H = ts_e2_d_H
        
    def fit(self, X, Y, **kwargs) -> None:
        return self

    def predict(self, X) -> None:
        rc_templates, ts_templates = get_templates_e2(
            rc_e2_d_nuc=self.rc_e2_d_nuc,
            rc_e2_d_leav=self.rc_e2_d_leav,
            rc_e2_d_H=self.rc_e2_d_H,
            ts_e2_d_nuc=self.ts_e2_d_nuc,
            ts_e2_d_leav=self.ts_e2_d_leav,
            ts_e2_d_H=self.ts_e2_d_H
        )

        arguments = []
        for (sub_smiles, nuc_smiles, idxs) in X:
            arguments.append((sub_smiles, nuc_smiles, idxs, rc_templates, ts_templates))

        with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
            results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

        preds = [np.min(energies[:, 0]) * HARTREE_TO_KCAL for energies in results]

        return preds

if __name__ == "__main__":
    reg_data = pd.read_csv('./data/regression_dataset.csv')
    reg_data = reg_data[reg_data['reaction_type'] == 'e2']

    X = [(row['smiles'].split('.')[0], row['smiles'].split('.')[1], ast.literal_eval(row['reaction_core'])) for _, row in reg_data.iterrows()]
    Y = [float(row['activation_energy']) for _, row in reg_data.iterrows()]

    def scoring_fn(estimator, Xtest, Ytest):
        _preds = estimator.predict(X)

        true, preds = [], []
        for t, pred in zip(Y, _preds):
            if pred < 1e5:
                true.append(t)
                preds.append(pred)
        corr = np.corrcoef(np.array(true).flatten(), np.array(preds).flatten())
        return corr[0, 1]

    # n_iter is total: iter * n_points
    opt = BayesSearchCV(
        XtbSimulationEstimator(),
        {
            'rc_e2_d_nuc': Real(1.0, 2.0, prior='uniform'),
            'rc_e2_d_leav': Real(1.0, 2.0, prior='uniform'),
            'rc_e2_d_H': Real(1.0, 2.0, prior='uniform'),
            'ts_e2_d_nuc': Real(1.0, 2.0, prior='uniform'),
            'ts_e2_d_leav': Real(1.0, 2.0, prior='uniform'),
            'ts_e2_d_H': Real(1.0, 2.0, prior='uniform'),
        },
        n_iter=100,
        n_points=5,
        # pre_dispatch=1,
        n_jobs=1,
        cv=[(slice(None), slice(None))],
        verbose=1,
        scoring=scoring_fn
    )

    def on_step(optim_result):
        print(f'Iter done\n')

    opt.fit(X, Y, callback=[on_step])
    print(opt.best_score_)
    print(opt.best_params_)