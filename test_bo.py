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

class XtbSimulationEstimator(BaseEstimator):

    def __init__(
        self,
        rc_sn2_d_nuc: float = 1.0,
        rc_sn2_d_leav: float = 1.0,
        rc_sn2_angle: int = 180,
        ts_sn2_d_nuc: float = 1.0,
        ts_sn2_d_leav: float = 1.0,
        ts_sn2_angle: int = 180,
        rc_e2_d_nuc: float = 1.0,
        rc_e2_d_leav: float = 1.0,
        rc_e2_d_H: float = 1.22,
        ts_e2_d_nuc: float = 1.0,
        ts_e2_d_leav: float = 1.0,
        ts_e2_d_H: float = 1.22,
    ) -> None:
        self.rc_sn2_d_nuc = rc_sn2_d_nuc
        self.rc_sn2_d_leav = rc_sn2_d_leav
        self.rc_sn2_angle = rc_sn2_angle
        self.ts_sn2_d_nuc = ts_sn2_d_nuc
        self.ts_sn2_d_leav = ts_sn2_d_leav
        self.ts_sn2_angle = ts_sn2_angle

        self.rc_e2_d_nuc = rc_e2_d_nuc
        self.rc_e2_d_leav = rc_e2_d_leav
        self.rc_e2_d_H = rc_e2_d_H
        self.ts_e2_d_nuc = ts_e2_d_nuc
        self.ts_e2_d_leav= ts_e2_d_leav
        self.ts_e2_d_H = ts_e2_d_H
        
    def fit(self, X, Y, **kwargs) -> None:
        return self

    def predict(self, X) -> None:
        rc_templates, ts_templates = construct_templates(
            rc_args=(self.rc_sn2_d_nuc, self.rc_sn2_d_leav, self.rc_sn2_angle, self.rc_e2_d_nuc, self.rc_e2_d_leav, self.rc_e2_d_H),
            ts_args=(self.ts_sn2_d_nuc, self.ts_sn2_d_leav, self.ts_sn2_angle, self.ts_e2_d_nuc, self.ts_e2_d_leav, self.ts_e2_d_H)
        )

        arguments = []
        for (sub_smiles, nuc_smiles, idxs) in X:
            arguments.append((sub_smiles, nuc_smiles, idxs, rc_templates, ts_templates))

        with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
            results = list(tqdm(executor.map(get_reaction_barriers, arguments), total=len(arguments)))

        sn2_energies, e2_energies = [], []
        for energies in results:
            sn2_energies.append(np.min(energies[:, 0]))
            e2_energies.append(np.min(energies[:, 1]))

        preds = []
        for idx in range(len(sn2_energies)):
            if e2_energies[idx] < sn2_energies[idx]:
                pred = 0 # e2
            else:
                pred = 1 # sn2
            preds.append(pred)       

        return preds

if __name__ == "__main__":
    class_data = pd.read_csv('./data/e2_sn2_classification_dataset.csv')

    X = [(row['smiles'].split('.')[0], row['smiles'].split('.')[1], ast.literal_eval(row['products_run'])) for _, row in class_data.iterrows()]
    Y = [int(len(ast.literal_eval(row['products_run'])[0]) == 3) for _, row in class_data.iterrows()]

    def scoring_fn(estimator, Xtest, Ytest):
        preds = estimator.predict(X)
        return roc_auc_score(Y, preds)

    opt = BayesSearchCV(
        XtbSimulationEstimator(),
        {
            'rc_sn2_d_nuc': Real(0.5, 2.2, prior='uniform'),
            'rc_sn2_d_leav': Real(0.5, 2.2, prior='uniform'),
            # 'rc_sn2_angle': Integer(120, 240, prior='uniform'),
            'ts_sn2_d_nuc': Real(0.5, 2.2, prior='uniform'),
            'ts_sn2_d_leav': Real(0.5, 2.2, prior='uniform'),
            # 'ts_sn2_angle': Integer(120, 240, prior='uniform'),
            'rc_e2_d_nuc': Real(0.5, 2.2, prior='uniform'),
            'rc_e2_d_leav': Real(0.5, 2.2, prior='uniform'),
            'rc_e2_d_H': Real(0.5, 2.2, prior='uniform'),
            'ts_e2_d_nuc': Real(0.5, 2.2, prior='uniform'),
            'ts_e2_d_leav': Real(0.5, 2.2, prior='uniform'),
            'ts_e2_d_H': Real(0.5, 2.2, prior='uniform'),
        },
        n_iter=12,
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