from typing import Dict, List, Literal, Tuple
import time
import pandas as pd
import ast
from rdkit import Chem
import numpy as np
import json
import pickle
import functools
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax
import jax.numpy as jnp
import optax

from huxel.huxel.beta_functions import _beta_r_linear
from huxel.huxel.huckel import f_energy
from huxel.huxel.molecule import myMolecule
from huxel.huxel.utils import get_default_params

DATASET_PATH = './data/datasets/sn2_dataset.csv'
XYZ_DATASET_PATH = './data/sn2_dataset.json'

def json_to_xyz(geom):
    return np.array([[atom[1], atom[2], atom[3]] for atom in geom])

def prepare_dataset(
    nucleophile: Literal["H+", "F-", "Cl-", "Br-"],
    batch_size: int
):
    with open(XYZ_DATASET_PATH) as json_file:
        xyz_dataset = json.load(json_file)

    reaction_labels = []
    smiles_list = []
    reaction_barriers_list = []
    
    df = pd.read_csv(DATASET_PATH)
    for _, row in df.iterrows():
        if row['substrates'].split('.')[-1] == f'[{nucleophile}]':
            reaction_labels.append(row['reaction_labels'])
            smiles_list.append(row['substrates'])
            try:
                barriers = ast.literal_eval(row["conformer_energies"])
            except:
                barriers = row['conformer_energies'].replace('[', '').replace(']', '').replace('\n', '')
                barriers = barriers.split(' ')
                barriers = list(filter(lambda x: len(x) > 2, barriers))
                barriers = [float(e) for e in barriers] 
            reaction_barriers_list.append(barriers)
    
    data = []
    for reaction_label, smiles, reaction_barriers in zip(
        reaction_labels, smiles_list, reaction_barriers_list
    ):
        if len(reaction_barriers) == len(xyz_dataset[reaction_label]['rc_conformers']):
            for conf_idx in range(len(reaction_barriers)):
                rdkit_mol = Chem.MolFromSmiles(smiles)
                reac = myMolecule(
                    id0=0,
                    smiles=smiles,
                    atom_types=[atom.GetSymbol() for atom in rdkit_mol.GetAtoms()],
                    connectivity_matrix=Chem.GetAdjacencyMatrix(rdkit_mol),
                    xyz=json_to_xyz(xyz_dataset[reaction_label]['rc_conformers'][conf_idx]),
                    homo_lumo_gap_ref=reaction_barriers[conf_idx]
                )
                reac.get_dm()
                ts = myMolecule(
                    id0=0,
                    smiles=smiles,
                    atom_types=[atom.GetSymbol() for atom in rdkit_mol.GetAtoms()],
                    connectivity_matrix=Chem.GetAdjacencyMatrix(rdkit_mol),
                    xyz=json_to_xyz(xyz_dataset[reaction_label]['ts'])
                )
                ts.get_dm()
                data.append((reac, ts))

    batches = []
    for index in np.arange(0, len(data), batch_size):
        batches.append(data[index:(index+batch_size)])

    return batches

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)

    batches = prepare_dataset('H+', 8)[:1]
    save_name = 'test'

    sys.stdout = open("stdout5.txt", "w", buffering=1)

    lr = 1e-2
    n_epochs = 150

    def loss_fn(
        params: Dict, 
        batch: List[Tuple[myMolecule]]
    ) -> float:
        f_beta = _beta_r_linear
        ref_barriers = []
        pred_barriers = []
        for datapoint in batch:
            reac, ts = datapoint
            ref_barriers.append(reac.homo_lumo_gap_ref)
            pred_barriers.append(
                f_energy(params, ts, f_beta) - f_energy(params, reac, f_beta)
            )
        return jnp.mean((jnp.array(ref_barriers) - jnp.array(pred_barriers))**2)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=False)

    params_init = get_default_params()
    params_init = {'h_x': params_init['h_x'], 'h_xy': params_init['h_xy'], 'r_xy': params_init['r_xy']}
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params_init)
    params = jax.tree_map(lambda x: jax.lax.convert_element_type(x, jnp.float32), params_init)

    @functools.partial(jax.jit, static_argnums=(2))
    def train_step(params, optimizer_state, batch):
        loss, grads = grad_fn(params, batch)
        updates, opt_state = optimizer.update(grads[0], optimizer_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    
    for epoch in range(n_epochs + 1):
        start_time_epoch = time.time()
        loss_tr_epoch = []
        for batch in batches:
            params, opt_state, loss_tr = train_step(params, opt_state, batch)
            loss_tr_epoch.append(loss_tr)

        loss_tr_mean = jnp.mean(jnp.asarray(loss_tr_epoch).ravel())

        time_epoch = time.time() - start_time_epoch
        print(epoch, loss_tr_mean, time_epoch)

    # with open(save_name, 'wb') as file:
    #     pickle.dump(params, file)
    