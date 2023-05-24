"""
Script to parse the EAS dataset
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def parse_eas_reactions(data):
    rxns = [
        AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]>>[C;R:1](Br)=[C,N;R;H1:2]'),
        AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]>>[C,R:1](Br)=[C,N;R;H0:2]')
    ]

    reaction_idxs = []
    substrates = []
    reaction_products = []
    label = []
    simulation_idx = []

    idx = 0
    for _, row in data.iterrows():
        substrate_smiles = row[0]
        pos_reacting_sites = [int(idx) for idx in row[1].split(',') if len(idx) > 0]

        # create mol
        mol = Chem.MolFromSmiles(substrate_smiles)
        Chem.Kekulize(mol, clearAromaticFlags=True)

        # check all possible reactions
        for rxn in rxns:
            products = rxn.RunReactants((mol,))
            for product in products:
                # sanitize mol
                product_mol = product[0]
                Chem.SanitizeMol(product_mol)

                # retrieve reacting site
                reacting_site_idx = None
                reactant_match = product_mol.GetSubstructMatches(Chem.MolFromSmiles(substrate_smiles))
                for _, atom in enumerate(product_mol.GetAtoms()):
                    if atom.GetIdx() not in reactant_match[0]: 
                        aromatic_site_idx = atom.GetNeighbors()[0].GetIdx()
                        reacting_site_idx = list(reactant_match[0]).index(aromatic_site_idx)
                        break

                # append reaction to dataset
                if reacting_site_idx is not None:
                    reaction_idxs.append(idx)
                    substrates.append(f'{substrate_smiles}.[Br+]')
                    reaction_products.append(Chem.MolToSmiles(product_mol))
                    label.append(int(reacting_site_idx in pos_reacting_sites))
                    simulation_idx.append(0) 
                    idx += 1

    return pd.DataFrame.from_dict({
        'uid': reaction_idxs,
        'reaction_idx': reaction_idxs,
        'substrates': substrates,
        'products': reaction_products,
        'label': label,
        'simulation_idx': simulation_idx
    })


if __name__ == "__main__":
    data = pd.read_csv('./data/eas_compound_smiles.csv', sep=' ', header=None)
    df = parse_eas_reactions(data)
    df.to_csv('./data/datasets/eas_dataset_2.csv')