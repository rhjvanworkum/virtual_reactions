import pandas as pd
import numpy as np

def construct_new_df(
    new_df: pd.DataFrame,
    source_df: pd.DataFrame,
):
    reaction_idxs = []
    substrates_list, products_list, reaction_smiles_list = [], [], []
    labels, simulation_idxs = [], []
    for _, row in new_df.iterrows():
        if row['reaction_smiles'] in source_df['reaction_smiles'].values:
            reaction_idx = source_df[source_df['reaction_smiles'] == row['reaction_smiles']]['reaction_idx'].values[0]
        else:
            if len(reaction_idxs) == 0:
                reaction_idx = max(source_df['reaction_idx'].values) + 1
            else:
                reaction_idx = max(reaction_idxs) + 1

        reaction_smiles = row['reaction_smiles']
        substrates, products = reaction_smiles.split('>>')
        label = row['label']
        simulation_idx = 1

        reaction_idxs.append(reaction_idx)
        substrates_list.append(substrates)
        products_list.append(products)
        reaction_smiles_list.append(reaction_smiles)
        labels.append(label)
        simulation_idxs.append(simulation_idx)
    
    simulated_df = pd.DataFrame({
        'reaction_idx': reaction_idxs,
        'uid': np.arange(max(source_df['uid'].values) + 1, max(source_df['uid'].values) + 1 + len(reaction_idxs)),
        'substrates': substrates_list,
        'products': products_list,
        'reaction_smiles': reaction_smiles_list,
        'label': labels,
        'simulation_idx': simulation_idxs
    })

    return pd.concat([source_df, simulated_df])

def construct_df(data: pd.DataFrame):
    reaction_idxs = {}
    reaction_idxs_list = []
    substrates_list, products_list, reaction_smiles_list = [], [], []
    labels, simulation_idxs = [], []
    i = 0
    for _, row in data.iterrows():
        reaction_smiles = row['reaction_smiles']
        substrates, products = reaction_smiles.split('>>')
        try:
            label = row['label']
        except:
            label = row['energies']
        simulation_idx = 1

        if reaction_smiles not in reaction_idxs.keys():
            reaction_idx = i
            reaction_idxs[reaction_smiles] = reaction_idx
            i += 1
        else:
            reaction_idx = reaction_idxs[reaction_smiles]

        reaction_idxs_list.append(reaction_idx)
        substrates_list.append(substrates)
        products_list.append(products)
        reaction_smiles_list.append(reaction_smiles)
        labels.append(label)
        simulation_idxs.append(simulation_idx)
    
    simulated_df = pd.DataFrame({
        'reaction_idx': reaction_idxs_list,
        'uid': np.arange(len(substrates_list)),
        'substrates': substrates_list,
        'products': products_list,
        'reaction_smiles': reaction_smiles_list,
        'label': labels,
        'simulation_idx': simulation_idxs
    })

    return simulated_df

if __name__ == "__main__":
    # add reaction core dataset
    dataset_name = 'DA_literature_reaction_core'

    source_data = pd.read_csv('./data/da/DA_literature.csv')    
    new_data = pd.read_csv('./data/da/diels_alder_reaction_cores_dataset_classification.csv')
    new_dataframe = construct_new_df(new_data, source_data)
    new_dataframe.to_csv(f'./data/da/{dataset_name}.csv')
    
    new_dataframe = construct_df(
        pd.read_csv('./data/da/diels_alder_reaction_cores_dataset_classification.csv')
    )
    new_dataframe.to_csv(f'./data/da/reaction_cores_class.csv')
    new_dataframe = construct_df(
        pd.read_csv('./data/da/diels_alder_reaction_cores_dataset_activation_energies.csv')
    )
    new_dataframe.to_csv(f'./data/da/reaction_cores.csv')


    # add reaction rps dataset
    dataset_name = 'DA_literature_reaxys_rps'

    source_data = pd.read_csv('./data/da/DA_literature.csv')    
    new_data = pd.read_csv('./data/da/diels_alder_reaxys_rps_dataset_classification.csv')
    new_dataframe = construct_new_df(new_data, source_data)
    new_dataframe.to_csv(f'./data/da/{dataset_name}.csv')

    new_dataframe = construct_df(
        pd.read_csv('./data/da/diels_alder_reaxys_rps_dataset_classification.csv')
    )
    new_dataframe.to_csv(f'./data/da/reaxys_rps_class.csv')
    new_dataframe = construct_df(
        pd.read_csv('./data/da/diels_alder_reaxys_rps_dataset_activation_energies.csv')
    )
    new_dataframe.to_csv(f'./data/da/reaxys_rps.csv')


    # add reaction core + rps dataset
    dataset_name = 'DA_literature_core_reaxys'

    source_data = pd.read_csv('./data/da/DA_literature.csv')    
    new_data = pd.concat([
        pd.read_csv('./data/da/diels_alder_reaxys_rps_dataset_classification.csv'),
        pd.read_csv('./data/da/diels_alder_reaction_cores_dataset_classification.csv')
    ])
    new_dataframe = construct_new_df(new_data, source_data)
    new_dataframe.to_csv(f'./data/da/{dataset_name}.csv')