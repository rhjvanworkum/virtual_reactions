import os
from typing import Any, List, Literal, Optional, Union
import pandas as pd
from rxnmapper import RXNMapper
import numpy as np
import ast
from tqdm import tqdm

import rdkit
from rdkit import Chem
import ast



SIMULATION_IDX_ATOM = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']

class Dataset:

    def __init__(
        self,
        csv_file_path: str,
    ) -> None:
        self.csv_file_path = os.path.join(os.environ['BASE_PATH'], f'data/datasets/{csv_file_path}')
        if os.path.exists(self.csv_file_path):
            self.generated = True
        else:
            self.generated = False

    @property
    def chemprop_csv_file_path(self) -> str:
        return self.csv_file_path.split('.')[0] + '_chemprop.csv'
    
    @property
    def chemprop_feature_file_path(self) -> str:
        return self.csv_file_path.split('.')[0] + '_chemprop_feat.csv'

    def load(
        self,
        uids: Optional[List[str]] = None,
        mode: Literal["regression", "classification"] = "classification",
    ) -> pd.DataFrame:
        if self.generated:
            df = pd.read_csv(self.csv_file_path)
            if uids is not None:
                return df[df['uid'].isin(uids)]
            else:
                return df
        else:
            raise ValueError("Dataset is not generated yet!")

    def load_chemprop_dataset(
        self,
        uids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if os.path.exists(self.chemprop_csv_file_path):
            df = pd.read_csv(self.chemprop_csv_file_path)
            if uids is not None:
                return df[df['uid'].isin(uids)]
            else:
                return df
        else:
            raise ValueError("Chemprop Dataset is not generated yet!")

    def load_chemprop_features(
        self,
        uids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if os.path.exists(self.chemprop_feature_file_path) and  os.path.exists(self.chemprop_csv_file_path):
            df = pd.read_csv(self.chemprop_csv_file_path)
            if uids is not None:
                smiles = df[df['uid'].isin(uids)]['smiles']
            else:
                smiles = df['smiles']

            feature_df = pd.read_csv(self.chemprop_feature_file_path)
            descriptors = [
                np.array(ast.literal_eval(
                    feature_df[feature_df['smiles'] == smi]['descriptors'].values[0]
                )) for smi in smiles
            ]
            return descriptors
        else:
            raise ValueError("Chemprop Feature Dataset is not generated yet!")

    def generate_chemprop_dataset(
        self,
        force: bool = False,
        simulation_idx_as_features: bool = False
    ) -> None:
        if not os.path.exists(self.chemprop_csv_file_path) or force:
            dataframe = self.load()
            rxn_mapper = RXNMapper()

            # parse reaction smiles + simulation idx
            smiles = []
            simulation_idxs = []
            for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Mapping rxn smiles.."):
                if simulation_idx_as_features:
                    simulation_idxs.append(row["simulation_idx"])
                    reaction_smiles = [f'{row["substrates"]}>>{row["products"]}']
                else:
                    reaction_smiles = [f'{row["substrates"]}.[{SIMULATION_IDX_ATOM[row["simulation_idx"]]}]>>{row["products"]}']
                
                atom_mapped_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps(reaction_smiles)[0]['mapped_rxn']
                smiles.append(atom_mapped_reaction_smiles)

            dataframe['smiles'] = smiles

            # export dataframe
            dataframe.to_csv(self.chemprop_csv_file_path)

            if simulation_idx_as_features:
                feat_dataframe = pd.DataFrame()
                feat_dataframe['smiles'] = smiles
                descriptors = []
                for idx, smiles in enumerate(smiles):
                    smi_reac, smi_prod = smiles.split('>>')
                    mol_reac, mol_prod = Chem.MolFromSmiles(smi_reac), Chem.MolFromSmiles(smi_prod)
                    n_atoms = max(mol_reac.GetNumAtoms(), mol_prod.GetNumAtoms())
                    descriptors.append([simulation_idxs[idx] for _ in range(n_atoms)])
                feat_dataframe['descriptors'] = descriptors
                feat_dataframe.to_csv(self.chemprop_feature_file_path)
        else:
            pass

class SimulatedDataset(Dataset):

    def __init__(
        self, 
        csv_file_path: str,
        n_simulations: int = 1,
        n_substrates: int = 1
    ) -> None:
        super().__init__(csv_file_path=csv_file_path)
        self.n_simulations = n_simulations
        self.n_substrates = n_substrates

    def _simulate_reactions(
        self,
        substrates: Union[str, List[str]],
        products: Union[str, List[str]],
        solvents: Union[str, List[str]],
        simulation_idx: int,
        n_cpus: int
    ) -> List[Any]:
        """
        Function to simulate a set of reactions
        """
        raise NotImplementedError

    def _select_reaction_to_simulate(
        self,
        source_dataset: Dataset
    ) -> None:
        """
        Function to select a set of reactions to simulate
        """
        raise NotImplementedError

    def generate(
        self,
        source_dataset: Dataset,
        n_cpus: int,
    ) -> None:
        source_data = pd.read_csv(source_dataset.csv_file_path)
        substrates, products, solvents, compute_product_only_list, reaction_idxs = self._select_reaction_to_simulate(source_dataset)

        substrate_energies_list = {}

        uids = []
        df_reaction_idxs = []
        df_substrates = []
        df_products = []
        labels = []
        conformer_energies = []
        simulation_idxs = []        
        for simulation_idx in range(self.n_simulations):
        
            simulation_results = self._simulate_reactions(substrates, products, solvents, compute_product_only_list, simulation_idx, n_cpus)
            assert len(simulation_results) == len(substrates) == len(compute_product_only_list)

            for idx, result in enumerate(simulation_results):
                
                # # look up results for earlier calculated substrate
                # if compute_product_only_list[idx]:
                #     result = [
                #         substrate_energies_list[substrates[idx]],
                #         result[1]
                #     ]
                # else:
                #     substrate_energies_list[substrates[idx]] = result[0]
                
                # add other properties
                uids.append(max(source_data['uid'].values) + len(uids) + 1)
                df_reaction_idxs.append(reaction_idxs[idx])
                df_substrates.append(substrates[idx])
                df_products.append(products[idx])
                labels.append(None)
                conformer_energies.append(result)
                simulation_idxs.append(simulation_idx + 1)
                idx += 1

        df = pd.DataFrame.from_dict({
            'uid': uids,
            'reaction_idx': df_reaction_idxs,
            'substrates': df_substrates,
            'products': df_products,
            'label': labels,
            'conformer_energies': conformer_energies,
            'simulation_idx': simulation_idxs
        })

        df = pd.concat([source_data, df])
        df.to_csv(self.csv_file_path)

        self.generated = True

    def _load_classification_dataset(
        self,
        aggregation_mode: Literal["avg", "low"] = "low",
        margin: float = 0.0,
    ) -> List[pd.DataFrame]:
        dataframe = pd.read_csv(self.csv_file_path)
        source_dataframe = dataframe[dataframe['simulation_idx'] == 0]
        
        dataframes = [source_dataframe]
        for simulation_idx in range(self.n_simulations):
            virtual_dataframe = dataframe[dataframe['simulation_idx'] == simulation_idx + 1]

            # filter out reactions that "failed"
            virtual_dataframe = virtual_dataframe.dropna(subset=['conformer_energies'])

            # drop label column
            virtual_dataframe = virtual_dataframe.drop(columns=['label'])

            # aggregate across conformers
            barriers = []
            for _, row in virtual_dataframe.iterrows():
                if self.n_substrates == 1:
                    sub_energies, ts_energies = ast.literal_eval(row['conformer_energies'])
                    sub_energies = [e for e in sub_energies if e is not None]
                    ts_energies = [e for e in ts_energies if e is not None]

                    if aggregation_mode == 'avg':
                        barrier = np.mean(np.array(ts_energies)) - np.mean(np.array(sub_energies))
                    elif aggregation_mode == 'low':
                        if len(sub_energies) > 0 and len(ts_energies) > 0:
                            barrier = np.min(np.array(ts_energies)) - np.min(np.array(sub_energies))
                        else:
                            barrier = None
                    else:
                        raise ValueError("aggregation_mode {aggregation_mode} doesn't exists")
                elif self.n_substrates == 2:
                    sub1_energies, sub2_energies, ts_energies = ast.literal_eval(row['conformer_energies'])
                    sub1_energies = [e for e in sub1_energies if e is not None]
                    sub2_energies = [e for e in sub2_energies if e is not None]
                    ts_energies = [e for e in ts_energies if e is not None]

                    if aggregation_mode == 'avg':
                        barrier = np.mean(np.array(ts_energies)) - (np.mean(np.array(sub1_energies)) + np.mean(np.array(sub2_energies)))
                    elif aggregation_mode == 'low':
                        if len(sub1_energies) > 0 and len(sub2_energies) > 0 and len(ts_energies) > 0:
                            barrier = np.min(np.array(ts_energies)) - (np.min(np.array(sub2_energies)) + np.min(np.array(sub2_energies)))
                        else:
                            barrier = None
                    else:
                        raise ValueError("aggregation_mode {aggregation_mode} doesn't exists")
                barriers.append(barrier)
            virtual_dataframe['barrier'] = barriers

            # assing labels based on barriers across substrates
            labels = []
            for _, row in virtual_dataframe.iterrows():
                barrier = row['barrier']
                other_barriers = virtual_dataframe[virtual_dataframe['substrates'] == row['substrates']]['barrier']
                label = int((barrier - margin <= other_barriers).all())
                labels.append(label)
            virtual_dataframe['label'] = labels

            dataframes.append(virtual_dataframe)

        return dataframes

    def _load_regression_dataset(
        self,
        aggregation_mode: Literal["avg", "low"] = "low",
    ) -> List[pd.DataFrame]:
        # load dataframe
        source_dataframe = pd.read_csv(self.csv_file_path)

        dataframes = []
        for simulation_idx in range(self.n_simulations + 1):
            dataframe = source_dataframe[source_dataframe['simulation_idx'] == simulation_idx]
            # filter out reactions that "failed"
            dataframe = dataframe.dropna(subset=['conformer_energies'])
            # drop label column
            if 'label' in dataframe.columns:
                dataframe = dataframe.drop(columns=['label'])
            # aggregate across conformers
            barrier_list = []
            for _, row in dataframe.iterrows():
                try:
                    barriers = ast.literal_eval(row['conformer_energies'])
                except:
                    barriers = row['conformer_energies'].replace('[', '').replace(']', '').replace('\n', '')
                    barriers = barriers.split(' ')
                    barriers = list(filter(lambda x: len(x) > 2, barriers))
                    barriers = [float(e) for e in barriers]
                barriers = [e for e in barriers if e is not None]

                if aggregation_mode == 'avg':
                    barrier = np.mean(barriers)
                elif aggregation_mode == 'low':
                    if len(barriers) > 0:
                        barrier = np.min(barriers)
                    else:
                        barrier = None
                else:
                    raise ValueError("aggregation_mode {aggregation_mode} doesn't exists")
                barrier_list.append(barrier)

            dataframe['label'] = barrier_list
            dataframe = dataframe[~dataframe['label'].isnull()]
            dataframe = dataframe[~dataframe['label'].isna()]
            
            dataframes.append(dataframe)

        return dataframes

    def load(
        self,
        aggregation_mode: Literal["avg", "low"] = "low",
        mode: Literal["regression", "classification"] = "classification",
        margin: float = 0.0,
        uids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if self.generated:
            # load simulated dataframes
            if mode == "classification":
                dataframes = self._load_classification_dataset(
                    aggregation_mode,
                    margin
                )
            elif mode == "regression":
                dataframes = self._load_regression_dataset(
                    aggregation_mode
                )

            # drop conformer energy column & duplicates
            dataframe = pd.concat(dataframes)
            dataframe = dataframe.drop(columns=['conformer_energies'])
            dataframe = dataframe.drop_duplicates(subset=['uid'])

            # check for selection
            if uids is not None:
                return dataframe[dataframe['uid'].isin(uids)]
            else:
                return dataframe
        else:
            raise ValueError("Dataset is not generate yet!")