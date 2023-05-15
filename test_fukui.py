import pandas as pd
import rdkit
from rdkit import Chem
from src.compound import Compound

da_mol_pattern = Chem.MolFromSmiles("C1=CCCCC1")
diene_mol_pattern = Chem.MolFromSmarts("[C,c][C,c]=[C,c][C,c]")
dienophile_mol_pattern = Chem.MolFromSmarts("[C,c][C,c]")


# df = pd.read_csv('./data/datasets/da/da_no_solvent_dataset.csv')
# compound = Compound.from_smiles(df['products'].values[1])

mol = Chem.MolFromSmiles('[H]C(=O)C1([H])C([H])([H])c2oc(=O)n(-c3c([H])c([H])c(C([H])([H])[H])c([H])c3[H])c2C([H])([H])C1([H])[H]')


# da_atoms = compound.rdkit_mol.GetSubstructMatch(da_mol_pattern)

# diene_matches = compound.rdkit_mol.GetSubstructMatches(diene_mol_pattern)
# for match in diene_matches:
#     if set(match).issubset(set(da_atoms)):
#         diene_atoms = match
#         break

# dienophile_matches = compound.rdkit_mol.GetSubstructMatches(dienophile_mol_pattern)
# for match in dienophile_matches:
#     if set(match).issubset(set(da_atoms)) and not set(match).intersection(set(diene_atoms)):
#         dienophile_atoms = match
#         break

# print(diene_atoms, dienophile_atoms)