import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def remove_identical_atoms(rdkit_mol, name_list, smiles_list, atom_list):
    idx_list = []
    rank_kept = []
    atom_rank = list(Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False))
    for idx, atom in enumerate(atom_list):
        if atom_rank[atom] not in rank_kept:
            rank_kept.append(atom_rank[atom])
            idx_list.append(idx)
    
    name_list = np.array(name_list)[idx_list].tolist()
    smiles_list = np.array(smiles_list)[idx_list].tolist()
    atom_list = np.array(atom_list)[idx_list].tolist()
    
    return name_list, smiles_list, atom_list


def generate_protonated_smiles(pmol, name):

    # Save a copy of the input to find identical atoms in mol
    rdkit_mol = copy.deepcopy(pmol)

    # __rxn1__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]>>[CH2:1][*H+:2]')
    # __rxn2__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]>>[CH2:1][*+;H0:2]')

    # Bromine
    __rxn1__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]>>[CH:1](Br)[*H+:2]')
    __rxn2__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]>>[CH:1](Br)[*+;H0:2]')

    name_list = []
    smiles_list = []
    atom_list = []

    aromatic_ch = pmol.GetSubstructMatches(Chem.MolFromSmarts('[c;H1]'))
    aromatic_ch = [element for tupl in aromatic_ch for element in tupl]

    Chem.Kekulize(pmol,clearAromaticFlags=True)

    # target = Chem.MolFromSmarts('[C;H1:1]=[C,N;H1:2]')
    target = Chem.MolFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]')
    atoms = pmol.GetSubstructMatches(target)

    # Convert tuple of tuple to one-dimensional list
    atoms = [element for tupl in atoms for element in tupl]

    i = 0
    ps = __rxn1__.RunReactants((pmol,))
    for x in ps:
        smiles = Chem.MolToSmiles(x[0])
        smiles = smiles.replace("NH2+","N+")
        i += 1

        name_list.append(name+"+_"+str(i))
        smiles_list.append(smiles)
        atom_list.append(atoms[i-1])


    # target = Chem.MolFromSmarts('[C;H1:1]=[C,N;H0:2]')
    target = Chem.MolFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]')
    atoms = pmol.GetSubstructMatches(target)

    # Convert tuple of tuple to one-dimensional list
    atoms = [element for tupl in atoms for element in tupl]

    isav = i
    ps = __rxn2__.RunReactants((pmol,))
    for x in ps:
        smiles = Chem.MolToSmiles(x[0])
        smiles = smiles.replace("NH2+","N+")
        i += 1

        name_list.append(name+"+_"+str(i))
        smiles_list.append(smiles)
        atom_list.append(atoms[2*(i-isav)-2])
    
    # Keep only one of the elements for identical atoms
    name_list, smiles_list, atom_list = remove_identical_atoms(rdkit_mol, name_list, smiles_list, atom_list)

    return name_list, smiles_list, atom_list



if __name__ == "__main__":
    
    import time
    start = time.perf_counter()

    input_mol = Chem.MolFromSmiles('c1ccccc1O')

    print(generate_protonated_smiles(input_mol, name='mol'))
    
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')