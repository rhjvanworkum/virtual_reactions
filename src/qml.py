from typing import List, Optional
import qml
import tempfile
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from src.b2r2_parallel import get_b2r2_a_parallel

def get_xyz_string_from_rdkit_conformer(
    rdkit_mol: Chem.Mol,
    conformer: Chem.Conformer
) -> str:
    xyz_string = f"{len(rdkit_mol.GetAtoms())} \n\n"

    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        positions = conformer.GetAtomPosition(i)
        xyz_string += f"{atom.GetSymbol()} {round(positions.x, 6)} {round(positions.y, 6)} {round(positions.z, 6)} \n"

    return xyz_string

def get_qml_compound_from_geometry(xyz_string: str) -> qml.Compound:
    with tempfile.NamedTemporaryFile(suffix='.xyz') as tmp:
        with open(tmp.name, 'w') as f:
            f.writelines(xyz_string)
        compound = qml.Compound(tmp.name)
        return compound

def get_qml_compound_from_smiles(smiles_string: str) -> qml.Compound:
    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=1,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True, 
        ETversion=2
    )

    if mol.GetNumConformers() > 0:
        AllChem.UFFOptimizeMolecule(mol)

    xyz_string = get_xyz_string_from_rdkit_conformer(mol, mol.GetConformer(0))

    return get_qml_compound_from_geometry(xyz_string)




atom_dict = {
    'H':  1,
    'O':  8,
    'N':  7,
    'Si': 14,
    'P': 15,
    'B': 5,
    'Se': 34,
    'C':  6,
    'S':  16,
    'Cl': 17,
    'Ge': 32,
    'Br': 35,
    'F':  9,
    'I':  53
}

def featurize(
    substrate_smiles: List[str],
    product_smiles: List[str],
    elements: Optional[List[int]] = None
) -> np.ndarray:
    reactant_mols = [
        [get_qml_compound_from_smiles(smiles) for smiles in reactants.split('.')]
        for reactants in substrate_smiles
    ]
    product_mols = [
        [get_qml_compound_from_smiles(smiles) for smiles in products.split('.')]
        for products in product_smiles
    ]
    if elements is None:
        elements = []
        for mols in reactant_mols + product_mols:
            for mol in mols:
                elements += mol.atomtypes
        elements = np.unique(np.array(elements))
        elements = [atom_dict[e] for e in elements]

    b2r2 = get_b2r2_a_parallel(
        reactants_ncharges=[[mol.nuclear_charges for mol in mols] for mols in reactant_mols],
        products_ncharges=[[mol.nuclear_charges for mol in mols] for mols in product_mols],
        reactants_coords=[[mol.coordinates for mol in mols] for mols in reactant_mols],
        products_coords=[[mol.coordinates for mol in mols] for mols in reactant_mols],
        elements=elements,
        r_cut=3.5,
        gridspace=0.03,
        n_processes=16
    )

    return b2r2, elements


def predict_KRR(X_train, y_train, X_test, sigma=100, l2reg=1e-6):

    g_gauss = 1.0 / (2 * sigma**2)

    K = rbf_kernel(X_train, X_train, gamma=g_gauss)
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)
    K_test = rbf_kernel(X_test, X_train, gamma=g_gauss)

    y_pred = np.dot(K_test, alpha)
    return y_pred