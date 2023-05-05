
from src.compound import Compound


compound = Compound.from_smiles('C=C')
compound.generate_conformers()
compound.optimize_lowest_conformer()
indices = compound.compute_fukui_indices()
print(indices)