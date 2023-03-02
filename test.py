import autode as ade
from autode.species.complex import Complex
from autode.opt.optimisers import PRFOptimiser
from autode.bonds import FormingBond, BreakingBond
from autode.wrappers.XTB import XTB
from autode.exceptions import NoMapping
from autode.bond_rearrangement import get_bond_rearrangs
from autode.transition_states.locate_tss import translate_rotate_reactant, _get_ts_neb_from_adaptive_path
from autode.path.adaptive import get_ts_adaptive_path
from autode.mol_graphs import get_mapping
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.values import Frequency, Distance, Allocation

from src.utils import write_xyz_file

BASE_DIR = '/home/ruard/code/virtual_reactions/calculations/'
ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
ade.Config.max_atom_displacement = Distance(7.0, units="Å")

rxn = ade.Reaction("CC(N)CCl.[F-]>>CC(N)=C.F.[Cl-]")
rxn.calculate_reaction_profile()

# xtb = XTB()

# for mol in rxn.reacs + rxn.prods:
#     # .find_lowest_energy_conformer works in conformers/
#     mol.find_lowest_energy_conformer(hmethod=xtb)

# reactant, product = rxn.reactant, rxn.product
# write_xyz_file(reactant.atoms, f'r_1.xyz')


# bond_rearr = get_bond_rearrangs(reactant, product, name=str(rxn))[0]
# bond_rearr.fbonds = [FormingBond(fbond, reactant) for fbond in bond_rearr.fbonds] 
# bond_rearr.bbonds = [BreakingBond(bbond, reactant) for bbond in bond_rearr.bbonds] 

# translate_rotate_reactant(
#     reactant,
#     bond_rearrangement=bond_rearr,
#     shift_factor=1.5 if reactant.charge == 0 else 2.5,
#     random_seed=42
# )
# write_xyz_file(reactant.atoms, f'r_2.xyz')

# try:
#     mapping = get_mapping(
#         graph1=product.graph,
#         graph2=reac_graph_to_prod_graph(reactant.graph, bond_rearr),
#     )
#     product.reorder_atoms(mapping=mapping)
# except NoMapping:
#     print("Could not find the expected bijection R -> P")

# reactant.solvent = "water"
# product.solvent = "water"

# reactant = _get_ts_neb_from_adaptive_path(reactant, product, xtb, bond_rearr, f"test", f"htest")
# write_xyz_file(reactant.atoms, f'r_3.xyz')


# ts_optimizer = PRFOptimiser(
#     maxiter=100,
#     gtol=1e-2,
#     etol=1e-2
# )
# ts_optimizer.optimise(
#     species=reactant,
#     method=xtb,
#     maxiter=100
# )

# write_xyz_file(reactant.atoms, f'r_4.xyz')
