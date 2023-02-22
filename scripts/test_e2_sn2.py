from typing import List
from scipy import spatial
import autode as ade
from autode.species.complex import Complex
from autode.values import Coordinates
from autode.bond_rearrangement import BondRearrangement
from autode.wrappers.XTB import XTB
from autode.conformers.conformer import Conformer
from autode.transition_states.locate_tss import translate_rotate_reactant
import numpy as np

def write_xyz_file(atoms: List[any], filename: str):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
        f.write(atom.atomic_symbol)
        for cartesian in ['x', 'y', 'z']:
            if getattr(atom.coord, cartesian) < 0:
                f.write('         ')
            else:
                f.write('          ')
            f.write("%.5f" % getattr(atom.coord, cartesian))
        f.write('\n')
    
    f.write('\n')

ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
xtb_method = XTB()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

if __name__ == "__main__":
    substrate = "[CH3:1][C@@H:2]([NH2:3])[CH2:4][Cl:5]"
    nucleophile = "[F-:6]"
    indices = [4, 3, 1, 5] # [[4, 3, 5], [4, 3, 1, 5]]
    n_conformers = 100

    dist_dict = {
        'H': 1.14,
        'F': 1.41,
        'Cl': 1.86,
        'Br': 2.04
    }
    hydrogen_dist_dict = {
        'H': 0.78,
        'F': 0.96,
        'Cl': 1.33,
        'Br': 2.48
    }

    nucleophile = ade.Molecule(smiles=nucleophile)
    substrate = ade.Molecule(smiles=substrate)
    substrate._generate_conformers(n_conformers)


    # add nucleophile to substrate
    conformers = []
    for conformer in substrate.conformers:
        # apply template
        diff_vec = conformer.atoms[indices[0]].coord - conformer.atoms[indices[1]].coord
        diff_vec /= np.linalg.norm(diff_vec)

        rot_vec = conformer.atoms[indices[1]].coord - conformer.atoms[indices[2]].coord
        rot_vec = rot_vec / np.linalg.norm(rot_vec) * np.radians(180)
        r = spatial.transform.Rotation.from_rotvec(rot_vec)
        
        distance = dist_dict[nucleophile.atoms[0].atomic_symbol] + 0.5
        nucleophile_coords = conformer.atoms[indices[1]].coord + r.apply(distance * diff_vec)
        constraints = {
            (indices[1], len(conformer.atoms)): distance
        }
        # TODO: fix this
        indices[-1] = len(conformer.atoms)

        new_atoms = conformer.atoms + [ade.Atom(
                atomic_symbol=nucleophile.atoms[0].atomic_symbol,
                x=nucleophile_coords[0],
                y=nucleophile_coords[1],
                z=nucleophile_coords[2]
            )]
        conformer._parent_atoms = [atom for atom in new_atoms]
        conformer._coordinates = Coordinates(np.array([atom.coord for atom in new_atoms]))
        conformer.charge = -1
        write_xyz_file(conformer.atoms, f'./results/test_{0}.xyz') 

        # constrained optimization
        conformer.constraints.distance = constraints
        conformer.optimise(method=xtb_method)
        write_xyz_file(conformer.atoms, f'./results/test_{1}.xyz') 

        # relaxed optimization
        conformer.constraints.distance = {}
        conformer.optimise(method=xtb_method)
        write_xyz_file(conformer.atoms, f'./results/test_{2}.xyz') 

        # check conditions
        conditions_checked = []
        conditions_checked.append(
            np.linalg.norm(conformer.atoms[indices[3]].coord - conformer.atoms[indices[1]].coord) > dist_dict[nucleophile.atoms[0].atomic_symbol]
        )
        h_distances = []
        for atom in conformer.atoms[:-1]:
            if atom.atomic_symbol == 'H':
                h_distances.append(np.linalg.norm(conformer.atoms[indices[3]].coord - atom.coord))
        conditions_checked.append(
            min(h_distances) > hydrogen_dist_dict[nucleophile.atoms[0].atomic_symbol]
        )

        # v1 = conformer.atoms[indices[0]].coord - conformer.atoms[indices[1]].coord
        # v2 = conformer.atoms[indices[3]].coord - conformer.atoms[indices[1]].coord
        # conditions_checked.append(
        #     angle_between(v1, v2) > 178
        # )

        # TODO: add another check if they are really disconnected

        if False not in conditions_checked:
            conformers.append(conformer)
        
    # write_xyz_file(conformers[0].atoms, f'./results/a.xyz') 

    r_energy = conformers[0].energy    
    """ TS initial guess """
    conformer_1 = Conformer(atoms=conformers[0].atoms[:-1])
    conformer_2 = Conformer(atoms=conformers[0].atoms[-1:], charge=-1)
    ts_guess = Complex(conformer_1, conformer_2)
    
    bond_rearr = BondRearrangement(
        forming_bonds=[(indices[1], indices[3])],
        breaking_bonds=[(indices[1], indices[0])]
    )
    translate_rotate_reactant(
        ts_guess,
        bond_rearrangement=bond_rearr,
        shift_factor=1.5 if ts_guess.charge == 0 else 2.5,
    )
    # write_xyz_file(ts_guess.atoms, f'./results/a.xyz') 

    r_2 = ts_guess.copy()
    r_2.optimise(method=xtb_method)
    r_2_energy = r_2.energy
    # write_xyz_file(ts_guess.atoms, f'./results/aa.xyz') 

    
    """ TS optimization """
    from autode.opt.optimisers import PRFOptimiser
    optimizer = PRFOptimiser(
        maxiter=100,
        gtol=1e-4,
        etol=1e-3
    )
    optimizer.optimise(
        species=ts_guess,
        method=xtb_method,
        maxiter=100
    )
    write_xyz_file(ts_guess.atoms, f'./results/aaa.xyz') 

   
    """ TS verification """
    # check imaginary modes
    ts_guess.calc_hessian(method=xtb_method)
    print(ts_guess.imaginary_frequencies)

    print(ts_guess.energy - r_energy)
    print(ts_guess.energy - r_2_energy)


    # nucleophile = ade.Molecule(smiles=nucleophile)

    # complex = Complex(substrate, nucleophile, name='new')
    # complex._generate_conformers()

    # print(len(complex.conformers))

    # # optimize conformers
    # conformers = []
    # for idx, conformer in enumerate(complex.conformers):
    #     try: 
    #         conformer.optimise(method=xtb_method)
    #         conformers.append(conformer)
    #     except:
    #         continue
    
    # print(len(conformers))

    # # filter out unique conformers

    # # filter conformers
    # filtered_conformers = []
    # for idx, conformer in enumerate(conformers):
    #     diff_vec = conformer.coordinates[indices[-1]] - conformer.coordinates[indices[1]]
    #     if np.linalg.norm(diff_vec) <= 3.5:
    #         filtered_conformers.append(conformer)

    # print(len(filtered_conformers))

    # for idx, conformer in enumerate(filtered_conformers):
    #     write_xyz_file(conformer.atoms, f'./results/test_{idx}.xyz')
    #     # conformer.calc_hessian(method=xtb_method)
    #     # print(conformer.frequencies)

    #     # print(conformer.atoms)

    #     # break

    # from autode.bond_rearrangement import get_bond_rearrangs
    # from autode.mol_graphs import species_are_isomorphic

    # def rr(reaction):
    #     reactant, product = reaction.reactant, reaction.product
    #     assert not species_are_isomorphic(reactant, product)
    #     bond_rearrs = get_bond_rearrangs(reactant, product, name=str(reaction))
    #     assert bond_rearrs is not None

    #     for bond_rearr in bond_rearrs:
    #         translate_rotate_reactant(
    #             reactant,
    #             bond_rearrangement=bond_rearr,
    #             shift_factor=1.5 if reactant.charge == 0 else 2.5,
    #         )
