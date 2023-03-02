from typing import Callable, List
import autode as ade
from autode.values import Coordinates
from autode.bond_rearrangement import BondRearrangement

from scipy import spatial
import numpy as np
from copy import copy

from src.utils import write_xyz_file


class E2Sn2ReactionIndices:
    def __init__(
        self,
        leaving_group_idx: int,
        central_atom_idx: int,
        attacked_atom_idx: int,
        nucleophile_idx: int
    ) -> None:
        self.leaving_group_idx = leaving_group_idx
        self.central_atom_idx = central_atom_idx
        self.attacked_atom_idx = attacked_atom_idx
        self.nucleophile_idx = nucleophile_idx


class Sn2ReactionComplexTemplate:
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

    def __init__(
        self,
        checks: List[Callable] = []
    ) -> None:
        super(Sn2ReactionComplexTemplate, self).__init__()
        self.checks = checks

    def _construct_bond_rearrangement(
        self,
        indices: E2Sn2ReactionIndices
    ):
        bond_rearr = BondRearrangement(
            forming_bonds=[(indices.central_atom_idx, indices.nucleophile_idx)],
            breaking_bonds=[(indices.central_atom_idx, indices.leaving_group_idx)]
        )
        return bond_rearr


    def generate_reaction_complex(
        self,
        substrate: ade.Species,
        nucleophile: ade.Species,
        indices: E2Sn2ReactionIndices,
        method: ade.methods
    ) -> ade.Species:   
        """
        Tries to generate a reaction complex
        """
        # generate coordinates along substitution center
        diff_vec = substrate.atoms[indices.leaving_group_idx].coord - substrate.atoms[indices.central_atom_idx].coord
        diff_vec /= np.linalg.norm(diff_vec)
        rot_vec = substrate.atoms[indices.central_atom_idx].coord - substrate.atoms[indices.attacked_atom_idx].coord
        rot_vec = rot_vec / np.linalg.norm(rot_vec) * np.radians(180)
        r = spatial.transform.Rotation.from_rotvec(rot_vec)
        distance = self.dist_dict[nucleophile.atoms[0].atomic_symbol] + 1.0
        nucleophile_coords = substrate.atoms[indices.central_atom_idx].coord + r.apply(distance * diff_vec)
        
        # generate constraints + bond_rearrangements
        constraints = {
            (indices.central_atom_idx, len(substrate.atoms)): distance
        }
        tmp_indices = copy(indices)
        tmp_indices.nucleophile_idx = len(substrate.atoms)
        bond_rearrangement = self._construct_bond_rearrangement(tmp_indices)

        # generate reaction complex
        reaction_complex = substrate.copy()
        new_atoms = substrate.atoms + [ade.Atom(
                atomic_symbol=nucleophile.atoms[0].atomic_symbol,
                x=nucleophile_coords[0],
                y=nucleophile_coords[1],
                z=nucleophile_coords[2]
            )]
        reaction_complex._parent_atoms = [atom for atom in new_atoms]
        reaction_complex._coordinates = Coordinates(np.array([atom.coord for atom in new_atoms]))
        reaction_complex.charge = -1

        # constrained optimization of reaction complex
        try:
            reaction_complex.constraints.distance = constraints
            reaction_complex.optimise(method=method)
        except Exception as e:
            print("failed at RC optimization")
            return False, None, None

        # relaxed optimization of reaction complex
        try:
            reaction_complex.constraints.distance = {}
            reaction_complex.optimise(method=method)
        except Exception as e:
            print("failed at RC optimization")
            return False, None, None

        # check conditions
        conditions_passed = []
        for check in self.checks:
            conditions_passed.append(
                check(
                    reaction_complex,
                    indices,
                    self.dist_dict,
                    self.hydrogen_dist_dict,
                    nucleophile.atoms[0].atomic_symbol
                )
            )
        
        return (False not in conditions_passed), reaction_complex, bond_rearrangement
