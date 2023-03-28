from functools import wraps
import os
from subprocess import PIPE, Popen
from typing import List, Optional, Sequence, Callable
import shutil
import numpy as np
from tempfile import mkdtemp


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


class Atom:
    def __init__(self, atomic_symbol, x, y, z) -> None:
        self.atomic_symbol = atomic_symbol
        self.x = x
        self.y = y
        self.z = z

    def to_list(self):
        return [self.atomic_symbol, self.x, self.y, self.z]
  
    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    @coordinates.setter
    def coordinates(self, coords):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]


def read_xyz_file(filename):
  atoms = []

  with open(filename) as f:
    n_atoms = int(f.readline())
    _ = f.readline()

    for i in range(n_atoms):
      data = f.readline().replace('\n', '').replace('\t', ' ').split(' ')
      data = list(filter(lambda a: a != '', data))
      atoms.append(Atom(data[0], float(data[1]), float(data[2]), float(data[3])))

  return atoms


def write_xyz_file(atoms: List[Atom], filename: str):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
        f.write(atom.atomic_symbol)
        for cartesian in ['x', 'y', 'z']:
            if getattr(atom, cartesian) < 0:
                f.write('         ')
            else:
                f.write('          ')
            f.write("%.5f" % getattr(atom, cartesian))
        f.write('\n')
    
    f.write('\n')


def work_in_tmp_dir(
    filenames_to_copy: Optional[Sequence[str]] = None,
    kept_file_exts: Optional[Sequence[str]] = None,
) -> Callable:

    if filenames_to_copy is None:
        filenames_to_copy = []

    if kept_file_exts is None:
        kept_file_exts = []

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            BASE_DIR = os.environ["BASE_DIR"]

            if BASE_DIR is not None:
                assert os.path.exists(BASE_DIR)

            tmpdir_path = mkdtemp(dir=BASE_DIR)

            for filename in filenames_to_copy:
                if filename.endswith("_mol.in"):
                    # MOPAC needs the file to be called this
                    shutil.move(filename, os.path.join(tmpdir_path, "mol.in"))
                else:
                    shutil.copy(filename, tmpdir_path)

            # # Move directories and execute
            os.chdir(tmpdir_path)

            try:
                result = func(*args, **kwargs)

                for filename in os.listdir(tmpdir_path):

                    if any([filename.endswith(ext) for ext in kept_file_exts]):
                        shutil.copy(filename, here)

            finally:
                os.chdir(here)

                shutil.rmtree(tmpdir_path)

            return result

        return wrapped_function

    return func_decorator




def run_in_tmp_environment(**kwargs) -> Callable:
    """
    Apply a set of environment variables, execute a function and reset them
    """

    class EnvVar:
        def __init__(self, name, val):
            self.name = str(name)
            self.val = os.getenv(str(name), None)
            self.new_val = str(val)

    env_vars = [EnvVar(k, v) for k, v in kwargs.items()]

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **_kwargs):

            for env_var in env_vars:
                os.environ[env_var.name] = env_var.new_val

            result = func(*args, **_kwargs)

            for env_var in env_vars:
                if env_var.val is None:
                    # Remove from the environment
                    os.environ.pop(env_var.name)
                else:
                    # otherwise set it back to the old value
                    os.environ[env_var.name] = env_var.val

            return result

        return wrapped_function

    return func_decorator



def run_external(
    params: List[str], output_filename: str, stderr_to_log: bool = True
):
    with open(output_filename, "w") as output_file:
        # /path/to/method input_filename > output_filename
        process = Popen(params, stdout=output_file, stderr=PIPE)

        with process.stderr:
            for line in iter(process.stderr.readline, b""):
                if stderr_to_log:
                    # logger.warning("STDERR: %r", line.decode())
                    pass

        process.wait()

    return None


from autode.log import logger
from autode.species import Complex
from autode.substitution import get_cost_rotate_translate
from autode.substitution import get_substc_and_add_dummy_atoms
from scipy.optimize import minimize


def translate_rotate_reactant(
    reactant, bond_rearrangement, shift_factor, random_seed, tolerance = 0.1, n_iters=10
):
    """
    Shift a molecule in the reactant complex so that the attacking atoms
    (a_atoms) are pointing towards the attacked atoms (l_atoms). Applied in
    place

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.complex.Complex):

        bond_rearrangement (autode.bond_rearrangement.BondRearrangement):

        shift_factor (float):

        n_iters (int): Number of iterations of translation/rotation to perform
                       to (hopefully) find the global minima
    """
    np.random.seed(random_seed)

    if not isinstance(reactant, Complex):
        logger.warning("Cannot rotate/translate component, not a Complex")
        return

    if reactant.n_molecules < 2:
        logger.info(
            "Reactant molecule does not need to be translated or " "rotated"
        )
        return

    logger.info("Rotating/translating into a reactive conformation... running")

    # This function can add dummy atoms for e.g. SN2' reactions where there
    # is not a A -- C -- Xattern for the substitution centre
    subst_centres = get_substc_and_add_dummy_atoms(
        reactant, bond_rearrangement, shift_factor=shift_factor
    )

    if all(
        sc.a_atom in reactant.atom_indexes(mol_index=0) for sc in subst_centres
    ):
        attacking_mol = 0
    else:
        attacking_mol = 1

    # Disable the logger to prevent rotation/translations printing
    logger.disabled = True

    # Find the global minimum for inplace rotation, translation and rotation
    min_cost, opt_x = None, None

    for _ in range(n_iters):
        res = minimize(
            get_cost_rotate_translate,
            x0=np.random.random(11),
            method="BFGS",
            tol=tolerance,
            args=(reactant, subst_centres, attacking_mol),
        )

        if min_cost is None or res.fun < min_cost:
            min_cost = res.fun
            opt_x = res.x

    # Re-enable the logger
    logger.disabled = False
    logger.info(f"Minimum cost for translating/rotating is {min_cost:.3f}")

    # Translate/rotation the attacking molecule optimally
    reactant.rotate_mol(
        axis=opt_x[:3], theta=opt_x[3], mol_index=attacking_mol
    )
    reactant.translate_mol(vec=opt_x[4:7], mol_index=attacking_mol)
    reactant.rotate_mol(
        axis=opt_x[7:10], theta=opt_x[10], mol_index=attacking_mol
    )

    logger.info("                                                 ... done")

    reactant.atoms.remove_dummy()
    reactant.print_xyz_file()

    return None





"""
openff toolkit replacement
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from cachetools import LRUCache, cached
from openff.units import unit
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from openff.toolkit.utils import base_wrapper
from openff.toolkit.utils.constants import (
    ALLOWED_AROMATICITY_MODELS,
    DEFAULT_AROMATICITY_MODEL,
)
from openff.toolkit.utils.exceptions import (
    InvalidAromaticityModelError,
    UndefinedStereochemistryError,
)

class EditedRDKitToolkitWrapper(RDKitToolkitWrapper):
    def __init__(self):
        super().__init__()

    def from_rdkit(
        self,
        rdmol,
        allow_undefined_stereo: bool = False,
        hydrogens_are_explicit: bool = False,
        _cls=None,
    ):
        """
        Create a Molecule from an RDKit molecule.

        Requires the RDKit to be installed.

        .. warning :: This API is experimental and subject to change.

        Parameters
        ----------
        rdmol : rkit.RDMol
            An RDKit molecule
        allow_undefined_stereo : bool, default=False
            If false, raises an exception if rdmol contains undefined stereochemistry.
        hydrogens_are_explicit : bool, default=False
            If False, RDKit will perform hydrogen addition using Chem.AddHs
        _cls : class
            Molecule constructor

        Returns
        -------
        molecule : openff.toolkit.topology.Molecule
            An OpenFF molecule

        Examples
        --------

        Create a molecule from an RDKit molecule

        >>> from rdkit import Chem
        >>> from openff.toolkit.tests.utils import get_data_file_path
        >>> rdmol = Chem.MolFromMolFile(get_data_file_path('systems/monomers/ethanol.sdf'))

        >>> toolkit_wrapper = RDKitToolkitWrapper()
        >>> molecule = toolkit_wrapper.from_rdkit(rdmol)

        """
        from rdkit import Chem

        if _cls is None:
            from openff.toolkit.topology.molecule import Molecule

            _cls = Molecule

        # Make a copy of the RDKit Mol as we'll need to change it (e.g. assign stereo).
        rdmol = Chem.Mol(rdmol)

        if not hydrogens_are_explicit:
            rdmol = Chem.AddHs(rdmol, addCoords=True)

        # Sanitizing the molecule. We handle aromaticity and chirality manually.
        # This SanitizeMol(...) calls cleanUp, updatePropertyCache, symmetrizeSSSR,
        # assignRadicals, setConjugation, and setHybridization.
        Chem.SanitizeMol(
            rdmol,
            (
                Chem.SANITIZE_ALL
                ^ Chem.SANITIZE_SETAROMATICITY
                ^ Chem.SANITIZE_ADJUSTHS
                ^ Chem.SANITIZE_CLEANUPCHIRALITY
                ^ Chem.SANITIZE_KEKULIZE
            ),
        )
        Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
        # SetAromaticity set aromatic bonds to 1.5, but Molecule.bond_order is an
        # integer (contrarily to fractional_bond_order) so we need the Kekule order.
        Chem.Kekulize(rdmol)

        # Make sure the bond stereo tags are set before checking for
        # undefined stereo. RDKit can figure out bond stereo from other
        # information in the Mol object like bond direction properties.
        # Do not overwrite eventual chiral tags provided by the user.
        Chem.AssignStereochemistry(rdmol, cleanIt=False)

        # Check for undefined stereochemistry.
        self._detect_undefined_stereo(
            rdmol,
            raise_warning=allow_undefined_stereo,
            err_msg_prefix="Unable to make OFFMol from RDMol: ",
        )

        # Create a new OpenFF Molecule
        offmol = _cls()

        # If RDMol has a title save it
        if rdmol.HasProp("_Name"):
            # raise Exception('{}'.format(rdmol.GetProp('name')))
            offmol.name = rdmol.GetProp("_Name")
        else:
            offmol.name = ""

        # Store all properties
        # TODO: Should there be an API point for storing properties?
        properties = rdmol.GetPropsAsDict()
        offmol._properties = properties

        # setting chirality in openeye requires using neighbor atoms
        # therefore we can't do it until after the atoms and bonds are all added
        map_atoms = {}
        map_bonds = {}
        # if we are loading from a mapped smiles extract the mapping
        atom_mapping = {}
        # We need the elements of the lanthanides, actinides, and transition
        # metals as we don't want to exclude radicals in these blocks.
        d_and_f_block_elements = {
            *range(21, 31),
            *range(39, 49),
            *range(57, 81),
            *range(89, 113),
        }
        for rda in rdmol.GetAtoms():
            # # See issues #1075 for some discussion on radicals
            # if (
            #     rda.GetAtomicNum() not in d_and_f_block_elements
            #     and rda.GetNumRadicalElectrons() != 0
            # ):
            #     raise RadicalsNotSupportedError(
            #         "The OpenFF Toolkit does not currently support parsing molecules with S- and P-block radicals. "
            #         f"Found {rda.GetNumRadicalElectrons()} radical electrons on molecule {Chem.MolToSmiles(rdmol)}."
            #     )

            rd_idx = rda.GetIdx()
            # if the molecule was made from a mapped smiles this has been hidden
            # so that it does not affect the sterochemistry tags
            try:
                map_id = int(rda.GetProp("_map_idx"))
            except KeyError:
                map_id = rda.GetAtomMapNum()

            # create a new atom
            # atomic_number = oemol.NewAtom(rda.GetAtomicNum())
            atomic_number = rda.GetAtomicNum()
            # implicit units of elementary charge
            formal_charge = rda.GetFormalCharge()
            is_aromatic = rda.GetIsAromatic()
            if rda.HasProp("_Name"):
                name = rda.GetProp("_Name")
            else:
                # check for PDB names
                try:
                    name = rda.GetMonomerInfo().GetName().strip()
                except AttributeError:
                    name = ""

            # If chiral, store the chirality to be set later
            stereochemistry = None
            # tag = rda.GetChiralTag()
            if rda.HasProp("_CIPCode"):
                stereo_code = rda.GetProp("_CIPCode")
                # if tag == Chem.CHI_TETRAHEDRAL_CCW:
                if stereo_code == "R":
                    stereochemistry = "R"
                # if tag == Chem.CHI_TETRAHEDRAL_CW:
                elif stereo_code == "S":
                    stereochemistry = "S"
                else:
                    raise UndefinedStereochemistryError(
                        "In from_rdkit: Expected atom stereochemistry of R or S. "
                        "Got {} instead.".format(stereo_code)
                    )

            res = rda.GetPDBResidueInfo()
            metadata = dict()
            if res is not None:
                metadata["residue_name"] = res.GetResidueName()
                metadata["residue_number"] = res.GetResidueNumber()
                metadata["insertion_code"] = res.GetInsertionCode()
                metadata["chain_id"] = res.GetChainId()

            atom_index = offmol._add_atom(
                atomic_number,
                formal_charge,
                is_aromatic,
                name=name,
                stereochemistry=stereochemistry,
                metadata=metadata,
                invalidate_cache=False,
            )
            map_atoms[rd_idx] = atom_index
            atom_mapping[atom_index] = map_id

        offmol._invalidate_cached_properties()

        # If we have a full / partial atom map add it to the molecule. Zeroes 0
        # indicates no mapping
        if {*atom_mapping.values()} != {0}:
            offmol._properties["atom_map"] = {
                idx: map_idx for idx, map_idx in atom_mapping.items() if map_idx != 0
            }

        # Similar to chirality, stereochemistry of bonds in OE is set relative to their neighbors
        for rdb in rdmol.GetBonds():
            rdb_idx = rdb.GetIdx()
            a1 = rdb.GetBeginAtomIdx()
            a2 = rdb.GetEndAtomIdx()

            # Determine bond aromaticity and Kekulized bond order
            is_aromatic = rdb.GetIsAromatic()
            order = rdb.GetBondTypeAsDouble()
            # Convert floating-point bond order to integral bond order
            order = int(order)

            # create a new bond
            bond_index = offmol._add_bond(
                map_atoms[a1],
                map_atoms[a2],
                order,
                is_aromatic=is_aromatic,
                invalidate_cache=False,
            )
            map_bonds[rdb_idx] = bond_index

        offmol._invalidate_cached_properties()

        # Now fill in the cached (structure-dependent) properties. We have to have the 2D
        # structure of the molecule in place first, because each call to add_atom and
        # add_bond invalidates all cached properties
        for rdb in rdmol.GetBonds():
            rdb_idx = rdb.GetIdx()
            offb_idx = map_bonds[rdb_idx]
            offb = offmol.bonds[offb_idx]
            # determine if stereochemistry is needed
            # Note that RDKit has 6 possible values of bond stereo: CIS, TRANS, E, Z, ANY, or NONE
            # The logic below assumes that "ANY" and "NONE" mean the same thing.
            stereochemistry = None
            tag = rdb.GetStereo()
            if tag == Chem.BondStereo.STEREOZ:
                stereochemistry = "Z"
            elif tag == Chem.BondStereo.STEREOE:
                stereochemistry = "E"
            elif tag == Chem.BondStereo.STEREOTRANS or tag == Chem.BondStereo.STEREOCIS:
                raise ValueError(
                    "Expected RDKit bond stereochemistry of E or Z, got {} instead".format(
                        tag
                    )
                )
            offb._stereochemistry = stereochemistry
            fractional_bond_order = None
            if rdb.HasProp("fractional_bond_order"):
                fractional_bond_order = rdb.GetDoubleProp("fractional_bond_order")
            offb.fractional_bond_order = fractional_bond_order

        # TODO: Save conformer(s), if present
        # If the rdmol has a conformer, store its coordinates
        if len(rdmol.GetConformers()) != 0:
            for conf in rdmol.GetConformers():
                n_atoms = offmol.n_atoms
                # Here we assume this always be angstrom
                positions = np.zeros((n_atoms, 3))
                for rd_idx, off_idx in map_atoms.items():
                    atom_coords = conf.GetPositions()[rd_idx, :]
                    positions[off_idx, :] = atom_coords
                offmol._add_conformer(unit.Quantity(positions, unit.angstrom))

        partial_charges = np.zeros(shape=offmol.n_atoms, dtype=np.float64)

        any_atom_has_partial_charge = False
        for rd_idx, rd_atom in enumerate(rdmol.GetAtoms()):
            off_idx = map_atoms[rd_idx]
            if rd_atom.HasProp("PartialCharge"):
                charge = rd_atom.GetDoubleProp("PartialCharge")
                partial_charges[off_idx] = charge
                any_atom_has_partial_charge = True
            else:
                # If some other atoms had partial charges but this one doesn't, raise an Exception
                if any_atom_has_partial_charge:
                    raise ValueError(
                        "Some atoms in rdmol have partial charges, but others do not."
                    )
        if any_atom_has_partial_charge:
            offmol.partial_charges = unit.Quantity(
                partial_charges, unit.elementary_charge
            )
        else:
            offmol.partial_charges = None
        return offmol

    to_rdkit_cache = LRUCache(maxsize=4096)

    @cached(to_rdkit_cache, key=base_wrapper._mol_to_ctab_and_aro_key)
    def _connection_table_to_rdkit(
        self, molecule, aromaticity_model=DEFAULT_AROMATICITY_MODEL
    ):
        from rdkit import Chem

        if aromaticity_model not in ALLOWED_AROMATICITY_MODELS:
            raise InvalidAromaticityModelError(
                f"Given aromaticity model {aromaticity_model} which is not in the set of allowed aromaticity models: "
                f"{ALLOWED_AROMATICITY_MODELS}"
            )

        # Create an editable RDKit molecule
        rdmol = Chem.RWMol()

        _bondtypes = {
            1: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
            5: Chem.BondType.QUINTUPLE,
            6: Chem.BondType.HEXTUPLE,
            7: Chem.BondType.ONEANDAHALF,
        }

        for index, atom in enumerate(molecule.atoms):
            rdatom = Chem.Atom(atom.atomic_number)
            rdatom.SetFormalCharge(atom.formal_charge.m_as(unit.elementary_charge))
            rdatom.SetIsAromatic(atom.is_aromatic)

            # Stereo handling code moved to after bonds are added
            if atom.stereochemistry == "S":
                rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            elif atom.stereochemistry == "R":
                rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

            # Stop rdkit from adding implicit hydrogens
            rdatom.SetNoImplicit(True)

            rd_index = rdmol.AddAtom(rdatom)

            # Let's make sure al the atom indices in the two molecules
            # are the same, otherwise we need to create an atom map.
            assert index == atom.molecule_atom_index
            assert index == rd_index

        for bond in molecule.bonds:
            atom_indices = (
                bond.atom1.molecule_atom_index,
                bond.atom2.molecule_atom_index,
            )
            rdmol.AddBond(*atom_indices)
            rdbond = rdmol.GetBondBetweenAtoms(*atom_indices)
            # Assign bond type, which is based on order unless it is aromatic
            if bond.is_aromatic:
                rdbond.SetBondType(_bondtypes[1.5])
                rdbond.SetIsAromatic(True)
            else:
                rdbond.SetBondType(_bondtypes[bond.bond_order])
                rdbond.SetIsAromatic(False)

        Chem.SanitizeMol(
            rdmol,
            Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS ^ Chem.SANITIZE_SETAROMATICITY,
        )

        if aromaticity_model == "OEAroModel_MDL":
            Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
        else:
            raise InvalidAromaticityModelError(
                f"Given aromaticity model {aromaticity_model} which is not in the set of allowed aromaticity models:"
                f"{ALLOWED_AROMATICITY_MODELS}"
            )

        # Assign atom stereochemsitry and collect atoms for which RDKit
        # can't figure out chirality. The _CIPCode property of these atoms
        # will be forcefully set to the stereo we want (see #196).
        undefined_stereo_atoms = {}
        for index, atom in enumerate(molecule.atoms):
            rdatom = rdmol.GetAtomWithIdx(index)

            # Skip non-chiral atoms.
            if atom.stereochemistry is None:
                continue

            # Let's randomly assign this atom's (local) stereo to CW
            # and check if this causes the (global) stereo to be set
            # to the desired one (S or R).
            rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            # We need to do force and cleanIt to recalculate CIP stereo.
            Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
            # If our random initial assignment worked, then we're set.
            if (
                rdatom.HasProp("_CIPCode")
                and rdatom.GetProp("_CIPCode") == atom.stereochemistry
            ):
                continue

            # Otherwise, set it to CCW.
            rdatom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            # We need to do force and cleanIt to recalculate CIP stereo.
            Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
            # Hopefully this worked, otherwise something's wrong
            if (
                rdatom.HasProp("_CIPCode")
                and rdatom.GetProp("_CIPCode") == atom.stereochemistry
            ):
                continue

            # Keep track of undefined stereo atoms. We'll force stereochemistry
            # at the end to avoid the next AssignStereochemistry to overwrite.
            if not rdatom.HasProp("_CIPCode"):
                undefined_stereo_atoms[rdatom] = atom.stereochemistry
                continue

            # Something is wrong.
            err_msg = (
                "Unknown atom stereochemistry encountered in to_rdkit. "
                "Desired stereochemistry: {}. Set stereochemistry {}".format(
                    atom.stereochemistry, rdatom.GetProp("_CIPCode")
                )
            )
            raise RuntimeError(err_msg)

        # Copy bond stereo info from molecule to rdmol.
        self._assign_rdmol_bonds_stereo(molecule, rdmol)

        # Cleanup the rdmol
        rdmol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(rdmol)

        # Forcefully assign stereo information on the atoms that RDKit
        # can't figure out. This must be done last as calling AssignStereochemistry
        # again will delete these properties (see #196).
        for rdatom, stereochemistry in undefined_stereo_atoms.items():
            rdatom.SetProp("_CIPCode", stereochemistry)

        return rdmol