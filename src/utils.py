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
      data = f.readline().replace('\n', '').split(' ')
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
            if getattr(atom.coord, cartesian) < 0:
                f.write('         ')
            else:
                f.write('          ')
            f.write("%.5f" % getattr(atom.coord, cartesian))
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