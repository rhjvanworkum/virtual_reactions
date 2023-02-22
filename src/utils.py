from functools import wraps
import os
from subprocess import PIPE, Popen
from typing import List, Optional, Sequence, Callable
import shutil
import numpy as np
from tempfile import mkdtemp


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