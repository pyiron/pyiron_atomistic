# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import os
from typing import Optional, Union

from ase.atoms import Atoms
from pyiron_base import Project, ProjectHDFio

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from pyiron_atomistics.lammps.interactive import LammpsInteractive

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Lammps(LammpsInteractive):
    """
    Class to setup and run and analyze LAMMPS simulations.

    Example:

    >>> job = pr.create.job.Lammps(job_name='lmp_example')
    >>> job.structure = pr.create.structure.bulk('Fe', cubic=True)
    >>> job.run()

    How to set potential: Look up potentials via `job.view_potentials()` (detailed data frame)
    or via `job.list_potentials()` (potential names). Assign the potential e.g. via:

    >>> job.potential = job.list_potentials()[0]

    Lammps has 3 modes: `static`, `md` and `minimize`. Set a mode e.g. via:

    >>> job.calc_minimize()

    Args:
        project (pyiron_atomistics.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (lammps.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(Lammps, self).__init__(project, job_name)

        self._executable_activate(enforce=True)


def lammps_function(
    working_directory: str,
    structure: Atoms,
    potential: str,
    calc_mode: str = "static",
    calc_kwargs: dict = {},
    cutoff_radius: Optional[float] = None,
    units: str = "metal",
    bonds_kwargs: dict = {},
    server_kwargs: dict = {},
    enable_h5md: bool = False,
    write_restart_file: bool = False,
    read_restart_file: bool = False,
    restart_file: str = "restart.out",
    executable_version: Optional[str] = None,
    executable_path: Optional[str] = None,
    input_control_file: Optional[Union[str, list, dict]] = None,
):
    """
    A single function to execute a LAMMPS calculation based on the LAMMPS job implemented in pyiron

    Examples:

    >>> import os
    >>> from ase.build import bulk
    >>> from pyiron_atomistics.lammps.lammps import lammps_function
    >>>
    >>> shell_output, parsed_output, job_crashed = lammps_function(
    ...     working_directory=os.path.abspath("lmp_working_directory"),
    ...     structure=bulk("Al", cubic=True),
    ...     potential='2009--Mendelev-M-I--Al-Mg--LAMMPS--ipr1',
    ...     calc_mode="md",
    ...     calc_kwargs={"temperature": 500.0, "pressure": 0.0, "n_ionic_steps": 1000, "n_print": 100},
    ...     cutoff_radius=None,
    ...     units="metal",
    ...     bonds_kwargs={},
    ...      enable_h5md=False,
    ... )

    Args:
        working_directory (str): directory in which the LAMMPS calculation is executed
        structure (Atoms): ase.atoms.Atoms - atomistic structure
        potential (str): Name of the LAMMPS potential based on the NIST database and the OpenKIM database
        calc_mode (str): select mode of calculation ["static", "md", "minimize", "vcsgc"]
        calc_kwargs (dict): key-word arguments for the calculate function, the input parameters depend on the calc_mode:
          "static": No parameters
          "md": "temperature", "pressure", "n_ionic_steps", "time_step", "n_print", "temperature_damping_timescale",
                "pressure_damping_timescale", "seed", "tloop", "initial_temperature", "langevin", "delta_temp",
                "delta_press"
          "minimize": "ionic_energy_tolerance", "ionic_force_tolerance", "max_iter", "pressure", "n_print", "style",
          "vcsgc": "mu", "ordered_element_list", "target_concentration", "kappa", "mc_step_interval", "swap_fraction",
                   "temperature_mc", "window_size", "window_moves", "temperature", "pressure", "n_ionic_steps",
                   "time_step", "n_print", "temperature_damping_timescale", "pressure_damping_timescale", "seed",
                   "initial_temperature", "langevin"
        cutoff_radius (float): cut-off radius for the interatomic potential
        units (str): Units for LAMMPS
        bonds_kwargs (dict): key-word arguments to create atomistic bonds:
          "species", "element_list", "cutoff_list", "max_bond_list", "bond_type_list", "angle_type_list",
        server_kwargs (dict): key-word arguments to create server object - the available parameters are:
          "user", "host", "run_mode", "queue", "qid", "cores", "threads", "new_h5", "structure_id", "run_time",
          "memory_limit", "accept_crash", "additional_arguments", "gpus", "conda_environment_name",
          "conda_environment_path"
        enable_h5md (bool): activate h5md mode for LAMMPS
        write_restart_file (bool): enable writing the LAMMPS restart file
        read_restart_file (bool): enable loading the LAMMPS restart file
        restart_file (str): file name of the LAMMPS restart file to copy
        executable_version (str): LAMMPS version to for the execution
        executable_path (str): path to the LAMMPS executable
        input_control_file (str|list|dict): Option to modify the LAMMPS input file directly

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    os.makedirs(working_directory, exist_ok=True)
    job = Lammps(
        project=ProjectHDFio(
            project=Project(working_directory),
            file_name="lmp_funct_job",
            h5_path=None,
            mode=None,
        ),
        job_name="lmp_funct_job",
    )
    job.structure = ase_to_pyiron(structure)
    job.potential = potential
    job.cutoff_radius = cutoff_radius
    server_dict = job.server.to_dict()
    server_dict.update(server_kwargs)
    job.server.from_dict(obj_dict=server_dict)
    job.units = units
    if calc_mode == "static":
        job.calc_static()
    elif calc_mode == "md":
        job.calc_md(**calc_kwargs)
    elif calc_mode == "minimize":
        job.calc_minimize(**calc_kwargs)
    elif calc_mode == "vcsgc":
        job.calc_vcsgc(**calc_kwargs)
    else:
        raise ValueError(
            f"calc_mode must be one of: static, md, minimize or vcsgc, not {calc_mode}"
        )
    if input_control_file is not None and isinstance(input_control_file, dict):
        for k, v in input_control_file.items():
            job.input.control[k] = v
    elif input_control_file is not None and (
        isinstance(input_control_file, str) or isinstance(input_control_file, list)
    ):
        job.input.control.load_string(input_str=input_control_file)
    if executable_path is not None:
        job.executable = executable_path
    if executable_version is not None:
        job.version = executable_version
    if enable_h5md:
        job.enable_h5md()
    if write_restart_file:
        job.write_restart_file(filename=restart_file)
    if read_restart_file:
        job.read_restart_file(filename=os.path.basename(restart_file))
        job.restart_file_list.append(restart_file)
    if len(bonds_kwargs) > 0:
        job.define_bonds(**bonds_kwargs)

    calculate_kwargs = job.calculate_kwargs
    calculate_kwargs["working_directory"] = working_directory
    return job.get_calculate_function()(**calculate_kwargs)
