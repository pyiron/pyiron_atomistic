# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import os
from typing import Optional, Union

from ase.atoms import Atoms
from pyiron_base import ProjectHDFio, Project

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from pyiron_atomistics.vasp.interactive import VaspInteractive

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Vasp(VaspInteractive):
    """
    Class to setup and run and analyze VASP simulations which is a derivative of pyiron_atomistics.objects.job.generic.GenericJob.
    The functions in these modules are written in such the function names and attributes are very generic
    (get_structure(), molecular_dynamics(), version) but the functions are written to handle VASP specific input/output.

    Args:
        project (pyiron_atomistics.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Examples:
        Let's say you need to run a vasp simulation where you would like to control the input parameters manually. To
        set up a static dft run with Gaussian smearing and a k-point MP mesh of [6, 6, 6]. You would have to set it up
        as shown below:

        >>> ham = Vasp(job_name="trial_job")
        >>> ham.input.incar[IBRION] = -1
        >>> ham.input.incar[ISMEAR] = 0
        >>> ham.input.kpoints.set_kpoints_file(size_of_mesh=[6, 6, 6])

        However, the according to pyiron's philosophy, it is recommended to avoid using code specific tags like IBRION,
        ISMEAR etc. Therefore the recommended way to set this calculation is as follows:

        >>> ham = Vasp(job_name="trial_job")
        >>> ham.calc_static()
        >>> ham.set_occupancy_smearing(smearing="gaussian")
        >>> ham.set_kpoints(mesh=[6, 6, 6])
        The exact same tags as in the first examples are set automatically.

    """

    def __init__(self, project, job_name):
        super(Vasp, self).__init__(project, job_name)

        self.__version__ = (
            None  # Reset the version number to the executable is set automatically
        )
        self._executable_activate(enforce=True)


def vasp_function(
    working_directory: str,
    structure: Atoms,
    plane_wave_cutoff: Optional[Union[float, int]] = None,
    exchange_correlation_functional: Optional[str] = None,
    spin_constraints: Optional[str] = None,
    write_electrostatic_potential: Optional[bool] = None,
    write_charge_density: Optional[bool] = None,
    write_wave_funct: Optional[bool] = None,
    write_resolved_dos: Optional[bool] = None,
    sorted_indices: Optional[list] = None,
    fix_spin_constraint: Optional[bool] = None,
    fix_symmetry: Optional[bool] = None,
    eddrmm_handling: str = "ignore",
    coulomb_interactions_kwargs: dict = {},
    algorithm_kwargs: dict = {},
    calc_mode: str = "static",
    calc_kwargs: dict = {},
    band_structure_calc_kwargs: dict = {},
    convergence_precision_kwargs: dict = {},
    dipole_correction_kwargs: dict = {},
    electric_field_kwargs: dict = {},
    occupancy_smearing_kwargs: dict = {},
    fft_mesh_kwargs: dict = {},
    mixing_parameters_kwargs: dict = {},
    n_empty_states: Optional[int] = None,
    rwigs_kwargs: dict = {},
    spin_constraint_kwargs: dict = {},
    kpoints_kwargs: dict = {},
    server_kwargs: dict = {},
    executable_version: Optional[str] = None,
    executable_path: Optional[str] = None,
    incar_file: Optional[Union[str, list, dict]] = None,
    kpoints_file: Optional[Union[str, list, dict]] = None,
):
    """

    Args:
        working_directory:
        structure:
        plane_wave_cutoff:
        exchange_correlation_functional:
        spin_constraints:
        write_electrostatic_potential:
        write_charge_density:
        write_wave_funct:
        write_resolved_dos:
        sorted_indices:
        fix_spin_constraint:
        fix_symmetry:
        eddrmm_handling:
        coulomb_interactions_kwargs:
        algorithm_kwargs:
        calc_mode:
        calc_kwargs:
        band_structure_calc_kwargs:
        convergence_precision_kwargs:
        dipole_correction_kwargs:
        electric_field_kwargs:
        occupancy_smearing_kwargs:
        fft_mesh_kwargs:
        mixing_parameters_kwargs:
        n_empty_states:
        rwigs_kwargs:
        spin_constraint_kwargs:
        kpoints_kwargs:
        server_kwargs:
        executable_version:
        executable_path:
        incar_file:
        kpoints_file:

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    os.makedirs(working_directory, exist_ok=True)
    job = Vasp(
        project=ProjectHDFio(
            project=Project(working_directory),
            file_name="lmp_funct_job",
            h5_path=None,
            mode=None,
        ),
        job_name="lmp_funct_job",
    )
    job.structure = ase_to_pyiron(structure)
    if plane_wave_cutoff is not None:
        job.plane_wave_cutoff = plane_wave_cutoff
    if exchange_correlation_functional is not None:
        job.exchange_correlation_functional = exchange_correlation_functional
    if spin_constraints is not None:
        job.spin_constraints = spin_constraints
    if write_electrostatic_potential is not None:
        job.write_electrostatic_potential = write_electrostatic_potential
    if write_charge_density is not None:
        job.write_charge_density = write_charge_density
    if write_wave_funct is not None:
        job.write_wave_funct = write_wave_funct
    if write_resolved_dos is not None:
        job.write_resolved_dos = write_resolved_dos
    if sorted_indices is not None:
        job.sorted_indices = sorted_indices
    if fix_spin_constraint is not None:
        job.fix_spin_constraint = fix_spin_constraint
    if fix_symmetry is not None:
        job.fix_symmetry = fix_symmetry
    job.set_eddrmm_handling(status=eddrmm_handling)
    if len(coulomb_interactions_kwargs) > 0:
        job.set_coulomb_interactions(**coulomb_interactions_kwargs)
    if len(algorithm_kwargs) > 0:
        job.set_algorithm(**algorithm_kwargs)
    if len(band_structure_calc_kwargs) > 0:
        job.set_for_band_structure_calc(**band_structure_calc_kwargs)
    if len(convergence_precision_kwargs) > 0:
        job.set_convergence_precision(**convergence_precision_kwargs)
    if len(dipole_correction_kwargs) > 0:
        job.set_dipole_correction(**dipole_correction_kwargs)
    if len(electric_field_kwargs) > 0:
        job.set_electric_field(**electric_field_kwargs)
    if len(occupancy_smearing_kwargs) > 0:
        job.set_occupancy_smearing(**occupancy_smearing_kwargs)
    if len(fft_mesh_kwargs) > 0:
        job.set_fft_mesh(**fft_mesh_kwargs)
    if len(mixing_parameters_kwargs) > 0:
        job.set_mixing_parameters(**mixing_parameters_kwargs)
    job.set_empty_states(n_empty_states=n_empty_states)
    if len(rwigs_kwargs) > 0:
        job.set_rwigs(rwigs_dict=rwigs_kwargs)
    if len(spin_constraint_kwargs) > 0:
        job.set_spin_constraint(**spin_constraint_kwargs)
    if len(kpoints_kwargs) > 0:
        job.set_kpoints(**kpoints_kwargs)
    server_dict = job.server.to_dict()
    server_dict.update(server_kwargs)
    job.server.from_dict(server_dict=server_dict)
    if calc_mode == "static":
        job.calc_static()
    elif calc_mode == "md":
        job.calc_md(**calc_kwargs)
    elif calc_mode == "minimize":
        job.calc_minimize(**calc_kwargs)
    else:
        raise ValueError()
    if incar_file is not None and isinstance(incar_file, dict):
        for k, v in incar_file.items():
            job.input.incar[k] = v
    elif incar_file is not None and (
        isinstance(incar_file, str) or isinstance(incar_file, list)
    ):
        job.input.incar.load_string(input_str=incar_file)
    if kpoints_file is not None and isinstance(kpoints_file, dict):
        for k, v in kpoints_file.items():
            job.input.kpoints[k] = v
    elif kpoints_file is not None and (
        isinstance(kpoints_file, str) or isinstance(kpoints_file, list)
    ):
        job.input.kpoints.load_string(input_str=kpoints_file)
    if executable_path is not None:
        job.executable = executable_path
    if executable_version is not None:
        job.version = executable_version

    calculate_kwargs = job.calculate_kwargs
    calculate_kwargs["working_directory"] = working_directory
    return job.get_calculate_function()(**calculate_kwargs)
