import posixpath
from shutil import copyfile

import numpy as np
import scipy.constants
from pyiron_base import DataContainer
from typing import Optional

BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)


def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_lower_camel_case(snake_str):
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return snake_str[0].lower() + camel_string[1:]


class Group(DataContainer):
    """
    Dictionary-like object to store SPHInX inputs.

    Attributes (sub-groups, parameters, & flags) can be set
    and accessed via dot notation, or as standard dictionary
    key/values.

    `to_{job_type}` converts the Group to the format
    expected by the given DFT code in its input files.
    """

    def to_sphinx(self, content="__self__", indent=0):
        if content == "__self__":
            content = self

        def format_value(v):
            if isinstance(v, bool):
                return f" = {v};".lower()
            elif isinstance(v, Group):
                if len(v) == 0:
                    return " {}"
                else:
                    return " {\n" + self.to_sphinx(v, indent + 1) + indent * "\t" + "}"
            else:
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                return " = {!s};".format(v)

        line = ""
        for k, v in content.items():
            if isinstance(v, Group) and len(v) > 0 and not v.has_keys():
                for vv in v.values():
                    line += indent * "\t" + str(k) + format_value(vv) + "\n"
            else:
                line += indent * "\t" + str(k) + format_value(v) + "\n"

        return line


def get_structure_group(
    positions,
    cell,
    elements,
    movable=None,
    labels=None,
    use_symmetry=True,
    keep_angstrom=False,
):
    """
    create a SPHInX Group object based on structure

    Args:
        positions ((n, 3)-list/numpy.ndarray): xyz-coordinates of the atoms
        cell ((3, 3)-list/numpy.ndarray): Simulation box cdimensions
        elements ((n,)-list/numpy.ndarray): Chemical symbols
        movable (None/(n, 3)-list/nump.ndarray): Whether to fix the
            movement of the atoms in given directions
        labels (None/(n,)-list/numpy.ndarray): Extra labels to distinguish
            atoms for symmetries (mainly for magnetic moments)
        use_symmetry (bool): Whether or not consider internal symmetry
        keep_angstrom (bool): Store distances in Angstroms or Bohr

    Returns:
        (Group): structure group
    """
    positions = np.array(positions)
    cell = np.array(cell)
    if not keep_angstrom:
        cell /= BOHR_TO_ANGSTROM
        positions /= BOHR_TO_ANGSTROM
    structure_group = Group({"cell": np.array(cell)})
    if movable is not None:
        movable = np.array(movable)
    else:
        movable = np.full(shape=positions.shape, fill_value=True)
    if positions.shape != movable.shape:
        raise ValueError("positions.shape != movable.shape")
    if labels is not None:
        labels = np.array(labels)
    else:
        labels = np.full(shape=(len(positions),), fill_value=None)
    if (len(positions),) != labels.shape:
        raise ValueError("len(positions) != labels.shape")
    species = structure_group.create_group("species")
    for elm_species in np.unique(elements):
        species.append(Group({"element": '"' + str(elm_species) + '"'}))
        elm_list = elements == elm_species
        atom_group = species[-1].create_group("atom")
        for elm_pos, elm_magmom, selective in zip(
            positions[elm_list],
            labels[elm_list],
            movable[elm_list],
        ):
            atom_group.append(Group())
            if elm_magmom is not None:
                atom_group[-1]["label"] = '"spin_' + str(elm_magmom) + '"'
            atom_group[-1]["coords"] = np.array(elm_pos)
            if all(selective):
                atom_group[-1]["movable"] = True
            elif any(selective):
                for xx in np.array(["X", "Y", "Z"])[selective]:
                    atom_group[-1]["movable" + xx] = True
    if not use_symmetry:
        structure_group.symmetry = Group(
            {"operator": {"S": "[[1,0,0],[0,1,0],[0,0,1]]"}}
        )
    return structure_group


def copy_potentials(origins, destinations):
    """
    Args:
        origins (list): list of paths to potentials
        destinations (list): list of paths to copy potentials to
    """
    for ori, des in zip(origins, destinations):
        copyfile(ori, des)


def write_spin_constraints(
    file_name="spins.in", cwd=None, magmoms=None, constraints=None
):
    """
    Write a text file containing a list of all spins named spins.in -
    which is used for the external control scripts.

    Args:
        file_name (str): name of the file to be written (optional)
        cwd (str): the current working directory (optinal)
        spins_list (list): list of spins
    """
    for spin in magmoms:
        if isinstance(spin, list) or isinstance(spin, np.ndarray):
            raise ValueError("SPHInX only supports collinear spins at the moment.")
    spins = np.array(magmoms).astype(str)
    spins[~np.asarray(constraints)] = "X"
    spins_str = "\n".join(spins) + "\n"
    if cwd is not None:
        file_name = posixpath.join(cwd, file_name)
    with open(file_name, "w") as f:
        f.write(spins_str)


def fill_values(group=None, **kwargs):
    if group is None:
        group = Group()
    for k, v in kwargs.items():
        if v is not None and v is not False:
            if isinstance(v, Group):
                group.create_group(k)
            group[k] = v
    return group


def get_CCG_group(
    d_energy: Optional[float] = None,
    max_steps: Optional[int] = None,
    print_steps: Optional[int] = None,
    initial_diag: Optional[bool] = None,
    final_diag: Optional[bool] = None,
    kappa: Optional[float] = None,
    keep_occ_fixed: Optional[bool] = None,
    ekt: Optional[float] = None,
    dipole_correction: Optional[bool] = None,
    no_rho_storage: Optional[bool] = None,
    no_wave_storage: Optional[bool] = None,
):
    """
    Args:
        d_energy (float): Energy convergence criterion
        max_steps (int): Maximum number of SCF steps
        print_steps (int): Print SCF steps
        initial_diag (bool): Initial diagonalization
        final_diag (bool): Final diagonalization
        kappa (float): Kappa parameter
        keep_occ_fixed (bool): Keep occupation fixed
        ekt (float): Temperature
        dipole_correction (bool): Dipole correction
        no_rho_storage (bool): Do not store density
        no_wave_storage (bool): Do not store wave functions
    """
    return fill_values(
        dEnergy=d_energy,
        maxSteps=max_steps,
        printSteps=print_steps,
        initialDiag=initial_diag,
        finalDiag=final_diag,
        kappa=kappa,
        keepOccFixed=keep_occ_fixed,
        ekt=ekt,
        dipoleCorrection=dipole_correction,
        noRhoStorage=no_rho_storage,
        noWaveStorage=no_wave_storage,
    )


def get_scf_CCG_group(
    d_rel_eps: Optional[float] = None,
    max_steps_CCG: Optional[int] = None,
    d_energy: Optional[float] = None,
):
    """
    Args:
        d_rel_eps (float): Relative energy convergence criterion
        max_steps_CCG (int): Maximum number of CCG steps
        d_energy (float): Energy convergence criterion
    """
    return fill_values(
        dRelEps=d_rel_eps,
        maxStepsCCG=max_steps_CCG,
        dEnergy=d_energy,
    )


def get_scf_block_CCG_group(
    d_rel_eps: Optional[float] = None,
    max_steps_CCG: Optional[int] = None,
    block_size: Optional[int] = None,
    n_sloppy: Optional[int] = None,
    d_energy: Optional[float] = None,
    verbose: Optional[bool] = None,
    numerical_limit: Optional[bool] = None,
):
    """
    Args:
        d_rel_eps (float): Relative energy convergence criterion
        max_steps_CCG (int): Maximum number of CCG steps
        block_size (int): Block size
        n_sloppy (int): Number of sloppy steps
        d_energy (float): Energy convergence criterion
        verbose (bool): Verbose output
        numerical_limit (bool): Numerical limit
    """
    return fill_values(
        dRelEps=d_rel_eps,
        maxStepsCCG=max_steps_CCG,
        blockSize=block_size,
        nSloppy=n_sloppy,
        dEnergy=d_energy,
        verbose=verbose,
        numericalLimit=numerical_limit,
    )


def get_preconditioner_group(
    type: Optional[str] = "KERKER",
    scaling: Optional[float] = None,
    spin_scaling: Optional[float] = None,
    kerker_camping: Optional[float] = None,
    dielec_constant: Optional[float] = None,
):
    return fill_values(
        type=type,
        scaling=scaling,
        spinScaling=spin_scaling,
        kerkerCamping=kerker_camping,
        dielecConstant=dielec_constant,
    )


def get_scf_diag_group(
    d_energy: Optional[float] = None,
    max_steps: Optional[int] = None,
    max_residue: Optional[float] = None,
    print_steps: Optional[int] = None,
    mixing_method: Optional[str] = None,
    n_pulay_steps: Optional[int] = None,
    rho_mixing: Optional[float] = None,
    spin_mixing: Optional[float] = None,
    keep_rho_fixed: Optional[bool] = None,
    keep_occ_fixed: Optional[bool] = None,
    keep_spin_fixed: Optional[bool] = None,
    ekt: Optional[float] = None,
    dipole_correction: Optional[bool] = None,
    d_spin_moment: Optional[float] = None,
    no_rho_storage: Optional[bool] = None,
    no_wave_storage: Optional[bool] = None,
    CCG: Optional[Group] = None,
    block_CCG: Optional[Group] = None,
    preconditioner: Optional[Group] = None,
):
    """
    Args:
        d_energy (float): Energy convergence criterion
        max_steps (int): Maximum number of SCF steps
        max_residue (float): Residue convergence criterion
        print_steps (int): Print SCF steps
        mixing_method (str): Mixing method
        n_pulay_steps (int): Number of Pulay steps
        rho_mixing (float): Density mixing parameter
        spin_mixing (float): Spin mixing parameter
        keep_rho_fixed (bool): Keep density fixed
        keep_occ_fixed (bool): Keep occupation fixed
        keep_spin_fixed (bool): Keep spin fixed
        ekt (float): Temperature
        dipole_correction (bool): Dipole correction
        d_spin_moment (float): Spin moment convergence criterion
        no_rho_storage (bool): Do not store density
        no_wave_storage (bool): Do not store wave functions
        CCG (Group): Conjugate gradient method
        block_CCG (Group): Block conjugate gradient method
        preconditioner (Group): Preconditioner
    """
    return fill_values(
        dEnergy=d_energy,
        maxSteps=max_steps,
        maxResidue=max_residue,
        printSteps=print_steps,
        mixingMethod=mixing_method,
        nPulaySteps=n_pulay_steps,
        rhoMixing=rho_mixing,
        spinMixing=spin_mixing,
        keepRhoFixed=keep_rho_fixed,
        keepOccFixed=keep_occ_fixed,
        keepSpinFixed=keep_spin_fixed,
        ekt=ekt,
        dipoleCorrection=dipole_correction,
        dSpinMoment=d_spin_moment,
        noRhoStorage=no_rho_storage,
        noWaveStorage=no_wave_storage,
        CCG=CCG,
        blockCCG=block_CCG,
        preconditioner=preconditioner,
    )


def get_born_oppenheimer_group(
    scf_diag: Optional[Group] = None,
)
    """
    Args:
        scf_diag (Group): SCF diagonalization
    """
    return fill_values(scfDiag=scf_diag)


def get_QN_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    max_step_length: Optional[float] = None,
    hessian: Optional[str] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[Group] = None,
):
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        max_step_length (float): Maximum step length
        hessian (str): Initialize Hessian from file
        drift_filter (bool): Drift filter
        born_oppenheimer (Group): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        maxStepLength=max_step_length,
        hessian=hessian,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )


def get_linQN_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    max_step_length: Optional[float] = None,
    n_projectors: Optional[int] = None,
    hessian: Optional[str] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[Group] = None,
):
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        max_step_length (float): Maximum step length
        n_projectors (int): Number of projectors
        hessian (str): Initialize Hessian from file
        drift_filter (bool): Drift filter
        born_oppenheimer (Group): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        maxStepLength=max_step_length,
        nProjectors=n_projectors,
        hessian=hessian,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )


def get_ricQN_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    max_step_length: Optional[float] = None,
    n_projectors: Optional[int] = None,
    soft_mode_damping: Optional[float] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[Group] = None,
):
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        max_step_length (float): Maximum step length
        n_projectors (int): Number of projectors
        soft_mode_damping (float): Soft mode damping
        drift_filter (bool): Drift filter
        born_oppenheimer (Group): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        maxStepLength=max_step_length,
        nProjectors=n_projectors,
        softModeDamping=soft_mode_damping,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )


def get_ric_group(
    max_dist: Optional[float] = None,
    typify_threshold: Optional[float] = None,
    rms_threshold: Optional[float] = None,
    plane_cut_limit: Optional[float] = None,
    with_angles: Optional[bool] = None,
    bvk_atoms: Optional[str] = None,
    born_oppenheimer: Optional[Group] = None,
):
    """
    Args:
        max_dist (float): Maximum distance
        typify_threshold (float): Typify threshold
        rms_threshold (float): RMS threshold
        plane_cut_limit (float): Plane cut limit
        with_angles (bool): With angles
        bvk_atoms (str): (experimental) List of atom ids (starting from 1) for
            which born-von-Karman transversal force constants are added. The
            comma-separated list must be enclosed by square brackets []. This
            adds a bond-directional coordinate to each bond of the atoms in the
            list.
        born_oppenheimer (Group): Born-Oppenheimer
    """
    return fill_values(
        maxDist=max_dist,
        typifyThreshold=typify_threshold,
        rmsThreshold=rms_threshold,
        planeCutLimit=plane_cut_limit,
        withAngles=with_angles,
        bvkAtoms=bvk_atoms,
        bornOppenheimer=born_oppenheimer,
    )


def get_ricTS_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    n_projectors: Optional[int] = None,
    max_step_length: Optional[float] = None,
    trans_curvature: Optional[float] = None,
    any_stationary_point: Optional[bool] = None,
    max_dir_rot: Optional[float] = None,
    scheme: Optional[int] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[Group] = None,
):
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        n_projectors (int): Number of projectors
        max_step_length (float): Maximum step length
        trans_curvature (float): Transversal curvature
        any_stationary_point (bool): Any stationary point
        max_dir_rot (float): Maximum direction rotation
        scheme (int): Scheme
        drift_filter (bool): Drift filter
        born_oppenheimer (Group): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        nProjectors=n_projectors,
        maxStepLength=max_step
        transCurvature=trans_curvature,
        anyStationaryPoint=any_stationary_point,
        maxDirRot=max_dir_rot,
        scheme=scheme,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )

