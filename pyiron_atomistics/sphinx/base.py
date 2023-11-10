# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH -Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function, division

import numpy as np
import os
import posixpath
import re
import stat
from shutil import copyfile, move as movefile
import scipy.constants
import warnings
import json
import spglib
import subprocess
from subprocess import PIPE
import tarfile
from tempfile import TemporaryDirectory

from pyiron_atomistics.dft.job.generic import GenericDFTJob
from pyiron_atomistics.dft.waves.electronic import ElectronicStructure
from pyiron_atomistics.vasp.potential import (
    VaspPotentialFile,
    strip_xc_from_potential_name,
    find_potential_file as find_potential_file_vasp,
    VaspPotentialSetter,
)
from pyiron_atomistics.sphinx.structure import read_atoms
from pyiron_atomistics.sphinx.potential import SphinxJTHPotentialFile
from pyiron_atomistics.sphinx.potential import (
    find_potential_file as find_potential_file_jth,
)
from pyiron_atomistics.sphinx.util import sxversions
from pyiron_atomistics.sphinx.volumetric_data import SphinxVolumetricData
from pyiron_base import state, DataContainer, job_status_successful_lst, deprecate

__author__ = "Osamu Waseda, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"

BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)
HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]
RYDBERG_TO_EV = HARTREE_TO_EV / 2
HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class SphinxBase(GenericDFTJob):
    """
    Class to setup and run SPHInX simulations.

    Inherits pyiron_atomistics.atomistics.job.generic.GenericJob. The functions in
    these modules are written such that the function names and attributes
    are very pyiron-generic (get_structure(), molecular_dynamics(),
    version) but internally handle SPHInX specific input and output.

    Alternatively, because SPHInX inputs are built on a group-based
    format, users have the option to set specific groups and parameters
    directly, e.g.

    ```python
    # Modify/add a new parameter via
    job.input.sphinx.PAWHamiltonian.nEmptyStates = 15
    job.input.sphinx.PAWHamiltonian.dipoleCorrection = True
    # or
    job.input.sphinx.PAWHamiltonian.set("nEmptyStates", 15)
    job.input.sphinx.PAWHamiltonian.set("dipoleCorrection", True)
    # Modify/add a sub-group via
    job.input.sphinx.initialGuess.rho.charged = {"charge": 2, "z": 25}
    # or
    job.input.sphinx.initialGuess.rho.set("charged", {"charge": 2, "z": 25})
    ```

    Args:
        project: Project object (defines path where job will be
                 created and stored)
        job_name (str): name of the job (must be unique within
                        this project path)
    """

    """Version of the data format in hdf5"""
    __hdf_version__ = "0.1.0"

    def __init__(self, project, job_name):
        super(SphinxBase, self).__init__(project, job_name)

        # keeps both the generic parameters as well as the sphinx specific
        # input groups
        self.input = Group(table_name="parameters", lazy=True)
        self.load_default_input()
        self.output = Output(job=self)
        self.input_writer = InputWriter()
        self._potential = VaspPotentialSetter([])
        if self.check_vasp_potentials():
            self.input["VaspPot"] = True  # use VASP potentials if available
        self._generic_input["restart_for_band_structure"] = False
        self._generic_input["path_name"] = None
        self._generic_input["n_path"] = None
        self._generic_input["fix_spin_constraint"] = False

    def update_sphinx(self):
        if self.output.old_version:
            _update_datacontainer(self)

    def __getitem__(self, item):
        if not isinstance(item, str):
            return super().__getitem__(item)
        result = None
        if item[-1] == "/":
            item = item[:-1]
        for tag in item.split("/"):
            try:  # horrible workaround to be removed when hdf output becomes consistent
                if result is None:
                    result = super().__getitem__(tag)
                else:
                    result = result[tag]
                if hasattr(result, "list_nodes") and "TYPE" in result.list_nodes():
                    result = result.to_object()
            except (ValueError, KeyError):
                return None
        return result

    @property
    def structure(self):
        """

        Returns:

        """
        return GenericDFTJob.structure.fget(self)

    @structure.setter
    def structure(self, structure):
        """

        Args:
            structure:

        Returns:

        """
        GenericDFTJob.structure.fset(self, structure)
        if structure is not None:
            self._potential = VaspPotentialSetter(
                element_lst=structure.get_species_symbols().tolist()
            )

    @property
    def id_pyi_to_spx(self):
        if self.input_writer.id_pyi_to_spx is None:
            self.input_writer.structure = self.structure
        return self.input_writer.id_pyi_to_spx

    @property
    def id_spx_to_pyi(self):
        if self.input_writer.id_spx_to_pyi is None:
            self.input_writer.structure = self.structure
        return self.input_writer.id_spx_to_pyi

    @property
    def plane_wave_cutoff(self):
        if "eCut" in self.input.sphinx.basis.keys():
            return self.input.sphinx.basis["eCut"] * RYDBERG_TO_EV
        else:
            return self.input["EnCut"]

    @property
    def fix_spin_constraint(self):
        return self._generic_input["fix_spin_constraint"]

    @fix_spin_constraint.setter
    def fix_spin_constraint(self, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("fix_spin_constraint has to be a boolean")
        self._generic_input["fix_spin_constraint"] = boolean
        self.structure.set_array(
            "spin_constraint", np.array(len(self.structure) * [boolean])
        )

    @plane_wave_cutoff.setter
    def plane_wave_cutoff(self, val):
        """
        Function to setup the energy cut-off for the SPHInX job in eV.

        Args:
            val (int): energy cut-off in eV
        """
        if val <= 0:
            raise ValueError("Cutoff radius value not valid")
        elif val < 50:
            warnings.warn(
                "The given cutoff is either very small (probably "
                + "too small) or was accidentally given in Ry. "
                + "Please make sure it is in eV (1eV = 13.606 Ry)."
            )
        self.input["EnCut"] = val
        self.input.sphinx.basis.eCut = self.input["EnCut"] / RYDBERG_TO_EV

    @property
    def exchange_correlation_functional(self):
        return self.input["Xcorr"]

    @exchange_correlation_functional.setter
    def exchange_correlation_functional(self, val):
        """
        Args:
            val:

        Returns:
        """
        if val.upper() in ["PBE", "LDA"]:
            self.input["Xcorr"] = val.upper()
        else:
            warnings.warn(
                "Exchange correlation function not recognized (\
                    recommended: PBE or LDA)",
                SyntaxWarning,
            )
            self.input["Xcorr"] = val
        if "xc" in self.input.sphinx.PAWHamiltonian.keys():
            self.input.sphinx.PAWHamiltonian.xc = self.input["Xcorr"]

    @property
    def potential_view(self):
        if self.structure is None:
            raise ValueError("Can't list potentials unless a structure is set")
        else:
            if self.input["VaspPot"]:
                potentials = VaspPotentialFile(xc=self.input["Xcorr"])
            else:
                potentials = SphinxJTHPotentialFile(xc=self.input["Xcorr"])
            df = potentials.find(self.structure.get_species_symbols().tolist())
            if len(df) > 0:
                df["Name"] = [
                    strip_xc_from_potential_name(n) for n in df["Name"].values
                ]
            return df

    @property
    def potential_list(self):
        return list(self.potential_view["Name"].values)

    @property
    def potential(self):
        return self._potential

    def get_kpoints(self):
        return self.input.KpointFolding

    def get_version_float(self):
        version_str = self.executable.version.split("_")[0]
        version_float = float(version_str.split(".")[0])
        if len(version_str.split(".")) > 1:
            version_float += float("0." + "".join(version_str.split(".")[1:]))
        return version_float

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes,
        but it has to be implemented in the individual classes.
        """
        super(SphinxBase, self).set_input_to_read_only()
        self.input.read_only = True

    def get_scf_group(
        self, maxSteps=None, keepRhoFixed=False, dEnergy=None, algorithm="blockCCG"
    ):
        """
        SCF group setting for SPHInX
        for all args refer to calc_static or calc_minimize
        """

        scf_group = Group()
        if algorithm.upper() == "CCG":
            algorithm = "CCG"
        elif algorithm.upper() != "BLOCKCCG":
            warnings.warn(
                "Algorithm not recognized -> setting to blockCCG. \
                    Alternatively, choose algorithm=CCG",
                SyntaxWarning,
            )
            algorithm = "blockCCG"

        if keepRhoFixed:
            scf_group["keepRhoFixed"] = True
        else:
            scf_group["rhoMixing"] = str(self.input["rhoMixing"])
            scf_group["spinMixing"] = str(self.input["spinMixing"])
            if "nPulaySteps" in self.input:
                scf_group["nPulaySteps"] = str(self.input["nPulaySteps"])
        if dEnergy is None:
            scf_group["dEnergy"] = self.input["Ediff"] / HARTREE_TO_EV
        else:
            scf_group["dEnergy"] = str(dEnergy)
        if maxSteps is None:
            scf_group["maxSteps"] = str(self.input["Estep"])
        else:
            scf_group["maxSteps"] = str(maxSteps)
        if "preconditioner" in self.input and self.input["preconditioner"] != "KERKER":
            scf_group.create_group("preconditioner")["type"] = self.input[
                "preconditioner"
            ]
        else:
            scf_group.create_group("preconditioner")["type"] = "KERKER"
            scf_group.preconditioner.set_parameter(
                "scaling", self.input["rhoResidualScaling"]
            )
            scf_group.preconditioner.set_parameter(
                "spinScaling", self.input["spinResidualScaling"]
            )
        scf_group.create_group(algorithm)
        if "maxStepsCCG" in self.input:
            scf_group[algorithm]["maxStepsCCG"] = self.input["maxStepsCCG"]
        if "blockSize" in self.input and algorithm == "blockCCG":
            scf_group[algorithm]["blockSize"] = self.input["blockSize"]
        if "nSloppy" in self.input and algorithm == "blockCCG":
            scf_group[algorithm]["nSloppy"] = self.input["nSloppy"]
        if self.input["WriteWaves"] is False:
            scf_group["noWavesStorage"] = True
        return scf_group

    def get_structure_group(self, keep_angstrom=False):
        """
        create a SPHInX Group object based on self.structure

        Args:
            keep_angstrom (bool): Store distances in Angstroms or Bohr
        """
        return get_structure_group(
            structure=self.structure,
            use_symmetry=self.fix_symmetry,
            keep_angstrom=keep_angstrom,
        )

    def load_default_input(self):
        """
        Set defaults for generic parameters and create SPHInX input groups.
        """

        sph = self.input.create_group("sphinx")
        sph.create_group("pawPot")
        sph.create_group("structure")
        sph.create_group("basis")
        sph.create_group("PAWHamiltonian")
        sph.create_group("initialGuess")
        sph.create_group("main")

        self.input.EnCut = 340
        self.input.KpointCoords = [0.5, 0.5, 0.5]
        self.input.KpointFolding = [4, 4, 4]
        self.input.EmptyStates = "auto"
        self.input.MethfesselPaxton = 1
        self.input.Sigma = 0.2
        self.input.Xcorr = "PBE"
        self.input.VaspPot = False
        self.input.Estep = 100
        self.input.Ediff = 1.0e-4
        self.input.WriteWaves = True
        self.input.KJxc = False
        self.input.SaveMemory = True
        self.input.rhoMixing = 1.0
        self.input.spinMixing = 1.0
        self.input.rhoResidualScaling = 1.0
        self.input.spinResidualScaling = 1.0
        self.input.CheckOverlap = True
        self.input.THREADS = 1
        self.input.use_on_the_fly_cg_optimization = True

    def load_structure_group(self, keep_angstrom=False):
        """
        Build + load the structure group based on self.structure

        Args:
            keep_angstrom (bool): Store distances in Angstroms or Bohr
        """
        self.input.sphinx.structure = self.get_structure_group(
            keep_angstrom=keep_angstrom
        )

    def load_species_group(self, check_overlap=True, potformat="VASP"):
        """
        Build the species Group object based on self.structure

        Args:
            check_overlap (bool): Whether to check overlap
                (see set_check_overlap)
            potformat (str): type of pseudopotentials that will be
                read. Options are JTH or VASP.
        """

        self.input.sphinx.pawPot = Group({"species": []})
        for species_obj in self.structure.get_species_objects():
            if species_obj.Parent is not None:
                elem = species_obj.Parent
            else:
                elem = species_obj.Abbreviation
            if potformat == "JTH":
                self.input.sphinx.pawPot["species"].append(
                    Group(
                        {
                            "name": '"' + elem + '"',
                            "potType": '"AtomPAW"',
                            "element": '"' + elem + '"',
                            "potential": f'"{elem}_GGA.atomicdata"',
                        }
                    )
                )
            elif potformat == "VASP":
                self.input.sphinx.pawPot["species"].append(
                    Group(
                        {
                            "name": '"' + elem + '"',
                            "potType": '"VASP"',
                            "element": '"' + elem + '"',
                            "potential": '"' + elem + "_POTCAR" + '"',
                        }
                    )
                )
            else:
                raise ValueError("Potential must be JTH or VASP")
        if not check_overlap:
            self.input.sphinx.pawPot["species"][-1]["checkOverlap"] = "false"
        if self.input["KJxc"]:
            self.input.sphinx.pawPot["kjxc"] = True

    def load_main_group(self):
        """
        Load the main Group.

        The group is populated based on the type of calculation and settings in
        the self.input.
        """

        if (
            len(self.restart_file_list) != 0
            and not self._generic_input["restart_for_band_structure"]
        ):
            self.input.sphinx.main.get("scfDiag", create=True).append(
                self.get_scf_group(maxSteps=10, keepRhoFixed=True, dEnergy=1.0e-4)
            )
        if "Istep" in self.input:
            optimizer = "linQN"
            if self.input.use_on_the_fly_cg_optimization:
                optimizer = "ricQN"
            self.input.sphinx.main[optimizer] = Group(table_name="input")
            self.input.sphinx.main[optimizer]["maxSteps"] = str(self.input["Istep"])
            self.input.sphinx.main[optimizer]["maxStepLength"] = str(
                0.1 / BOHR_TO_ANGSTROM
            )
            if "dE" in self.input:
                self.input.sphinx.main[optimizer]["dEnergy"] = str(
                    self.input["dE"] / HARTREE_TO_EV
                )
            if "dF" in self.input:
                self.input.sphinx.main[optimizer]["dF"] = str(
                    self.input["dF"] / HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM
                )
            self.input.sphinx.main[optimizer].create_group("bornOppenheimer")
            self.input.sphinx.main[optimizer]["bornOppenheimer"][
                "scfDiag"
            ] = self.get_scf_group()
        else:
            scf = self.input.sphinx.main.get("scfDiag", create=True)
            if self._generic_input["restart_for_band_structure"]:
                scf.append(self.get_scf_group(keepRhoFixed=True))
            else:
                scf.append(self.get_scf_group())
            if self.executable.version is not None:
                if self.get_version_float() > 2.5:
                    self.input.sphinx.main.create_group("evalForces")[
                        "file"
                    ] = '"relaxHist.sx"'
            else:
                warnings.warn("executable version could not be identified")

    def load_basis_group(self):
        """
        Load the basis Group.

        The group is populated using setdefault to avoid
        overwriting values that were previously (intentionally)
        modified.
        """
        self.input.sphinx.basis.setdefault("eCut", self.input["EnCut"] / RYDBERG_TO_EV)
        self.input.sphinx.basis.get("kPoint", create=True)
        if "KpointCoords" in self.input:
            self.input.sphinx.basis.kPoint.setdefault(
                "coords", np.array(self.input["KpointCoords"])
            )
        self.input.sphinx.basis.kPoint.setdefault("weight", 1)
        self.input.sphinx.basis.kPoint.setdefault("relative", True)
        if "KpointFolding" in self.input:
            self.input.sphinx.basis.setdefault(
                "folding", np.array(self.input["KpointFolding"])
            )
        self.input.sphinx.basis.setdefault("saveMemory", self.input["SaveMemory"])

    def load_hamilton_group(self):
        """
        Load the PAWHamiltonian Group.

        The group is populated using setdefault to avoid
        overwriting values that were previously (intentionally)
        modified.
        """
        self.input.sphinx.PAWHamiltonian.setdefault(
            "nEmptyStates", self.input["EmptyStates"]
        )
        self.input.sphinx.PAWHamiltonian.setdefault("ekt", self.input["Sigma"])
        for k in ["MethfesselPaxton", "FermiDirac"]:
            if k in self.input.list_nodes():
                self.input.sphinx.PAWHamiltonian.setdefault(k, self.input[k])
                break
        self.input.sphinx.PAWHamiltonian.setdefault("xc", self.input["Xcorr"])
        self.input.sphinx.PAWHamiltonian["spinPolarized"] = self._spin_enabled

    def load_guess_group(self, update_spins=True):
        """
        Load the initialGuess Group.

        The group is populated using setdefault to avoid
        overwriting values that were previously (intentionally)
        modified.

        Args:
            update_spins (bool): whether or not to reload the
                atomicSpin groups based on the latest structure.
                Defaults to True.
        """

        charge_density_file = None
        for ff in self.restart_file_list:
            if "rho.sxb" in ff.split("/")[-1]:
                charge_density_file = ff
        wave_function_file = None
        for ff in self.restart_file_list:
            if "waves.sxb" in ff.split("/")[-1]:
                wave_function_file = ff

        # introduce short alias for initialGuess group
        guess = self.input.sphinx.initialGuess

        guess.setdefault("waves", Group())
        guess.waves.setdefault("pawBasis", True)
        if wave_function_file is None:
            guess.waves.setdefault("lcao", Group())
        else:
            guess.waves.setdefault("file", '"' + wave_function_file + '"')
            # TODO: only for hybrid functionals
            guess.setdefault("exchange", Group())
            guess.exchange.setdefault("file", '"' + wave_function_file + '"')

        guess.setdefault("rho", Group())
        if charge_density_file is None:
            if wave_function_file is None:
                guess.rho.setdefault("atomicOrbitals", True)
                if self._spin_enabled:
                    init_spins = self.structure.get_initial_magnetic_moments()
                    # --- validate that initial spin moments are scalar
                    for spin in init_spins:
                        if isinstance(spin, list) or isinstance(spin, np.ndarray):
                            raise ValueError("SPHInX only supports collinear spins.")
                    guess.rho.get("atomicSpin", create=True)
                    if update_spins:
                        guess.rho.atomicSpin.clear()
                    # --- create initial spins if needed
                    if len(guess.rho.atomicSpin) == 0:
                        # set initial spin via label for each unique value of spin
                        # dict.from_keys (...).keys () deduplicates
                        for spin in dict.fromkeys(init_spins).keys():
                            guess.rho["atomicSpin"].append(
                                Group(
                                    {
                                        "label": '"spin_' + str(spin) + '"',
                                        "spin": str(spin),
                                    }
                                )
                            )
            else:
                guess.rho.setdefault("fromWaves", True)
        else:
            guess.rho.setdefault("file", '"' + charge_density_file + '"')

        if "noWavesStorage" not in guess:
            guess["noWavesStorage"] = not self.input["WriteWaves"]

    def calc_static(
        self,
        electronic_steps=100,
        algorithm=None,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
    ):
        """
        Setup the hamiltonian to perform a static SCF run.

        Loads defaults for all SPHInX input groups, including a static
        main Group.

        Args:
            electronic_steps (float): max # of electronic steps
            retain_electrostatic_potential:
            retain_charge_density:
            algorithm (str): CCG or blockCCG (not implemented)
            electronic_steps (int): maximum number of electronic steps
                which can be used to achieve convergence
        """
        if electronic_steps is not None:
            self.input["Estep"] = electronic_steps
        for arg in ["Istep", "dF", "dE"]:
            if arg in self.input:
                del self.input[arg]
        super(SphinxBase, self).calc_static(
            electronic_steps=electronic_steps,
            algorithm=algorithm,
            retain_charge_density=retain_charge_density,
            retain_electrostatic_potential=retain_electrostatic_potential,
        )
        self.load_default_groups()

    def calc_minimize(
        self,
        electronic_steps=60,
        ionic_steps=None,
        max_iter=None,
        pressure=None,
        algorithm=None,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
        ionic_energy=None,
        ionic_forces=None,
        ionic_energy_tolerance=None,
        ionic_force_tolerance=None,
        volume_only=False,
    ):
        """
        Setup the hamiltonian to perform ionic relaxations.

        The convergence goal can be set using either the
        ionic_energy_tolerance as a limit for fluctuations in energy or the
        ionic_force_tolerance.

        Loads defaults for all SPHInX input groups, including a
        ricQN-based main Group.

        .. warning::
            Sphinx does not support volume minimizations!  Calling this method with `pressure` or `volume_only` results
            in an error.

        Args:
            retain_electrostatic_potential:
            retain_charge_density:
            algorithm:
            pressure:
            max_iter:
            electronic_steps (int): maximum number of electronic steps
                                    per electronic convergence
            ionic_steps (int): maximum number of ionic steps
            ionic_energy (float): convergence goal in terms of
                                  energy (depreciated use ionic_energy_tolerance instead)
            ionic_energy_tolerance (float): convergence goal in terms of
                                  energy (optional)
            ionic_forces (float): convergence goal in terms of
                                  forces (depreciated use ionic_force_tolerance instead)
            ionic_force_tolerance (float): convergence goal in terms of
                                  forces (optional)
            volume_only (bool):
        """
        if pressure is not None or volume_only:
            raise NotImplementedError(
                "pressure minimization is not implemented in SPHInX"
            )
        if electronic_steps is not None:
            self.input["Estep"] = electronic_steps
        if ionic_steps is not None:
            self.input["Istep"] = ionic_steps
        elif "Istep" not in self.input:
            self.input["Istep"] = 100
        if ionic_force_tolerance is not None:
            if ionic_force_tolerance < 0:
                raise ValueError("ionic_force_tolerance must be a positive integer")
            self.input["dF"] = float(ionic_force_tolerance)
        if ionic_energy_tolerance is not None:
            if ionic_energy_tolerance < 0:
                raise ValueError("ionic_force_tolerance must be a positive integer")
            self.input["dE"] = float(ionic_energy_tolerance)
        super(SphinxBase, self).calc_minimize(
            electronic_steps=electronic_steps,
            ionic_steps=ionic_steps,
            max_iter=max_iter,
            pressure=pressure,
            algorithm=algorithm,
            retain_charge_density=retain_charge_density,
            retain_electrostatic_potential=retain_electrostatic_potential,
            ionic_energy_tolerance=ionic_energy_tolerance,
            ionic_force_tolerance=ionic_force_tolerance,
            volume_only=volume_only,
        )
        self.load_default_groups()

    def calc_md(
        self,
        temperature=None,
        n_ionic_steps=1000,
        n_print=1,
        time_step=1.0,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
        **kwargs,
    ):
        raise NotImplementedError("calc_md() not implemented in SPHInX.")

    def restart_for_band_structure_calculations(self, job_name=None):
        """
        Restart a new job created from an existing calculation
        by reading the charge density for band structures.

        Args:
            job_name (str/None): Job name

        Returns:
            pyiron_atomistics.sphinx.sphinx.sphinx: new job instance
        """
        return self.restart_from_charge_density(
            job_name=job_name, band_structure_calc=True
        )

    def restart_from_charge_density(
        self, job_name=None, job_type="Sphinx", band_structure_calc=False
    ):
        """
        Restart a new job created from an existing calculation
        by reading the charge density.

        Args:
            job_name (str/None): Job name
            job_type (str/None): Job type. If not specified a SPHInX job type is assumed (actually
                this is all that's currently supported)
            band_structure_calc (bool): has to be True for band structure calculations.

        Returns:
            pyiron_atomistics.sphinx.sphinx.sphinx: new job instance
        """
        ham_new = self.restart(
            job_name=job_name,
            job_type=job_type,
            from_wave_functions=False,
            from_charge_density=True,
        )
        if band_structure_calc:
            ham_new._generic_input["restart_for_band_structure"] = True
            # --- clean up minimization related settings
            for setting in ["Istep", "dF", "dE"]:
                if setting in ham_new.input:
                    del ham_new.input[setting]
            # remove optimization-related stuff from GenericDFTJob
            super(SphinxBase, self).calc_static()
            # --- recreate main group
            del ham_new.input.sphinx["main"]
            ham_new.input.sphinx.create_group("main")
            ham_new.load_main_group()
        return ham_new

    def restart_from_wave_functions(
        self,
        job_name=None,
        job_type="Sphinx",
    ):
        """
        Restart a new job created from an existing calculation
        by reading the wave functions.

        Args:
            job_name (str): Job name
            job_type (str): Job type. If not specified a SPHInX job type is assumed (actually
                this is all that's currently supported.)

        Returns:
            pyiron_atomistics.sphinx.sphinx.sphinx: new job instance
        """
        return self.restart(
            job_name=job_name,
            job_type=job_type,
            from_wave_functions=True,
            from_charge_density=False,
        )

    def restart(
        self,
        job_name=None,
        job_type=None,
        from_charge_density=True,
        from_wave_functions=True,
    ):
        if not self.status.finished and not self.is_compressed():
            # self.decompress()
            with warnings.catch_warnings(record=True) as w:
                try:
                    self.collect_output()
                except AssertionError as orig_error:
                    if from_charge_density or from_wave_functions:
                        raise AssertionError(
                            orig_error.message
                            + "\nCowardly refusing to use density or wavefunctions for restart.\n"
                            + "Solution: set from_charge_density and from_wave_functions to False."
                        )
                if len(w) > 0:
                    self.status.not_converged = True
        new_job = super(SphinxBase, self).restart(job_name=job_name, job_type=job_type)

        new_job.input = self.input.copy()

        recreate_guess = False
        if from_charge_density and os.path.isfile(
            posixpath.join(self.working_directory, "rho.sxb")
        ):
            new_job.restart_file_list.append(
                posixpath.join(self.working_directory, "rho.sxb")
            )
            del new_job.input.sphinx.initialGuess["rho"]
            recreate_guess = True

        elif from_charge_density:
            self._logger.warning(
                msg=f"A charge density from job: {self.job_name} "
                + "is not generated and therefore it can't be read."
            )
        if from_wave_functions and os.path.isfile(
            posixpath.join(self.working_directory, "waves.sxb")
        ):
            new_job.restart_file_list.append(
                posixpath.join(self.working_directory, "waves.sxb")
            )
            try:
                del new_job.input.sphinx.initialGuess["rho"]
            except KeyError:
                pass
            del new_job.input.sphinx.initialGuess["waves"]
            recreate_guess = True

        elif from_wave_functions:
            self._logger.warning(
                msg="No wavefunction file (waves.sxb) was found for "
                + f"job {self.job_name} in {self.working_directory}."
            )
        if recreate_guess:
            new_job.load_guess_group()
        return new_job

    def relocate_hdf5(self, h5_path=None):
        self.input._force_load()
        super().relocate_hdf5(h5_path=h5_path)

    def to_hdf(self, hdf=None, group_name=None):
        """
        Stores the instance attributes into the hdf5 file

        Args:
            hdf (str): Path to the hdf5 file
            group_name (str): Name of the group which contains the object
        """
        super(SphinxBase, self).to_hdf(hdf=hdf, group_name=group_name)
        self._structure_to_hdf()
        with self._hdf5.open("input") as hdf:
            self.input.to_hdf(hdf)
        self.output.to_hdf(self._hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Recreates instance from the hdf5 file

        Args:
            hdf (str): Path to the hdf5 file
            group_name (str): Name of the group which contains the object
        """
        if "HDF_VERSION" not in self._hdf5.keys():
            from pyiron_base import GenericParameters

            super(SphinxBase, self).from_hdf(hdf=hdf, group_name=group_name)
            self._structure_from_hdf()
            gp = GenericParameters(table_name="input")
            gp.from_hdf(self._hdf5)
            for k in gp.keys():
                self.input[k] = gp[k]
        elif self._hdf5["HDF_VERSION"] == "0.1.0":
            super(SphinxBase, self).from_hdf(hdf=hdf, group_name=group_name)
            self._structure_from_hdf()
            with self._hdf5.open("input") as hdf:
                self.input.from_hdf(hdf, group_name="parameters")
        self.output.from_hdf(self._hdf5)

    def from_directory(self, directory, file_name="structure.sx"):
        try:
            if not self.status.finished:
                file_path = posixpath.join(directory, file_name)
                if os.path.isfile(file_path):
                    self.structure = read_atoms(file_path)
                else:
                    raise ValueError(
                        f"File {file_path} not found. "
                        "Please double check the directory and file name."
                    )

                self.output.collect(directory=directory)
                self.to_hdf(self._hdf5)
            else:
                self.output.from_hdf(self._hdf5)
            self.status.finished = True
        except Exception as err:
            print(err)
            self.status.aborted = True

    def set_check_overlap(self, check_overlap=True):
        """
        Args:
            check_overlap (bool): Whether to check overlap

        Comments:
            Certain PAW-pseudo-potentials have an intrinsic pathology:
            their PAW overlap operator is not generally positive definite
            (i.e., the PAW-corrected norm of a wavefunction could become
            negative). SPHInX usually refuses to use such problematic
            potentials. This behavior can be overridden by setting
            check_overlap to False.
        """
        if not isinstance(check_overlap, bool):
            raise TypeError("check_overlap has to be a boolean")

        if self.get_version_float() < 2.51 and not check_overlap:
            warnings.warn(
                "SPHInX executable version has to be 2.5.1 or above "
                + "in order for the overlap to be considered. "
                + "Change it via job.executable.version"
            )
        self.input.CheckOverlap = check_overlap

    def set_mixing_parameters(
        self,
        method=None,
        n_pulay_steps=None,
        density_mixing_parameter=None,
        spin_mixing_parameter=None,
        density_residual_scaling=None,
        spin_residual_scaling=None,
    ):
        """
        Further information can be found on the website:
        https://sxrepo.mpie.de
        """
        method_list = ["PULAY", "KERKER", "LINEAR"]
        if method is not None and method.upper() not in method_list:
            raise ValueError("Mixing method has to be PULAY or KERKER")
        if method is not None:
            self.input["mixingMethod"] = method.upper().replace("KERKER", "LINEAR")
        if n_pulay_steps is not None:
            self.input["nPulaySteps"] = int(n_pulay_steps)
        if density_mixing_parameter is not None:
            if density_mixing_parameter > 1.0 or density_mixing_parameter < 0:
                raise ValueError(
                    "density_mixing_parameter has to be between 0 and 1 "
                    + "(default value is 1)"
                )
            self.input["rhoMixing"] = density_mixing_parameter
        if spin_mixing_parameter is not None:
            if spin_mixing_parameter > 1.0 or spin_mixing_parameter < 0:
                raise ValueError(
                    "spin_mixing_parameter has to be between 0 and 1 "
                    + "(default value is 1)"
                )
            self.input["spinMixing"] = spin_mixing_parameter
        if density_residual_scaling is not None:
            if density_residual_scaling <= 0:
                raise ValueError("density_residual_scaling must be a positive value")
            self.input["rhoResidualScaling"] = density_residual_scaling
        if spin_residual_scaling is not None:
            if spin_residual_scaling <= 0:
                raise ValueError("spin_residual_scaling   must be a positive value")
            self.input["spinResidualScaling"] = spin_residual_scaling

    set_mixing_parameters.__doc__ = (
        GenericDFTJob.set_mixing_parameters.__doc__ + set_mixing_parameters.__doc__
    )

    def set_occupancy_smearing(self, smearing=None, width=None, order=1):
        """
        Set how the finite temperature smearing is applied in
        determining partial occupancies

        Args:
            smearing (str): Type of smearing, 'FermiDirac' or 'MethfesselPaxton'
            width (float): Smearing width (eV) (default: 0.2)
            order (int): Smearing order
        """
        if smearing is not None:
            if not isinstance(smearing, str):
                raise ValueError("Smearing must be a string")
            if smearing.lower().startswith("meth"):
                self.input.MethfesselPaxton = order
                if "FermiDirac" in self.input.list_nodes():
                    del self.input["FermiDirac"]
            elif smearing.lower().startswith("fermi"):
                self.input.FermiDirac = order
                if "MethfesselPaxton" in self.input.list_nodes():
                    del self.input["MethfesselPaxton"]
        if width is not None and width < 0:
            raise ValueError("Smearing value must be a float >= 0")
        if width is not None:
            self.input["Sigma"] = width

    @deprecate(
        ionic_forces="Use ionic_force_tolerance",
        ionic_energy="use ionic_energy_tolerance",
    )
    def set_convergence_precision(
        self,
        ionic_energy_tolerance=None,
        ionic_force_tolerance=None,
        ionic_energy=None,
        electronic_energy=None,
        ionic_forces=None,
    ):
        """
        Sets the electronic and ionic convergence precision.

        For ionic convergence either the energy or the force
        precision is required.

        Args:
            ionic_energy (float): Ionic energy convergence precision
                                  (depreciated use ionic_energy_tolerance instead)
            ionic_energy_tolerance (float): Ionic energy convergence precision
            electronic_energy (float): Electronic energy convergence
                                       precision
            ionic_forces (float): Ionic force convergence precision
                                  (depreciated use ionic_force_tolerance instead)
            ionic_force_tolerance (float): Ionic force convergence precision
        """
        if ionic_forces is not None:
            ionic_force_tolerance = ionic_forces
        if ionic_energy is not None:
            ionic_energy_tolerance = ionic_energy
        assert (
            ionic_energy_tolerance is None or ionic_energy_tolerance > 0
        ), "ionic_energy_tolerance must be a positive float"
        assert (
            ionic_force_tolerance is None or ionic_force_tolerance > 0
        ), "ionic_force_tolerance must be a positive float"
        assert (
            electronic_energy is None or electronic_energy > 0
        ), "electronic_energy must be a positive float"
        if ionic_energy_tolerance is not None or ionic_force_tolerance is not None:
            # self.input["dE"] = ionic_energy_tolerance
            # self.input["dF"] = ionic_force_tolerance
            print("Setting calc_minimize")
            self.calc_minimize(
                ionic_energy_tolerance=ionic_energy_tolerance,
                ionic_force_tolerance=ionic_force_tolerance,
            )
        if electronic_energy is not None:
            self.input["Ediff"] = electronic_energy

    def set_empty_states(self, n_empty_states=None):
        """
        Function to set the number of empty states.

        Args:
            n_empty_states (int/None): Number of empty states.
            If None, sets it to 'auto'.

        Comments:
            If this number is too low, the algorithm will not be
            able to able to swap wave functions near the chemical
            potential. If the number is too high, computation time
            will be wasted for the higher energy states and
            potentially lead to a memory problem.

            In contrast to VASP, this function sets only the number
            of empty states and not the number of total states.

            The default value is 0.5*NIONS+3 for non-magnetic systems
            and 1.5*NIONS+3 for magnetic systems
        """
        if n_empty_states is None:
            # will be converted later; see load_default_groups
            self.input["EmptyStates"] = "auto"
        else:
            if n_empty_states < 0:
                raise ValueError("Number of empty states must be greater than 0")
            self.input["EmptyStates"] = n_empty_states
        self.input.sphinx.PAWHamiltonian.nEmptyStates = self.input["EmptyStates"]

    def _set_kpoints(
        self,
        mesh=None,
        scheme="MP",
        center_shift=None,
        symmetry_reduction=True,
        manual_kpoints=None,
        weights=None,
        reciprocal=True,
        n_path=None,
        path_name=None,
    ):
        """
        Function to setup the k-points for the SPHInX job

        Args:
            reciprocal (bool): Tells if the supplied values are in
                               reciprocal (direct) or cartesian coordinates
                               (in reciprocal space) (not implemented)
            weights (list): Manually supplied weights to each k-point in
                            case of the manual mode (not implemented)
            manual_kpoints (list): Manual list of k-points (not implemented)
            symmetry_reduction (bool): Tells if the symmetry reduction is
                                       to be applied to the k-points
            scheme (str): Type of k-point generation scheme ('MP' or 'Line')
            mesh (list): Size of the mesh (in the MP scheme)
            center_shift (list): Shifts the center of the mesh from the
                                 gamma point by the given vector
            n_path (int): Number of points per trace part for line mode
            path_name (str): Name of high symmetry path used for band
                             structure calculations.
        """
        if not isinstance(symmetry_reduction, bool):
            raise ValueError("symmetry_reduction has to be a boolean")
        if manual_kpoints is not None:
            raise ValueError(
                "manual_kpoints is not yet implemented in pyiron for SPHInX"
            )
        if weights is not None:
            raise ValueError(
                "manual weights are not yet implmented in Pyiron for " + "SPHInX"
            )

        if scheme == "MP":
            # Remove kPoints and set kPoint
            if "kPoints" in self.input.sphinx.basis:
                del self.input.sphinx.basis.kPoints
            self.input.sphinx.basis.get("kPoint", create=True)
            if mesh is not None:
                self.input["KpointFolding"] = list(mesh)
                self.input.sphinx.basis["folding"] = np.array(
                    self.input["KpointFolding"]
                )
            if center_shift is not None:
                self.input["KpointCoords"] = list(center_shift)
                self.input.sphinx.basis["kPoint"]["coords"] = np.array(
                    self.input["KpointCoords"]
                )
                self.input.sphinx.basis.kPoint["weight"] = 1
                self.input.sphinx.basis.kPoint["relative"] = True

        elif scheme == "Line":
            # Remove Kpoint and set Kpoints

            if "kPoint" in self.input.sphinx.basis:
                del self.input.sphinx.basis["kPoint"]
                del self.input["KpointFolding"]
                del self.input["KpointCoords"]
                if "folding" in self.input.sphinx.basis:
                    del self.input.sphinx.basis["folding"]
            if n_path is None and self._generic_input["n_path"] is None:
                raise ValueError("'n_path' has to be defined")
            if n_path is None:
                n_path = self._generic_input["n_path"]
            else:
                self._generic_input["n_path"] = n_path

            if self.structure.get_high_symmetry_points() is None:
                raise ValueError("no 'high_symmetry_points' defined for 'structure'.")

            if path_name is None and self._generic_input["path_name"] is None:
                raise ValueError("'path_name' has to be defined")
            if path_name is None:
                path_name = self._generic_input["path_name"]
            else:
                self._generic_input["path_name"] = path_name

            try:
                path = self.structure.get_high_symmetry_path()[path_name]
            except KeyError:
                raise AssertionError("'{}' is not a valid path!".format(path_name))

            def make_point(point, n_path):
                return Group(
                    {
                        "coords": np.array(
                            self.structure.get_high_symmetry_points()[point]
                        ),
                        "nPoints": n_path,
                        "label": '"{}"'.format(point.replace("'", "p")),
                    }
                )

            kpoints = Group({"relative": True})
            kpoints["from"] = make_point(path[0][0], None)
            # from nodes are not supposed to have a nPoints attribute
            del kpoints["from/nPoints"]

            kpoints.create_group("to").append(make_point(path[0][1], n_path))

            for segment in path[1:]:
                # if the last node on the so far is not the same as the first
                # node of this path segment, then we need to insert another
                # node into the path to alert sphinx that we want a cut in our
                # band structure (n_path = 0)
                if '"{}"'.format(segment[0]) != kpoints.to[-1].label:
                    kpoints["to"].append(make_point(segment[0], 0))

                kpoints["to"].append(make_point(segment[1], n_path))

            self.input.sphinx.basis["kPoints"] = kpoints
        else:
            raise ValueError(
                "only Monkhorst-Pack mesh and Line mode\
                are currently implemented in Pyiron for SPHInX"
            )

    def load_default_groups(self):
        """
        Populates input groups with the default values.

        Nearly every default simply points to a variable stored in
        self.input.

        Does not load job.input.structure or job.input.pawPot.
        These groups should usually be modified via job.structure,
        in which case they will be set at the last minute when
        the job is run. These groups can be synced to job.structure
        at any time using job.load_structure_group() and
        job.load_species_group().
        """

        if self.structure is None:
            raise AssertionError(
                f"{self.job_name} has not been assigned "
                + "a structure. Please load one first (e.g. "
                + f"{self.job_name}.structure = ...)"
            )

        if self.input["EmptyStates"] == "auto":
            if self._spin_enabled:
                self.input["EmptyStates"] = int(1.5 * len(self.structure) + 3)
            else:
                self.input["EmptyStates"] = int(len(self.structure) + 3)

        if not self.input.sphinx.basis.read_only:
            self.load_basis_group()
        if not self.input.sphinx.structure.read_only:
            self.load_structure_group()
        if self.input["VaspPot"]:
            potformat = "VASP"
        else:
            potformat = "JTH"
        if not self.input.sphinx.pawPot.read_only:
            self.load_species_group(
                check_overlap=self.input.CheckOverlap, potformat=potformat
            )
        if not self.input.sphinx.initialGuess.read_only:
            self.load_guess_group()
        if not self.input.sphinx.PAWHamiltonian.read_only:
            self.load_hamilton_group()
        if not self.input.sphinx.main.read_only:
            self.load_main_group()

    def list_potentials(self):
        """
        Lists all the possible POTCAR files for the elements in the structure depending on the XC functional

        Returns:
           list: a list of available potentials
        """
        return self.potential_list

    def write_input(self):
        """
        Generate all the required input files for the SPHInX job.

        Creates:
        structure.sx: structure associated w/ job
        all pseudopotential files
        spins.in (if necessary): constrained spin moments
        input.sx: main input file with all sub-groups

        Automatically called by job.run()
        """

        # If the structure group was not modified directly by the
        # user, via job.input.structure (which is likely True),
        # load it based on job.structure.
        structure_sync = str(self.input.sphinx.structure) == str(
            self.get_structure_group()
        )
        if not structure_sync and not self.input.sphinx.structure.read_only:
            self.load_structure_group()

        # copy potential files to working directory
        if self.input["VaspPot"]:
            potformat = "VASP"
        else:
            potformat = "JTH"
        # If the species group was not modified directly by the user,
        # via job.input.pawPot (which is likely True),
        # load it based on job.structure.
        if not structure_sync and not self.input.sphinx.pawPot.read_only:
            self.load_species_group(
                check_overlap=self.input.CheckOverlap, potformat=potformat
            )

        modified_elements = {
            key: value
            for key, value in self._potential.to_dict().items()
            if value is not None
        }

        self.input_writer.structure = self.structure
        self.input_writer.copy_potentials(
            potformat=potformat,
            xc=self.input["Xcorr"],
            cwd=self.working_directory,
            modified_elements=modified_elements,
        )

        # Write spin constraints, if set via _generic_input.
        all_groups = [
            self.input.sphinx.pawPot,
            self.input.sphinx.structure,
            self.input.sphinx.basis,
            self.input.sphinx.PAWHamiltonian,
            self.input.sphinx.initialGuess,
            self.input.sphinx.main,
        ]

        if self._generic_input["fix_spin_constraint"]:
            self.input.sphinx.spinConstraint = Group()
            all_groups.append(self.input.sphinx.spinConstraint)
            self.input_writer.write_spin_constraints(cwd=self.working_directory)
            self.input.sphinx.spinConstraint.setdefault("file", '"spins.in"')

        # In case the entire group was
        # set/overwritten as a normal dict.
        for group in all_groups:
            group = Group(group)

        # write input.sx
        file_name = posixpath.join(self.working_directory, "input.sx")
        with open(file_name, "w") as f:
            f.write(f"//{self.job_name}\n")
            f.write("//SPHInX input file generated by pyiron\n\n")
            f.write("format paw;\n")
            f.write("include <parameters.sx>;\n\n")
            f.write(self.input.sphinx.to_sphinx(indent=0))

    @property
    def _spin_enabled(self):
        return self.structure.has("initial_magmoms")

    def get_charge_density(self):
        """
        Gets the charge density from the hdf5 file. This value is normalized by the volume

        Returns:
            pyiron_atomistics.atomistics.volumetric.generic.VolumetricData
        """
        if self.status not in job_status_successful_lst:
            return
        else:
            with self.project_hdf5.open("output") as ho:
                cd_obj = SphinxVolumetricData()
                cd_obj.from_hdf(ho, "charge_density")
            cd_obj.atoms = self.get_structure(-1)
            return cd_obj

    def get_electrostatic_potential(self):
        """
        Gets the electrostatic potential from the hdf5 file.

        Returns:
            pyiron_atomistics.atomistics.volumetric.generic.VolumetricData
        """
        if self.status not in job_status_successful_lst:
            return
        else:
            with self.project_hdf5.open("output") as ho:
                es_obj = SphinxVolumetricData()
                es_obj.from_hdf(ho, "electrostatic_potential")
            es_obj.atoms = self.get_structure(-1)
            return es_obj

    def collect_output(self, force_update=False, compress_files=True):
        """
        Collects the outputs and stores them to the hdf file
        """
        if self.is_compressed():
            warnings.warn("Job already compressed - output not collected")
            return
        self.output.collect(directory=self.working_directory)
        self.output.to_hdf(self._hdf5, force_update=force_update)
        if compress_files:
            self.compress()

    def convergence_check(self):
        """
        Checks for electronic and ionic convergence according to the user specified tolerance

        Returns:

            bool: True if converged

        """
        # Checks if sufficient empty states are present
        if not self.nbands_convergence_check():
            return False
        return self.output.generic.dft.scf_convergence[-1]

    def collect_logfiles(self):
        """
        Collect errors and warnings.
        """
        self.collect_errors()
        self.collect_warnings()

    def collect_warnings(self):
        """
        Collects warnings from the SPHInX run
        """
        self._logger.info(
            "collect_warnings() is not yet \
            implemented for SPHInX"
        )

    def collect_errors(self):
        """
        Collects errors from the SPHInX run
        """
        self._logger.info("collect_errors() is not yet implemented for SPHInX")

    def get_n_ir_reciprocal_points(
        self, is_time_reversal=True, symprec=1e-5, ignore_magmoms=False
    ):
        lattice = self.structure.cell
        positions = self.structure.get_scaled_positions()
        numbers = self.structure.get_atomic_numbers()
        if ignore_magmoms:
            magmoms = np.zeros(len(positions))
        else:
            magmoms = self.structure.get_initial_magnetic_moments()
        mag_num = np.array(list(zip(magmoms, numbers)))
        satz = np.unique(mag_num, axis=0)
        numbers = []
        for nn in np.all(satz == mag_num[:, np.newaxis], axis=-1):
            numbers.append(np.arange(len(satz))[nn][0])
        mapping, _ = spglib.get_ir_reciprocal_mesh(
            mesh=[int(self.input["KpointFolding"][k]) for k in range(3)],
            cell=(lattice, positions, numbers),
            is_shift=np.dot(self.structure.cell, np.array(self.input["KpointCoords"])),
            is_time_reversal=is_time_reversal,
            symprec=symprec,
        )
        return len(np.unique(mapping))

    def check_setup(self):
        with warnings.catch_warnings(record=True) as w:
            # Check for parameters that were not modified but
            # possibly should have (encut, kpoints, smearing, etc.),
            # or were set to nonsensical values.

            if (
                not (
                    isinstance(self.input.sphinx.basis["eCut"], int)
                    or isinstance(self.input.sphinx.basis["eCut"], float)
                )
                or round(self.input.sphinx.basis["eCut"] * RYDBERG_TO_EV, 0) == 340
            ):
                warnings.warn(
                    "Energy cut-off value wrong or not modified from default "
                    + "340 eV; change it via job.set_encut()"
                )
            if not (
                isinstance(self.input.sphinx.basis["kPoint"]["coords"], np.ndarray)
                or len(self.input.sphinx.basis["kPoint"]["coords"]) != 3
            ):
                warnings.warn("K point coordinates seem to be inappropriate")
            if (
                not (
                    isinstance(self.input.sphinx.PAWHamiltonian["ekt"], int)
                    or isinstance(self.input.sphinx.PAWHamiltonian["ekt"], float)
                )
                or round(self.input.sphinx.PAWHamiltonian["ekt"], 1) == 0.2
            ):
                warnings.warn(
                    "Fermi smearing value wrong or not modified from default "
                    + "0.2 eV; change it via job.set_occupancy_smearing()"
                )
            if not (
                isinstance(self.input.sphinx.basis["folding"], np.ndarray)
                or len(self.input.sphinx.basis["folding"]) != 3
            ) or self.input.sphinx.basis["folding"].tolist() == [4, 4, 4]:
                warnings.warn(
                    "K point folding wrong or not modified from default "
                    + "[4,4,4]; change it via job.set_kpoints()"
                )
            if self.get_n_ir_reciprocal_points() < self.server.cores:
                warnings.warn(
                    "Number of cores exceed number of irreducible "
                    + "reciprocal points: "
                    + str(self.get_n_ir_reciprocal_points())
                )
            if self.input["EmptyStates"] == "auto":
                if self._spin_enabled:
                    warnings.warn(
                        "Number of empty states was not specified. Default: "
                        + "3+NIONS*1.5 for magnetic systems. "
                    )
                else:
                    warnings.warn(
                        "Number of empty states was not specified. Default: "
                        + "3+NIONS for non-magnetic systems"
                    )

            return len(w) == 0

    def validate_ready_to_run(self):
        """
        Checks whether parameters are set appropriately. It does not
        mean the simulation won't run if it returns False.
        """

        all_groups = {
            "job.input.pawPot": self.input.sphinx.pawPot,
            "job.input.structure": self.input.sphinx.structure,
            "job.input.basis": self.input.sphinx.basis,
            "job.input.PAWHamiltonian": self.input.sphinx.PAWHamiltonian,
            "job.input.initialGuess": self.input.sphinx.initialGuess,
            "job.input.main": self.input.sphinx.main,
        }

        if np.any([len(all_groups[group]) == 0 for group in all_groups]):
            self.load_default_groups()

        if self.structure is None:
            raise AssertionError(
                "Structure not set; set it via job.structure = "
                + "Project().create_structure()"
            )
        if self.input["THREADS"] > self.server.cores:
            raise AssertionError(
                "Number of cores cannot be smaller than the number "
                + "of OpenMP threads"
            )
        with warnings.catch_warnings(record=True) as w:
            # Warn about discrepancies between values in
            # self.input and individual groups, in case
            # a user modified them directly
            if round(self.input["EnCut"], 0) != round(
                self.input.sphinx.basis.eCut * RYDBERG_TO_EV, 0
            ):
                warnings.warn(
                    "job.input.basis.eCut was modified directly. "
                    "It is recommended to set it via job.set_encut()"
                )

            if round(self.input["Sigma"], 1) != round(
                self.input.sphinx.PAWHamiltonian.ekt, 1
            ):
                warnings.warn(
                    "job.input.PAWHamiltonian.ekt was modified directly. "
                    "It is recommended to set it via "
                    "job.set_occupancy_smearing()"
                )

            if self.input["Xcorr"] != self.input.sphinx.PAWHamiltonian.xc:
                warnings.warn(
                    "job.input.PAWHamiltonian.xc was modified directly. "
                    "It is recommended to set it via "
                    "job.exchange_correlation_functional()"
                )

            if (
                self.input["EmptyStates"]
                != self.input.sphinx.PAWHamiltonian.nEmptyStates
            ):
                warnings.warn(
                    "job.input.PAWHamiltonian.nEmptyStates was modified "
                    "directly. It is recommended to set it via "
                    "job.set_empty_states()"
                )

            if (
                "KpointCoords" in self.input
                and np.array(self.input.KpointCoords).tolist()
                != np.array(self.input.sphinx.basis.kPoint.coords).tolist()
            ) or (
                "KpointFolding" in self.input
                and np.array(self.input.KpointFolding).tolist()
                != np.array(self.input.sphinx.basis.folding).tolist()
            ):
                warnings.warn(
                    "job.input.basis.kPoint was modified directly. "
                    "It is recommended to set all k-point settings via "
                    "job.set_kpoints()"
                )

            structure_sync = str(self.input.sphinx.structure) == str(
                self.get_structure_group()
            )
            if not structure_sync and not self.input.sphinx.structure.read_only:
                warnings.warn(
                    "job.input.structure != job.structure. "
                    "The current job.structure will overwrite "
                    "any changes you may might have made to "
                    "job.input.structure in the meantime. "
                    "To disable this overwrite, "
                    "set job.input.structure.read_only = True. "
                    "To disable this warning, call "
                    "job.load_structure_group() after making changes "
                    "to job.structure."
                )

            if len(w) > 0:
                print("WARNING:")
                for ww in w:
                    print(ww.message)
                return False
            else:
                return True

    def compress(self, files_to_compress=None):
        """
        Compress the output files of a job object.

        Args:
            files_to_compress (list): A list of files to compress (optional)
        """
        # delete empty files
        if files_to_compress is None:
            files_to_compress = [
                f
                for f in list(self.list_files())
                if (
                    f not in ["rho.sxb", "waves.sxb"]
                    and not stat.S_ISFIFO(
                        os.stat(os.path.join(self.working_directory, f)).st_mode
                    )
                )
            ]
        for f in list(self.list_files()):
            filename = os.path.join(self.working_directory, f)
            if (
                f not in files_to_compress
                and os.path.exists(filename)
                and os.stat(filename).st_size == 0
            ):
                os.remove(filename)
        super(SphinxBase, self).compress(files_to_compress=files_to_compress)

    @staticmethod
    def check_vasp_potentials():
        return any(
            [
                os.path.exists(
                    os.path.join(p, "vasp", "potentials", "potpaw", "Fe", "POTCAR")
                )
                for p in state.settings.resource_paths
            ]
        )

    def run_addon(
        self,
        addon,
        args=None,
        from_tar=None,
        silent=False,
        log=True,
        version=None,
        debug=False,
    ):
        """Run a SPHInX addon

        addon          - name of addon (str)
        args           - arguments (str or list)
        from_tar       - if job is compressed, extract these files (list)
        silent         - do not print output for successful runs?
        log            - produce log file?
        version        - which sphinx version to load (str or None)
        debug          - return subprocess.CompletedProcess ?

        """
        if self.is_compressed() and from_tar is None:
            raise FileNotFoundError(
                "Cannot run add-on on compressed job without 'from_tar' parameter.\n"
                + "   Solution 1: Run .decompress () first.\n"
                + "   Solution 2: specify from_tar list to run in temporary directory\n"
                + "   Solution 3: run with from_tar=[] if no files from tar are needed\n"
            )

        # prepare argument list
        if args is None:
            args = ""
        elif isinstance(args, list):
            args = " ".join(args)
        if log:
            args += " --log"

        cmd = addon + " " + args

        # --- handle versions
        sxv = sxversions()
        if (
            version is None
            and self.executable.version is not None
            and self.executable.version in sxv.keys()
        ):
            version = self.executable.version
            if not silent:
                print("Taking version '" + version + "' from job._executable version")
        if isinstance(version, str):
            if version in sxv.keys():
                cmd = sxv[version] + " && " + cmd
            elif version != "":
                raise KeyError(
                    "version '"
                    + version
                    + "' not found. Available versions are: '"
                    + "', '".join(sxv.keys())
                    + "'."
                )
            # version="" overrides job.executable_version
        elif version is not None:
            raise TypeError("version must be str or None")

        if isinstance(from_tar, str):
            from_tar = [from_tar]
        if self.is_compressed() and isinstance(from_tar, list):
            # run addon in temporary directory
            with TemporaryDirectory() as tempd:
                if not silent:
                    print("Running {} in temporary directory {}".format(addon, tempd))

                # --- extract files from list
                # note: tf should be obtained from JobCore to ensure encapsulation
                tarfilename = os.path.join(
                    self.working_directory, self.job_name + ".tar.bz2"
                )
                with tarfile.open(tarfilename, "r:bz2") as tf:
                    for file in from_tar:
                        try:
                            tf.extract(file, path=tempd)
                        except:
                            print("Cannot extract " + file + " from " + tarfilename)

                # --- link other files
                linkfiles = []
                for file in self.list_files():
                    linkfile = os.path.join(tempd, file)
                    if not os.path.isfile(linkfile):
                        os.symlink(
                            os.path.join(self.working_directory, file),
                            linkfile,
                            target_is_directory=True,
                        )
                        linkfiles.append(linkfile)
                # now run
                out = subprocess.run(
                    cmd, cwd=tempd, shell=True, stdout=PIPE, stderr=PIPE, text=True
                )

                # now clean tempdir
                for file in from_tar:
                    try:
                        os.remove(os.path.join(tempd, file))
                    except FileNotFoundError:
                        pass
                for linkfile in linkfiles:
                    if os.path.islink:
                        os.remove(linkfile)

                # move output to working directory for successful runs
                if out.returncode == 0:
                    for file in os.listdir(tempd):
                        movefile(os.path.join(tempd, file), self.working_directory)
                        if not silent:
                            print("Copying " + file + " to " + self.working_directory)
                else:
                    print(addon + " crashed - potential output files are not kept.")

        else:
            out = subprocess.run(
                cmd,
                cwd=self.working_directory,
                shell=True,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )
            if out.returncode != 0:
                print(addon + " crashed.")

        # print output
        if not silent or out.returncode != 0:
            if out.returncode != 0:
                print(addon + " output:\n\n")
            print(out.stdout)
            if out.returncode != 0:
                print(out.stderr)
        return out if debug else None


def get_structure_group(structure, use_symmetry=True, keep_angstrom=False):
    """
    create a SPHInX Group object based on structure

    Args:
        structure (pyiron_atomistics.atomistics.structure.atoms) structure
        use_symmetry (bool): Whether or not consider internal symmetry
        keep_angstrom (bool): Store distances in Angstroms or Bohr

    Returns:
        (Group): structure group
    """
    if keep_angstrom:
        structure_group = Group({"cell": np.array(structure.cell)})
    else:
        structure_group = Group(
            {
                "cell": np.array(structure.cell * 1 / BOHR_TO_ANGSTROM),
            }
        )
    if "selective_dynamics" in structure.arrays:
        selective_dynamics_list = list(structure.selective_dynamics)
    else:
        selective_dynamics_list = [3 * [False]] * len(structure.positions)
    species = structure_group.create_group("species")
    for elm_species in structure.get_species_objects():
        if elm_species.Parent:
            element = elm_species.Parent
        else:
            element = elm_species.Abbreviation
        species.append(Group({"element": '"' + str(element) + '"'}))
        elm_list = np.array(
            structure.get_chemical_symbols() == elm_species.Abbreviation
        )
        atom_group = species[-1].create_group("atom")
        for elm_pos, elm_magmon, selective in zip(
            structure.positions[elm_list],
            np.array(structure.get_initial_magnetic_moments())[elm_list],
            np.array(selective_dynamics_list)[elm_list],
        ):
            atom_group.append(Group())
            if structure.has("initial_magmoms"):
                atom_group[-1]["label"] = '"spin_' + str(elm_magmon) + '"'
            if keep_angstrom:
                atom_group[-1]["coords"] = np.array(elm_pos)
            else:
                atom_group[-1]["coords"] = np.array(elm_pos * 1 / BOHR_TO_ANGSTROM)
            if all(selective):
                atom_group[-1]["movable"] = True
            elif any(selective):
                for ss, xx in zip(selective, ["X", "Y", "Z"]):
                    if ss:
                        atom_group[-1]["movable" + xx] = True
    if not use_symmetry:
        structure_group.symmetry = Group(
            {"operator": {"S": "[[1,0,0],[0,1,0],[0,0,1]]"}}
        )
    return structure_group


class InputWriter(object):
    """
    The SPHInX Input writer is called to write the
    SPHInX specific input files.
    """

    def __init__(self):
        self.structure = None
        self._id_pyi_to_spx = []
        self._id_spx_to_pyi = []
        self.file_dict = {}

    def copy_potentials(
        self,
        potformat="JTH",
        xc=None,
        cwd=None,
        pot_path_dict=None,
        modified_elements=None,
    ):
        """
        Copy potential files

        Args:
            potformat (str):
            xc (str/None):
            cwd (str/None):
            pot_path_dict (dict):
            modified_elements (dict):
        """

        if pot_path_dict is None:
            pot_path_dict = {}

        if potformat == "JTH":
            potentials = SphinxJTHPotentialFile(xc=xc)
            find_potential_file = find_potential_file_jth
            pot_path_dict.setdefault("PBE", "jth-gga-pbe")
        elif potformat == "VASP":
            potentials = VaspPotentialFile(xc=xc)
            find_potential_file = find_potential_file_vasp
            pot_path_dict.setdefault("PBE", "paw-gga-pbe")
            pot_path_dict.setdefault("LDA", "paw-lda")
        else:
            raise ValueError("Only JTH and VASP potentials are supported!")

        for species_obj in self.structure.get_species_objects():
            if species_obj.Parent is not None:
                elem = species_obj.Parent
            else:
                elem = species_obj.Abbreviation

            if "pseudo_potcar_file" in species_obj.tags.keys():
                new_element = species_obj.tags["pseudo_potcar_file"]
                potentials.add_new_element(parent_element=elem, new_element=new_element)
                potential_path = find_potential_file(
                    path=potentials.find_default(new_element)["Filename"].values[0][0]
                )
                assert os.path.isfile(
                    potential_path
                ), "such a file does not exist in the pp directory"
            elif elem in modified_elements.keys():
                new_element = modified_elements[elem]
                if os.path.isabs(new_element):
                    potential_path = new_element
                else:
                    potentials.add_new_element(
                        parent_element=elem, new_element=new_element
                    )
                    potential_path = find_potential_file(
                        path=potentials.find_default(new_element)["Filename"].values[0][
                            0
                        ]
                    )
            else:
                potential_path = find_potential_file(
                    path=potentials.find_default(elem)["Filename"].values[0][0]
                )
            if potformat == "JTH":
                copyfile(potential_path, posixpath.join(cwd, elem + "_GGA.atomicdata"))
            else:
                copyfile(potential_path, posixpath.join(cwd, elem + "_POTCAR"))

    @property
    def id_spx_to_pyi(self):
        if self.structure is None:
            return None
        if len(self._id_spx_to_pyi) == 0:
            self._initialize_order()
        return self._id_spx_to_pyi

    @property
    def id_pyi_to_spx(self):
        if self.structure is None:
            return None
        if len(self._id_pyi_to_spx) == 0:
            self._initialize_order()
        return self._id_pyi_to_spx

    def _initialize_order(self):
        for elm_species in self.structure.get_species_objects():
            self._id_pyi_to_spx.append(
                np.arange(len(self.structure))[
                    self.structure.get_chemical_symbols() == elm_species.Abbreviation
                ]
            )
        self._id_pyi_to_spx = np.array(
            [ooo for oo in self._id_pyi_to_spx for ooo in oo]
        )
        self._id_spx_to_pyi = np.array([0] * len(self._id_pyi_to_spx))
        for i, p in enumerate(self._id_pyi_to_spx):
            self._id_spx_to_pyi[p] = i

    def write_spin_constraints(self, file_name="spins.in", cwd=None, spins_list=None):
        """
        Write a text file containing a list of all spins named spins.in -
        which is used for the external control scripts.

        Args:
            file_name (str): name of the file to be written (optional)
            cwd (str): the current working directory (optinal)
            spins_list (list): the input to write, if no input is
                given the default input will be written. (optional)
        """
        state.logger.debug(f"Writing {file_name}")
        state.logger.debug(
            "Getting magnetic moments via \
            get_initial_magnetic_moments"
        )
        if self.structure.has("initial_magmoms"):
            if any(
                [
                    True
                    if isinstance(spin, list) or isinstance(spin, np.ndarray)
                    else False
                    for spin in self.structure.get_initial_magnetic_moments()
                ]
            ):
                raise ValueError("SPHInX only supports collinear spins at the moment.")
            else:
                constraint = self.structure.spin_constraint[self.id_pyi_to_spx]
                if spins_list is None or len(spins_list) == 0:
                    spins_list = self.structure.get_initial_magnetic_moments()
                spins = spins_list[self.id_pyi_to_spx].astype(str)
                spins[~np.asarray(constraint)] = "X"
                spins_str = "\n".join(spins) + "\n"
                if cwd is not None:
                    file_name = posixpath.join(cwd, file_name)
                with open(file_name, "w") as f:
                    f.write(spins_str)
        else:
            state.logger.debug("No magnetic moments")


class Group(DataContainer):
    """
    Dictionary-like object to store SPHInX inputs.

    Attributes (sub-groups, parameters, & flags) can be set
    and accessed via dot notation, or as standard dictionary
    key/values.

    `to_{job_type}` converts the Group to the format
    expected by the given DFT code in its input files.
    """

    def set(self, name, content):
        self[name] = content

    def set_group(self, name, content=None):
        """
        Set a new group in SPHInX input.

        Args:
            name (str): name of the group
            content: content to append

        This creates an input group of the type `name { content }`.
        """
        if content is None:
            self.create_group(name)
        else:
            self.set(name, content)

    def set_flag(self, flag, val=True):
        """
        Set a new flag in SPHInX input.

        Args:
            flag (str): name of the flag
            val (bool): boolean value

        This creates an input flag of the type `name = val`.
        """
        self.set(flag, val)

    def set_parameter(self, parameter, val):
        """
        Set a new parameter in SPHInX input.

        Args:
            parameter (str): name of the flag
            val (float): parameter value

        This creates an input parameter of the type `parameter = val`.
        """
        self.set(parameter, val)

    def remove(self, name):
        if name in self.keys():
            del self[name]

    def to_sphinx(self, content="__self__", indent=0):
        if content == "__self__":
            content = self

        def format_value(v):
            if isinstance(v, bool):
                if v:
                    return ";"
                else:
                    return " = false;"
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


def splitter(arr, counter):
    if len(arr) == 0 or len(counter) == 0:
        return []
    arr_new = []
    spl_loc = list(np.where(np.array(counter) == min(counter))[0])
    spl_loc.append(None)
    for ii, ll in enumerate(spl_loc[:-1]):
        arr_new.append(np.array(arr[ll : spl_loc[ii + 1]]).tolist())
    return arr_new


class _SphinxLogParser:
    def __init__(self, log_file):
        self.log_file = log_file
        self._check_enter_scf()
        self._log_main = None
        self._counter = None
        self._n_atoms = None
        self._n_steps = None
        self._spin_enabled = None

    @property
    def spin_enabled(self):
        return len(re.findall("The spin for the label", self.log_file)) > 0

    @property
    def log_main(self):
        if self._log_main is None:
            match = re.search("Enter Main Loop", self.log_file)
            self._log_main = match.end() + 1
        return self.log_file[self._log_main :]

    @property
    def job_finished(self):
        if (
            len(re.findall("Program exited normally.", self.log_file, re.MULTILINE))
            == 0
        ):
            warnings.warn("scf loops did not converge")
            return False
        return True

    def _check_enter_scf(self):
        if len(re.findall("Enter Main Loop", self.log_file, re.MULTILINE)) == 0:
            raise AssertionError("Log file created but first scf loop not reached")

    def get_n_valence(self):
        log = self.log_file.split("\n")
        return {
            log[ii - 1].split()[1]: int(ll.split("=")[-1])
            for ii, ll in enumerate(log)
            if ll.startswith("| Z=")
        }

    @property
    def log_k_points(self):
        start_match = re.search(
            "-ik-     -x-      -y-       -z-    \|  -weight-    -nG-    -label-",
            self.log_file,
        )
        log_part = self.log_file[start_match.end() + 1 :]
        log_part = log_part[: re.search("^\n", log_part, re.MULTILINE).start()]
        return log_part.split("\n")[:-2]

    def get_bands_k_weights(self):
        return np.array([float(kk.split()[6]) for kk in self.log_k_points])

    @property
    def _rec_cell(self):
        log_extract = re.findall("b[1-3]:.*$", self.log_file, re.MULTILINE)
        return (
            np.array([ll.split()[1:4] for ll in log_extract]).astype(float)
            / BOHR_TO_ANGSTROM
        )[:3]

    def get_kpoints_cartesian(self):
        return np.einsum("ni,ij->nj", self.k_points, self._rec_cell)

    @property
    def k_points(self):
        return np.array(
            [[float(kk.split()[i]) for i in range(2, 5)] for kk in self.log_k_points]
        )

    def get_volume(self):
        volume = re.findall("Omega:.*$", self.log_file, re.MULTILINE)
        if len(volume) > 0:
            volume = float(volume[0].split()[1])
            volume *= BOHR_TO_ANGSTROM**3
        else:
            volume = 0
        return np.array(self.n_steps * [volume])

    @property
    def counter(self):
        if self._counter is None:
            self._counter = [
                int(re.sub("[^0-9]", "", line.split("=")[0]))
                for line in re.findall("F\(.*$", self.log_main, re.MULTILINE)
            ]
        return self._counter

    def get_energy_free(self):
        return splitter(
            [
                float(line.split("=")[1]) * HARTREE_TO_EV
                for line in re.findall("F\(.*$", self.log_main, re.MULTILINE)
            ],
            self.counter,
        )

    def get_energy_int(self):
        return splitter(
            [
                float(line.replace("=", " ").replace(",", " ").split()[1])
                * HARTREE_TO_EV
                for line in re.findall("^eTot\([0-9].*$", self.log_main, re.MULTILINE)
            ],
            self.counter,
        )

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            self._n_atoms = len(
                np.unique(re.findall("^Species.*\{", self.log_main, re.MULTILINE))
            )
        return self._n_atoms

    def get_forces(self, spx_to_pyi=None):
        forces = [
            float(re.split("{|}", line)[1].split(",")[i])
            * HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM
            for line in re.findall("^Species.*$", self.log_main, re.MULTILINE)
            for i in range(3)
        ]
        if len(forces) != 0:
            forces = np.array(forces).reshape(-1, self.n_atoms, 3)
            if spx_to_pyi is not None:
                for ii, ff in enumerate(forces):
                    forces[ii] = ff[spx_to_pyi]
        return forces

    def get_magnetic_forces(self, spx_to_pyi=None):
        magnetic_forces = [
            HARTREE_TO_EV * float(line.split()[-1])
            for line in re.findall("^nu\(.*$", self.log_main, re.MULTILINE)
        ]
        if len(magnetic_forces) != 0:
            magnetic_forces = np.array(magnetic_forces).reshape(-1, self.n_atoms)
            if spx_to_pyi is not None:
                for ii, mm in enumerate(magnetic_forces):
                    magnetic_forces[ii] = mm[spx_to_pyi]
        return splitter(magnetic_forces, self.counter)

    @property
    def n_steps(self):
        if self._n_steps is None:
            self._n_steps = len(
                re.findall("\| SCF calculation", self.log_file, re.MULTILINE)
            )
        return self._n_steps

    def _parse_band(self, term):
        arr = np.loadtxt(re.findall(term, self.log_main, re.MULTILINE))
        shape = (-1, len(self.k_points), arr.shape[-1])
        if self.spin_enabled:
            arr = np.vstack((arr[::2], arr[1::2]))
            shape = (-1, 2, len(self.k_points), shape[-1])
        return arr.reshape(shape)

    def get_band_energy(self):
        return self._parse_band("final eig \[eV\]:(.*)$")

    def get_occupancy(self):
        return self._parse_band("final focc:(.*)$")

    def get_convergence(self):
        conv_dict = {
            "WARNING: Maximum number of steps exceeded": False,
            "Convergence reached.": True,
        }
        key = "|".join(list(conv_dict.keys())).replace(".", "\.")
        items = re.findall(key, self.log_main, re.MULTILINE)
        convergence = [conv_dict[k] for k in items]
        diff = self.n_steps - len(convergence)
        for _ in range(diff):
            convergence.append(False)
        return convergence

    def get_fermi(self):
        return np.array(
            [
                float(line.split()[2])
                for line in re.findall("Fermi energy:.*$", self.log_main, re.MULTILINE)
            ]
        )


class Output:
    """
    Handles the output from a SPHInX simulation.
    """

    def __init__(self, job):
        self._job = job
        self.generic = DataContainer(table_name="output/generic")
        self.charge_density = SphinxVolumetricData()
        self.electrostatic_potential = SphinxVolumetricData()
        self.generic.create_group("dft")
        self.old_version = False

    def collect_spins_dat(self, file_name="spins.dat", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        if not os.path.isfile(posixpath.join(cwd, file_name)):
            return None
        spins = np.loadtxt(posixpath.join(cwd, file_name))
        self.generic.dft.atom_scf_spins = splitter(
            np.array([ss[self._job.id_spx_to_pyi] for ss in spins[:, 1:]]), spins[:, 0]
        )

    def collect_energy_dat(self, file_name="energy.dat", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        if not os.path.isfile(posixpath.join(cwd, file_name)):
            return None
        energies = np.loadtxt(posixpath.join(cwd, file_name))
        self.generic.dft.scf_computation_time = splitter(energies[:, 1], energies[:, 0])
        self.generic.dft.scf_energy_int = splitter(
            energies[:, 2] * HARTREE_TO_EV, energies[:, 0]
        )

        def en_split(e, counter=energies[:, 0]):
            return splitter(e * HARTREE_TO_EV, counter)

        if len(energies[0]) == 7:
            self.generic.dft.scf_energy_free = en_split(energies[:, 3])
            self.generic.dft.scf_energy_zero = en_split(energies[:, 4])
            self.generic.dft.scf_energy_band = en_split(energies[:, 5])
            self.generic.dft.scf_electronic_entropy = en_split(energies[:, 6])
        else:
            self.generic.dft.scf_energy_band = en_split(energies[:, 3])

    def collect_residue_dat(self, file_name="residue.dat", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        if not os.path.isfile(posixpath.join(cwd, file_name)):
            return None
        residue = np.loadtxt(posixpath.join(cwd, file_name))
        if len(residue) == 0:
            return None
        if len(residue[0]) == 2:
            self.generic.dft.scf_residue = splitter(residue[:, 1], residue[:, 0])
        else:
            self.generic.dft.scf_residue = splitter(residue[:, 1:], residue[:, 0])

    def collect_eps_dat(self, file_name="eps.dat", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        if isinstance(file_name, str):
            file_name = [file_name]
        values = []
        for f in file_name:
            file_tmp = posixpath.join(cwd, f)
            if not os.path.isfile(file_tmp):
                return
            values.append(np.loadtxt(file_tmp)[..., 1:])
        values = np.stack(values, axis=0)
        if "bands_eigen_values" not in self.generic.dft.list_nodes():
            self.generic.dft.bands_eigen_values = values.reshape((-1,) + values.shape)

    def collect_energy_struct(self, file_name="energy-structOpt.dat", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        file_name = posixpath.join(cwd, file_name)
        if os.path.isfile(file_name):
            self.generic.dft.energy_free = (
                np.loadtxt(file_name).reshape(-1, 2)[:, 1] * HARTREE_TO_EV
            )

    def collect_sphinx_log(self, file_name="sphinx.log", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """

        if not os.path.isfile(posixpath.join(cwd, file_name)):
            return None

        with open(posixpath.join(cwd, file_name), "r") as sphinx_log_file:
            log_file = "".join(sphinx_log_file.readlines())
        try:
            self._spx_log_parser = _SphinxLogParser(log_file)
        except AssertionError as e:
            self._job.status.aborted = True
            raise AssertionError(e)
        if not self._spx_log_parser.job_finished:
            self._job.status.aborted = True
        self.generic.dft.n_valence = self._spx_log_parser.get_n_valence()
        self.generic.dft.bands_k_weights = self._spx_log_parser.get_bands_k_weights()
        self.generic.dft.kpoints_cartesian = (
            self._spx_log_parser.get_kpoints_cartesian()
        )
        self.generic.volume = self._spx_log_parser.get_volume()
        self.generic.dft.bands_e_fermi = self._spx_log_parser.get_fermi()
        self.generic.dft.bands_occ = self._spx_log_parser.get_occupancy()
        self.generic.dft.bands_eigen_values = self._spx_log_parser.get_band_energy()
        self.generic.dft.scf_convergence = self._spx_log_parser.get_convergence()
        if "scf_energy_int" not in self.generic.dft.list_nodes():
            self.generic.dft.scf_energy_int = self._spx_log_parser.get_energy_int()
        if "scf_energy_free" not in self.generic.dft.list_nodes():
            self.generic.dft.scf_energy_free = self._spx_log_parser.get_energy_free()
        if "forces" in self.generic.list_nodes():
            self.generic.forces = self._spx_log_parser.get_forces(
                self._job.id_spx_to_pyi
            )
        if "scf_magnetic_forces" not in self.generic.dft.list_nodes():
            self.generic.dft.scf_magnetic_forces = (
                self._spx_log_parser.get_magnetic_forces(self._job.id_spx_to_pyi)
            )

    def collect_relaxed_hist(self, file_name="relaxHist.sx", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        file_name = posixpath.join(cwd, file_name)
        if not os.path.isfile(file_name):
            return None
        with open(file_name, "r") as f:
            file_content = "".join(f.readlines())
        natoms = len(self._job.id_spx_to_pyi)

        def get_value(term, file_content=file_content, natoms=natoms):
            value = (
                np.array(
                    [
                        re.split("\[|\]", line)[1].split(",")
                        for line in re.findall(term, file_content, re.MULTILINE)
                    ]
                )
                .astype(float)
                .reshape(-1, natoms, 3)
            )
            return np.array([ff[self._job.id_spx_to_pyi] for ff in value])

        self.generic.positions = get_value("coords.*$") * BOHR_TO_ANGSTROM
        self.generic.forces = (
            get_value("force.*$") * HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM
        )
        self.generic.cells = (
            np.array(
                [
                    json.loads(line)
                    for line in re.findall(
                        "cell =(.*?);", file_content.replace("\n", ""), re.MULTILINE
                    )
                ]
            )
            * BOHR_TO_ANGSTROM
        )

    def collect_charge_density(self, file_name, cwd):
        if (
            file_name in os.listdir(cwd)
            and os.stat(posixpath.join(cwd, file_name)).st_size != 0
        ):
            self.charge_density.from_file(
                filename=posixpath.join(cwd, file_name), normalize=True
            )

    def collect_electrostatic_potential(self, file_name, cwd):
        if (
            file_name in os.listdir(cwd)
            and os.stat(posixpath.join(cwd, file_name)).st_size != 0
        ):
            self.electrostatic_potential.from_file(
                filename=posixpath.join(cwd, file_name), normalize=False
            )

    def _get_electronic_structure_object(self):
        es = ElectronicStructure()
        if "bands_eigen_values" in self.generic.dft.list_nodes():
            eig_mat = self.generic.dft.bands_eigen_values[-1]
            occ_mat = self.generic.dft.bands_occ[-1]
            if len(eig_mat.shape) == 3:
                es.eigenvalue_matrix = eig_mat
                es.occupancy_matrix = occ_mat
            else:
                es.eigenvalue_matrix = np.array([eig_mat])
                es.occupancy_matrix = np.array([occ_mat])
            es.efermi = self.generic.dft.bands_e_fermi[-1]
            es.n_spins = len(es.occupancy_matrix)
            es.kpoint_list = self.generic.dft.kpoints_cartesian
            es.kpoint_weights = self.generic.dft.bands_k_weights
            es.generate_from_matrices()
        return es

    def collect(self, directory=os.getcwd()):
        """
        The collect function, collects all the output from a SPHInX simulation.

        Args:
            directory (str): the directory to collect the output from.
        """
        self.collect_energy_struct(file_name="energy-structOpt.dat", cwd=directory)
        self.collect_sphinx_log(file_name="sphinx.log", cwd=directory)
        self.collect_energy_dat(file_name="energy.dat", cwd=directory)
        self.collect_residue_dat(file_name="residue.dat", cwd=directory)
        if self._job._spin_enabled:
            self.collect_eps_dat(file_name="eps.dat", cwd=directory)
        else:
            self.collect_eps_dat(
                file_name=[f"eps.{i}.dat" for i in [0, 1]], cwd=directory
            )
        self.collect_spins_dat(file_name="spins.dat", cwd=directory)
        self.collect_relaxed_hist(file_name="relaxHist.sx", cwd=directory)
        self.collect_electrostatic_potential(file_name="vElStat-eV.sxb", cwd=directory)
        self.collect_charge_density(file_name="rho.sxb", cwd=directory)

    def to_hdf(self, hdf, force_update=False):
        """
        Store output in an HDF5 file

        Args:
            hdf: HDF5 group
            force_update(bool):
        """

        def get_last(arr):
            return np.array([vv[-1] for vv in arr])

        for k in self.generic.dft.list_nodes():
            if "scf" in k and k != "scf_convergence":
                self.generic.dft[k.replace("scf_", "")] = get_last(self.generic.dft[k])
        if "energy_free" in self.generic.dft.list_nodes():
            self.generic.energy_tot = self.generic.dft.energy_free
            self.generic.energy_pot = self.generic.dft.energy_free
        if "positions" not in self.generic.list_nodes():
            self.generic.positions = np.array([self._job.structure.positions])
        if "cells" not in self.generic.list_nodes():
            self.generic.cells = np.array([self._job.structure.cell.tolist()])
        self.generic.to_hdf(hdf=hdf)

        with hdf.open("output") as hdf5_output:
            if self.electrostatic_potential.total_data is not None:
                self.electrostatic_potential.to_hdf(
                    hdf5_output, group_name="electrostatic_potential"
                )
            if self.charge_density.total_data is not None:
                self.charge_density.to_hdf(hdf5_output, group_name="charge_density")
            if "bands_eigen_values" in self.generic.dft.list_nodes():
                es = self._get_electronic_structure_object()
                if len(es.kpoint_list) > 0:
                    es.to_hdf(hdf5_output)
            with hdf5_output.open("electronic_structure") as hdf5_es:
                if "dos" not in hdf5_es.list_groups():
                    hdf5_es.create_group("dos")
                with hdf5_es.open("dos") as hdf5_dos:
                    warning_message = " is not stored in SPHInX; use job.get_density_of_states instead"
                    for k in ["energies", "int_densities", "tot_densities"]:
                        hdf5_dos[k] = k + warning_message

    def from_hdf(self, hdf):
        """
        Load output from an HDF5 file
        """
        try:
            self.generic.from_hdf(hdf=hdf)
        except ValueError:
            warnings.warn(
                "You are using an old version of SPHInX output - update via job.update_sphinx()"
            )
            self.old_version = True
            pass


def _update_datacontainer(job):
    job.output.generic.create_group("dft")
    for node in job["output/generic/dft"].list_nodes():
        job.output.generic.dft[node] = job["output/generic/dft"][node]
    for node in job["output/generic"].list_nodes():
        job.output.generic[node] = job["output/generic"][node]
    job["output/generic"].remove_group()
    job.output.generic.to_hdf(hdf=job.project_hdf5)
    job.output.old_version = False
