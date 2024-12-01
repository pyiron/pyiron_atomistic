# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function, unicode_literals

import ast
import os
import warnings
from typing import Optional

import numpy as np
import pandas
from pyiron_base import state
from pyiron_snippets.deprecate import deprecate
from pyiron_snippets.logger import logger

from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_atomistics.lammps.control import LammpsControl
from pyiron_atomistics.lammps.output import (
    parse_lammps_output,
    remap_indices,
)
from pyiron_atomistics.lammps.potential import (
    LammpsPotential,
    LammpsPotentialFile,
    PotentialAvailable,
    list_potentials,
    view_potentials,
)
from pyiron_atomistics.lammps.structure import (
    LammpsStructure,
    UnfoldingPrism,
    structure_to_lammps,
)
from pyiron_atomistics.lammps.units import LAMMPS_UNIT_CONVERSIONS

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class LammpsBase(AtomisticGenericJob):
    """
    Class to setup and run and analyze LAMMPS simulations which is a derivative of
    atomistics.job.generic.GenericJob. The functions in these modules are written in such the function names and
    attributes are very generic (get_structure(), molecular_dynamics(), version) but the functions are written to handle
    LAMMPS specific input/output.

    Args:
        project (pyiron_atomistics.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (lammps.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(LammpsBase, self).__init__(project, job_name)
        self.input = Input()
        self._job_with_calculate_function = True
        self._cutoff_radius = None
        self._is_continuation = None
        self._compress_by_default = True
        self._prism = None
        self._collect_output_funct = parse_lammps_output
        state.publications.add(self.publication)

    @property
    def bond_dict(self):
        """
        A dictionary which defines the nature of LAMMPS bonds that are to be drawn between atoms. To set the values, use
        the function `define_bonds`.

        Returns:
            dict: Dictionary of the bond properties for every species

        """
        return self.input.bond_dict

    @property
    def cutoff_radius(self):
        """

        Returns:

        """
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, cutoff):
        """

        Args:
            cutoff:

        Returns:

        """
        self._cutoff_radius = cutoff

    @property
    def potential(self):
        """
        Execute view_potential() or list_potential() in order to see the pre-defined potential files

        Returns:

        """
        return self.input.potential.df

    @potential.setter
    def potential(self, potential_filename):
        """
        Execute view_potential() or list_potential() in order to see the pre-defined potential files

        Args:
            potential_filename:

        Returns:

        """
        potential = self._potential_file_to_potential(potential_filename)
        if self.structure is not None:
            self._check_potential_elements(
                self.structure.get_species_symbols(), list(potential["Species"])[0]
            )
        self.input.potential.df = potential
        if "Citations" in potential.columns.values:
            state.publications.add(self._get_potential_citations(potential))
        for val in ["units", "atom_style", "dimension"]:
            v = self.input.potential[val]
            if v is not None:
                self.input.control[val] = v
        if self.input.potential["units"] not in ("metal", None):
            warnings.warn(
                "Non-'metal' units are not fully supported. Your calculation should run OK, but "
                "results may not be saved in pyiron units."
            )
        self.input.potential.remove_structure_block()

    @property
    def potential_available(self):
        return PotentialAvailable(list_of_potentials=self.potential_list)

    @property
    def potential_list(self):
        """
        List of interatomic potentials suitable for the current atomic structure.

        use self.potentials_view() to get more details.

        Returns:
            list: potential names
        """

        return self.list_potentials()

    @property
    def potential_view(self):
        """
        List all interatomic potentials for the current atomistic sturcture including all potential parameters.

        To quickly get only the names of the potentials you can use: self.potentials_list()

        Returns:
            pandas.Dataframe: Dataframe including all potential parameters.
        """
        return self.view_potentials()

    @property
    def units(self):
        """
        Type of LAMMPS units used in the calculations. Can be either of 'metal', 'real', 'si', 'cgs', and 'lj'

        Returns:
            str: Type of LAMMPS unit
        """
        if self.input.control["units"] is not None:
            return self.input.control["units"]
        else:
            # Default to metal units
            return "metal"

    @units.setter
    def units(self, val):
        allowed_types = LAMMPS_UNIT_CONVERSIONS.keys()
        if val in allowed_types:
            self.input.control["units"] = val
        else:
            raise ValueError("'{}' is not a valid LAMMPS unit")

    @property
    def publication(self):
        return {
            "lammps": {
                "lammps": {
                    "title": "Fast Parallel Algorithms for Short-Range Molecular Dynamics",
                    "journal": "Journal of Computational Physics",
                    "volume": "117",
                    "number": "1",
                    "pages": "1-19",
                    "year": "1995",
                    "issn": "0021-9991",
                    "doi": "10.1006/jcph.1995.1039",
                    "url": "http://www.sciencedirect.com/science/article/pii/S002199918571039X",
                    "author": ["Steve Plimpton"],
                }
            }
        }

    def clear_bonds(self) -> None:
        """
        Clears all pre-defined bonds
        """
        self.input.bond_dict = {}

    def define_bonds(
        self,
        species,
        element_list,
        cutoff_list,
        max_bond_list,
        bond_type_list,
        angle_type_list=None,
    ):
        """
        Define the nature of bonds between different species. Make sure that the bonds between two species are defined
        only once (no double counting).

        Args:
            species (str): Species for which the bonds are to be drawn (e.g. O, H, C ..)
            element_list (list): List of species to which the bonds are to be made (e.g. O, H, C, ..)
            cutoff_list (list): Draw bonds only for atoms within this cutoff distance
            max_bond_list (list): Maximum number of bonds drawn from each molecule
            bond_type_list (list): Type of the bond as defined in the LAMMPS potential file
            angle_type_list (list): Type of the angle as defined in the LAMMPS potential file

        Example:
            The command below defined bonds between O and H atoms within a cutoff raduis of 2 $\AA$ with the bond and
            angle types 1 defined in the potential file used

            >> job_lammps.define_bonds(species="O", element_list-["H"], cutoff_list=[2.0], bond_type_list=[1],
            angle_type_list=[1])

        """
        if isinstance(species, str):
            if len(element_list) == len(cutoff_list) == bond_type_list == max_bond_list:
                self.input.bond_dict[species] = dict()
                self.input.bond_dict[species]["element_list"] = element_list
                self.input.bond_dict[species]["cutoff_list"] = cutoff_list
                self.input.bond_dict[species]["bond_type_list"] = bond_type_list
                self.input.bond_dict[species]["max_bond_list"] = max_bond_list
                if angle_type_list is not None:
                    self.input.bond_dict[species]["angle_type_list"] = angle_type_list
                else:
                    self.input.bond_dict[species]["angle_type_list"] = [None]
            else:
                raise ValueError(
                    "The element list, cutoff list, max bond list, and the bond type list"
                    " must have the same length"
                )

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(LammpsBase, self).set_input_to_read_only()
        self.input.control.read_only = True
        self.input.potential.read_only = True

    def validate_ready_to_run(self):
        """
        Validating input parameters before LAMMPS run
        """
        super(LammpsBase, self).validate_ready_to_run()
        if self.potential is None:
            lst_of_potentials = self.list_potentials()
            if len(lst_of_potentials) > 0:
                self.potential = lst_of_potentials[0]
                warnings.warn(
                    "No potential set via job.potential - use default potential, "
                    + lst_of_potentials[0]
                )
            else:
                raise ValueError(
                    "This job does not contain a valid potential: {}".format(
                        self.job_name
                    )
                )
        scaled_positions = self.structure.get_scaled_positions(wrap=False)
        # Check if atoms located outside of non periodic box
        conditions = [
            (
                np.min(scaled_positions[:, i]) < 0.0
                or np.max(scaled_positions[:, i]) > 1.0
            )
            and not self.structure.pbc[i]
            for i in range(3)
        ]
        if any(conditions):
            raise ValueError(
                "You have atoms located outside the non-periodic boundaries "
                "of the defined simulation box"
            )

    def get_potentials_for_structure(self):
        """

        Returns:

        """
        return self.list_potentials()

    @deprecate("use get_structure() instead")
    def get_final_structure(self):
        """

        Returns:

        """
        return self.get_structure(iteration_step=-1)

    def view_potentials(self) -> pandas.DataFrame:
        """
        List all interatomic potentials for the current atomistic structure including all potential parameters.

        To quickly get only the names of the potentials you can use: self.list_potentials()

        Returns:
            pandas.Dataframe: Dataframe including all potential parameters.
        """
        if not self.structure:
            raise ValueError("No structure set.")
        return view_potentials(self.structure)

    def list_potentials(self) -> list:
        """
        List of interatomic potentials suitable for the current atomic structure.

        use self.view_potentials() to get more details.

        Returns:
            list: potential names
        """
        return list_potentials(self.structure)

    def enable_h5md(self):
        """

        Returns:

        """
        del self.input.control["dump_modify___1"]
        del self.input.control["dump___1"]
        self.input.control["dump___1"] = (
            "all h5md ${dumptime} dump.h5 position force create_group yes"
        )

    def get_input_parameter_dict(self):
        """
        Get an hierarchical dictionary of input files. On the first level the dictionary is divided in file_to_create
        and files_to_copy. Both are dictionaries use the file names as keys. In file_to_create the values are strings
        which represent the content which is going to be written to the corresponding file. In files_to_copy the values
        are the paths to the source files to be copied.

        The get_input_file_dict() function is called before the write_input() function to convert the input specified on
        the job object to strings which can be written to the working directory as well as files which are copied to the
        working directory. After the write_input() function wrote the input files the executable is called.

        Returns:
            dict: hierarchical dictionary of input files
        """
        self.validate_ready_to_run()
        input_file_dict = super().get_input_parameter_dict()
        if self.structure is None:
            raise ValueError("Input structure not set. Use method set_structure()")
        lmp_structure = self._get_lammps_structure(
            structure=self.structure, cutoff_radius=self.cutoff_radius
        )
        update_input_hdf5 = False
        if not all(self.structure.pbc):
            self.input.control["boundary"] = " ".join(
                ["p" if coord else "f" for coord in self.structure.pbc]
            )
            update_input_hdf5 = True
        self._set_selective_dynamics()
        if update_input_hdf5:
            self.input.to_hdf(self._hdf5)
        if self.input.potential.files is not None:
            input_file_dict["files_to_copy"].update(
                {os.path.basename(f): f for f in self.input.potential.files}
            )
        input_file_dict["files_to_create"].update(
            {
                "structure.inp": lmp_structure._string_input,
                "control.inp": "".join(self.input.control.get_string_lst()),
                "potential.inp": "".join(self.input.potential.get_string_lst()),
            }
        )
        return input_file_dict

    def get_output_parameter_dict(self):
        return {
            "structure": self.structure,
            "potential_elements": self.input.potential.get_element_lst(),
            "units": self.units,
            "prism": self._prism,
            "dump_h5_file_name": "dump.h5",
            "dump_out_file_name": "dump.out",
            "log_lammps_file_name": "log.lammps",
        }

    def save_output(
        self, output_dict: Optional[dict] = None, shell_output: Optional[str] = None
    ):
        _ = shell_output
        self.input.from_hdf(self._hdf5)
        hdf_dict = resolve_hierachical_dict(
            data_dict=output_dict,
            group_name="output",
        )
        if len(self.structure) == len(hdf_dict["output/generic/indices"][-1]):
            final_structure = self.structure.copy()
            final_structure.indices = hdf_dict["output/generic/indices"][-1]
            final_structure.positions = hdf_dict["output/generic/positions"][-1]
            final_structure.cell = hdf_dict["output/generic/cells"][-1]
            hdf_dict.update(
                {
                    "output/structure/" + k: v
                    for k, v in final_structure.to_dict().items()
                }
            )
        else:
            logger.warning(
                "The number of atoms changed during the simulation. This can be a sign of massive issues in your simulation.\n"
                "Not storing 'output/structure' to HDF"
            )
        self.project_hdf5.write_dict_to_hdf(data_dict=hdf_dict)

    def convergence_check(self):
        if self._generic_input["calc_mode"] == "minimize":
            if (
                self._generic_input["max_iter"] + 1
                <= len(self["output/generic/energy_tot"])
                or len(
                    [l for l in self["log.lammps"] if "linesearch alpha is zero" in l]
                )
                != 0
            ):
                return False
            else:
                return True
        else:
            return True

    def collect_logfiles(self):
        """

        Returns:

        """
        return

    def remap_indices(self, lammps_indices):
        """
        Give the Lammps-dumped indices, re-maps these back onto the structure's indices to preserve the species.

        The issue is that for an N-element potential, Lammps dumps the chemical index from 1 to N based on the order
        that these species are written in the Lammps input file. But the indices for a given structure are based on the
        order in which chemical species were added to that structure, and run from 0 up to the number of species
        currently in that structure. Therefore we need to be a little careful with mapping.

        Args:
            indices (numpy.ndarray/list): The Lammps-dumped integers.

        Returns:
            numpy.ndarray: Those integers mapped onto the structure.
        """
        return remap_indices(
            lammps_indices=lammps_indices,
            potential_elements=self.input.potential.get_element_lst(),
            structure=self.structure,
        )

    def calc_minimize(
        self,
        ionic_energy_tolerance=0.0,
        ionic_force_tolerance=1e-4,
        e_tol=None,
        f_tol=None,
        max_iter=1000000,
        pressure=None,
        n_print=100,
        style="cg",
    ):
        rotation_matrix = self._get_rotation_matrix(pressure=pressure)
        # Docstring set programmatically -- Ensure that changes to signature or defaults stay consistent!
        if e_tol is not None:
            ionic_energy_tolerance = e_tol
        if f_tol is not None:
            ionic_force_tolerance = f_tol
        super(LammpsBase, self).calc_minimize(
            ionic_energy_tolerance=ionic_energy_tolerance,
            ionic_force_tolerance=ionic_force_tolerance,
            e_tol=e_tol,
            f_tol=f_tol,
            max_iter=max_iter,
            pressure=pressure,
            n_print=n_print,
        )
        self.input.control.calc_minimize(
            ionic_energy_tolerance=ionic_energy_tolerance,
            ionic_force_tolerance=ionic_force_tolerance,
            max_iter=max_iter,
            pressure=pressure,
            n_print=n_print,
            style=style,
            rotation_matrix=rotation_matrix,
        )

    calc_minimize.__doc__ = LammpsControl.calc_minimize.__doc__

    def calc_static(self):
        """

        Returns:

        """
        super(LammpsBase, self).calc_static()
        self.input.control.calc_static()

    def calc_md(
        self,
        temperature=None,
        pressure=None,
        n_ionic_steps=1000,
        time_step=1.0,
        n_print=100,
        temperature_damping_timescale=100.0,
        pressure_damping_timescale=1000.0,
        seed=None,
        tloop=None,
        initial_temperature=None,
        langevin=False,
        delta_temp=None,
        delta_press=None,
    ):
        # Docstring set programmatically -- Ensure that changes to signature or defaults stay consistent!
        if self.server.run_mode.interactive_non_modal:
            warnings.warn(
                "calc_md() is not implemented for the non modal interactive mode use calc_static()!"
            )
        rotation_matrix = self._get_rotation_matrix(pressure=pressure)
        super(LammpsBase, self).calc_md(
            temperature=temperature,
            pressure=pressure,
            n_ionic_steps=n_ionic_steps,
            time_step=time_step,
            n_print=n_print,
            temperature_damping_timescale=temperature_damping_timescale,
            pressure_damping_timescale=pressure_damping_timescale,
            seed=seed,
            tloop=tloop,
            initial_temperature=initial_temperature,
            langevin=langevin,
        )
        self.input.control.calc_md(
            temperature=temperature,
            pressure=pressure,
            n_ionic_steps=n_ionic_steps,
            time_step=time_step,
            n_print=n_print,
            temperature_damping_timescale=temperature_damping_timescale,
            pressure_damping_timescale=pressure_damping_timescale,
            seed=seed,
            tloop=tloop,
            initial_temperature=initial_temperature,
            langevin=langevin,
            delta_temp=delta_temp,
            delta_press=delta_press,
            job_name=self.job_name,
            rotation_matrix=rotation_matrix,
        )

    calc_md.__doc__ = LammpsControl.calc_md.__doc__

    def calc_vcsgc(
        self,
        mu=None,
        target_concentration=None,
        kappa=1000.0,
        mc_step_interval=100,
        swap_fraction=0.1,
        temperature_mc=None,
        window_size=None,
        window_moves=None,
        temperature=None,
        pressure=None,
        n_ionic_steps=1000,
        time_step=1.0,
        n_print=100,
        temperature_damping_timescale=100.0,
        pressure_damping_timescale=1000.0,
        seed=None,
        initial_temperature=None,
        langevin=False,
    ):
        """
        Run variance-constrained semi-grand-canonical MD/MC for a binary system. In addition to VC-SGC arguments, all
        arguments for a regular MD calculation are also accepted.

        https://vcsgc-lammps.materialsmodeling.org

        Note:
            For easy visualization later (with `get_structure`), it is highly recommended that the initial structure
            contain at least one atom of each species.

        Warning:
            - The fix does not yet support non-orthogonal simulation boxes; using one will give a runtime error.

        Args:
            mu (dict): A dictionary of chemical potentials, one for each element the potential treats, where the
                dictionary keys are just the chemical symbol. Note that only the *relative* chemical potentials are used
                here, such that the swap acceptance probability is influenced by the chemical potential difference
                between the two species (a more negative value increases the odds of swapping *to* that element.)
                (Default is None, all elements have the same chemical potential.)
            target_concentration: A dictionary of target simulation domain concentrations for each species *in the
                potential*. Dictionary keys should be the chemical symbol of the corresponding species, and the sum of
                all concentrations must be 1. (Default is None, which runs regular semi-grand-canonical MD/MC without
                any variance constraint.)
            kappa: Variance constraint for the MC. Larger value means a tighter adherence to the target concentrations.
                (Default is 1000.)
            mc_step_interval (int): How many steps of MD between each set of MC moves. (Default is 100.) Must divide the
                number of ionic steps evenly.
            swap_fraction (float): The fraction of atoms whose species is swapped at each MC phase. (Default is 0.1.)
            temperature_mc (float): The temperature for accepting MC steps. (Default is None, which uses the MD
                temperature.)
            window_size (float): The size of the sampling window for parallel calculations as a fraction of something
                unspecified in the VC-SGC docs, but it must lie between 0.5 and 1. (Default is None, window is
                determined automatically.)
            window_moves (int): The number of times the sampling window is moved during one MC cycle. (Default is None,
                number of moves is determined automatically.)
        """
        rotation_matrix = self._get_rotation_matrix(pressure=pressure)
        if mu is None:
            mu = {}
            for el in self.input.potential.get_element_lst():
                mu[el] = 0.0

        self._generic_input["calc_mode"] = "vcsgc"
        self._generic_input["mu"] = mu
        if target_concentration is not None:
            self._generic_input["target_concentration"] = target_concentration
            self._generic_input["kappa"] = kappa
        self._generic_input["mc_step_interval"] = mc_step_interval
        self._generic_input["swap_fraction"] = swap_fraction
        self._generic_input["temperature"] = temperature
        self._generic_input["temperature_mc"] = temperature_mc
        if window_size is not None:
            self._generic_input["window_size"] = window_size
        if window_moves is not None:
            self._generic_input["window_moves"] = window_moves
        self._generic_input["n_ionic_steps"] = n_ionic_steps
        self._generic_input["n_print"] = n_print
        self._generic_input.remove_keys(["max_iter"])
        self.input.control.calc_vcsgc(
            mu=mu,
            ordered_element_list=self.input.potential.get_element_lst(),
            target_concentration=target_concentration,
            kappa=kappa,
            mc_step_interval=mc_step_interval,
            swap_fraction=swap_fraction,
            temperature_mc=temperature_mc,
            window_size=window_size,
            window_moves=window_moves,
            temperature=temperature,
            pressure=pressure,
            n_ionic_steps=n_ionic_steps,
            time_step=time_step,
            n_print=n_print,
            temperature_damping_timescale=temperature_damping_timescale,
            pressure_damping_timescale=pressure_damping_timescale,
            seed=seed,
            initial_temperature=initial_temperature,
            langevin=langevin,
            job_name=self.job_name,
            rotation_matrix=rotation_matrix,
        )

    def to_dict(self):
        data_dict = super(LammpsBase, self).to_dict()
        lammps_dict = self._structure_to_dict() | self.input.to_dict()
        data_dict.update({"input/" + k: v for k, v in lammps_dict.items()})
        return data_dict

    def from_dict(self, obj_dict):
        super().from_dict(obj_dict=obj_dict)
        self._structure_from_dict(obj_dict=obj_dict)
        self.input.from_dict(obj_dict=obj_dict["input"])

    def write_restart_file(self, filename="restart.out"):
        """

        Args:
            filename:

        Returns:

        """
        self.input.control.modify(write_restart=filename, append_if_not_present=True)

    def compress(self, files_to_compress=None):
        """
        Compress the output files of a job object.

        Args:
            files_to_compress (list):
        """
        if files_to_compress is None:
            files_to_compress = [
                f for f in self.files.list() if f not in ["restart.out"]
            ]
        super(LammpsBase, self).compress(files_to_compress=files_to_compress)

    def next(self, job_name=None, job_type=None):
        """
        Restart a new job created from an existing Lammps calculation.
        Args:
            project (pyiron_atomistics.project.Project instance): Project instance at which the new job should be created
            job_name (str): Job name
            job_type (str): Job type. If not specified a Lammps job type is assumed

        Returns:
            new_ham (lammps.lammps.Lammps instance): New job
        """
        return super(LammpsBase, self).restart(job_name=job_name, job_type=job_type)

    def read_restart_file(self, filename="restart.out"):
        """

        Args:
            filename:

        Returns:

        """
        self._is_continuation = True
        self.input.control.set(read_restart=filename)
        self.input.control["reset_timestep"] = 0
        self.input.control.remove_keys(
            ["dimension", "read_data", "boundary", "atom_style", "velocity"]
        )

    def restart(self, job_name=None, job_type=None):
        """
        Restart a new job created from an existing Lammps calculation.
        Args:
            project (pyiron_atomistics.project.Project instance): Project instance at which the new job should be created
            job_name (str): Job name
            job_type (str): Job type. If not specified a Lammps job type is assumed

        Returns:
            lammps.lammps.Lammps instance: New job
        """
        new_ham = super(LammpsBase, self).restart(job_name=job_name, job_type=job_type)
        if new_ham.__name__ == self.__name__:
            new_ham.potential = self.potential
            if "restart.out" in self.files.list():
                new_ham.read_restart_file(filename="restart.out")
                new_ham.restart_file_list.append(self.files.restart_out)
        return new_ham

    def set_potential(self, file_name):
        """

        Args:
            file_name:

        Returns:

        """
        print("This function is outdated use the potential setter instead!")
        self.potential = file_name

    @staticmethod
    def _structure_to_lammps(structure):
        """
        Convert structure to LAMMPS compatible lower triangle format
        Args:
            structure (pyiron_atomistics.atomistics.structure.atoms.Atoms): Current structure

        Returns:
            pyiron_atomistics.atomistics.structure.atoms.Atoms: Converted structure
        """
        return structure_to_lammps(structure=structure)

    @staticmethod
    def _potential_file_to_potential(potential_filename):
        if isinstance(potential_filename, str):
            potential_filename = potential_filename.split(".lmp")[0]
            return LammpsPotentialFile().find_by_name(potential_filename)
        elif isinstance(potential_filename, pandas.DataFrame):
            return potential_filename
        elif hasattr(potential_filename, "get_df"):
            return potential_filename.get_df()
        else:
            raise TypeError("Potentials have to be strings or pandas dataframes.")

    @staticmethod
    def _check_potential_elements(structure_elements, potential_elements):
        if not set(structure_elements).issubset(potential_elements):
            raise ValueError(
                f"Potential {potential_elements} does not support elements "
                f"in structure {structure_elements}."
            )

    @staticmethod
    def _get_potential_citations(potential):
        pot_pub_dict = {}
        pub_lst = potential["Citations"].values[0]
        if isinstance(pub_lst, str) and len(pub_lst) > 0:
            for p in ast.literal_eval(pub_lst):
                for k in p.keys():
                    pot_pub_dict[k] = p[k]
        return {"lammps_potential": pot_pub_dict}

    @staticmethod
    def _modify_structure_to_allow_requested_deformation(
        structure, pressure, prism=None
    ):
        """
        Lammps will not allow xy/xz/yz cell deformations in minimization or MD for non-triclinic cells. In case the
        requested pressure for a calculation has these non-diagonal entries, we need to make sure it will run. One way
        to do this is by invoking the lammps `change_box` command, but it is easier to just force our box to to be
        triclinic by adding a very small cell perturbation (in the case where it isn't triclinic already).

        Args:
            pressure (float/int/list/numpy.ndarray/tuple): Between three and six pressures for the x, y, z, xy, xz, and
                yz directions, in that order, or a single value.
        """
        if hasattr(pressure, "__len__"):
            non_diagonal_pressures = np.any([p is not None for p in pressure[3:]])

            if prism is None:
                prism = UnfoldingPrism(structure.cell)

            if non_diagonal_pressures:
                try:
                    if not prism.is_skewed():
                        skew_structure = structure.copy()
                        skew_structure.cell[0, 1] += 2 * prism.acc
                        return skew_structure
                except AttributeError:
                    warnings.warn(
                        "WARNING: Setting a calculation type which uses pressure before setting the structure risks "
                        + "constraining your cell shape evolution if non-diagonal pressures are used but the structure "
                        + "is not triclinic from the start of the calculation."
                    )
        return structure

    def _set_selective_dynamics(self):
        if "selective_dynamics" in self.structure.arrays.keys():
            sel_dyn = np.logical_not(np.stack(self.structure.selective_dynamics))
            # Enter loop only if constraints present
            if len(np.argwhere(np.any(sel_dyn, axis=1)).flatten()) != 0:
                all_indices = np.arange(len(self.structure), dtype=int)
                constraint_xyz = np.argwhere(np.all(sel_dyn, axis=1)).flatten()
                not_constrained_xyz = np.setdiff1d(all_indices, constraint_xyz)
                # LAMMPS starts counting from 1
                constraint_xyz += 1
                ind_x = np.argwhere(sel_dyn[not_constrained_xyz, 0]).flatten()
                ind_y = np.argwhere(sel_dyn[not_constrained_xyz, 1]).flatten()
                ind_z = np.argwhere(sel_dyn[not_constrained_xyz, 2]).flatten()
                constraint_xy = not_constrained_xyz[np.intersect1d(ind_x, ind_y)] + 1
                constraint_yz = not_constrained_xyz[np.intersect1d(ind_y, ind_z)] + 1
                constraint_zx = not_constrained_xyz[np.intersect1d(ind_z, ind_x)] + 1
                constraint_x = (
                    not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_x, ind_y), ind_z)]
                    + 1
                )
                constraint_y = (
                    not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_y, ind_z), ind_x)]
                    + 1
                )
                constraint_z = (
                    not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_z, ind_x), ind_y)]
                    + 1
                )
                if len(constraint_xyz) > 0:
                    self.input.control["group___constraintxyz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_xyz]
                    )
                    self.input.control["fix___constraintxyz"] = (
                        "constraintxyz setforce 0.0 0.0 0.0"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constraintxyz"] = (
                            "set 0.0 0.0 0.0"
                        )
                if len(constraint_xy) > 0:
                    self.input.control["group___constraintxy"] = "id " + " ".join(
                        [str(ind) for ind in constraint_xy]
                    )
                    self.input.control["fix___constraintxy"] = (
                        "constraintxy setforce 0.0 0.0 NULL"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constraintxy"] = (
                            "set 0.0 0.0 NULL"
                        )
                if len(constraint_yz) > 0:
                    self.input.control["group___constraintyz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_yz]
                    )
                    self.input.control["fix___constraintyz"] = (
                        "constraintyz setforce NULL 0.0 0.0"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constraintyz"] = (
                            "set NULL 0.0 0.0"
                        )
                if len(constraint_zx) > 0:
                    self.input.control["group___constraintxz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_zx]
                    )
                    self.input.control["fix___constraintxz"] = (
                        "constraintxz setforce 0.0 NULL 0.0"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constraintxz"] = (
                            "set 0.0 NULL 0.0"
                        )
                if len(constraint_x) > 0:
                    self.input.control["group___constraintx"] = "id " + " ".join(
                        [str(ind) for ind in constraint_x]
                    )
                    self.input.control["fix___constraintx"] = (
                        "constraintx setforce 0.0 NULL NULL"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constraintx"] = (
                            "set 0.0 NULL NULL"
                        )
                if len(constraint_y) > 0:
                    self.input.control["group___constrainty"] = "id " + " ".join(
                        [str(ind) for ind in constraint_y]
                    )
                    self.input.control["fix___constrainty"] = (
                        "constrainty setforce NULL 0.0 NULL"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constrainty"] = (
                            "set NULL 0.0 NULL"
                        )
                if len(constraint_z) > 0:
                    self.input.control["group___constraintz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_z]
                    )
                    self.input.control["fix___constraintz"] = (
                        "constraintz setforce NULL NULL 0.0"
                    )
                    if self._generic_input["calc_mode"] == "md":
                        self.input.control["velocity___constraintz"] = (
                            "set NULL NULL 0.0"
                        )

    def _get_lammps_structure(self, structure=None, cutoff_radius=None):
        lmp_structure = LammpsStructure(bond_dict=self.input.bond_dict, job=self)
        lmp_structure._force_skewed = self.input.control._force_skewed
        lmp_structure.potential = self.input.potential
        lmp_structure.atom_type = self.input.control["atom_style"]
        if cutoff_radius is not None:
            lmp_structure.cutoff_radius = cutoff_radius
        else:
            lmp_structure.cutoff_radius = self.cutoff_radius
        lmp_structure.el_eam_lst = self.input.potential.get_element_lst()

        if structure is not None:
            lmp_structure.structure = structure_to_lammps(structure)
        else:
            lmp_structure.structure = structure_to_lammps(self.structure)
        if not set(lmp_structure.structure.get_species_symbols()).issubset(
            set(lmp_structure.el_eam_lst)
        ):
            raise ValueError(
                "The selected potentials do not support the given combination of elements."
            )
        return lmp_structure

    def _get_rotation_matrix(self, pressure):
        """

        Args:
            pressure:

        Returns:

        """
        if self.structure is not None:
            if self._prism is None:
                self._prism = UnfoldingPrism(self.structure.cell)

            self.structure = self._modify_structure_to_allow_requested_deformation(
                pressure=pressure, structure=self.structure, prism=self._prism
            )
            rotation_matrix = self._prism.R
        else:
            warnings.warn("No structure set, can not validate the simulation cell!")
            rotation_matrix = None
        return rotation_matrix


class Input:
    def __init__(self):
        self.control = LammpsControl()
        self.potential = LammpsPotential()
        self.bond_dict = dict()
        # Set default bond parameters
        self._load_default_bond_params()

    def _load_default_bond_params(self):
        """
        Function to automatically load a few default bond params (wont automatically write them)

        """
        # Default bond properties of a water molecule
        self.bond_dict["O"] = dict()
        self.bond_dict["O"]["element_list"] = ["H"]
        self.bond_dict["O"]["cutoff_list"] = [2.0]
        self.bond_dict["O"]["max_bond_list"] = [2]
        self.bond_dict["O"]["bond_type_list"] = [1]
        self.bond_dict["O"]["angle_type_list"] = [1]

    def from_dict(self, obj_dict):
        self.control.from_dict(obj_dict=obj_dict[self.control.table_name])
        self.potential.from_dict(obj_dict=obj_dict[self.potential.table_name])
        if "bond_dict" in obj_dict.keys():
            self.bond_dict = obj_dict["bond_dict"]

    def to_dict(self):
        return {
            self.control.table_name + "/" + k: v
            for k, v in self.control.to_dict().items()
        } | {
            self.potential.table_name + "/" + k: v
            for k, v in self.potential.to_dict().items()
        }


def resolve_hierachical_dict(data_dict, group_name=""):
    return_dict = {}
    if len(group_name) > 0 and group_name[-1] != "/":
        group_name = group_name + "/"
    for k, v in data_dict.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                return_dict[group_name + k + "/" + sk] = sv
        else:
            return_dict[group_name + k] = v
    return return_dict
