# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from collections import OrderedDict
import numpy as np
import warnings
import scipy.constants
import re

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

KBAR_TO_EVA = (
    scipy.constants.physical_constants["joule-electron volt relationship"][0] / 1e22
)


# derives from ValueError, because that was the exception previously raised
class OutcarCollectError(ValueError):
    pass


class Outcar(object):
    """
    This module is used to parse VASP OUTCAR files.

    Attributes:

        parse_dict (dict): A dictionary with all the useful quantities parsed from an OUTCAR file after from_file() is
                           executed

    """

    def __init__(self):
        self.parse_dict = dict()

    def from_file(self, filename="OUTCAR"):
        """
        Parse and store relevant quantities from the OUTCAR file into parse_dict.

        Args:
            filename (str): Filename of the OUTCAR file to parse

        """
        with open(filename, "r") as f:
            lines = f.readlines()
        energies = self.get_total_energies(filename=filename, lines=lines)
        energies_int = self.get_energy_without_entropy(filename=filename, lines=lines)
        energies_zero = self.get_energy_sigma_0(filename=filename, lines=lines)
        scf_energies = self.get_all_total_energies(filename=filename, lines=lines)
        n_atoms = self.get_number_of_atoms(filename=filename, lines=lines)
        forces = self.get_forces(filename=filename, lines=lines, n_atoms=n_atoms)
        positions = self.get_positions(filename=filename, lines=lines, n_atoms=n_atoms)
        cells = self.get_cells(filename=filename, lines=lines)
        steps = self.get_steps(filename=filename, lines=lines)
        temperatures = self.get_temperatures(filename=filename, lines=lines)
        time = self.get_time(filename=filename, lines=lines)
        fermi_level = self.get_fermi_level(filename=filename, lines=lines)
        scf_moments = self.get_dipole_moments(filename=filename, lines=lines)
        kin_energy_error = self.get_kinetic_energy_error(filename=filename, lines=lines)
        stresses = self.get_stresses(filename=filename, si_unit=False, lines=lines)
        n_elect = self.get_nelect(filename=filename, lines=lines)
        e_fermi_list, vbm_list, cbm_list = self.get_band_properties(
            filename=filename, lines=lines
        )
        elastic_constants = self.get_elastic_constants(filename=filename, lines=lines)
        energy_components = self.get_energy_components(filename=filename, lines=lines)
        cpu_time = self.get_cpu_time(filename=filename, lines=lines)
        user_time = self.get_user_time(filename=filename, lines=lines)
        system_time = self.get_system_time(filename=filename, lines=lines)
        elapsed_time = self.get_elapsed_time(filename=filename, lines=lines)
        memory_used = self.get_memory_used(filename=filename, lines=lines)
        vasp_version = self.get_vasp_version(filename=filename, lines=lines)
        full_array, ml_index, dft_index = self.get_index_type(
            filename=filename,
            lines=lines,
        )  # ml and dft indices from all outcar indices

        try:
            (
                irreducible_kpoints,
                ir_kpt_weights,
                plane_waves,
            ) = self.get_irreducible_kpoints(filename=filename, lines=lines)
        except ValueError:
            print("irreducible kpoints not parsed !")
            irreducible_kpoints = None
            ir_kpt_weights = None
            plane_waves = None
        magnetization, final_magmom_lst = self.get_magnetization(
            filename=filename, lines=lines
        )
        broyden_mixing = self.get_broyden_mixing_mesh(filename=filename, lines=lines)
        self.parse_dict["vasp_version"] = vasp_version

        self.parse_dict["ml_index"] = ml_index
        self.parse_dict["dft_index"] = dft_index
        outcar_index = [steps[i] for i in np.union1d(dft_index, ml_index)]
        is_ml_vec = full_array[outcar_index]
        self.parse_dict["steps"] = np.arange(len(is_ml_vec))

        self.parse_dict["ml_steps"] = self.parse_dict["steps"][is_ml_vec]
        self.parse_dict["dft_steps"] = self.parse_dict["steps"][~is_ml_vec]

        ## quantities from both DFT and ML
        self.parse_dict["energies"] = [
            energies[i] for i in np.union1d(dft_index, ml_index)
        ]  # remove zeroes

        ## quantites from DFT
        self.parse_dict["dft_energies"] = [energies[i] for i in dft_index]
        self.parse_dict["energies_int"] = [energies_int[i] for i in dft_index]
        self.parse_dict["energies_zero"] = [energies_zero[i] for i in dft_index]
        self.parse_dict["scf_energies"] = [scf_energies[i] for i in dft_index]

        ## quantites from ML
        self.parse_dict["ml_energies"] = [energies[i] for i in ml_index]
        self.parse_dict["energies"] = energies
        self.parse_dict["energies_int"] = energies_int
        self.parse_dict["energies_zero"] = energies_zero
        self.parse_dict["scf_energies"] = scf_energies

        self.parse_dict["forces"] = forces
        self.parse_dict["positions"] = positions
        self.parse_dict["cells"] = cells
        # self.parse_dict["steps"] = steps
        self.parse_dict["temperatures"] = temperatures
        self.parse_dict["time"] = time
        self.parse_dict["fermi_level"] = fermi_level
        self.parse_dict["scf_dipole_moments"] = scf_moments
        self.parse_dict["kin_energy_error"] = kin_energy_error
        self.parse_dict["stresses"] = stresses * KBAR_TO_EVA
        self.parse_dict["irreducible_kpoints"] = irreducible_kpoints
        self.parse_dict["irreducible_kpoint_weights"] = ir_kpt_weights
        self.parse_dict["number_plane_waves"] = plane_waves
        self.parse_dict["magnetization"] = magnetization
        self.parse_dict["final_magmoms"] = final_magmom_lst
        self.parse_dict["broyden_mixing"] = broyden_mixing
        self.parse_dict["n_elect"] = n_elect
        self.parse_dict["e_fermi_list"] = e_fermi_list
        self.parse_dict["vbm_list"] = vbm_list
        self.parse_dict["cbm_list"] = cbm_list
        self.parse_dict["elastic_constants"] = elastic_constants
        self.parse_dict["energy_components"] = energy_components
        self.parse_dict["resources"] = {
            "cpu_time": cpu_time,
            "user_time": user_time,
            "system_time": system_time,
            "elapsed_time": elapsed_time,
            "memory_used": memory_used,
        }
        try:
            self.parse_dict["pressures"] = (
                np.average(stresses[:, 0:3], axis=1) * KBAR_TO_EVA
            )
        except IndexError:
            self.parse_dict["pressures"] = np.zeros(len(steps))

    def to_hdf(self, hdf, group_name="outcar"):
        """
        Store output in an HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.FileHDFio): HDF5 group or file
            group_name (str): Name of the HDF5 group
        """
        with hdf.open(group_name) as hdf5_output:
            for key in self.parse_dict.keys():
                hdf5_output[key] = self.parse_dict[key]

    def to_hdf_minimal(self, hdf, group_name="outcar"):
        """
        Store minimal output in an HDF5 file (output unique to OUTCAR)

        Args:
            hdf (pyiron_base.generic.hdfio.FileHDFio): HDF5 group or file
            group_name (str): Name of the HDF5 group
        """
        with hdf.open(group_name) as hdf5_output:
            for k, v in self.to_dict_minimal().items():
                hdf5_output[k] = v

    def to_dict_minimal(self):
        hdf5_output = {}
        unique_quantities = [
            "kin_energy_error",
            "broyden_mixing",
            "stresses",
            "irreducible_kpoints",
            "irreducible_kpoint_weights",
            "number_plane_waves",
            "energy_components",
            "resources",
        ]
        for key in self.parse_dict.keys():
            if key in unique_quantities:
                hdf5_output[key] = self.parse_dict[key]
        return hdf5_output

    def from_hdf(self, hdf, group_name="outcar"):
        """
        Load output from an HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.FileHDFio): HDF5 group or file
            group_name (str): Name of the HDF5 group
        """
        with hdf.open(group_name) as hdf5_output:
            for key in hdf5_output.list_nodes():
                self.parse_dict[key] = hdf5_output[key]

    def get_vasp_version(self, filename="OUTCAR", lines=None):
        return lines[0].lstrip().split(sep=" ")[0]

    def get_positions_and_forces(self, filename="OUTCAR", lines=None, n_atoms=None):
        """
        Gets the forces and positions for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file
            n_atoms (int/None): number of ions in OUTCAR

        Returns:
            [positions, forces] (sequence)
            numpy.ndarray: A Nx3xM array of positions in $\AA$
            numpy.ndarray: A Nx3xM array of forces in $eV / \AA$

            where N is the number of atoms and M is the number of time steps
        """
        if n_atoms is None:
            n_atoms = self.get_number_of_atoms(filename=filename, lines=lines)
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger="TOTAL-FORCE (eV/Angst)"
        )
        return self._get_positions_and_forces_parser(
            lines=lines,
            trigger_indices=trigger_indices,
            n_atoms=n_atoms,
            pos_flag=True,
            force_flag=True,
        )

    def get_positions(self, filename="OUTCAR", lines=None, n_atoms=None):
        """
        Gets the positions for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file
            n_atoms (int/None): number of ions in OUTCAR

        Returns:
            numpy.ndarray: A Nx3xM array of positions in $\AA$

            where N is the number of atoms and M is the number of time steps
        """
        if n_atoms is None:
            n_atoms = self.get_number_of_atoms(filename=filename, lines=lines)
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger="TOTAL-FORCE (eV/Angst)"
        )
        return self._get_positions_and_forces_parser(
            lines=lines,
            trigger_indices=trigger_indices,
            n_atoms=n_atoms,
            pos_flag=True,
            force_flag=False,
        )

    def get_forces(self, filename="OUTCAR", lines=None, n_atoms=None):
        """
        Gets the forces for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file
            n_atoms (int/None): number of ions in OUTCAR

        Returns:

            numpy.ndarray: A Nx3xM array of forces in $eV / \AA$

            where N is the number of atoms and M is the number of time steps
        """
        if n_atoms is None:
            n_atoms = self.get_number_of_atoms(filename=filename, lines=lines)
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger="TOTAL-FORCE (eV/Angst)"
        )
        return self._get_positions_and_forces_parser(
            lines=lines,
            trigger_indices=trigger_indices,
            n_atoms=n_atoms,
            pos_flag=False,
            force_flag=True,
        )

    def get_cells(self, filename="OUTCAR", lines=None):
        """
        Gets the cell size and shape for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: A 3x3xM array of the cell shape in $\AA$

            where M is the number of time steps
        """
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger="VOLUME and BASIS-vectors are now :"
        )
        return self._get_cells_praser(lines=lines, trigger_indices=trigger_indices)

    @staticmethod
    def get_stresses(filename="OUTCAR", lines=None, si_unit=True):
        """

        Args:
            filename (str): Input filename
            lines (list/None): lines read from the file
            si_unit (bool): True SI units are used

        Returns:
            numpy.ndarray: An array of stress values

        """
        trigger_indices, lines = _get_trigger(
            lines=lines,
            filename=filename,
            trigger="FORCE on cell =-STRESS in cart. coord.  units (eV):",
        )
        stress_lst = []
        for j in trigger_indices:
            # search for '------...' delimiters of the stress table
            # setting a constant offset into `lines` does not work, because the number of stress contributions may vary
            # depending on the VASP configuration (e.g. with or without van der Waals interactions)
            jj = j
            while set(lines[jj].strip()) != {"-"}:
                jj += 1
            jj += 1
            # there's two delimiters, so search again
            while set(lines[jj].strip()) != {"-"}:
                jj += 1
            try:
                if si_unit:
                    stress = [float(l) for l in lines[jj + 1].split()[1:7]]
                else:
                    stress = [float(l) for l in lines[jj + 2].split()[2:8]]
            except ValueError:
                stress = [float("NaN")] * 6
            # VASP outputs the stresses in XX, YY, ZZ, XY, YZ, ZX order
            #                               0,  1,  2,  3,  4,  5
            stressm = np.diag(stress[:3])
            stressm[0, 1] = stressm[1, 0] = stress[3]
            stressm[1, 2] = stressm[2, 1] = stress[4]
            stressm[0, 2] = stressm[2, 0] = stress[5]
            stress_lst.append(stressm)
        return np.array(stress_lst)

    @staticmethod
    def get_irreducible_kpoints(
        filename="OUTCAR", reciprocal=True, weight=True, planewaves=True, lines=None
    ):
        """
        Function to extract the irreducible kpoints from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            reciprocal (bool): Get either the reciprocal or the cartesian coordinates
            weight (bool): Get the weight assigned to the irreducible kpoints
            planewaves (bool): Get the planewaves assigned to the irreducible kpoints
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: An array of k-points
        """
        kpoint_lst = []
        weight_lst = []
        planewaves_lst = []
        trigger_number_str = "Subroutine IBZKPT returns following result:"
        trigger_plane_waves_str = "k-point  1 :"
        trigger_number = 0
        trigger_plane_waves = 0
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if trigger_number_str in line:
                trigger_number = int(i)
            elif planewaves:
                if trigger_plane_waves_str in line:
                    trigger_plane_waves = int(i)
        number_irr_kpoints = int(lines[trigger_number + 3].split()[1])
        if reciprocal:
            trigger_start = trigger_number + 7
        else:
            trigger_start = trigger_number + 10 + number_irr_kpoints
        for line in lines[trigger_start : trigger_start + number_irr_kpoints]:
            line = line.strip()
            line = _clean_line(line)
            kpoint_lst.append([float(l) for l in line.split()[0:3]])
            if weight:
                weight_lst.append(float(line.split()[3]))
        if planewaves and trigger_plane_waves != 0:
            for line in lines[
                trigger_plane_waves : trigger_plane_waves + number_irr_kpoints
            ]:
                line = line.strip()
                line = _clean_line(line)
                planewaves_lst.append(int(line.split()[-1]))
        if weight and planewaves:
            return np.array(kpoint_lst), np.array(weight_lst), np.array(planewaves_lst)
        elif weight:
            return np.array(kpoint_lst), np.array(weight_lst)
        elif planewaves:
            return np.array(kpoint_lst), np.array(planewaves_lst)
        else:
            return np.array(kpoint_lst)

    @staticmethod
    def get_total_energies(filename="OUTCAR", lines=None):
        """
        Gets the total energy for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: A 1xM array of the total energies in $eV$

            where M is the number of time steps
        """

        def get_total_energies_from_line(line):
            return float(_clean_line(line.strip()).split()[-2])

        trigger_indices, lines = _get_trigger(
            lines=lines,
            filename=filename,
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        )
        return np.array(
            [get_total_energies_from_line(lines[j + 2]) for j in trigger_indices]
        )

    @staticmethod
    def get_energy_without_entropy(filename="OUTCAR", lines=None):
        """
        Gets the total energy for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: A 1xM array of the total energies in $eV$

            where M is the number of time steps
        """

        def get_energy_without_entropy_from_line(line):
            return float(_clean_line(line.strip()).split("=")[1].split()[0])

        trigger_indices, lines = _get_trigger(
            lines=lines,
            filename=filename,
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        )
        return np.array(
            [
                get_energy_without_entropy_from_line(lines[j + 4])
                for j in trigger_indices
            ]
        )

    @staticmethod  # -> new function
    def get_index_type(filename="OUTCAR", lines=None):
        """
        - In the outcar file you have two output for each DFT step (one of which is always zero for the energy and labelled ML) and one output for each ML step
        - This function read all steps from the outcar file and then returns back what is the index of the DFT steps and the index of each ML step
        - return indices of each DFT steps and index of DFT steps (excluding the empty ML output before DFT) from all steps
        """
        trigger_indices, lines = _get_trigger(
            lines=lines,
            filename=filename,
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        )
        full_array = np.array(
            ##boolean array, length: ml_steps + 2 x dft steps, values: true if ml false if dft and additional true for dummy ml steps printed withing dft steps
            ["ML" in lines[j] for j in trigger_indices]
        )
        outcar_ml_index = np.setdiff1d(
            np.where(full_array)[0], np.where(~full_array)[0] - 1
        )  ## ml indices of of the full_array dummy ml in dft is removed
        outcar_dft_index = np.where(~full_array)[0]  ## dft indiceds of the full_array

        # outcar_index = [steps[i] for i in np.union1d(dft_index, ml_index)]
        return full_array, outcar_ml_index, outcar_dft_index  # ml indices, dft indices

        # full_array, ml_index, dft_index = self.get_index_type(filename=filename, lines=lines)  # ml and dft indices from all outcar indices
        # outcar_index = [steps[i] for i in np.union1d(dft_index, ml_index)]
        # is_ml_vec = full_array[outcar_index]

    @staticmethod
    def get_energy_sigma_0(filename="OUTCAR", lines=None):
        """
        Gets the total energy for every ionic step from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: A 1xM array of the total energies in $eV$

            where M is the number of time steps
        """

        def get_energy_sigma_0_from_line(line):
            return float(_clean_line(line.strip()).split()[-1])

        trigger_indices, lines = _get_trigger(
            lines=lines,
            filename=filename,
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        )
        return np.array(
            [get_energy_sigma_0_from_line(lines[j + 4]) for j in trigger_indices]
        )

    @staticmethod
    def get_all_total_energies(filename="OUTCAR", lines=None):
        """
        Gets the energy at every electronic step

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            list: A list of energie for every electronic step at every ionic step
        """
        ind_ionic_lst, lines = _get_trigger(
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            filename=filename,
            lines=lines,
            return_lines=True,
        )
        ind_elec_lst = _get_trigger(
            trigger="free energy    TOTEN  =",
            filename=None,
            lines=lines,
            return_lines=False,
        )
        ind_combo_lst = _split_indices(
            ind_ionic_lst=ind_ionic_lst, ind_elec_lst=ind_elec_lst
        )
        return [
            np.array(
                [float(_clean_line(lines[ind].strip()).split()[-2]) for ind in ind_lst]
            )
            for ind_lst in ind_combo_lst
        ]

    @staticmethod
    def get_magnetization(filename="OUTCAR", lines=None):
        """
        Gets the magnetization

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            list: A list with the mgnetization values
        """
        ionic_trigger = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"
        electronic_trigger = "eigenvalue-minimisations"
        nion_trigger = "NIONS ="
        mag_lst = list()
        local_spin_trigger = False
        n_atoms = None
        mag_dict = dict()
        mag_dict["x"] = list()
        mag_dict["y"] = list()
        mag_dict["z"] = list()
        lines = _get_lines_from_file(filename=filename, lines=lines)
        istep_energies = list()
        final_magmom_lst = list()
        for i, line in enumerate(lines):
            line = line.strip()
            if ionic_trigger in line:
                mag_lst.append(np.array(istep_energies))
                istep_energies = list()
            if "Atomic Wigner-Seitz radii" in line:
                local_spin_trigger = True

            if electronic_trigger in line:
                try:
                    line = lines[i + 2].split("magnetization")[-1]
                    if line != " \n":
                        spin_str_lst = line.split()
                        spin_str_len = len(spin_str_lst)
                        if spin_str_len == 1:
                            ene = float(line)
                        elif spin_str_len == 3:
                            ene = [
                                float(spin_str_lst[0]),
                                float(spin_str_lst[1]),
                                float(spin_str_lst[2]),
                            ]
                        else:
                            warnings.warn("Unrecognized spin configuration.")
                            return mag_lst, final_magmom_lst
                        istep_energies.append(ene)
                except ValueError:
                    warnings.warn("Something went wrong in parsing the magnetization")
            if n_atoms is None:
                if nion_trigger in line:
                    n_atoms = int(line.split(nion_trigger)[-1])
            if local_spin_trigger:
                try:
                    for ind_dir, direc in enumerate(["x", "y", "z"]):
                        if "magnetization ({})".format(direc) in line:
                            mag_dict[direc].append(
                                [
                                    float(lines[i + 4 + atom_index].split()[-1])
                                    for atom_index in range(n_atoms)
                                ]
                            )
                except ValueError:
                    warnings.warn(
                        "Something went wrong in parsing the magnetic moments"
                    )
        if len(mag_dict["x"]) > 0:
            if len(mag_dict["y"]) == 0:
                final_mag = np.array(mag_dict["x"])
            else:
                n_ionic_steps = np.array(mag_dict["x"]).shape[0]
                final_mag = np.abs(np.zeros((n_ionic_steps, n_atoms, 3)))
                final_mag[:, :, 0] = np.array(mag_dict["x"])
                final_mag[:, :, 1] = np.array(mag_dict["y"])
                final_mag[:, :, 2] = np.array(mag_dict["z"])
            final_magmom_lst = final_mag.tolist()
        return mag_lst, final_magmom_lst

    @staticmethod
    def get_broyden_mixing_mesh(filename="OUTCAR", lines=None):
        """
        Gets the Broyden mixing mesh size

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            int: Mesh size
        """
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger="gives a total of "
        )
        if len(trigger_indices) > 0:
            line_ngx = lines[trigger_indices[0] - 2]
        else:
            warnings.warn(
                "Unable to parse the Broyden mixing mesh. Returning 0 instead"
            )
            return 0
        # Exclude all alphabets, and spaces. Then split based on '='
        str_list = re.sub(
            r"[a-zA-Z]", r"", line_ngx.replace(" ", "").replace("\n", "")
        ).split("=")
        return np.prod([int(val) for val in str_list[1:]])

    @staticmethod
    def get_temperatures(filename="OUTCAR", lines=None):
        """
        Gets the temperature at each ionic step (applicable for MD)

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: An array of temperatures in Kelvin
        """
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger="kin. lattice  EKIN_LAT= "
        )
        temperatures = []
        if len(trigger_indices) > 0:
            for j in trigger_indices:
                line = lines[j].strip()
                line = _clean_line(line)
                output_string = line.split("temperature")[-1].split()[0]
                try:
                    temperatures.append(float(output_string))
                except ValueError:
                    warnings.warn(f"Temperature too high. Vasp output: {line}")
                    temperatures.append(np.nan)
        else:
            temperatures = np.zeros(
                len(
                    _get_trigger(
                        lines=lines,
                        trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
                        return_lines=False,
                    )
                )
            )
        return np.array(temperatures)

    @staticmethod
    def get_steps(filename="OUTCAR", lines=None):
        """

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: Steps during the simulation
        """
        nblock_regex = re.compile(r"NBLOCK =\s+(\d+);")
        trigger = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"
        steps = 0
        nblock = None
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for line in lines:
            if trigger in line:
                steps += 1
            if nblock is None and "NBLOCK" in line:
                line = line.strip()
                line = _clean_line(line)
                nblock = int(nblock_regex.findall(line)[0])
        if nblock is None:
            nblock = 1
        return np.arange(0, steps * nblock, nblock)

    def get_time(self, filename="OUTCAR", lines=None):
        """
        Time after each simulation step (for MD)

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            numpy.ndarray: An array of time values in fs

        """
        potim_trigger = "POTIM  ="
        potim = 1.0
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            if potim_trigger in line:
                line = line.strip()
                line = _clean_line(line)
                potim = float(line.split(potim_trigger)[1].strip().split()[0])
                break
        return potim * self.get_steps(filename)

    @staticmethod
    def get_kinetic_energy_error(filename="OUTCAR", lines=None):
        """
        Get the kinetic energy error

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            float: The kinetic energy error in eV
        """
        trigger = "kinetic energy error for atom="
        e_kin_err = list()
        n_species_list = list()
        nion_trigger = "ions per type ="
        tot_kin_error = 0.0
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if trigger in line:
                e_kin_err.append(float(line.split()[5]))
            if nion_trigger in line:
                n_species_list = [
                    float(val) for val in line.split(nion_trigger)[-1].strip().split()
                ]
        if len(n_species_list) > 0 and len(n_species_list) == len(e_kin_err):
            tot_kin_error = np.sum(np.array(n_species_list) * np.array(e_kin_err))
        return tot_kin_error

    @staticmethod
    def get_fermi_level(filename="OUTCAR", lines=None):
        """
        Getting the Fermi-level (Kohn_Sham) from the OUTCAR file

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            float: The Kohn-Sham Fermi level in eV
        """
        trigger = "E-fermi :"
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger=trigger
        )
        if len(trigger_indices) != 0:
            try:
                return float(lines[trigger_indices[-1]].split(trigger)[-1].split()[0])
            except ValueError:
                return
        else:
            return

    @staticmethod
    def get_dipole_moments(filename="OUTCAR", lines=None):
        """
        Get the electric dipole moment at every electronic step

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file

        Returns:
            list: A list of dipole moments in (eA) for each electronic step

        """
        moment_trigger = "dipolmoment"
        istep_trigger = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"
        dip_moms = list()
        lines = _get_lines_from_file(filename=filename, lines=lines)
        istep_mom = list()
        for i, line in enumerate(lines):
            line = line.strip()
            if istep_trigger in line:
                dip_moms.append(np.array(istep_mom))
                istep_mom = list()
            if moment_trigger in line:
                line = _clean_line(line)
                mom = np.array([float(val) for val in line.split()[1:4]])
                istep_mom.append(mom)
        return dip_moms

    @staticmethod
    def get_nelect(filename="OUTCAR", lines=None):
        """
        Returns the number of electrons in the simulation

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            float: The number of electrons in the simulation

        """
        nelect_trigger = "NELECT"
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if nelect_trigger in line:
                return float(line.split()[2])

    @staticmethod
    def get_cpu_time(filename="OUTCAR", lines=None):
        """
        Returns the total CPU time in seconds

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            float: CPU time in seconds

        """
        nelect_trigger = "Total CPU time used (sec):"
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if nelect_trigger in line:
                return float(line.split()[-1])

    @staticmethod
    def get_user_time(filename="OUTCAR", lines=None):
        """
        Returns the User time in seconds

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            float: User time in seconds

        """
        nelect_trigger = "User time (sec):"
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if nelect_trigger in line:
                return float(line.split()[-1])

    @staticmethod
    def get_system_time(filename="OUTCAR", lines=None):
        """
        Returns the system time in seconds

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            float: system time in seconds

        """
        nelect_trigger = "System time (sec):"
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if nelect_trigger in line:
                return float(line.split()[-1])

    @staticmethod
    def get_elapsed_time(filename="OUTCAR", lines=None):
        """
        Returns the elapsed time in seconds

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            float: elapsed time in seconds

        """
        nelect_trigger = "Elapsed time (sec):"
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if nelect_trigger in line:
                return float(line.split()[-1])

    @staticmethod
    def get_memory_used(filename="OUTCAR", lines=None):
        """
        Returns the maximum memory used during the simulation in kB

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            float: Maximum memory used in kB

        """
        nelect_trigger = "Maximum memory used (kb):"
        lines = _get_lines_from_file(filename=filename, lines=lines)
        for i, line in enumerate(lines):
            line = line.strip()
            if nelect_trigger in line:
                return float(line.split()[-1])

    @staticmethod
    def get_number_of_atoms(filename="OUTCAR", lines=None):
        """
        Returns the number of ions in the simulation

        Args:
            filename (str): OUTCAR filename
            lines (list/None): lines read from the file

        Returns:
            int: The number of ions in the simulation

        """
        ions_trigger = "NIONS ="
        trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger=ions_trigger
        )
        if len(trigger_indices) != 0:
            return int(lines[trigger_indices[0]].split(ions_trigger)[-1])
        else:
            raise OutcarCollectError(
                "Failed to read number of atoms, can't find NIONS!"
            )

    @staticmethod
    def get_band_properties(filename="OUTCAR", lines=None):
        fermi_trigger = "E-fermi"
        fermi_trigger_indices, lines = _get_trigger(
            lines=lines, filename=filename, trigger=fermi_trigger
        )
        fermi_level_list = list()
        vbm_level_dict = OrderedDict()
        cbm_level_dict = OrderedDict()
        for ind in fermi_trigger_indices:
            fermi_level_list.append(float(lines[ind].strip().split()[2]))
        band_trigger = "band No.  band energies     occupation"
        is_spin_polarized = False
        for n, ind in enumerate(fermi_trigger_indices):
            if n == len(fermi_trigger_indices) - 1:
                trigger_indices, lines_new = _get_trigger(
                    lines=lines[ind:-1], filename=filename, trigger=band_trigger
                )
            else:
                trigger_indices, lines_new = _get_trigger(
                    lines=lines[ind : fermi_trigger_indices[n + 1]],
                    filename=filename,
                    trigger=band_trigger,
                )
            band_data = list()
            for ind in trigger_indices:
                if "spin component" in lines_new[ind - 3]:
                    is_spin_polarized = True
                for line in lines_new[ind + 1 :]:
                    data = line.strip().split()
                    # This if "Fermi" bypass needs to exist because of VASP changing it's OUTCAR format after 6.1.0
                    # In all versions prior, searching "Fermi" will only yield 2 mentions in the OUTCAR
                    # In the new versions, a Fermi energy is printed immediately after each spin-component k-point text block:
                    # i.e. Fermi energy: XXXXX
                    # This breaks the (old) parser, and thus this bypass is necessary to skip this last line at each spin component block
                    # Ugly and hacky, but parsing the OUTCAR is a ugly and hacky endeavour in general
                    if "Fermi" in data:
                        continue
                    if len(data) != 3:
                        break
                    band_data.append([float(d) for d in data[1:]])
            if is_spin_polarized:
                band_data_per_spin = [
                    np.array(band_data[0 : int(len(band_data) / 2)]).tolist(),
                    np.array(band_data[int(len(band_data) / 2) :]).tolist(),
                ]
            else:
                band_data_per_spin = [band_data]
            for spin, band_data in enumerate(band_data_per_spin):
                if spin in cbm_level_dict.keys():
                    pass
                else:
                    cbm_level_dict[spin] = list()
                if spin in vbm_level_dict.keys():
                    pass
                else:
                    vbm_level_dict[spin] = list()
                if len(band_data) > 0:
                    band_energy, band_occ = [
                        np.array(band_data)[:, i] for i in range(2)
                    ]
                    args = np.argsort(band_energy)
                    band_occ = band_occ[args]
                    band_energy = band_energy[args]
                    cbm_bool = np.abs(band_occ) < 1e-6
                    if any(cbm_bool):
                        cbm_level_dict[spin].append(
                            band_energy[np.abs(band_occ) < 1e-6][0]
                        )
                    else:
                        cbm_level_dict[spin].append(band_energy[-1])
                    # If spin channel is completely empty, setting vbm=cbm
                    if all(cbm_bool):
                        vbm_level_dict[spin].append(cbm_level_dict[spin][-1])
                    else:
                        vbm_level_dict[spin].append(band_energy[~cbm_bool][-1])
        return (
            np.array(fermi_level_list),
            np.array([val for val in vbm_level_dict.values()]),
            np.array([val for val in cbm_level_dict.values()]),
        )

    @staticmethod
    def get_elastic_constants(filename="OUTCAR", lines=None):
        lines = _get_lines_from_file(filename=filename, lines=lines)
        trigger_indices = _get_trigger(
            lines=lines,
            filename=filename,
            trigger="TOTAL ELASTIC MODULI (kBar)",
            return_lines=False,
        )
        if len(trigger_indices) != 1:
            return None
        else:
            start_index = trigger_indices[0] + 3
            end_index = start_index + 6
            elastic_constants = []
            for line in lines[start_index:end_index]:
                elastic_constants.append(line.split()[1:])
            elastic_GPa = np.array(elastic_constants, dtype=float) / 10
            return elastic_GPa

    @staticmethod
    def _get_positions_and_forces_parser(
        lines, trigger_indices, n_atoms, pos_flag=True, force_flag=True
    ):
        """
        Parser to get the forces and or positions for every ionic step from the OUTCAR file

        Args:
            lines (list): lines read from the file
            trigger_indices (list): list of line indices where the trigger was found.
            n_atoms (int): number of atoms
            pos_flag (bool): parse position
            force_flag (bool): parse forces

        Returns:
            [positions, forces] (sequence)
            numpy.ndarray: A Nx3xM array of positions in $\AA$
            numpy.ndarray: A Nx3xM array of forces in $eV / \AA$

            where N is the number of atoms and M is the number of time steps

        """
        positions = []
        forces = []
        for j in trigger_indices:
            pos = []
            force = []
            for line in lines[j + 2 : j + n_atoms + 2]:
                line = line.strip()
                line = _clean_line(line)
                if pos_flag:
                    pos.append([float(l) for l in line.split()[0:3]])
                if force_flag:
                    force.append([float(l) for l in line.split()[3:]])
            forces.append(force)
            positions.append(pos)
        if pos_flag and force_flag:
            return np.array(positions), np.array(forces)
        elif pos_flag:
            return np.array(positions)
        elif force_flag:
            return np.array(forces)

    @staticmethod
    def _get_cells_praser(lines, trigger_indices):
        """
        Parser to get the cell size and shape for every ionic step from the OUTCAR file

        Args:
            lines (list): lines read from the file
            trigger_indices (list): list of line indices where the trigger was found.
            n_atoms (int): number of atoms

        Returns:
            numpy.ndarray: A 3x3xM array of the cell shape in $\AA$

            where M is the number of time steps

        """
        cells = []
        try:
            for j in trigger_indices:
                cell = []
                for line in lines[j + 5 : j + 8]:
                    line = line.strip()
                    line = _clean_line(line)
                    cell.append([float(l) for l in line.split()[0:3]])
                cells.append(cell)
            return np.array(cells)
        except ValueError:
            warnings.warn("Unable to parse the cells from the OUTCAR file")
            return

    @staticmethod
    def get_energy_components(filename="OUTCAR", lines=None):
        """
        Gets the individual components of the free energy energy for every electronic step from the OUTCAR file

        alpha Z        PSCENC =        -0.19957337
        Ewald energy   TEWEN  =       -73.03212173
        -Hartree energ DENC   =        -0.10933240
        -exchange      EXHF   =         0.00000000
        -V(xc)+E(xc)   XCENC  =       -26.17018410
        PAW double counting   =       168.82497547     -136.88269783
        entropy T*S    EENTRO =        -0.00827174
        eigenvalues    EBANDS =        10.35379785
        atomic energy  EATOM  =        53.53616173
        Solvation  Ediel_sol  =         0.00000000

        Args:
            filename (str): Filename of the OUTCAR file to parse
            lines (list/None): lines read from the file
        Returns:
            numpy.ndarray: A 1xM array of the total energies in $eV$
            where M is the number of time steps
        """
        ind_ionic_lst, lines = _get_trigger(
            trigger="FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
            filename=filename,
            lines=lines,
            return_lines=True,
        )
        ind_elec_lst = _get_trigger(
            trigger="Free energy of the ion-electron system (eV)",
            filename=None,
            lines=lines,
            return_lines=False,
        )
        ind_combo_lst = _split_indices(
            ind_ionic_lst=ind_ionic_lst, ind_elec_lst=ind_elec_lst
        )
        try:
            return [
                np.array(
                    [
                        np.hstack(
                            [
                                (
                                    float(lines[ind + i].split()[-1])
                                    if i != 7
                                    else [
                                        float(lines[ind_lst[-1] + 7].split()[-2]),
                                        float(lines[ind_lst[-1] + 7].split()[-1]),
                                    ]
                                )
                                for i in range(2, 12)
                            ]
                        )
                        for ind in ind_lst
                    ]
                ).T
                for ind_lst in ind_combo_lst
            ]
        except ValueError:
            return []


def _clean_line(line):
    return line.replace("-", " -")


def _get_trigger(trigger, filename=None, lines=None, return_lines=True):
    """
    Find the lines where a specific trigger appears.

    Args:
        trigger (str): string pattern to search for
        lines (list/None): list of lines
        filename (str/None): file to read lines from

    Returns:
        list: indicies of the lines where the trigger string was found and list of lines
    """
    lines = _get_lines_from_file(filename=filename, lines=lines)
    trigger_indicies = [i for i, line in enumerate(lines) if trigger in line.strip()]
    if return_lines:
        return trigger_indicies, lines
    else:
        return trigger_indicies


def _split_indices(ind_ionic_lst, ind_elec_lst):
    """
    Combine ionic pattern matches and electronic pattern matches

    Args:
        ind_ionic_lst (list): indices of lines which matched the iconic pattern
        ind_elec_lst (list): indices of lines which matched the electronic pattern

    Returns:
        list: nested list of electronic pattern matches within an ionic pattern match
    """
    ind_elec_array = np.array(ind_elec_lst)
    return [
        (
            ind_elec_array[(ind_elec_array < j2) & (j1 < ind_elec_array)]
            if j1 < j2
            else ind_elec_array[(ind_elec_array < j2)]
        )
        for j1, j2 in zip(np.roll(ind_ionic_lst, 1), ind_ionic_lst)
    ]


def _get_lines_from_file(filename, lines=None):
    """
    If lines is None read the lines from the file with the filename filename.

    Args:
        filename (str): file to read lines from
        lines (list/ None): list of lines

    Returns:
        list: list of lines
    """
    if lines is None:
        with open(filename, "r") as f:
            lines = f.readlines()
    return lines
