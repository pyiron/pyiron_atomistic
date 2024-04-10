# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from typing import Optional

from atomistics.workflows.evcurve.debye import DebyeModel
from atomistics.workflows.evcurve.fit import (
    EnergyVolumeFit,
    fitfunction,
    get_error,
    fit_leastsq_eos,
)
from atomistics.workflows.evcurve.workflow import _strain_axes
import matplotlib.pyplot as plt
import numpy as np

from pyiron_atomistics.atomistics.master.parallel import AtomisticParallelMaster
from pyiron_atomistics.atomistics.structure.atoms import Atoms, ase_to_pyiron
from pyiron_base import JobGenerator
from pyiron_base.jobs.job.util import _get_safe_job_name

__author__ = "Joerg Neugebauer, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class MurnaghanDebyeModel(DebyeModel):
    """
    Calculate Thermodynamic Properties based on the Murnaghan output
    """

    def __init__(self, murnaghan, num_steps=50):
        self._murnaghan = murnaghan
        fit_dict = self._murnaghan.fit_dict.copy()
        fit_dict["volume"] = self._murnaghan["output/volume"]
        fit_dict["energy"] = self._murnaghan["output/energy"]
        super().__init__(
            fit_dict=fit_dict,
            masses=self._murnaghan.structure.get_masses(),
            num_steps=num_steps,
        )

    def polynomial(self, poly_fit=None, volumes=None):
        if poly_fit is None:
            self._murnaghan.fit_polynomial()  # TODO: include polyfit in output
            poly_fit = self._murnaghan.fit_dict["poly_fit"]
        p_fit = np.poly1d(poly_fit)
        if volumes is None:
            return p_fit(self.volume)
        return p_fit(volumes)

    @property
    def publication(self):
        return {
            "debye_model": {
                "Moruzzi1988": {
                    "title": "Calculated thermal properties of metals",
                    "author": ["Moruzzi, V. L.", "Janak, J. F.", "Schwarz, K"],
                    "journal": "Phys. Rev. B",
                    "volume": "37",
                    "issue": "2",
                    "pages": "790--799",
                    "numpages": "0",
                    "month": "jan",
                    "publisher": "American Physical Society",
                    "doi": "10.1103/PhysRevB.37.790",
                    "url": "https://link.aps.org/doi/10.1103/PhysRevB.37.790",
                }
            }
        }


class MurnaghanJobGenerator(JobGenerator):
    def _get_strains(self):
        strains = self._master.input.get("strains")
        if strains is None:
            strains = np.linspace(
                -self._master.input["vol_range"],
                self._master.input["vol_range"],
                int(self._master.input["num_points"]),
            )
        return strains

    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        parameter_lst = []
        for strain in self._get_strains():
            basis = _strain_axes(
                structure=self._master.structure,
                axes=self._master.input["axes"],
                volume_strain=strain,
            )
            parameter_lst.append([1 + np.round(strain, 7), basis])
        return parameter_lst

    def job_name(self, parameter):
        # writing the strain to float precision can blow up the job name of the
        # children; so figure out how much we really need here
        # find the smallest change in strain
        delta = np.min(np.abs(np.diff(self._get_strains())))
        # we need at most log(delta) decimal places to give unique children
        # names; if delta >= 1, rounding to 0 is sufficient
        ndigits = max(-int(np.floor(np.log10(delta))), 0)
        return _get_safe_job_name((self._master.job_name, parameter[0]),
                                  ndigits=ndigits)

    def modify_job(self, job, parameter):
        job.structure = parameter[1]
        return job


# ToDo: not all abstract methods implemented
class Murnaghan(AtomisticParallelMaster):
    """
    Murnghan calculation to obtain the minimum energy volume and bulk modulus.

    Example:

    >>> pr = Project('my_project')
    >>> ref_job = pr.create_job('Lammps', 'lmp')
    >>> ref_job.structure = structure_of_your_choice
    >>> murn = ref_job.create_job('Murnaghan', 'murn')
    >>> murn.run()

    The minimum energy volume and bulk modulus are stored in `ref_job['output/equilibrium_volume']`
    and `ref_job['output/equilibrium_bulk_modulus/']`.
    """

    def __init__(self, project, job_name):
        """

        Args:
            project:
            job_name:
        """
        super(Murnaghan, self).__init__(project, job_name)

        self.__version__ = "0.3.0"

        # print ("h5_path: ", self.project_hdf5._h5_path)

        # define default input
        self.input["num_points"] = (11, "number of sample points")
        self.input["fit_type"] = (
            "polynomial",
            "['polynomial', 'birch', 'birchmurnaghan', 'murnaghan', 'pouriertarantola', 'vinet']",
        )
        self.input["fit_order"] = (3, "order of the fit polynom")
        self.input["vol_range"] = (
            0.1,
            "relative volume variation around volume defined by ref_ham",
        )
        self.input["axes"] = (
            ("x", "y", "z"),
            "Axes along which the strain will be applied",
        )
        self.input["strains"] = (
            None,
            "List of strains that should be calculated.  If given vol_range and num_points take no effect.",
        )
        self.input["allow_aborted"] = (
            0,
            "The number of child jobs that are allowed to abort, before the whole job is considered aborted.",
        )

        self.debye_model = None
        self.fit_module = EnergyVolumeFit()

        self.fit_dict = None
        self._debye_T = None
        self._job_generator = MurnaghanJobGenerator(self)

    def convergence_check(self) -> bool:
        """
        Checks if the Murnaghan job has cnverged or not

        Note: Currently, a 3rd order polynomial is fit to check if there is any convergence

        Returns:
            bool: True if the calculation is converged
        """
        if super().convergence_check():
            e_vol = self["output/equilibrium_volume"]
            return e_vol is not None
        else:
            return False

    @property
    def fit(self):
        if self.debye_model is None and self.fit_dict is None:
            raise ValueError(
                "The fit object is only available after fitting the energy volume curve."
            )
        elif self.debye_model is None:
            self.debye_model = MurnaghanDebyeModel(self)
        return self.debye_model

    @property
    def equilibrium_volume(self):
        return self.fit_dict["volume_eq"]

    @property
    def equilibrium_energy(self):
        return self.fit_dict["energy_eq"]

    def fit_polynomial(self, fit_order=3, vol_erg_dic=None):
        return self.poly_fit(fit_order=fit_order, vol_erg_dic=vol_erg_dic)

    def fit_murnaghan(self, vol_erg_dic=None):
        return self._fit_eos_general(vol_erg_dic=vol_erg_dic, fittype="murnaghan")

    def fit_birch_murnaghan(self, vol_erg_dic=None):
        return self._fit_eos_general(vol_erg_dic=vol_erg_dic, fittype="birchmurnaghan")

    def fit_vinet(self, vol_erg_dic=None):
        return self._fit_eos_general(vol_erg_dic=vol_erg_dic, fittype="vinet")

    def _final_struct_to_hdf(self):
        with self._hdf5.open("output") as hdf5:
            structure = self.get_structure(frame=-1)
            if not isinstance(structure, Atoms):
                structure = ase_to_pyiron(structure)
            structure.to_hdf(hdf5)

    def _store_fit_in_hdf(self, fit_dict):
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)
        with self.project_hdf5.open("output") as hdf5:
            hdf5["equilibrium_energy"] = fit_dict["energy_eq"]
            hdf5["equilibrium_volume"] = fit_dict["volume_eq"]
            hdf5["equilibrium_bulk_modulus"] = fit_dict["bulkmodul_eq"]
            hdf5["equilibrium_b_prime"] = fit_dict["b_prime_eq"]
        self._final_struct_to_hdf()

    def _fit_eos_general(self, vol_erg_dic=None, fittype="birchmurnaghan"):
        self._set_fit_module(vol_erg_dic=vol_erg_dic)
        fit_dict = self.fit_module.fit_eos_general(fittype=fittype)
        self.input["fit_type"] = fit_dict["fit_type"]
        self.input["fit_order"] = 0
        self._store_fit_in_hdf(fit_dict=fit_dict)
        self.fit_dict = fit_dict
        return fit_dict

    def _fit_leastsq(self, volume_lst, energy_lst, fittype="birchmurnaghan"):
        return fit_leastsq_eos(
            volume_lst=volume_lst, energy_lst=energy_lst, fittype=fittype
        )

    def _set_fit_module(self, vol_erg_dic=None):
        if vol_erg_dic is not None:
            if "volume" in vol_erg_dic.keys() and "energy" in vol_erg_dic.keys():
                self.fit_module = EnergyVolumeFit(
                    volume_lst=vol_erg_dic["volume"], energy_lst=vol_erg_dic["energy"]
                )
            else:
                raise KeyError
        else:
            df = self.output_to_pandas()
            self.fit_module = EnergyVolumeFit(
                volume_lst=df["volume"].values, energy_lst=df["energy"].values
            )

    def poly_fit(self, fit_order=3, vol_erg_dic=None):
        self._set_fit_module(vol_erg_dic=vol_erg_dic)
        fit_dict = self.fit_module.fit_polynomial(fit_order=fit_order)
        if fit_dict is None:
            self._logger.warning("Minimum could not be found!")
        else:
            self.input["fit_type"] = fit_dict["fit_type"]
            self.input["fit_order"] = fit_dict["fit_order"]
            self._store_fit_in_hdf(fit_dict=fit_dict)
            self.fit_dict = fit_dict
        return fit_dict

    def list_structures(self):
        if self.ref_job.structure is not None:
            return [parameter[1] for parameter in self._job_generator.parameter_list]
        else:
            return []

    def validate_ready_to_run(self):
        axes = self.input["axes"]
        if len(set(axes)) != len(axes):
            raise ValueError('input["axes"] may not contain duplicate entries!')
        if not (1 <= len(axes) <= 3):
            raise ValueError('input["axes"] must contain one to three entries!')
        if set(axes).union(["x", "y", "z"]) != {"x", "y", "z"}:
            raise ValueError('input["axes"] entries must be one of "x", "y" or "z"!')

    def collect_output(self):
        if self.ref_job.server.run_mode.interactive:
            ham = self.project_hdf5.inspect(self.child_ids[0])
            erg_lst = ham["output/generic/energy_tot"]
            vol_lst = ham["output/generic/volume"]
            arg_lst = np.argsort(vol_lst)

            self._output["volume"] = vol_lst[arg_lst]
            self._output["energy"] = erg_lst[arg_lst]
        else:
            erg_lst, vol_lst, err_lst, id_lst = [], [], [], []
            allowed_aborted_children = self.input.get("allow_aborted", 0)
            for job_id in self.child_ids:
                ham = self.project_hdf5.inspect(job_id)
                if ham.status == "aborted":
                    if allowed_aborted_children == 0:
                        raise ValueError(f"Child {ham.name}({job_id}) is aborted!")
                    allowed_aborted_children -= 1
                    continue
                if "energy_tot" in ham["output/generic"].list_nodes():
                    energy = ham["output/generic/energy_tot"][-1]
                elif "energy_pot" in ham["output/generic"].list_nodes():
                    energy = ham["output/generic/energy_pot"][-1]
                else:
                    raise ValueError("Neither energy_pot or energy_tot was found.")
                volume = ham["output/generic/volume"][-1]
                erg_lst.append(np.mean(energy))
                err_lst.append(np.var(energy))
                vol_lst.append(volume)
                id_lst.append(job_id)
            aborted_children = (
                self.input.get("allow_aborted", 0) - allowed_aborted_children
            )
            if aborted_children > 0:
                self.logger.warning(
                    f"{aborted_children} aborted, but proceeding anyway."
                )
            vol_lst = np.array(vol_lst)
            erg_lst = np.array(erg_lst)
            err_lst = np.array(err_lst)
            id_lst = np.array(id_lst)
            arg_lst = np.argsort(vol_lst)

            self._output["volume"] = vol_lst[arg_lst]
            self._output["energy"] = erg_lst[arg_lst]
            self._output["error"] = err_lst[arg_lst]
            self._output["id"] = id_lst[arg_lst]

        with self.project_hdf5.open("output") as hdf5_out:
            for key, val in self._output.items():
                hdf5_out[key] = val
        if self.input["fit_type"] == "polynomial":
            self.fit_polynomial(fit_order=self.input["fit_order"])
        else:
            self._fit_eos_general(fittype=self.input["fit_type"])

    def plot(
        self,
        per_atom: bool = False,
        num_steps: int = 100,
        plt_show: bool = True,
        ax=None,
        plot_kwargs: Optional[dict] = None,
    ):
        """
        Plot E-V curve.

        Args:
            per_atom (optional, bool): normalize energy and volume by number of atoms in structure before plotting
            num_steps (optional, int): number of sample points to interpolate the calculated values on
            plt_show (optional, bool): call `matplotlib.pyplot.show()` after plotting (only necessary when running pyiron from scripts)
            ax (optional, plt.Axes): if given plot onto this axis, otherwise create new figure for the plot
            plot_kwargs (optional, dict): arguments passed verbatim to `matplotlib.pyplot.plot()`

        Returns:
            ax: The axis plotted onto

        Raises:
            ValueError: if job is not finished when calling this method
        """
        if not (self.status.finished or self.status.not_converged):
            raise ValueError(
                "Job must have finished executing before calling this method."
            )

        if ax is None:
            ax = plt.subplot(111)
        else:
            plt_show = False
        if not self.fit_dict:
            if self.input["fit_type"] == "polynomial":
                self.fit_polynomial(fit_order=self.input["fit_order"])
            else:
                self._fit_eos_general(fittype=self.input["fit_type"])
        df = self.output_to_pandas()
        vol_lst, erg_lst = df["volume"].values, df["energy"].values
        x_i = np.linspace(np.min(vol_lst), np.max(vol_lst), num_steps)

        if plot_kwargs is None:
            plot_kwargs = {}

        if "color" in plot_kwargs.keys():
            color = plot_kwargs["color"]
            del plot_kwargs["color"]
        else:
            color = "blue"

        if "marker" in plot_kwargs.keys():
            del plot_kwargs["marker"]

        if "label" in plot_kwargs.keys():
            label = plot_kwargs["label"]
            del plot_kwargs["label"]
        else:
            label = self.input["fit_type"]

        normalization = 1 if not per_atom else len(self.structure)
        if self.fit_dict is not None:
            if self.input["fit_type"] == "polynomial":
                p_fit = np.poly1d(self.fit_dict["poly_fit"])
                least_square_error = get_error(vol_lst, erg_lst, p_fit)
                ax.set_title("Murnaghan: error: " + str(least_square_error))
                ax.plot(
                    x_i / normalization,
                    p_fit(x_i) / normalization,
                    "-",
                    label=label,
                    color=color,
                    linewidth=3,
                    **plot_kwargs,
                )
            else:
                V0 = self.fit_dict["volume_eq"]
                E0 = self.fit_dict["energy_eq"]
                B0 = self.fit_dict["bulkmodul_eq"]
                BP = self.fit_dict["b_prime_eq"]
                eng_fit_lst = fitfunction(
                    parameters=[E0, B0, BP, V0], vol=x_i, fittype=self.input["fit_type"]
                )
                ax.plot(
                    x_i / normalization,
                    eng_fit_lst / normalization,
                    "-",
                    label=label,
                    color=color,
                    linewidth=3,
                    **plot_kwargs,
                )

        ax.plot(
            vol_lst / normalization,
            erg_lst / normalization,
            "x",
            color=color,
            markersize=20,
            **plot_kwargs,
        )
        ax.legend()
        ax.set_xlabel("Volume ($\AA^3$)")
        ax.set_ylabel("energy (eV)")
        if plt_show:
            plt.show()
        return ax

    def _get_structure(self, frame=-1, wrap_atoms=True):
        """
        Gives original structure or final one with equilibrium volume.

        Args:
            iteration_step (int): if 0 return original structure; if 1/-1 structure with equilibrium volume

        Returns:
            :class:`pyiron_atomistics.atomistics.structure.atoms.Atoms`: requested structure
        """
        if frame == 1:
            old_vol = self.structure.get_volume()
            new_vol = self["output/equilibrium_volume"]
            vol_strain = new_vol / old_vol - 1
            return _strain_axes(
                structure=self.structure,
                axes=self.input["axes"],
                volume_strain=vol_strain,
            )
        elif frame == 0:
            return self.structure

    def _number_of_structures(self):
        if self.structure is None:
            return 0
        elif not self.status.finished:
            return 1
        else:
            return 2
