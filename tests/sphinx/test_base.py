# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import sys
import shutil

import numpy as np
import unittest
import tempfile
from pathlib import Path
import warnings
import scipy.constants
from pyiron_atomistics.project import Project
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.sphinx.base import Group

BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)
HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]
HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class TestSphinx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, "../static/sphinx"))
        cls.sphinx = cls.project.create_job("Sphinx", "job_sphinx_base")
        cls.sphinx_band_structure = cls.project.create_job("Sphinx", "sphinx_test_bs")
        cls.sphinx_2_3 = cls.project.create_job("Sphinx", "sphinx_test_2_3")
        cls.sphinx_2_5 = cls.project.create_job("Sphinx", "sphinx_test_2_5")
        cls.sphinx_aborted = cls.project.create_job("Sphinx", "sphinx_test_aborted")
        basis = Atoms(
            elements=2 * ["Fe"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=2.6 * np.eye(3),
        )
        basis.set_initial_magnetic_moments(2 * [0.5])
        cls.sphinx.structure = basis
        cls.sphinx.fix_spin_constraint = True
        cls.sphinx_band_structure.structure = cls.project.create.structure.bulk("Fe")
        cls.sphinx_band_structure.structure = (
            cls.sphinx_band_structure.structure.create_line_mode_structure()
        )
        cls.sphinx_2_3.structure = Atoms(
            elements=["Fe", "Fe"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=2.6 * np.eye(3),
        )
        cls.sphinx_2_5.structure = Atoms(
            elements=["Fe", "Ni"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=2.83 * np.eye(3),
        )
        cls.sphinx_2_5.structure.set_initial_magnetic_moments([2, 2])
        cls.sphinx_aborted.structure = Atoms(
            elements=32 * ["Fe"],
            scaled_positions=np.arange(32 * 3).reshape(-1, 3) / (32 * 3),
            cell=3.5 * np.eye(3),
        )
        cls.sphinx_aborted.status.aborted = True
        cls.current_dir = os.path.abspath(os.getcwd())
        cls.sphinx._create_working_directory()
        cls.sphinx_2_3._create_working_directory()
        cls.sphinx.input["VaspPot"] = False
        cls.sphinx.structure.add_tag(selective_dynamics=(True, True, True))
        cls.sphinx.structure.selective_dynamics[1] = (False, False, False)
        cls.sphinx.fix_symmetry = False
        cls.sphinx.load_default_groups()
        cls.sphinx.write_input()
        cls.sphinx_2_3.to_hdf()
        cls.sphinx_2_3.decompress()
        cls.sphinx_2_5.decompress()
        cls.sphinx_2_5.collect_output()

    @classmethod
    def tearDownClass(cls):
        cls.sphinx_2_3.decompress()
        cls.sphinx_2_5.decompress()
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        os.remove(
            os.path.join(
                cls.file_location,
                "../static/sphinx/job_sphinx_base_hdf5/job_sphinx_base/input.sx",
            )
        )
        os.remove(
            os.path.join(
                cls.file_location,
                "../static/sphinx/job_sphinx_base_hdf5/job_sphinx_base/spins.in",
            )
        )
        os.remove(
            os.path.join(
                cls.file_location,
                "../static/sphinx/job_sphinx_base_hdf5/job_sphinx_base/Fe_GGA.atomicdata",
            )
        )
        shutil.rmtree(
            os.path.join(cls.file_location, "../static/sphinx/job_sphinx_base_hdf5")
        )
        os.remove(
            os.path.join(cls.file_location, "../static/sphinx/sphinx_test_2_3.h5")
        )

    def test_id_pyi_to_spx(self):
        spx = self.project.create_job("Sphinx", "check_order")
        elements = ["Fe", "C", "Ni", "Fe"]
        x = np.random.random((len(elements), 3))
        c = np.eye(3)
        spx.structure = Atoms(elements=elements, positions=x, cell=c)
        self.assertTrue(np.all(spx.id_spx_to_pyi == [1, 0, 3, 2]))
        self.assertTrue(
            np.all(spx.id_pyi_to_spx[spx.id_spx_to_pyi] == np.arange(len(elements)))
        )

    def test_potential(self):
        self.assertEqual(["Fe_GGA"], self.sphinx.list_potentials())
        self.assertEqual(["Fe_GGA"], self.sphinx_2_3.list_potentials())
        # The following sphinx_2_5.list_potentials test depends on the environment
        # [this probably applies to all list_potentials tests]
        # Thoughts by C. Freysoldt, 2022-10-24:
        #   - I think this is unhealthy, as the test environment is NOT controlled by the test
        #   - there is a discrepancy between mybinder/pyiron_atomistics environment and the
        #     github/CI environment.
        # next line is for an environment with Fe_GGA and Ni_GGA (but no other Fe/Ni)
        # self.assertEqual(['Fe_GGA', 'Ni_GGA'], sorted(self.sphinx_2_5.list_potentials()))
        # next line is for the github/CI test environment (only Fe_GGA, no Ni, no other Fe)
        self.assertEqual(["Fe_GGA"], self.sphinx_2_5.list_potentials())
        self.sphinx_2_3.potential.Fe = "Fe_GGA"
        self.sphinx_2_5.potential["Fe"] = "Fe_GGA"
        self.assertEqual(
            "Fe_GGA", list(self.sphinx_2_3.potential.to_dict().values())[0]
        )
        self.assertEqual(
            "Fe_GGA", list(self.sphinx_2_5.potential.to_dict().values())[0]
        )

    def test_write_input(self):
        file_content = [
            "//job_sphinx_base\n",
            "//SPHInX input file generated by pyiron\n",
            "\n",
            "format paw;\n",
            "include <parameters.sx>;\n",
            "\n",
            "pawPot {\n",
            "\tspecies {\n",
            '\t\tname = "Fe";\n',
            '\t\tpotType = "AtomPAW";\n',
            '\t\telement = "Fe";\n',
            '\t\tpotential = "Fe_GGA.atomicdata";\n',
            "\t}\n",
            "}\n",
            "structure {\n",
            "\tcell = [[4.913287924027003, 0.0, 0.0], [0.0, 4.913287924027003, 0.0], [0.0, 0.0, 4.913287924027003]];\n",
            "\tspecies {\n",
            '\t\telement = "Fe";\n',
            "\t\tatom {\n",
            '\t\t\tlabel = "spin_0.5";\n',
            "\t\t\tcoords = [0.0, 0.0, 0.0];\n",
            "\t\t\tmovable = true;\n",
            "\t\t}\n",
            "\t\tatom {\n",
            '\t\t\tlabel = "spin_0.5";\n',
            "\t\t\tcoords = [2.4566439620135014, 2.4566439620135014, 2.4566439620135014];\n",
            "\t\t}\n",
            "\t}\n",
            "\tsymmetry {\n",
            "\t\toperator {\n",
            "\t\t\tS = [[1,0,0],[0,1,0],[0,0,1]];\n",
            "\t\t}\n",
            "\t}\n",
            "}\n",
            "basis {\n",
            "\teCut = 24.989539079445393;\n",
            "\tkPoint {\n",
            "\t\tcoords = [0.5, 0.5, 0.5];\n",
            "\t\tweight = 1;\n",
            "\t\trelative = true;\n",
            "\t}\n",
            "\tfolding = [4, 4, 4];\n",
            "\tsaveMemory = true;\n",
            "}\n",
            "PAWHamiltonian {\n",
            "\tnEmptyStates = 6;\n",
            "\tekt = 0.2;\n",
            "\tMethfesselPaxton = 1;\n",
            "\txc = PBE;\n",
            "\tspinPolarized = true;\n",
            "}\n",
            "initialGuess {\n",
            "\twaves {\n",
            "\t\tpawBasis = true;\n",
            "\t\tlcao {}\n",
            "\t}\n",
            "\trho {\n",
            "\t\tatomicOrbitals = true;\n",
            "\t\tatomicSpin {\n",
            '\t\t\tlabel = "spin_0.5";\n',
            "\t\t\tspin = 0.5;\n",
            "\t\t}\n",
            "\t}\n",
            "\tnoWavesStorage = false;\n",
            "}\n",
            "main {\n",
            "\tscfDiag {\n",
            "\t\trhoMixing = 1.0;\n",
            "\t\tspinMixing = 1.0;\n",
            "\t\tdEnergy = 3.674932217565499e-06;\n",
            "\t\tmaxSteps = 100;\n",
            "\t\tpreconditioner {\n",
            "\t\t\ttype = KERKER;\n",
            "\t\t\tscaling = 1.0;\n",
            "\t\t\tspinScaling = 1.0;\n",
            "\t\t}\n",
            "\t\tblockCCG {}\n",
            "\t}\n",
            "\tevalForces {\n",
            '\t\tfile = "relaxHist.sx";\n',
            "\t}\n",
            "}\n",
            "spinConstraint {\n",
            '\tfile = "spins.in";\n',
            "}\n",
        ]
        file_name = os.path.join(
            self.file_location,
            "../static/sphinx/job_sphinx_base_hdf5/job_sphinx_base/input.sx",
        )
        with open(file_name) as input_sx:
            lines = input_sx.readlines()
        self.assertEqual(file_content, lines)

    def test_plane_wave_cutoff(self):
        with self.assertRaises(ValueError):
            self.sphinx.plane_wave_cutoff = -1

        with warnings.catch_warnings(record=True) as w:
            self.sphinx.plane_wave_cutoff = 25
            self.assertEqual(len(w), 1)

        self.sphinx.plane_wave_cutoff = 340
        self.assertEqual(self.sphinx.plane_wave_cutoff, 340)

    def test_set_kpoints(self):
        mesh = [2, 3, 4]
        center_shift = [0.1, 0.1, 0.1]

        trace = {"my_path": [("GAMMA", "H"), ("H", "N"), ("P", "H")]}

        kpoints_group = Group(
            {
                "relative": True,
                "from": {"coords": np.array([0.0, 0.0, 0.0]), "label": '"GAMMA"'},
                "to": [
                    {
                        "coords": np.array([0.5, -0.5, 0.5]),
                        "nPoints": 20,
                        "label": '"H"',
                    },
                    {
                        "coords": np.array([0.0, 0.0, 0.5]),
                        "nPoints": 20,
                        "label": '"N"',
                    },
                    {
                        "coords": np.array([0.25, 0.25, 0.25]),
                        "nPoints": 0,
                        "label": '"P"',
                    },
                    {
                        "coords": np.array([0.5, -0.5, 0.5]),
                        "nPoints": 20,
                        "label": '"H"',
                    },
                ],
            }
        )

        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(symmetry_reduction="pyiron rules!")
        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(scheme="no valid scheme")
        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(scheme="Line", path_name="my_path")

        self.sphinx_band_structure.structure.add_high_symmetry_path(trace)
        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(scheme="Line", n_path=20)
        with self.assertRaises(AssertionError):
            self.sphinx_band_structure.set_kpoints(
                scheme="Line", path_name="wrong name", n_path=20
            )

        self.sphinx_band_structure.set_kpoints(
            scheme="Line", path_name="my_path", n_path=20
        )
        self.assertTrue("kPoint" not in self.sphinx_band_structure.input.sphinx.basis)
        self.assertEqual(
            self.sphinx_band_structure.input.sphinx.to_sphinx(kpoints_group),
            self.sphinx_band_structure.input.sphinx.basis.kPoints.to_sphinx(),
        )

        self.sphinx_band_structure.set_kpoints(
            scheme="MP", mesh=mesh, center_shift=center_shift
        )
        self.assertTrue("kPoints" not in self.sphinx_band_structure.input.sphinx.basis)
        self.assertEqual(self.sphinx_band_structure.input.KpointFolding, mesh)
        self.assertEqual(self.sphinx_band_structure.input.KpointCoords, center_shift)
        self.assertEqual(
            self.sphinx_band_structure.get_k_mesh_by_cell(2 * np.pi / 2.81), [1, 1, 1]
        )

    def test_set_empty_states(self):
        with self.assertRaises(ValueError):
            self.sphinx.set_empty_states(-1)
        self.sphinx.set_empty_states(666)
        self.assertEqual(self.sphinx.input["EmptyStates"], 666)
        self.sphinx.set_empty_states()
        self.assertEqual(self.sphinx.input["EmptyStates"], "auto")

    def test_fix_spin_constraint(self):
        self.assertTrue(self.sphinx.fix_spin_constraint)
        with self.assertRaises(ValueError):
            self.sphinx.fix_spin_constraint = 3
        self.sphinx.fix_spin_constraint = False
        self.assertIsInstance(self.sphinx.fix_spin_constraint, bool)

    def test_calc_static(self):
        self.sphinx.calc_static()
        self.assertFalse("keepRho" in self.sphinx.input.sphinx.main.to_sphinx())
        self.assertTrue("blockCCG" in self.sphinx.input.sphinx.main.to_sphinx())
        self.sphinx.restart_file_list.append("randomfile")
        self.sphinx.calc_static()
        self.assertTrue("keepRho" in self.sphinx.input.sphinx.main.to_sphinx())
        self.assertEqual(self.sphinx.input["Estep"], 100)
        self.assertTrue("CCG" in self.sphinx.input.sphinx.main.to_sphinx())

    def test_calc_minimize(self):
        self.sphinx.calc_minimize(electronic_steps=100, ionic_steps=50)
        self.assertEqual(self.sphinx.input["Estep"], 100)
        self.assertEqual(self.sphinx.input["Istep"], 50)
        self.assertEqual(self.sphinx.input.sphinx.main["ricQN"]["maxSteps"], "50")

    def test_get_scf_group(self):
        with warnings.catch_warnings(record=True) as w:
            test_scf = self.sphinx_band_structure.get_scf_group(algorithm="wrong")
            self.assertEqual(len(w), 1)
            ref_scf = {
                "rhoMixing": "1.0",
                "spinMixing": "1.0",
                "dEnergy": 3.674932217565499e-06,
                "preconditioner": {
                    "type": "KERKER",
                    "scaling": 1.0,
                    "spinScaling": 1.0,
                },
                "maxSteps": "100",
                "blockCCG": {},
            }
            self.assertEqual(test_scf, ref_scf)

        ref_scf = {
            "rhoMixing": "1.0",
            "spinMixing": "1.0",
            "nPulaySteps": "0",
            "dEnergy": 3.674932217565499e-06,
            "maxSteps": "100",
            "preconditioner": {"type": 0},
            "blockCCG": {"maxStepsCCG": 0, "blockSize": 0, "nSloppy": 0},
            "noWavesStorage": True,
        }

        self.sphinx_band_structure.input["nPulaySteps"] = 0
        self.sphinx_band_structure.input["preconditioner"] = 0
        self.sphinx_band_structure.input["maxStepsCCG"] = 0
        self.sphinx_band_structure.input["blockSize"] = 0
        self.sphinx_band_structure.input["nSloppy"] = 0
        self.sphinx_band_structure.input["WriteWaves"] = False
        test_scf = self.sphinx_band_structure.get_scf_group()
        self.assertEqual(test_scf, ref_scf)

    def test_check_setup(self):
        self.assertFalse(self.sphinx.check_setup())

        self.sphinx_band_structure.load_default_groups()
        self.sphinx_band_structure.input.sphinx.basis.kPoint = {
            "coords": "0.5, 0.5, 0.5"
        }
        self.assertFalse(self.sphinx_band_structure.check_setup())

        self.sphinx_band_structure.load_default_groups()
        self.sphinx_band_structure.server.cores = 2000
        self.assertFalse(self.sphinx_band_structure.check_setup())

        self.sphinx_band_structure.input["EmptyStates"] = "auto"
        self.assertFalse(self.sphinx_band_structure.check_setup())
        self.sphinx_band_structure.structure.add_tag(spin=None)
        for i in range(len(self.sphinx_band_structure.structure)):
            self.sphinx_band_structure.structure.spin[i] = 4
        self.assertFalse(self.sphinx_band_structure.check_setup())

    def test_set_check_overlap(self):
        self.assertRaises(TypeError, self.sphinx_band_structure.set_check_overlap, 0)

    def test_set_occupancy_smearing(self):
        self.assertRaises(
            ValueError, self.sphinx_band_structure.set_occupancy_smearing, 0.1, 0.1
        )
        self.assertRaises(
            ValueError, self.sphinx_band_structure.set_occupancy_smearing, "fermi", -0.1
        )
        self.sphinx_band_structure.set_occupancy_smearing("fermi", 0.1)
        self.assertTrue("FermiDirac" in self.sphinx_band_structure.input)
        self.sphinx_band_structure.set_occupancy_smearing("methfessel", 0.1)
        self.assertTrue("MethfesselPaxton" in self.sphinx_band_structure.input)

    def test_load_default_groups(self):
        backup = self.sphinx_band_structure.structure.copy()
        self.sphinx_band_structure.structure = None
        self.assertRaises(
            AssertionError, self.sphinx_band_structure.load_default_groups
        )
        self.sphinx_band_structure.structure = backup

    def test_load_guess_group(self):
        guess = self.sphinx_band_structure.input.sphinx.initialGuess

        def reload_guess(filelist):
            # ugly hack: restart_file_list cannot be emptied otherwise
            self.sphinx_band_structure._restart_file_list = list()
            self.sphinx_band_structure.restart_file_list = filelist
            del guess["rho"]
            del guess["waves"]
            self.sphinx_band_structure.load_guess_group()

        with tempfile.TemporaryDirectory() as tmpdir:
            # create temporary files to pass file existence checks
            rho_file = os.path.join(tmpdir, "rho.sxb")
            waves_file = os.path.join(tmpdir, "waves.sxb")
            Path(rho_file).touch()
            Path(waves_file).touch()

            # test loading rho
            reload_guess([rho_file])
            self.assertEqual(guess.rho.file, '"' + rho_file + '"')

            # test loading waves
            reload_guess([waves_file])
            self.assertEqual(guess.waves.file, '"' + waves_file + '"')
            self.assertEqual(guess.rho.fromWaves, True)

            # test loading rho + waves
            reload_guess([rho_file, waves_file])
            self.assertEqual(guess.waves.file, '"' + waves_file + '"')
            self.assertEqual(guess.rho.file, '"' + rho_file + '"')

            # test default
            reload_guess([])
            self.assertEqual(guess.rho.atomicOrbitals, True)
            self.assertTrue("lcao" in guess.waves)

    def test_validate_ready_to_run(self):
        backup = self.sphinx_band_structure.structure.copy()
        self.sphinx_band_structure.structure = None
        self.assertRaises(
            AssertionError, self.sphinx_band_structure.validate_ready_to_run
        )
        self.sphinx_band_structure.structure = backup

        self.sphinx_band_structure.input["THREADS"] = 20
        self.sphinx_band_structure.server.cores = 10
        self.assertRaises(
            AssertionError, self.sphinx_band_structure.validate_ready_to_run
        )

        self.sphinx_band_structure.input.sphinx.main.clear()
        self.assertRaises(
            AssertionError, self.sphinx_band_structure.validate_ready_to_run
        )

        backup = self.sphinx.input.sphinx.basis.eCut
        self.sphinx.input.sphinx.basis.eCut = 400
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.basis.eCut = backup

        backup = self.sphinx.input.sphinx.basis.kPoint.copy()
        self.sphinx.input.sphinx.basis.kPoint.clear()
        self.sphinx.input.sphinx.basis.kPoint.coords = [0.5, 0.5, 0.25]
        self.sphinx.input.sphinx.basis.kPoint.weight = 1
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.basis.kPoint = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.ekt
        self.sphinx.input.sphinx.PAWHamiltonian.ekt = 0.0001
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.ekt = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.xc
        self.sphinx.input.sphinx.PAWHamiltonian.xc = "Wrong"
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.xc = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.xc
        self.sphinx.input.sphinx.PAWHamiltonian.xc = "Wrong"
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.xc = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.nEmptyStates
        self.sphinx.input.sphinx.PAWHamiltonian.nEmptyStates = 100
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.nEmptyStates = backup

        backup = self.sphinx.input.sphinx.structure.copy()
        self.sphinx.input.sphinx.structure.cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.structure = backup

        self.assertTrue(self.sphinx.validate_ready_to_run())

    def test_set_mixing_parameters(self):
        self.assertRaises(
            ValueError, self.sphinx.set_mixing_parameters, "LDA", 7, 1.0, 1.0
        )
        self.assertRaises(
            ValueError, self.sphinx.set_mixing_parameters, "PULAY", 7, -0.1, 1.0
        )
        self.assertRaises(
            ValueError, self.sphinx.set_mixing_parameters, "PULAY", 7, 1.0, 2.0
        )
        self.assertRaises(
            ValueError,
            self.sphinx.set_mixing_parameters,
            "PULAY",
            7,
            1.0,
            1.0,
            -0.1,
            0.5,
        )
        self.assertRaises(
            ValueError,
            self.sphinx.set_mixing_parameters,
            "PULAY",
            7,
            1.0,
            1.0,
            0.1,
            -0.5,
        )
        self.sphinx.set_mixing_parameters(
            method="PULAY",
            n_pulay_steps=7,
            density_mixing_parameter=0.5,
            spin_mixing_parameter=0.2,
            density_residual_scaling=0.1,
            spin_residual_scaling=0.3,
        )
        self.assertEqual(self.sphinx.input["rhoMixing"], 0.5)
        self.assertEqual(self.sphinx.input["spinMixing"], 0.2)
        self.assertEqual(self.sphinx.input["rhoResidualScaling"], 0.1)
        self.assertEqual(self.sphinx.input["spinResidualScaling"], 0.3)

    def test_exchange_correlation_functional(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.sphinx.exchange_correlation_functional = "llda"
            self.assertEqual(len(w), 1)
            self.assertIsInstance(w[-1].message, SyntaxWarning)
        self.sphinx.exchange_correlation_functional = "pbe"
        self.assertEqual(self.sphinx.exchange_correlation_functional, "PBE")

    def test_write_structure(self):
        cell = (self.sphinx.structure.cell / BOHR_TO_ANGSTROM).tolist()
        pos_2 = (self.sphinx.structure.positions[1] / BOHR_TO_ANGSTROM).tolist()

        file_content = [
            f"cell = {cell};\n",
            "species {\n",
            '\telement = "Fe";\n',
            "\tatom {\n",
            '\t\tlabel = "spin_0.5";\n',
            "\t\tcoords = [0.0, 0.0, 0.0];\n",
            "\t\tmovable = true;\n",
            "\t}\n",
            "\tatom {\n",
            '\t\tlabel = "spin_0.5";\n',
            "\t\tcoords = [2.4566439620135014, 2.4566439620135014, 2.4566439620135014];\n",
            "\t}\n",
            "}\n",
            "symmetry {\n",
            "\toperator {\n",
            "\t\tS = [[1,0,0],[0,1,0],[0,0,1]];\n",
            "\t}\n",
            "}\n",
        ]
        self.assertEqual(
            "".join(file_content), self.sphinx.input.sphinx.structure.to_sphinx()
        )

    def test_collect_aborted(self):
        self.sphinx_aborted.collect_output()
        self.sphinx_aborted.decompress()
        self.assertTrue(self.sphinx_aborted.status.aborted)

    def test_collect_2_5(self):
        output = self.sphinx_2_5.output
        output.collect(directory=self.sphinx_2_5.working_directory)
        self.assertTrue(all(np.diff(output.generic.dft.computation_time) > 0))
        self.assertTrue(
            all(output.generic.dft.energy_free - output.generic.dft.energy_int < 0)
        )
        self.assertTrue(
            all(output.generic.dft.energy_free - output.generic.dft.energy_zero < 0)
        )
        list_values = [
            "scf_energy_int",
            "scf_energy_zero",
            "scf_energy_free",
            "scf_convergence",
            "scf_electronic_entropy",
            "atom_scf_spins",
        ]
        for list_one in list_values:
            for list_two in list_values:
                self.assertEqual(
                    len(output.generic.dft[list_one]), len(output.generic.dft[list_two])
                )

        rho = self.sphinx_2_5.output.charge_density
        vel = self.sphinx_2_5.output.electrostatic_potential
        self.assertIsNotNone(rho.total_data)
        self.assertIsNotNone(vel.total_data)

    def test_check_band_occupancy(self):
        self.assertTrue(self.sphinx_2_5.output.check_band_occupancy())
        self.assertTrue(self.sphinx_2_5.nbands_convergence_check())

    def test_collect_2_3(self):
        file_location = os.path.join(
            self.file_location, "../static/sphinx/sphinx_test_2_3_hdf5/sphinx_test_2_3/"
        )
        residue_lst = np.loadtxt(file_location + "residue.dat")[:, 1].reshape(1, -1)
        residue_lst = (residue_lst).tolist()
        energy_int_lst = np.loadtxt(file_location + "energy.dat")[:, 2].reshape(1, -1)
        energy_int_lst = (energy_int_lst * HARTREE_TO_EV).tolist()
        with open(file_location + "sphinx.log") as ffile:
            energy_free_lst = [
                [
                    float(line.split("=")[-1]) * HARTREE_TO_EV
                    for line in ffile
                    if line.startswith("F(")
                ]
            ]
        eig_lst = [np.loadtxt(file_location + "eps.dat")[:, 1:].tolist()]
        self.sphinx_2_3.collect_output()
        self.assertEqual(residue_lst, self.sphinx_2_3.output.generic.dft["scf_residue"])
        self.assertEqual(
            energy_int_lst, self.sphinx_2_3.output.generic.dft["scf_energy_int"]
        )
        self.assertEqual(
            eig_lst,
            self.sphinx_2_3.output.generic.dft["bands_eigen_values"].tolist()[0],
        )
        self.assertEqual(
            energy_free_lst,
            self.sphinx_2_3.output.generic.dft["scf_energy_free"],
        )
        self.assertEqual(
            21.952 * BOHR_TO_ANGSTROM**3,
            self.sphinx_2_3.output.generic["volume"],
        )

    def test_structure_parsing(self):
        self.sphinx_2_3.output.collect_relaxed_hist(
            file_name="relaxedHist_2.sx", cwd=self.sphinx_2_3.working_directory
        )

    def test_density_of_states(self):
        dos = self.sphinx_2_5.get_density_of_states()
        self.assertLess(dos["grid"][dos["dos"][0].argmax()], 0)

    def test_convergence_precision(self):
        job = self.project.create_job(job_type="Sphinx", job_name="energy_convergence")
        job.structure = self.project.create.structure.ase.bulk("Al", "fcc", 3.5)
        job.set_convergence_precision(
            ionic_energy_tolerance=1e-5, electronic_energy=1e-8
        )
        job.calc_minimize(ionic_steps=250, electronic_steps=200)
        self.assertAlmostEqual(
            float(job.input.sphinx.main.ricQN.bornOppenheimer.scfDiag.dEnergy)
            * HARTREE_TO_EV,
            1e-8,
        )
        self.assertAlmostEqual(
            float(job.input.sphinx.main.ricQN.dEnergy) * HARTREE_TO_EV, 1e-5
        )
        self.assertEqual(int(job.input.sphinx.main.ricQN.maxSteps), 250)
        self.assertEqual(
            int(job.input.sphinx.main.ricQN.bornOppenheimer.scfDiag.maxSteps), 200
        )

    def test_partial_constraint(self):
        spx = self.project.create.job.Sphinx("spx_partial_constraint")
        spx.structure = self.project.create.structure.bulk("Fe", cubic=True)
        spx.structure.set_initial_magnetic_moments(2 * [2])
        spx.fix_spin_constraint = True
        spx.structure.spin_constraint[-1] = False
        spx.calc_static()
        spx.server.run_mode.manual = True
        spx.run()
        self.assertEqual(spx["spins.in"], ["2\n", "X\n"])

    @unittest.skipIf(
        "linux" not in sys.platform, "Running of the addon is only supported on linux"
    )
    def test_run_addon(self):
        def try_remove(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        logfile_name = os.path.join(self.sphinx.working_directory, "sxcheckinput.log")
        # test addons from compressed job
        self.sphinx.compress()
        try:
            out = self.sphinx.run_addon(
                "sxcheckinput",
                from_tar="input.sx",
                log=False,
                version="fake_addon",
                debug=True,
            )
            self.assertTrue(out.returncode == 0, msg=out.stdout + out.stderr)
            self.assertIn("Checking for dublets...ok\n", out.stdout)
            with self.assertRaises(FileNotFoundError):
                self.sphinx.run_addon("sxcheckinput")
            with self.assertRaises(KeyError):
                self.sphinx.run_addon("sxcheckinput", from_tar=[], version="inexistent")
            with self.assertRaises(TypeError):
                self.sphinx.run_addon("sxcheckinput", from_tar=[], version=3.14)
            try_remove(logfile_name)
            self.sphinx.run_addon(
                "sxcheckinput",
                ["", "", ""],  # fake arguments
                from_tar=["inexistentFile", "input.sx"],
                silent=True,
                version="fake_addon",
            )
            self.assertTrue(os.path.exists(logfile_name))
            # check that addon doesn't run without input.sx
            out = self.sphinx.run_addon(
                "sxcheckinput", from_tar=[], log=False, version="fake_addon", debug=True
            )
            self.assertTrue(out.returncode != 0)
        finally:
            self.sphinx.decompress()
            try_remove(logfile_name)

        # test addon from decompressed job (with log file)
        self.sphinx.run_addon("sxcheckinput", silent=True, version="fake_addon")
        with open(logfile_name) as logfile:
            lines = logfile.readlines()
        self.assertIn("Checking for dublets...ok\n", lines)
        try_remove(logfile_name)


if __name__ == "__main__":
    unittest.main()
