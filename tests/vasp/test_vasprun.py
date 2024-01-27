# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
import posixpath
import numpy as np
from pyiron_atomistics.vasp.vasprun import Vasprun, VasprunError
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.dft.waves.electronic import ElectronicStructure

__author__ = "surendralal"


class TestVasprun(unittest.TestCase):

    """
    Testing the Vasprun() module.
    """

    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.vp_list = list()
        cls.direc = os.path.join(
            cls.file_location, "../static/vasp_test_files/vasprun_samples"
        )
        file_list = sorted(os.listdir(cls.direc))
        del file_list[file_list.index("vasprun_spoilt.xml")]
        cls.num_species = [3, 1, 2, 2, 3, 4, 4, 4, 10, 2]
        for f in file_list:
            vp = Vasprun()
            filename = posixpath.join(cls.direc, f)
            vp.from_file(filename)
            cls.vp_list.append(vp)

    def test_from_file(self):
        vp = Vasprun()
        filename = posixpath.join(self.direc, "vasprun_spoilt.xml")
        self.assertRaises(VasprunError, vp.from_file, filename)

    def test_get_potentiostat_output(self):
        for i, vp in enumerate(self.vp_list):
            if i == 8:
                self.assertIsInstance(vp.get_potentiostat_output(), dict)
            else:
                self.assertIsNone(vp.get_potentiostat_output())

    def test_parse_generator(self):
        vasprun_data = {
                        'vasprun_1.xml': {
                            'program': 'vasp',
                            'version': '4.6.28',
                            'subversion': '25Jul05 complex  parallel',
                            'platform': 'LinuxIFC',
                            'date': '2016 04 10',
                            'time': '17:29:20'
                        },
                        'vasprun_2.xml': {
                            'program': 'vasp',
                            'version': '5.3.5',
                            'subversion': '31Mar14 (build Apr 17 2014 15:55:21) complex                          parallel',
                            'platform': 'LinuxIFC',
                            'date': '2016 06 22',
                            'time': '20:08:33'
                        },
                        'vasprun_3.xml': {
                            'program': 'vasp',
                            'version': '5.3.5',
                            'subversion': '31Mar14 (build Dec 16 2016 13:46:59) complex                          parallel',
                            'platform': 'LinuxIFC',
                            'date': '2017 04 25',
                            'time': '19:09:22'
                        },
                        'vasprun_4.xml': {
                            'program': 'vasp',
                            'version': '5.3.5',
                            'subversion': '31Mar14 (build Dec 16 2016 13:46:59) complex                          parallel',
                            'platform': 'LinuxIFC',
                            'date': '2017 04 25',
                            'time': '19:47:01'
                        },
                        'vasprun_5.xml': {
                            'program': 'vasp',
                            'version': '5.3.5',
                            'subversion': '31Mar14 (build Dec 16 2016 13:46:59) complex                          parallel',
                            'platform': 'LinuxIFC',
                            'date': '2017 04 26',
                            'time': '12:55:54'
                        },
                        'vasprun_6.xml': {
                            'program': 'vasp',
                            'version': '5.4.1',
                            'subversion': '24Jun15 (build Dec 18 2016 09:48:16) gamma-only                       parallel',
                            'platform': 'IFC91_ompi',
                            'date': '2017 11 21',
                            'time': '15:01:21'
                        },
                        'vasprun_7.xml': {
                            'program': 'vasp',
                            'version': '5.4.4.18Apr17-6-g9f103f2a35',
                            'subversion': '(build Oct 10 2019 11:41:53) complex            parallel',
                            'platform': 'LinuxIFC',
                            'date': '2019 12 02',
                            'time': '06:30:52'
                        },
                        'vasprun_8.xml': {
                            'program': 'vasp',
                            'version': '5.4.4.18Apr17-6-g9f103f2a35',
                            'subversion': '(build Oct 10 2019 11:41:53) complex            parallel',
                            'platform': 'LinuxIFC',
                            'date': '2019 12 02',
                            'time': '06:30:52'
                        },
                        'vasprun_9.xml': {
                            'program': 'vasp',
                            'version': '5.4.4.18Apr17-6-g9f103f2a35',
                            'subversion': '(build Mar 04 2021 11:38:41) gamma-only         parallel',
                            'platform': 'LinuxIFC',
                            'date': '2021 03 04',
                            'time': '11:39:31'
                        },
                        'vasprun_line.xml': {
                            'program': 'vasp',
                            'version': '5.4.1',
                            'subversion': '24Jun15 (build Dec 18 2016 09:36:42) complex                          parallel',
                            'platform': 'IFC91_ompi',
                            'date': '2019 03 19',
                            'time': '16:54:22'
                        }
                    }

        for vp in self.vp_list:
            self.assertIsInstance(vp.vasprun_dict["generator"], dict)
            self.assertTrue("program" in vp.vasprun_dict["generator"].keys())
            self.assertTrue("version" in vp.vasprun_dict["generator"].keys())
            self.assertTrue("time" in vp.vasprun_dict["generator"].keys())
            self.assertTrue("subversion" in vp.vasprun_dict["generator"].keys())
            self.assertTrue("platform" in vp.vasprun_dict["generator"].keys())
            self.assertTrue("date" in vp.vasprun_dict["generator"].keys())
            self.assertEqual(vp.vasprun_dict["generator"], vasprun_data[os.path.basename(vp.filename)])


    def test_parse_incar(self):
        expected_incar_outputs = {
            'vasprun_1.xml': {'SYSTEM': 'passivated_AlN_vacuum_118.0', 'ISTART': 0, 'PREC': 'accurate', 'ALGO': 'Fast', 'ICHARG': 2, 'NELM': 40, 'NELMDL': -5, 'NELMIN': 8, 'EDIFF': 1e-05, 'EDIFFG': 0.001, 'IBRION': 1, 'NSW': 10, 'ISIF': 2, 'ENCUT': 500.0, 'LREAL': False, 'ISMEAR': 2, 'SIGMA': 0.1, 'LWAVE': False, 'LCHARG': False, 'LVTOT': False, 'REF_TYPE': 'none'},
            'vasprun_2.xml': {'SYSTEM': 'ToDo', 'ISTART': 0, 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 300, 'IBRION': -1, 'ENCUT': 400.0, 'LREAL': False, 'SIGMA': 0.1, 'LCHARG': True, 'LVTOT': True, 'LORBIT': False, 'LDIPOL': True, 'IDIPOL': 3, 'DIP%EFIELD': -0.01},
            'vasprun_3.xml': {'SYSTEM': 'ToDo', 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 60, 'IBRION': -1, 'ENCUT': 250.0, 'LREAL': False, 'LWAVE': False, 'LORBIT': False},
            'vasprun_4.xml': {'SYSTEM': 'ToDo', 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 60, 'IBRION': -1, 'ENCUT': 250.0, 'LREAL': False, 'LWAVE': False, 'LORBIT': False},
            'vasprun_5.xml': {'SYSTEM': 'ToDo', 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 60, 'IBRION': -1, 'ENCUT': 250.0, 'LREAL': False, 'LWAVE': False, 'LORBIT': False},
            'vasprun_6.xml': {'SYSTEM': 'ToDo', 'ISTART': 1, 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 200, 'IBRION': -1, 'EDIFF': 1e-06, 'ISYM': 0, 'ENCUT': 500.0, 'LREAL': 'Auto', 'ISMEAR': -1, 'SIGMA': 0.1, 'LWAVE': True, 'LCHARG': True, 'LVTOT': True, 'LVHAR': True, 'LORBIT': False, 'KPOINT_BSE': [-1, 0, 0, 0], 'LDIPOL': True, 'IDIPOL': 3, 'DIPOL': [0.0, 0.0, 0.31870518]},
            'vasprun_7.xml': {'SYSTEM': 'pseudo_H', 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 400, 'IBRION': -1, 'LREAL': False, 'LWAVE': False, 'LORBIT': 0, 'KPOINT_BSE': [-1, 0, 0, 0]},
            'vasprun_8.xml': {'SYSTEM': 'pseudo_H', 'PREC': 'accurate', 'ALGO': 'Fast', 'NELM': 400, 'IBRION': -1, 'LREAL': False, 'LWAVE': False, 'LORBIT': 0, 'KPOINT_BSE': [-1, 0, 0, 0]},
            'vasprun_9.xml': {'SYSTEM': 'Ge.water', 'PREC': 'normal', 'ALGO': 'FAST', 'NELM': 100, 'IBRION': 0, 'EDIFF': 1e-05, 'EDIFFG': -0.01, 'NSW': 2, 'IWAVPR': 10, 'ISYM': 0, 'ENCUT': 400.0, 'POTIM': 0.5, 'TEBEG': 400.0, 'SMASS': -3.0, 'LREAL': 'A', 'ISMEAR': 0, 'SIGMA': 0.05, 'LWAVE': False, 'LCHARG': False, 'LVTOT': True, 'LVHAR': True, 'MDALGO': 5, 'CSVRTAU': 400.0, 'ICCE': 4, 'CCETYP': 3, 'POTTAU': 100.0, 'UBEG': 0.0, 'POTCAP': 0.03339719, 'ILOC': 3, 'KPOINT_BSE': [-1, 0, 0, 0], 'LDIPOL': True, 'IDIPOL': 3, 'DIPOL': [0.0, 0.0, 0.337978]},
            'vasprun_line.xml': {'SYSTEM': 'Test2', 'ISTART': 0, 'PREC': 'high', 'ALGO': 'FAST', 'ICHARG': 11, 'NELM': 400, 'EDIFF': 1e-07, 'ENCUT': 450.0, 'LREAL': False, 'ISMEAR': 1, 'SIGMA': 0.1, 'LWAVE': False, 'LCHARG': True, 'LORBIT': False, 'KPOINT_BSE': [-1, 0, 0, 0]}
        }

        for _, vp in enumerate(self.vp_list):
            filename = os.path.basename(vp.filename)  # Extract filename
            self.assertIn(filename, expected_incar_outputs)  # Ensure expected output is defined for this file
            expected_incar = expected_incar_outputs[filename]
            actual_incar = vp.vasprun_dict["incar"]
            self.assertEqual(actual_incar, expected_incar)

    def test_parse_kpoints(self):
        for vp in self.vp_list:
            d = vp.vasprun_dict["kpoints"]
            self.assertIsInstance(d, dict)
            self.assertIsInstance(d["generation"], dict)
            if d["generation"]["scheme"] in ["Monkhorst-Pack", "Gamma"]:
                self.assertIsInstance(d["generation"]["divisions"], np.ndarray)
                self.assertIsInstance(d["generation"]["genvec"], np.ndarray)
                self.assertIsInstance(d["generation"]["shift"], np.ndarray)
                self.assertIsInstance(d["generation"]["usershift"], np.ndarray)
                self.assertTrue(len(d["generation"]["divisions"]) == 3)
                self.assertTrue(len(d["generation"]["genvec"]) == 3)
                self.assertTrue(len(d["generation"]["genvec"].T) == 3)
                self.assertTrue(len(d["generation"]["shift"]) == 3)
                self.assertTrue(len(d["generation"]["usershift"]) == 3)
            if d["generation"]["scheme"] in ["listgenerated"]:
                self.assertIsInstance(d["line_mode_kpoints"], np.ndarray)
                self.assertTrue(len(d["line_mode_kpoints"][-1]), 3)
            self.assertIsInstance(d["kpoint_list"], np.ndarray)
            self.assertIsInstance(d["kpoint_weights"], np.ndarray)
            self.assertEqual(len(d["kpoint_list"]), len(d["kpoint_weights"]))
            self.assertTrue(len(d["kpoint_list"].T) == 3)
            self.assertIsInstance(d["kpoint_weights"][-1], float)

    def test_parse_atom_info(self):
        for vp in self.vp_list:
            d = vp.vasprun_dict["atominfo"]
            self.assertIsInstance(d, dict)
            self.assertIsInstance(d["species_dict"], dict)
            self.assertIsInstance(d["species_list"], list)
            n_atoms = 0
            for key, value in d["species_dict"].items():
                n_atoms += d["species_dict"][key]["n_atoms"]
            self.assertEqual(n_atoms, len(d["species_list"]))

    def test_parse_structure(self):
        for vp in self.vp_list:
            for pos_tag in ["init_structure", "final_structure"]:
                d = vp.vasprun_dict[pos_tag]
                self.assertIsInstance(d["positions"], np.ndarray)
                if "selective_dynamics" in d.keys():
                    self.assertIsInstance(d["selective_dynamics"], np.ndarray)

    def test_parse_sc_step(self):
        for vp in self.vp_list:
            d = vp.vasprun_dict
            self.assertIsInstance(d["scf_energies"], list)
            self.assertIsInstance(d["scf_dipole_moments"], list)

    def test_parse_calc(self):
        for vp in self.vp_list:
            d = vp.vasprun_dict
            self.assertIsInstance(d["positions"], np.ndarray)
            self.assertIsInstance(d["forces"], np.ndarray)
            self.assertIsInstance(d["cells"], np.ndarray)
            self.assertEqual(len(d["cells"]), len(d["positions"]))
            self.assertEqual(len(d["forces"]), len(d["positions"]))
            self.assertEqual(len(d["total_energies"]), len(d["positions"]))
            self.assertEqual(len(d["forces"]), len(d["scf_energies"]))
            self.assertEqual(len(d["forces"]), len(d["scf_dipole_moments"]))
            self.assertEqual(len(d["forces"].T), 3)
            self.assertEqual(len(d["positions"].T), 3)
            self.assertFalse(len(d["positions"][d["positions"] > 1.01]) > 0)
            self.assertFalse(np.max(d["positions"]) > 1.01)
            self.assertEqual(np.shape(d["cells"][0]), np.shape(np.eye(3)))
            self.assertIsInstance(d["grand_eigenvalue_matrix"], np.ndarray)
            self.assertIsInstance(d["grand_occupancy_matrix"], np.ndarray)
            self.assertEqual(
                np.shape(d["grand_occupancy_matrix"]),
                np.shape(d["grand_eigenvalue_matrix"]),
            )
            [n_spin, n_kpts, n_bands] = np.shape(d["grand_occupancy_matrix"])
            self.assertEqual(len(d["kpoints"]["kpoint_list"]), n_kpts)
            self.assertEqual(len(d["kpoints"]["kpoint_list"]), n_kpts)
            self.assertGreaterEqual(n_spin, 1)
            self.assertGreaterEqual(n_bands, 1)

    def test_pdos_parser(self):
        for vp in self.vp_list:
            d = vp.vasprun_dict
            if "efermi" in d.keys():
                self.assertIsInstance(d["efermi"], float)
            if "grand_dos_matrix" in d.keys():
                [n_spin0, n_kpts0, n_bands0] = np.shape(d["grand_occupancy_matrix"])
                [n_spin, n_kpts, n_bands, n_atoms, n_orbitals] = np.shape(
                    d["grand_dos_matrix"]
                )
                self.assertEqual(len(d["kpoints"]["kpoint_list"]), n_kpts)
                self.assertEqual(len(d["positions"][0]), n_atoms)
                self.assertEqual(n_spin, n_spin0)
                self.assertEqual(n_bands, n_bands0)
                self.assertEqual(n_kpts, n_kpts0)
                self.assertIsInstance(d["grand_dos_matrix"], np.ndarray)
                self.assertIsInstance(d["orbital_dict"], dict)
                self.assertEqual(len(d["orbital_dict"].keys()), n_orbitals)

    def test_parse_parameters(self):
        for vp in self.vp_list:
            d = vp.vasprun_dict
            self.assertIsInstance(d, dict)

    def test_get_initial_structure(self):
        for vp in self.vp_list:
            basis = vp.get_initial_structure()
            self.assertIsInstance(basis, Atoms)
            self.assertTrue(np.max(basis.get_scaled_positions()) < 1.01)

    def test_get_final_structure(self):
        for vp in self.vp_list:
            basis = vp.get_final_structure()
            self.assertIsInstance(basis, Atoms)
            self.assertTrue(np.max(basis.get_scaled_positions()) < 1.01)
            self.assertFalse(np.max(basis.positions) < 1.01)

    def test_get_electronic_structure(self):
        for vp in self.vp_list:
            es_obj = vp.get_electronic_structure()
            self.assertIsInstance(es_obj, ElectronicStructure)
            if "grand_dos_matrix" in vp.vasprun_dict.keys():
                [_, n_kpts, n_bands, _, _] = np.shape(
                    vp.vasprun_dict["grand_dos_matrix"]
                )
                self.assertEqual(len(es_obj.kpoints), n_kpts)
                self.assertEqual(len(es_obj.kpoints[0].bands[0]), n_bands)

    def test_species_info(self):
        for i, vp in enumerate(self.vp_list):
            self.assertEqual(
                len(vp.vasprun_dict["atominfo"]["species_dict"].keys()),
                self.num_species[i],
            )
            self.assertEqual(
                vp.get_initial_structure().get_number_of_species(), self.num_species[i]
            )
            self.assertEqual(
                vp.get_final_structure().get_number_of_species(), self.num_species[i]
            )

    def test_energies(self):
        for i, vp in enumerate(self.vp_list):
            self.assertIsInstance(vp.vasprun_dict["total_0_energies"], np.ndarray)
            if i == 7:
                self.assertEqual(vp.vasprun_dict["scf_fr_energies"][0][0], 0.0)


if __name__ == "__main__":
    unittest.main()
