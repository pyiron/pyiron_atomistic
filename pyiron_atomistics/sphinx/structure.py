# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import re
import numpy as np
import scipy.constants
from pyiron_atomistics.atomistics.structure.parser_base import KeywordTreeParser
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable

__author__ = "Christoph Freysoldt"
__copyright__ = (
    "Copyright 2023, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "2.0"
__maintainer__ = "Christoph Freysoldt"
__email__ = "freysoldt@mpie.de"
__status__ = "production"
__date__ = "Dec 8, 2023"

BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)


class StructParser(KeywordTreeParser):
    """
    This class reads one or more structures in sx format.
    """

    def __init__(self, file):
        super().__init__({"structure": self.parse_structure})
        self.configs = []
        self.parse(file)

    def parse_structure(self):
        """Parses structure{} blocks"""
        self.keylevels.append({"cell": self.parse_cell, "species": self.parse_species})
        self.extract_via_regex("structure")
        # --- initialize for next structure
        self.cell = None
        self.positions = []
        self.species = []
        self.indices = []
        self.ispecies = -1
        # continue parsing
        yield
        # create Atoms object and append it to configs
        pse = PeriodicTable()
        atoms = Atoms(
            species=[pse.element(s) for s in self.species],
            indices=self.indices,
            cell=self.cell * BOHR_TO_ANGSTROM,
            positions=np.array(self.positions) * BOHR_TO_ANGSTROM,
            pbc=True,
        )
        self.configs.append(atoms)

    def parse_cell(self):
        """Read the cell"""
        txt = self.extract_var("cell")
        self.cell = self.get_vector("cell", txt).reshape(3, 3)

    def parse_species(self):
        """Parses species{} blocks"""
        self.extract_via_regex("species")
        self.keylevels.append({"element": self.get_element, "atom": self.read_atom})
        self.ispecies += 1

    def get_element(self):
        """Read element"""
        txt = self.extract_var("element")
        self.species.append(re.sub('.*"([^"]*)".*', r"\1", txt))

    def read_atom(self):
        """Read atomic coordinates from an atom block"""
        txt = self.extract_var("atom", "{}")
        self.positions.append(self.get_vector("coords", txt))
        self.indices.append(self.ispecies)
        if "label" in txt:
            label = re.sub(r'.*label\s*=\s*"([^"]+)"\s*;.*', r"\1", txt)
            print(f"atom {len(self.positions)} label={label}")


def read_atoms(filename="structure.sx"):
    """
    Args:
        filename (str): Filename of the sphinx structure file

    Returns:
        pyiron_atomistics.objects.structure.atoms.Atoms instance (or a list of them)

    """
    configs = StructParser(filename).configs
    return configs[0] if len(configs) == 1 else configs
