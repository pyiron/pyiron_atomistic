# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from structuretoolkit.build import get_grainboundary_info, grainboundary
import functools


__author__ = "Ujjal Saikia"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Feb 26, 2021"


class AimsgbFactory:
    @staticmethod
    @functools.wraps(get_grainboundary_info)
    def info(*args, table_view=False, **kwargs):
        if table_view:
            print(get_grainboundary_info(*args, **kwargs).__str__())
        else:
            return get_grainboundary_info(*args, **kwargs)

    @staticmethod
    @functools.wraps(grainboundary)
    def build(*args, **kwargs):
        return ase_to_pyiron(grainboundary(*args, **kwargs))
