# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
grates - Gravity Field Analysis Tools for Earth System Studies
==============================================================

What is grates
--------------

grates is a free Open Source software package for computing mass anomalies from
time variable gravity field solutions. It is tailored for products of the GRACE
mission and its successor GRACE-FO.

The features of grates are:

 * File I/O for common data formats including GFC files and netCDF files
 * Basic arithmetic operations for sets of potential coefficients
 * Propagation of spherical harmonic coefficients to gridded mass anomalies
 * Spatial filtering of potential coefficients

Modules
-------

.. autosummary::
    :toctree: _generated
    :template: grates_module.rst

    grates.data
    grates.experimental
    grates.filter
    grates.gravityfield
    grates.grid
    grates.io
    grates.kernel
    grates.lstsq
    grates.plot
    grates.time
    grates.transport
    grates.utilities

"""

from . import data
from . import filter
from . import gravityfield
from . import grid
from . import io
from . import kernel
from . import lstsq
from . import time
from . import transport
from . import utilities
try:
    from . import plot
except ImportError:
    __all__ = ['data', 'filter', 'gravityfield', 'grid', 'io', 'kernel', 'lstsq', 'time', 'transport', 'utilities']
else:
    __all__ = ['data', 'filter', 'gravityfield', 'grid', 'io', 'kernel', 'lstsq', 'time', 'transport', 'utilities', 'plot']
