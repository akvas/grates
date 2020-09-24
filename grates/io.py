# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
File I/O for gravity field representations and gridded data.
"""

import datetime as dt
import tarfile
from grates.gravityfield import PotentialCoefficients, TimeSeries


def __parse_gfc_entry(line):
    """Return the values for both coefficients in a GFC file line."""
    sline = line.split()
    n = int(sline[1])
    m = int(sline[2])

    return n, m, float(sline[3]), float(sline[4])


def loadgfc(file_name, max_degree=None):
    """
    Read a set of potential coefficients from a GFC file.

    Parameters
    ----------
    file_name : str
        name of GFC file
    max_degree : int
        truncate PotentialCoefficients instance at degree max_degree  (Default: return the full field)

    Returns
    -------
    gf : PotentialCoefficients
        PotentialCoefficients instance
    """
    gf = PotentialCoefficients(3.986004415E+14, 6378136.3)

    with open(file_name, 'r') as f:

        for line in f:
            if line.startswith('gfc'):
                n, m, cnm, snm = __parse_gfc_entry(line)
                if max_degree and n > max_degree:
                    continue
                gf.append('c', n, m, cnm)
                gf.append('s', n, m, snm)

            elif line.startswith('radius'):
                gf.R = float(line.split()[-1])
            elif line.startswith('earth_gravity_constant'):
                gf.GM = float(line.split()[-1])

    return gf


def loadtn13(file_name, GM=3.986004415E+14, R=6378136.3):
    """
    Read GRACE Technical Note 13 (degree 1, geocenter motion)

    Parameters
    ----------
    file_name : str
        name of ASCII file
    GM : float
        geocentric gravitational constant
    R : float
        reference radius

    Returns
    -------
    time_series : TimeSeries
        time series of PotentialCoefficients instances
    """
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith('GRCOF2'):
                sline = line.split()
                gf = PotentialCoefficients(GM, R)
                gf.append('c', int(sline[1]), int(sline[2]), float(sline[3]))

                time_start = dt.datetime.strptime(sline[7], '%Y%m%d.%H%M')
                time_end = dt.datetime.strptime(sline[8], '%Y%m%d.%H%M')
                gf.epoch = time_start + (time_end-time_start)*0.5

                sline = f.readline().split()
                gf.append('c', int(sline[1]), int(sline[2]), float(sline[3]))
                gf.append('s', int(sline[1]), int(sline[2]), float(sline[4]))

                data.append(gf)

    return TimeSeries(data)


def loadesm(file_name, min_degree=0, max_degree=None):
    """
    Read spherical harmonics coefficients from an ESA ESM archive.

    Parameters
    ----------
    file_name : str
        archive file name
    min_degree : int
        return coefficients starting from min_degree
    max_degree : int
        return coefficients up to and including max_degree
    """
    tar = tarfile.open(file_name, 'r:gz')
    data = []
    for member in tar.getmembers():

        gf = PotentialCoefficients(3.986004415E+14, 6378136.3)
        epoch = dt.datetime.strptime(member.name[-15:-4], '%Y%m%d_%H')
        gf.epoch = epoch

        f = tar.extractfile(member)
        for line in f:
            if line.startswith(b'gfc'):
                n, m, cnm, snm = __parse_gfc_entry(line.replace(b'D', b'e'))
                if max_degree and n > max_degree or n < min_degree:
                    continue
                gf.append('c', n, m, cnm)
                gf.append('s', n, m, snm)

            elif line.startswith(b'radius'):
                gf.R = float(line.split()[-1].replace(b'D', b'e'))
            elif line.startswith(b'earth_gravity_constant'):
                gf.GM = float(line.split()[-1].replace(b'D', b'e'))

        data.append(gf)

    return TimeSeries(data)



