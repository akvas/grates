# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
File I/O for gravity field representations and gridded data.
"""

import datetime as dt
import tarfile
import abc
import gzip
import numpy as np
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


class SINEXBlock(metaclass=abc.ABCMeta):
    """
    Base class for blocks of a (spherical harmonics) SINEX file.
    """

    @staticmethod
    def epoch2datetime(line_remainder):
        """
        Converts a SINEX time stamp into a datetime object.

        Parameters
        ----------
        line_remainder : bytes
            slice of the line where the time stamp starts.

        Returns
        -------
        epoch : datetime
            time stamp as datetime object
        """
        sline = line_remainder.split(b':')
        year = int(sline[0])
        if year < 100:
            format = '%y'
        else:
            format = '%Y'
        epoch = dt.datetime.strptime(sline[0].decode(), format)
        epoch += dt.timedelta(days=int(sline[1]) - 1)
        epoch += dt.timedelta(seconds=int(sline[2][0:5]))

        return epoch

    @staticmethod
    def max_degree():
        return None

    @staticmethod
    def parameter_count():
        return None

    @abc.abstractmethod
    def read(self, f):
        """
        Read the block (meta-) data from the file object f.

        Parameters
        ----------
        f : file object
            file object ot be read.
        """
        pass


class SINEXSphericalHarmonicsVector(SINEXBlock):
    """
    A SINEX block containing a spherical harmonic vector, possibly with uncertainties.

    Parameters
    ----------
    id_line : bytes
        starting line of block in file
    """
    def __init__(self, id_line):

        self.type = id_line[1:].decode()
        self.anm = np.empty((0, 0))
        self.sigma = np.empty((0, 0))
        self.index = np.empty((0, 0), dtype=int)

    def __parse_line(self, line):
        """
        Parse a line within the block.

        Parameters
        ----------
        line : bytes
            line containing the vector entries
        """
        ptype = line[7:13].strip()

        if ptype not in [b'CN', b'SN']:
            raise ValueError('Parameter type <' + ptype.decode() + '> not supported.')

        index = int(line[1:6]) - 1
        degree = int(line[14:18].strip())
        order = int(line[22:26].strip())

        row_index = degree if ptype == b'CN' else order - 1
        col_index = order if ptype == b'CN' else degree

        if degree > self.anm.shape[0] - 1:
            tmp_anm = np.zeros((degree + 1, degree + 1))
            tmp_sigma = np.zeros((degree + 1, degree + 1))
            tmp_index = np.full((degree + 1, degree + 1), -1, dtype=int)
            tmp_anm[0:self.anm.shape[0], 0:self.anm.shape[1]] = self.anm.copy()
            tmp_sigma[0:self.sigma.shape[0], 0:self.sigma.shape[1]] = self.sigma.copy()
            tmp_index[0:self.index.shape[0], 0:self.index.shape[1]] = self.index.copy()

            self.anm = tmp_anm
            self.sigma = tmp_sigma
            self.index = tmp_index

        self.index[row_index, col_index] = index
        self.anm[row_index, col_index] = float(line[47:68])
        if not self.type.startswith('SOLUTION/NORMAL_EQUATION_VECTOR'):
            self.sigma[row_index, col_index] = float(line[69:80])

    def read(self, f):
        """
        Read the block (meta-) data from the file object f.

        Parameters
        ----------
        f : file object
            file object ot be read.
        """
        for line in f:
            if not line or line.startswith(b'*'):
                continue
            if line.startswith(b'-'):
                break
            self.__parse_line(line)

    def max_degree(self):
        """
        Return the maximum degree of the parsed spherical harmonics. This only yields sensible results after the block
        data is read from file.

        Returns
        -------
        max_degree : int
            maximum degree
        """
        return self.anm.shape[0] - 1

    def parameter_count(self):
        """
        Return the parameter count of the parsed spherical harmonics. This only yields sensible results after the block
        data is read from file.

        Returns
        -------
        parameter_count : int
            number of parameters in the vector
        """
        return np.max(self.index) + 1

    def to_vector(self):
        """
        Return the spherical harmonic coefficients as vector in their original order.

        Returns
        -------
        x : ndarray(parameter_count)
            unravelled spherical harmonic coefficients
        """
        x = np.zeros(self.parameter_count())
        for coeff, idx in zip(self.anm.flatten(), self.index.flatten()):
            if idx == -1:
                continue
            x[idx] = coeff

        return x


class SINEXSymmetricMatrix(SINEXBlock):
    """
    SINEX block holding a symmetric matrix, for example, normal equation matrices or covariance matrices.

    Parameters
    ----------
    id_line : bytes
        starting line of block in file
    parameter_count : int or None
        if not None, the matrix is preallocated to size (parameter_count, parameter_count)
    """
    def __init__(self, id_line, parameter_count):

        self.type = id_line[1:-2].decode()
        self.__parameter_count = parameter_count
        self.matrix = np.zeros((0, 0))

    def read(self, f):
        """
        Read the block (meta-) data from the file object f.

        Parameters
        ----------
        f : file object
            file object ot be read.
        """
        if self.__parameter_count is not None:
            self.matrix = np.zeros((self.__parameter_count, self.__parameter_count))

        for line in f:
            if not line or line.startswith(b'*'):
                continue
            if line.startswith(b'-'):
                break
            sline = line.split()
            row = int(sline[0]) - 1
            col_start = int(sline[1]) - 1

            parameter_count = max(row + 1, col_start + len(sline) - 2)
            if parameter_count > self.matrix.shape[0]:
                tmp = np.zeros((parameter_count, parameter_count))
                tmp[0:self.matrix.shape[0], 0:self.matrix.shape[0]] = self.matrix.copy()
                self.matrix = tmp

            for k, v in enumerate(sline[2:]):
                self.matrix[row, col_start + k] = float(v)
                self.matrix[col_start + k, row] = self.matrix[row, col_start + k]


class SINEXStatistics(SINEXBlock):
    """
    SINEX least squares adjustment statistics block.

    Parameters
    ----------
    id_line : bytes
        starting line of block in file
    """
    def __init__(self, id_line):

        self.type = id_line[1:].decode()

        self.degrees_of_freedom = 0
        self.observation_count = 0
        self.parameters = 0
        self.observation_square_sum = 0.0

    def read(self, f):
        """
        Read the block (meta-) data from the file object f.

        Parameters
        ----------
        f : file object
            file object ot be read.
        """
        for line in f:
            if not line or line.startswith(b'*'):
                continue
            if line.startswith(b'-'):
                break
            if line[1:].startswith(b'NUMBER OF DEGREES OF FREEDOM'):
                self.degrees_of_freedom = int(line[32:])
            elif line[1:].startswith(b'NUMBER OF OBSERVATIONS'):
                self.observation_count = int(line[32:])
            elif line[1:].startswith(b'NUMBER OF UNKNOWNS'):
                self.parameters = int(line[32:])
            elif line[1:].startswith(b'WEIGHTED SQUARE SUM OF O-C'):
                self.observation_square_sum = float(line[32:])


class SINEXBlockPlaceholder(SINEXBlock):
    """
    A place holder for not yet implemented SINEX blocks. This class just reads all lines from the beginning to the end
    of an unknown block and drops all information.
    """
    def __init__(self):
        self.type = 'PLACEHOLDER'

    def read(self, f):
        """
        Reads all lines from the beginning to the end  of an unknown block and drops all information.

        Parameters
        ----------
        f : file object
            file object ot be read.
        """
        for line in f:
            if not line or line.startswith(b'*'):
                continue
            if line.startswith(b'-'):
                break


def read_sinex_block(start_line, f, parameter_count):
    """
    Reads different SINEX blocks from a file object.

    Parameters
    ----------
    start_line : bytes
        line with the block definition
    f : file object
        file stream to read from
    parameter_count : int or None
        some SINEX blocks benefit from pre-allocating arrays by passing their expected parameter count

    Returns
    -------
    block : SINEXBlock subclass
        successfully parsed SINEX block
    """
    sinex_block = SINEXBlockPlaceholder()
    if start_line.startswith(b'+SOLUTION/ESTIMATE') or start_line.startswith(b'+SOLUTION/APRIORI') or start_line.startswith(b'+SOLUTION/NORMAL_EQUATION_VECTOR'):
        sinex_block = SINEXSphericalHarmonicsVector(start_line)
    elif start_line.startswith(b'+SOLUTION/NORMAL_EQUATION_MATRIX'):
        sinex_block = SINEXSymmetricMatrix(start_line, parameter_count)
    elif start_line.startswith(b'+SOLUTION/STATISTICS'):
        sinex_block = SINEXStatistics(start_line)

    sinex_block.read(f)

    return sinex_block


def loadsinex(file_name):
    """
    General purpose function to read SINEX blocks from a file.

    Parameters
    ----------
    file_name : str
        name of the SINEX file to be parsed.

    Returns
    -------
    blocks : list
        returns all recognized blocks in the SINEX file as a list.
    """
    try:
        f = gzip.open(file_name, 'r')
    except OSError:
        f = open(file_name, 'r')

    header_line = f.readline()
    if not header_line.startswith(b'%'):
        f.seek(0)

    blocks = []
    parameter_count = None
    for line in f:

        sline = line.rstrip()

        if not sline or sline.startswith(b'*'):
            continue

        if sline.startswith(b'%'):
            break

        if sline.startswith(b'+'):
            block = read_sinex_block(sline, f, parameter_count)
            if parameter_count is None:
                parameter_count = block.parameter_count()
            if not isinstance(block, SINEXBlockPlaceholder):
                blocks.append(block)

    f.close()

    return blocks


def loadsinexnormals(file_name):
    """

    Parameters
    ----------
    file_name : str
        file name of the SINEX file

    Returns
    -------
    N : array_like(parameter_count, parameter_count)
        coefficient matrix of normal equations
    n : array_like(parameter_count, ŕhs_count)
        right hand side(s)
    lPl : array_like(rhs_count,)
        square sum of observations for each right hand side
    obs_count : int
        observation count

    """
    blocks = loadsinex(file_name)
    block_dict = {b.type: b for b in blocks}
    block_types = set(block_dict.keys())

    required_blocks_6b = {'SOLUTION/MATRIX_APRIORI', 'SOLUTION/NORMAL_EQUATION_MATRIX',
                          'SOLUTION/NORMAL_EQUATION_VECTOR', 'SOLUTION/STATISTICS'}

    required_blocks_6c = {'SOLUTION/NORMAL_EQUATION_MATRIX', 'SOLUTION/NORMAL_EQUATION_VECTOR', 'SOLUTION/STATISTICS'}

    if required_blocks_6b.issubset(block_types) or required_blocks_6c.issubset(block_types):

        N = block_dict['SOLUTION/NORMAL_EQUATION_MATRIX'].matrix
        n = block_dict['SOLUTION/NORMAL_EQUATION_VECTOR'].to_vector()[:, np.newaxis]
        lPl = np.atleast_1d(block_dict['SOLUTION/STATISTICS'].observation_square_sum)
        obs_count = block_dict['SOLUTION/STATISTICS'].observation_count

        return N, n, lPl, obs_count
    else:
        raise ValueError('SINEX file does not conform to storage schemes 6b or 6c for normal equations.')