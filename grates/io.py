# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
File I/O for gravity field representations and gridded data.
"""

import datetime as dt
import tarfile
import abc
import gzip
import bz2
import numpy as np
import grates
from grates.gravityfield import PotentialCoefficients, TimeSeries, SurfaceMasCons
from grates.grid import CSRMasconGridRL06, RegularGrid
from grates.kernel import WaterHeight
import scipy.spatial
import netCDF4
import h5py
import os
import contextlib
import io
import yaml


class InputFile:
    """
    Class for flexible input file handling.

    Parameters
    ----------
    file_name : file, str, os.PathLike
        File or filename to read. If the filename extension is .gz or .bz2, the file is first decompressed.
    """
    def __init__(self, file_name):

        if isinstance(file_name, os.PathLike):
            file_name = os.fspath(file_name)

        if isinstance(file_name, str):

            if file_name.endswith('.gz'):
                self.__stream = gzip.open(file_name, 'rb')
            elif file_name.endswith('.bz2'):
                self.__stream = bz2.open(file_name, 'rb')
            else:
                self.__stream = open(file_name, 'rb')

            self.__is_stream_owner = True
        elif isinstance(file_name, (io.BufferedIOBase, io.TextIOBase)):
            self.__is_stream_owner = False
            self.__stream = file_name
        else:
            raise ValueError('file_name must be a string, PathLike object or file object')

        if isinstance(self.__stream, io.BufferedIOBase):
            self.__is_binary = True
        elif isinstance(self.__stream, io.TextIOBase):
            self.__is_binary = False
        else:
            raise ValueError('file stream must be a binary or text stream')

        if not self.__stream.readable():
            raise ValueError('file stream must be readable')

    def readline(self):
        """
        Read a line from input file.

        Returns
        -------
        line : bytes
            encoded line in file
        """
        if self.__is_binary:
            return self.__stream.readline()
        else:
            return self.__stream.readline().encode(self.__stream.encoding)

    def close(self):
        """
        Close file stream if it is generated in the constructor.
        """
        if self.__is_stream_owner:
            self.__stream.close()

    def read(self, size=-1):
        self.__stream.read(size)

    @property
    def stream(self):
        return self.__stream

    def seek(self, offset, whence=0):
        self.__stream.seek(offset, whence)

    @staticmethod
    @contextlib.contextmanager
    def open(file_name):
        """
        Create an InputFile instance.

        Parameters
        ----------
        file_name : file, str, os.PathLike
            File or filename to read. If the filename extension is .gz or .bz2, the file is first decompressed.

        Returns
        -------
        input_file : GeneratorContextManager
            input file as context manager
        """
        try:
            input_file = InputFile(file_name)
        except Exception as e:
            raise e
        else:
            yield input_file
            input_file.close()

    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                break
            yield line


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

    with InputFile.open(file_name) as f:

        for line in f:
            if line.startswith(b'gfc'):
                sline = line.split()
                n, m, cnm, snm = int(sline[1]), int(sline[2]), float(sline[3]), float(sline[4])
                if max_degree and n > max_degree:
                    continue
                gf.append('c', n, m, cnm)
                gf.append('s', n, m, snm)

            elif line.startswith(b'radius'):
                gf.R = float(line.split()[-1])
            elif line.startswith(b'earth_gravity_constant'):
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
                gf.epoch = time_start + (time_end - time_start) * 0.5

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

    Returns
    -------
    time_series : grates.gravityfield.TimeSeries
        TimeSeries instance
    """
    tar = tarfile.open(file_name, 'r:gz')
    data = []
    for member in tar.getmembers():
        if member.isdir():
            continue

        gf = PotentialCoefficients(3.986004415E+14, 6378136.3)
        epoch = dt.datetime.strptime(member.name[-15:-4], '%Y%m%d_%H')
        gf.epoch = epoch

        f = tar.extractfile(member)
        for line in f:
            if line.startswith(b'gfc'):
                sline = line.replace(b'D', b'e').split()
                n, m, cnm, snm = int(sline[1]), int(sline[2]), float(sline[3]), float(sline[4])
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


class SINEXFile:
    """
    Class representation of a SINEX file.

    Parameters
    ----------
    file_name : str
        SINEX file name. If the file name ends with '.gz', a gzip stream is openend.
    mode : str
        file open mode
    """
    def __init__(self, file_name, mode):
        self.is_output = 'w' in mode

        if file_name.endswith('.gz'):
            if 't' not in mode:
                mode += 't'
            self.f = gzip.open(file_name, mode)
        else:
            self.f = open(file_name, mode)

    def close(self):
        """
        Close the file stream. If it is an output file, append the SINEX trailer (%ENDSNX).
        """
        if self.is_output:
            self.f.write('%ENDSNX' + os.linesep)
        self.f.close()

    @staticmethod
    def datetime2sinex(t):
        """
        Convert a datetime object to SINEX date/time format.

        Parameters
        ----------
        t : datetime
            datetime object
        """
        start_year = dt.datetime(t.year, 1, 1)
        time_delta = t - start_year

        return '{0:2s}:{1:03d}:{2:05d}'.format(start_year.strftime('%y'), time_delta.days + 1, time_delta.seconds)

    def write_header(self, agency, time_start, time_end, parameter_count, techniques='C'):
        """
        Write the mandatory SINEX header line. Version and constraint code are hard coded to 2.02 and 2 at the moment.

        Parameters
        ----------
        agency : str
            3-character agency code
        time_start : datetime
            solution start time
        time_end : datetime
            solution end time
        parameter_count : int
            number of estimated parameters in the file
        techniques : str
            techniques used to obtain the estimate
        """
        creation_time = dt.datetime.now()

        header_line = '%=SNX 2.02 {0:3s} {1:12s} {0:3s} {2:12s} {3:12s} {4:1s} {5:05d} 2      '.format(agency, SINEXFile.datetime2sinex(creation_time),
                      SINEXFile.datetime2sinex(time_start), SINEXFile.datetime2sinex(time_end), techniques, parameter_count)

        self.f.write(header_line + os.linesep)

    def write_reference(self, description=None, output=None, contact=None, software=None, hardware=None, input=None):
        """
        Write the mandatory FILE/REFERENCE block.
        """
        self.f.write('+FILE/REFERENCE' + os.linesep)
        if description is not None:
            self.f.write(' {0:18s} {1:60s}'.format('DESCRIPTION', description) + os.linesep)
        if output is not None:
            self.f.write(' {0:18s} {1:60s}'.format('OUTPUT', output) + os.linesep)
        if contact is not None:
            self.f.write(' {0:18s} {1:60s}'.format('CONTACT', contact) + os.linesep)
        if software is not None:
            self.f.write(' {0:18s} {1:60s}'.format('SOFTWARE', software) + os.linesep)
        if hardware is not None:
            self.f.write(' {0:18s} {1:60s}'.format('HARDWARE', hardware) + os.linesep)
        if input is not None:
            self.f.write(' {0:18s} {1:60s}'.format('INPUT', input) + os.linesep)
        self.f.write('-FILE/REFERENCE' + os.linesep)


    @staticmethod
    @contextlib.contextmanager
    def open(file_name, mode):
        """
        Open a SINEX file in a context.

        Parameters
        ----------
        file_name : str
            SINEX file name. If the file name ends with '.gz', a gzip stream is openend.
        mode : str
            file open mode

        Returns
        -------
        snx_file : SINEXFile
            SINEXFine instance
        """
        try:
            snx_file = SINEXFile(file_name, mode)
            yield snx_file
        finally:
            snx_file.close()

    def read_blocks(self):
        """
        Read all SINEX blocks in the file.

        Returns
        -------
        blocks : list of SINEXBlock
            blocks in the SINEX file in the order they are read
        """
        header_line = self.f.readline()
        if not header_line.startswith(b'%'):
            self.f.seek(0)

        blocks = []
        parameter_count = None
        for line in self.f:

            sline = line.rstrip()

            if not sline or sline.startswith(b'*'):
                continue

            if sline.startswith('%'):
                break

            if sline.startswith('+'):
                block = read_sinex_block(sline, self.f, parameter_count)
                if parameter_count is None:
                    parameter_count = block.parameter_count()
                if not isinstance(block, SINEXBlockPlaceholder):
                    blocks.append(block)

        return blocks

    def write_block(self, block):
        """
        Write a block to a SINEX file.

        Parameters
        ----------
        block : SINEXBlock
            SINEXBlock instance
        """
        block.write(self.f)


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
        sline = line_remainder.split(':')
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
    def parameter_count():
        return None


class SINEXSphericalHarmonicsVector(SINEXBlock):
    """
    A SINEX block containing a spherical harmonic vector, possibly with uncertainties.

    Parameters
    ----------
    id_line : bytes
        starting line of block in file
    """
    def __init__(self, numbering, x, sigmax=None, reference_epoch=None, index=None, block_type=None):

        self.numbering = numbering
        self.x = x
        if sigmax is None:
            self.sigmax = np.zeros(x.shape)
        else:
            self.sigmax = sigmax
        if reference_epoch is None:
            self.reference_epoch = dt.datetime(2000, 1, 1, 12)
        else:
            self.reference_epoch = reference_epoch

        if index is None:
            self.index = np.array(range(x.size))
        else:
            self.index = index

        self.block_type = block_type

    @staticmethod
    def from_file(f, block_type):

        x = []
        sigmax = []
        index = []
        coefficients = []
        for line in f:
            if not line or line.startswith(b'*'):
                continue
            if line.startswith(b'-'):
                break

            ptype = line[7:13].strip()

            if ptype not in [b'CN', b'SN']:
                raise ValueError('Parameter type <' + ptype + '> not supported.')

            degree = int(line[14:18].strip())
            order = int(line[22:26].strip())

            coefficients.append(grates.gravityfield.CoefficientSequence.Coefficient(np.int8(0) if ptype == 'CN' else np.int8(1), degree, order))
            index.append(int(line[1:6]) - 1)

            x.append(float(line[47:68]))
            if not block_type.startswith(b'SOLUTION/NORMAL_EQUATION_VECTOR'):
                sigmax.append(float(line[69:80]))

        if len(sigmax) == 0:
            sigmax = None
        else:
            sigmax = np.array(sigmax)

        return SINEXSphericalHarmonicsVector(grates.gravityfield.CoefficientSequence(coefficients), np.array(x), sigmax)

    def write(self, f):
        """
        Write a spherical harmonic vector to SINEX file.

        Parameters
        ----------
        f : file object
            file object ot be read
        x : ndarray
            solution vector
        numbering : CoefficientSequence
            corresponding spherical harmonic coefficient order
        reference_epoch : datetime or None
            reference epoch of solution (default: J2000)
        """
        start_year = dt.datetime(self.reference_epoch.year, 1, 1)
        time_delta = self.reference_epoch - start_year

        f.write('+' + self.block_type + os.linesep)
        for k in range(self.x.size):
            coeff = self.numbering.coefficients[k]
            cs = 'CN' if coeff.basis_function == 0 else 'SN'

            f.write(' {0:5d} {1:6s} {2:4d} -- {3:4d}'.format(k + 1, cs, coeff.degree, coeff.order))
            f.write(' {0:2s}:{1:03d}:{2:05d}'.format(start_year.strftime('%y'), time_delta.days + 1, time_delta.seconds))
            f.write(' ---- 2 {0:21.14e}'.format(self.x[k]))
            if not self.block_type.startswith(b'SOLUTION/NORMAL_EQUATION_VECTOR'):
                f.write(' {0:10.5e}'.format(self.sigmax[k]) + os.linesep)
            else:
                f.write(os.linesep)

        f.write('-' + self.block_type + os.linesep)

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
    def __init__(self, matrix, lower=False, block_type=None):

        self.matrix = matrix
        self.lower = lower
        self.block_type = block_type


    @staticmethod
    def from_file(f, block_type, parameter_count):
        """
        Read the block (meta-) data from the file object f.

        Parameters
        ----------
        f : file object
            file object ot be read.
        """
        if parameter_count is not None:
            matrix = np.zeros((parameter_count, parameter_count))

        for line in f:
            if not line or line.startswith(b'*'):
                continue
            if line.startswith(b'-'):
                break
            sline = line.split()
            row = int(sline[0]) - 1
            col_start = int(sline[1]) - 1

            count = max(row + 1, col_start + len(sline) - 2)
            if count > matrix.shape[0]:
                tmp = np.zeros((count, count))
                tmp[0:matrix.shape[0], 0:matrix.shape[0]] = matrix.copy()
                matrix = tmp

            for k, v in enumerate(sline[2:]):
                matrix[row, col_start + k] = float(v)
                matrix[col_start + k, row] = matrix[row, col_start + k]

        return SINEXSymmetricMatrix(matrix, False, block_type)

    def write(self, f):
        """
        Write a matrix to the file object f.

        Parameters
        ----------
        f : file object
            file object to be written to
        matrix : ndarray
            (symmetric) matrix to be written to file
        lower : bool
            whether to access the lower or upper triangle
        """
        f.write('+' + self.block_type + (' L' if self.lower else ' U') + os.linesep)
        if self.lower:
            for row in range(self.matrix.shape[0]):
                for column in range(0, row + 1, 3):
                    f.write(' {0:5d} {1:5d}'.format(row + 1, column + 1))
                    for k in range(column, min(column + 3, row + 1)):
                        f.write(' {0:21.14e}'.format(self.matrix[row, k]))
                    f.write(os.linesep)
        else:
            for row in range(self.matrix.shape[0]):
                for column in range(row, self.matrix.shape[1], 3):
                    f.write(' {0:5d} {1:5d}'.format(row + 1, column + 1))
                    for k in range(column, min(column + 3, self.matrix.shape[1])):
                        f.write(' {0:21.14e}'.format(self.matrix[row, k]))
                    f.write(os.linesep)

        f.write('-' + self.block_type + (' L' if self.lower else ' U') + os.linesep)


class SINEXStatistics(SINEXBlock):
    """
    SINEX least squares adjustment statistics block.

    Parameters
    ----------
    id_line : bytes
        starting line of block in file
    """
    def __init__(self, degrees_of_freedom, observation_count, parameters, observation_square_sum, block_type):

        self.block_type = block_type

        self.degrees_of_freedom = degrees_of_freedom
        self.observation_count = observation_count
        self.parameters = parameters
        self.observation_square_sum = observation_square_sum

    @staticmethod
    def from_file(f, block_type):
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
                degrees_of_freedom = int(float(line[32:]))
            elif line[1:].startswith(b'NUMBER OF OBSERVATIONS'):
                observation_count = int(float(line[32:]))
            elif line[1:].startswith(b'NUMBER OF UNKNOWNS'):
                parameters = int(float(line[32:]))
            elif line[1:].startswith(b'WEIGHTED SQUARE SUM OF O-C'):
                observation_square_sum = float(line[32:])

        return SINEXStatistics(degrees_of_freedom, observation_count, parameters, observation_square_sum, block_type)


class SINEXBlockPlaceholder(SINEXBlock):
    """
    A place holder for not yet implemented SINEX blocks. This class just reads all lines from the beginning to the end
    of an unknown block and drops all information.
    """
    def __init__(self):
        self.type = 'PLACEHOLDER'

    @staticmethod
    def from_file(f):
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

        return SINEXBlockPlaceholder()


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
    if start_line.startswith(b'+SOLUTION/ESTIMATE') or start_line.startswith(b'+SOLUTION/APRIORI') or start_line.startswith(b'+SOLUTION/NORMAL_EQUATION_VECTOR'):
        block_type = start_line[1:]
        sinex_block = SINEXSphericalHarmonicsVector.from_file(f, block_type)

    elif start_line.startswith(b'+SOLUTION/NORMAL_EQUATION_MATRIX'):
        block_type = start_line[1:-2]
        sinex_block = SINEXSymmetricMatrix.from_file(f, block_type, parameter_count)

    elif start_line.startswith(b'+SOLUTION/MATRIX_ESTIMATE'):
        block_type = start_line[1:-2]
        sinex_block = SINEXSymmetricMatrix.from_file(f, block_type, parameter_count)

    elif start_line.startswith(b'+SOLUTION/STATISTICS'):
        block_type = start_line[1:]
        sinex_block = SINEXStatistics.from_file(f, block_type)
    else:
        sinex_block = SINEXBlockPlaceholder.from_file(f)

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
    with InputFile.open(file_name) as f:

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


def loadcsr06mascons(file_name):
    """
    Read CSR RL06 mascon grid from a netCDF file. The mascon data is mapped to the original estimation grid, rather than
    using the 0.25 grid the mascons are provided on.

    Parameters
    ----------
    file_name : str
        netCDF file name

    Returns
    -------
    time_series : grates.gravityfield.TimeSeries
        TimeSeries instance
    """
    output_grid = CSRMasconGridRL06()

    dataset = netCDF4.Dataset(file_name)
    longitude = np.deg2rad(dataset['lon'])
    latitude = np.deg2rad(dataset['lat'])
    times = np.asarray(dataset['time'])

    base_grid = RegularGrid(longitude, latitude, a=output_grid.semimajor_axis, f=output_grid.flattening)

    tree = scipy.spatial.cKDTree(base_grid.cartesian_coordinates())
    _, index = tree.query(output_grid.cartesian_coordinates(), k=1)

    data = []
    for k in range(times.size):
        values = dataset['lwe_thickness'][k, :, :].flatten() * 1e-2

        mascons = SurfaceMasCons(output_grid.copy(), kernel=WaterHeight)
        mascons.values = np.array(values[index], dtype=float)
        mascons.epoch = dt.datetime(2002, 1, 1) + dt.timedelta(days=float(times[k]))
        data.append(mascons)

    return TimeSeries(data)


def loadrl06mascongrids(file_name, scale=1e-2, data_layer='lwe_thickness'):
    """
    Read GRACE/GRACE-FO RL06 mascon grids from a NetCDF file.

    Parameters
    ----------
    file_name : str
        NetCDF file name
    scale : float
        scale is applied to the data_layer (default: 1e-2, centimeters to meters)
    data_layer : str
        name of the data layer to return (default: lwe_thickness)

    Returns
    -------
    time_series : grates.gravityfield.TimeSeries
        TimeSeries instance
    """
    dataset = netCDF4.Dataset(file_name)
    longitude = np.deg2rad(dataset['lon'])
    longitude[longitude > np.pi] -= 2 * np.pi
    idx_lon = np.argsort(longitude, kind='stable')
    longitude = longitude[idx_lon]
    latitude = np.deg2rad(dataset['lat'])
    idx_lat = np.argsort(latitude)[::-1]
    latitude = latitude[idx_lat]
    times = np.asarray(dataset['time'])

    base_grid = grates.grid.RegularGrid(longitude, latitude, a=grates.gravityfield.WGS84.R, f=grates.gravityfield.WGS84.flattening)

    data = []
    for k in range(times.size):
        grid = base_grid.copy()
        values = dataset[data_layer][k, :, :] * scale
        grid.value_array =  values[np.ix_(idx_lat, idx_lon)]
        grid.epoch = dt.datetime(2002, 1, 1) + dt.timedelta(days=float(times[k]))
        data.append(grid)

    return TimeSeries(data)


def loadgsfc06mascons(file_name, scale=1e-2, data_layer='cmwe'):
    """
    Read GSFC RL06 mascon solutions from a HDF5 file. The mascon data is returned on the original estimation grid.

    Parameters
    ----------
    file_name : str
        netCDF file name

    Returns
    -------
    time_series : grates.gravityfield.TimeSeries
        TimeSeries instance
    """
    data = []
    with h5py.File(file_name, 'r') as f:

        lons = np.deg2rad(f['mascon']['lon_center']).squeeze()
        lons[lons > np.pi] -= 2 * np.pi
        lats = np.deg2rad(f['mascon']['lat_center']).squeeze()
        areas = f['mascon']['area_km2'][:].squeeze()
        areas /= np.sum(areas) * 4 * np.pi
        base_grid = grates.grid.IrregularGrid(lons, lats, area_element=areas)

        times = f['time']['ref_days_middle'][:].squeeze()
        epochs = [dt.datetime(2002, 1, 1) + dt.timedelta(days=tk - 1) for tk in times]
        dataset = f['solution'][data_layer]
        for k in range(dataset.shape[1]):
            grid = base_grid.copy()
            grid.values = dataset[:, k] * scale
            grid.epoch = epochs[k]
            data.append(grid)

    return TimeSeries(data)



def loadgsm(file_name):
    """
    Read spherical harmonics coefficients from an GRACE/GRACE-FO SDS GSM file.
    The epoch of the gravity field is defined as the midpoint between start and end of data coverage.

    Parameters
    ----------
    file_name : str
        archive file name

    Returns
    -------
    coefficients : grates.gravityfield.PotentialCoefficients
        PotentialCoefficients instance
    """
    with InputFile.open(file_name) as f:

        header = b''
        for line in f:
            if line.startswith(b'# End of YAML header'):
                break
            header += line

        yaml_header = yaml.safe_load(header)

        max_degree = yaml_header['header']['dimensions']['degree']
        R = yaml_header['header']['non-standard_attributes']['mean_equator_radius']['value']
        GM = yaml_header['header']['non-standard_attributes']['earth_gravity_param']['value']

        time_start = yaml_header['header']['global_attributes']['time_coverage_start']
        time_end = yaml_header['header']['global_attributes']['time_coverage_start']
        epoch = time_start + (time_end - time_start) * 0.5

        anm = np.zeros((max_degree + 1, max_degree + 1))
        for line in f:
            if line.startswith(b'GRCOF2'):
                sline = line.split()
                n = int(sline[1])
                m = int(sline[2])

                anm[n, m] = float(sline[3])
                if m > 0:
                    anm[m - 1, n] = float(sline[4])

        coeffs = grates.gravityfield.PotentialCoefficients(GM, R)
        coeffs.anm = anm
        coeffs.epoch = epoch

        return coeffs
