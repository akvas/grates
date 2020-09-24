# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Python wrappers for GROOPS file in-/output.
"""

import numpy as np
from os.path import isfile, split, isdir
import warnings
import datetime as dt
import groopsiobase as giocpp


class GroopsIOError(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def loadmat(fname):
    """
    Read GROOPS Matrix file format.

    Imports GROOPS matrices as numpy array. Note that
    triangular and symmetric matrices are stored fully.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    mat : array_like(m, n)
        2d ndarray containing the matrix data

    Raises
    ------
    FileNotFoundError
        if file is nonexistent

    Examples
    --------
     >>> import groopsio.io as gio
     >>> A = gio.loadmat('A.dat')

    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    return giocpp.loadmat(fname)


def savemat(fname, M, mtype='general', uplo='upper'):
    """
    Write Numpy ndarray to GROOPS Matrix file

    Save numeric array in GROOPS matrix file format. Per default
    the array is saved as general matrix. To account for matrix
    properties the keyword arguments mtype and uplo can be used.

    Parameters
    ----------
    fname : str
        file name
    M : array_like(m, n)
        2d ndarray to be written to file
    mtype : str
        matrix type {'general', 'symmetric', 'triangular'}, (default: 'general')
    uplo : str
        chose which triangle is stored (only applies if mtype is not 'general') {'upper', 'lower'} (default: 'upper')

    Raises
    ------
    FileNotFoundError
        if directory is nonexistent or not writeable

    Examples
    --------
    >>> import numpy as np
    >>> import groopsio.io as gio
    >>> A = np.eye(10)  # 10x10 identity matrix
    >>>
    >>> gio.savemat('A.dat', A) # A is saved as general 10x10 matrix
    >>> gio.savemat('A.dat', A, mtype='symmetric') # A is saved as symmetric 10x10 matrix
    >>> gio.savemat('A.dat', A, mtype='triangular', uplo='lower') # A is saved as lwoer triangular 10x10 matrix

    """
    if M.ndim == 0:
        warnings.warn('0-dimensional array treated as 1x1 matrix.')
        M = np.atleast_2d(M)

    elif M.ndim == 1:
        warnings.warn('1-dimensional array treated as column vector.')
        M = M[:, np.newaxis]

    elif M.ndim > 2:
        raise ValueError('ndarray must have at most two dimensions (has {0:d}).'.format(M.ndim))

    if split(fname)[0] and not isdir(split(fname)[0]):
        raise FileNotFoundError('Directory ' + split(fname)[0] + ' does not exist.')

    if mtype.lower() not in ('general', 'symmetric', 'triangular'):
        raise ValueError("Matrix type must be 'general', 'symmetric' or 'triangular'.")

    if uplo.lower() not in ('upper', 'lower'):
        raise ValueError("Matrix triangle must be 'upper' or 'lower'.")

    giocpp.savemat(fname, M, mtype.lower(), uplo.lower())


def loadgridrectangular(fname):
    """
    Read GROOPS GriddedDataRectangular file format.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    data : list of array_like(m, n)
        2d ndarray containing the grid values
    lon : array_like(n,)
        1d ndarray containing the longitude values in radians
    lat : array_like(m,)
        1d ndarray containing the latitude values in radians
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid

    Raises
    ------
    FileNotFoundError
        if file is nonexistent

    Examples
    --------
    >>> import numpy as np
    >>> import groopsio.io as gio
    >>> data, a, f = gio.loadgrid('grids/aod1b_RL04.dat')

    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    return_tuple = giocpp.loadgridrectangular(fname)
    data_count = len(return_tuple) - 4
    return list(return_tuple[0:data_count]), return_tuple[-4].flatten(), return_tuple[-3].flatten(), return_tuple[-2], return_tuple[-1]


def loadgrid(fname):
    """
    Read GROOPS GriddedData file format.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    data : array_like(m, n)
        2d ndarray containing the grid coordinates and values. Columns 0-3 contain geometry (lon, lat, h, area),
        columns 4-(n-1) contain the corresponding point values
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid

    Raises
    ------
    FileNotFoundError
        if file is nonexistent

    Examples
    --------
    >>> import numpy as np
    >>> import groopsio.io as gio
    >>> data, a, f = gio.loadgrid('grids/aod1b_RL04.dat')

    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    data, a, f = giocpp.loadgrid(fname)

    return data, a, f


def savegrid(fname, data, a=6378137.0, f=298.2572221010**-1):
    """
    Write grid to GROOPS GriddedData file

    Parameters
    ----------
    fname : str
        file name
    data : array_like(m, n)
        2d ndarray containing the grid coordinates and values. Columns 0-3 contain geometry (lon, lat, h, area),
        columns 4-(n-1) contain the corresponding point values
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid

    Raises
    ------
    FileNotFoundError
        if directory is nonexistent or not writeable

    Examples
    --------
    >>> import groopsio.io as gio
    >>> G, a, f = gio.loadgrid('grids/aod1b_RL04.dat')
    >>> # manipulate grid
    >>> gio.savegrid('grids/aod1b_RL04_mod.dat', G, a, f)

    """
    if split(fname)[0] and not isdir(split(fname)[0]):
        raise FileNotFoundError('Directory ' + split(fname)[0] + ' does not exist.')

    giocpp.savegrid(fname, data, a, f)


def loadinstrument(fname, concat_arcs=False):
    """
    Read GROOPS Instrument file format.

    Instrument data is returned as saved in the file, time stamps are given in MJD.

    Parameters
    ----------
    fname : str
        file name
    concat_arcs : bool
        flag whether to concatenate all arcs (default: False)

    Returns
    -------
    arcs : tuple of array_like(m, n) or array_like(m, n)
        tuple of 2d ndarrays containing the arc data or a single ndarray if concate_arcs=True
    epoch_Type : int
        enum of instrument type

    Raises
    ------
    FileNotFoundError
        if file is nonexistent

    Examples
    --------
    >>> import numpy as np
    >>> import groopsio.io as gio
    >>> pod2, pod2_type = gio.loadinstrument('satellite/grace2_pod_2008-05.dat')
    >>> pod2, pod2_type = gio.loadinstrument('satellite/grace2_pod_2008-05.dat', concat_arcs=True)

    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    arcs, epoch_type = giocpp.loadinstrument(fname)

    if concat_arcs:
        arcs = np.hstack(arcs)

    return arcs, epoch_type


def loadstarcamera(fname):
    """
    Read rotation matrices from StarCameraFile.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    times : array_like(m,)
        time stamps in MJD
    data : tuple of array_like(3,3)
        rotation matrices for each epoch

    Raises
    ------
    FileNotFoundError
        if file is nonexistent
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    times, data = giocpp.loadstarcamera(fname)

    return times.flatten(), data


def saveinstrument(fname, arcs, epoch_type=None):
    """
    Save arcs to  GROOPS Instrument file format.

    Parameters
    ----------
    fname : str
        file name
    arcs : list of array_like(m, n) or array_like(m, n)
        arc-wise data as ndarray, or single ndarray
    epoch_type : int
        enum of epoch type (Default: MISCVALUES)

    Raises
    ------
    FileNotFoundError
        if directory is nonexistent or not writeable

    Examples
    --------
    >>> import numpy as np
    >>> import groopsio.io as gio
    >>> pod2, pod2_type = gio.loadinstrument('satellite/grace2_pod_2008-05.dat')
    >>> gio.saveinstrument('tmp/grace2_pod_2008-05_arcs_1-5-17.dat', pod2, pod2_type)

    """
    if split(fname)[0] and not isdir(split(fname)[0]):
        raise FileNotFoundError('Directory ' + split(fname)[0] + ' does not exist.')

    if type(arcs) is not list:
        arcs = [arcs]

    epoch_type = -1 if epoch_type is None else epoch_type

    giocpp.saveinstrument(fname, [arc for arc in arcs], epoch_type)


def loadgravityfield(fname):
    """
    Read SphericalHarmonics from gfc-file

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    GM : float
        Geocentric gravitational constant
    R : float
        Reference radius
    anm : array_like(nmax+1, nmax+1)
        Potential coefficients as ndarray. cosine coefficients are stored in the lower triangle, sine coefficients
        above the superdiagonal
    sigma2anm : array_like(nmax+1, nmax+1)
        Variances of potential coefficients, in the same structure as anm. If the gfc file does not provide accuracies,
        a NAN array is returned.

    Raises
    ------
    FileNotFoundError
        if file is nonexistent
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    GM, R, anm, sigma2anm = giocpp.loadgravityfield(fname)

    return GM, R, anm, sigma2anm


def savegravityfield(fname, GM, R, anm, sigma2anm=None):
    """
    Write GravityField instance to gfc-file

    Parameters
    ----------
    fname : str
        file name
    GM : float
        Geocentric gravitational constant
    R : float
        Reference radius
    anm : array_like(nmax+1, nmax+1)
        Potential coefficients as ndarray. cosine coefficients are stored in the lower triangle, sine coefficients
        above the superdiagonal
    sigma2anm : array_like(nmax+1, nmax+1)
        Variances of potential coefficients, in the same structure as anm. Default behavior is to not save accuracies
        (sigma2anm = None).

    Raises
    ------
    FileNotFoundError
        if directory is nonexistent or not writeable

    """
    if split(fname)[0] and not isdir(split(fname)[0]):
        raise FileNotFoundError('Directory ' + split(fname)[0] + ' does not exist.')

    has_sigmas = sigma2anm is not None

    giocpp.savegravityfield(fname, GM, R, anm, has_sigmas, sigma2anm if has_sigmas else None)


def loadtimesplines(fname, time):
    """
    Read potential coefficients from TimeSplines file


    Parameters
    ----------
    fname : str
        file name
    time : float or datetime.datetime
        evaluation time of TimeSplines file as MJD (float) or datetime object

    Returns
    -------
    GM : float
        Geocentric gravitational constant
    R : float
        Reference radius
    anm : array_like(nmax+1, nmax+1)
        Potential coefficients as ndarray. cosine coefficients are stored in the lower triangle, sine coefficients
        above the superdiagonal

    Raises
    ------
    FileNotFoundError
        if file is nonexistent
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    if isinstance(time, dt.datetime):
        delta = time-dt.datetime(1858, 11, 17)
        time = delta.days + delta.seconds/86400.0

    GM, R, anm = giocpp.loadtimesplines(fname, time)

    return GM, R, anm


def loadnormalsinfo(fname, return_full_info=False):
    """
    Read metadata of normal equation file.

    Parameters
    ----------
    fname : str
        file name
    return_full_info : bool
        if true, return lPl, observation count, parameter names, block index and used blocks, else (default)
        return only lPl, observation count and parameter names

    Returns
    -------
    lPl : array_like(rhs_count,)
        square sum of observations for each right hand side
    obs_count : int
        observation count
    names : tuple of str
        string representation of parameter names
    block_index : array_like(block_count+1,)
        beginning/end of normal equation blocks. Only returned if return_full_info is true.
    used_blocks : array_like(block_count, block_count)
        boolean array representing the sparsity structure of the normal equations. Only returned if return_full_info
        is true.
    """
    lPl, obs_count, names, block_index, used_blocks = giocpp.loadnormalsinfo(fname)

    if return_full_info:
        return lPl, obs_count, names, block_index.flatten().astype(int), used_blocks.astype(bool)
    else:
        return lPl, obs_count, names


def loadnormals(fname):
    """
    Read normal equations from file file.

    Parameters
    ----------
    fname : str
        file name

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
    return giocpp.loadnormals(fname)


def savenormals(fname, N, n, lPl, obs_count):
    """
    Read normal equations from file file.

    Parameters
    ----------
    fname : str
        file name
    N : array_like(m, m)
        normal equation matrix
    n : array_like(m, k)
        right hand side
    lPl : array_like(k,)
        square sum of observations
    obs_count : int
        number of observations

    Raises
    ------
    ValueError
        if dimensions of passed arguments do not match

    """
    if (N.ndim != 2) or (N.shape[0] != N.shape[1]):
        raise ValueError('Square normal equation coefficient matrix required.')

    if (n.ndim != 2) or (n.shape[0] != N.shape[0]):
        raise ValueError('Number of parameters in normal equation coefficient matrix and right hand side do not match.')

    if lPl.size != n.shape[1]:
        raise ValueError('Number of right hand sides in observation square sum and right hand side vector do not match.')

    return giocpp.savenormals(fname, N, n, lPl, obs_count)


def loadarclist(fname):
    """
    Read GROOPS arcList file.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    arc_intervals : tuple of int
        interval bounds in arc indices
    time_intervals : tuple of float
        interval bounds in MJD

    Raises
    ------
    FileNotFoundError
        if file does not exist
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    return giocpp.loadarclist(fname)


def loadtimeseries(fname):
    """
    Read Time Series from matrix/instrument file (based on loadmat)

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    t : array_like(m,)
        time stamps in MJD
    X : array_like(m, n)
        data values associated with each time stamp
    Raises
    ------
    FileNotFoundError
        if file not found
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    ts = loadmat(fname)

    return ts[:, 0], ts[:, 1::]


def loadfilter(fname):
    """
    Read digital filter from file.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    b : array_like(p,)
        MA coefficients
    a : array_like(q,)
        AR coefficients
    start_index : int
        positive integer which determines the time shift (non-causality) of the MA coefficients

    Raises
    ------
    FileNotFoundError
        if file does not exist
    ValueError
        if a negative index of an AR coefficient is found (AR part must be causal)
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    A = giocpp.loadmat(fname)
    idx_bk = A[A[:, 1] != 0, 0].astype(int)
    idx_ak = A[A[:, 2] != 0, 0].astype(int)

    if np.any(idx_ak < 0):
        raise ValueError('Negative indices for AR coefficients not allowed (causal-filter).')

    start_index = max(-np.min(idx_bk), 0)

    b = np.zeros(np.max(idx_bk) + start_index + 1)
    b[idx_bk + start_index] = A[A[:, 1] != 0, 1]
    a = np.zeros(np.max(idx_ak) + 1 if idx_ak.size > 0 else 1)
    a[idx_ak] = A[A[:, 2] != 0, 2]
    if a[0] != 1.0:
        a[0] = 1.0
        warnings.warn('a0 coefficient set to one.')

    return b, a, start_index


def savefilter(fname, b, a=np.ones(1), start_index=0):
    """
    Save filter coefficients to file.

    Parameters
    ----------
    fname : str
        file name
    b : array_like(p,)
        MA coefficients
    a = array_like(q,)
        AR coefficients (Default: [1], pure MA filter)
    start_index : int
        positive integer which determines the time shift (non-causality) of the MA coefficients (Default: 0)

    Raises
    ------
    ValueError
        if a negative start_index is passed
    FileNotFoundError
        if directory is not writeable
    """
    if start_index < 0:
        raise ValueError('start_index must be positive')

    if split(fname)[0] and not isdir(split(fname)[0]):
        raise FileNotFoundError('Directory ' + split(fname)[0] + ' does not exist.')

    idx = np.arange(-start_index, -start_index + max(b.size, a.size))
    print(idx)

    A = np.zeros((idx.size, 3))
    A[:, 0] = idx
    A[0:b.size, 1] = b
    A[start_index:start_index + a.size, 2] = a

    giocpp.savemat(fname, A, 0, 0)


def loadpolygon(fname):
    """
    Read  a polygon list from file.

    Parameters
    ----------
    fname : str
        file name

    Returns
    -------
    polygons : tuple of array_likes(p, 2)
        tuple of 2-d arrays representing the vertices (longitude, latitude) of each polygon in radians.

    Raises
    ------
    FileNotFoundError
        if file does not exist
    """
    if not isfile(fname):
        raise FileNotFoundError('File ' + fname + ' does not exist.')

    return giocpp.loadpolygon(fname)
