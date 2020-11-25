# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Auxiliary functions.
"""

import numpy as np
import abc
import grates.time
import pkg_resources


def legendre_functions(max_degree, colat):
    """
    Associated fully normalized Legendre functions (1st kind).

    Parameters
    ----------
    max_degree : int
        maximum spherical harmonic degree to compute
    colat : float, array_like(m,)
        co-latitude of evaluation points in radians

    Returns
    -------
    Pnm : array_like(m, nmax + 1, nmax + 1)
        Array containing the fully normalized Legendre functions. Pnm[:, n, m] returns the
        Legendre function of degree n and order m for all points, as does Pnm[:, m-1, n] (for m > 0).

    """
    theta = np.atleast_1d(colat)
    function_array = np.empty((theta.size, max_degree + 1, max_degree + 1))
    if max_degree == 0:
        function_array[:, 0, 0] = 1.0
        return function_array

    function_array[:, 0, 0] = 1.0  # initial values for recursion
    function_array[:, 1, 0] = np.sqrt(3) * np.cos(theta)
    function_array[:, 1, 1] = np.sqrt(3) * np.sin(theta)

    for n in range(2, max_degree + 1):
        function_array[:, n, n] = np.sqrt((2.0 * n + 1.0) / (2.0 * n)) * np.sin(theta) * \
                                  function_array[:, n - 1, n - 1]

    index = np.arange(max_degree + 1)
    function_array[:, index[2:], index[1:-1]] = np.sqrt(2 * index[2:] + 1) * np.cos(theta[:, np.newaxis]) * \
                                                function_array[:, index[1:-1], index[1:-1]]

    for row in range(2, max_degree + 1):
        n = index[row:]
        m = index[0:-row]
        function_array[:, n, m] = np.sqrt((2.0 * n - 1.0) / (n - m) * (2.0 * n + 1.0) / (n + m)) * \
                                  np.cos(theta[:, np.newaxis]) * function_array[:, n - 1, m] - \
                                  np.sqrt((2.0 * n + 1.0) / (2.0 * n - 3.0) * (n - m - 1.0) / (n - m) *
                                          (n + m - 1.0) / (n + m)) * function_array[:, n - 2, m]

    for m in range(1, max_degree + 1):
        function_array[:, m - 1, m:] = function_array[:, m:, m]

    return function_array


def legendre_functions_per_order(max_degree, order, colat):
    """
    Compute fully normalized associated Legendre functions for a specific order.

    Parameters
    ----------
    max_degree : int
       maximum spherical harmonic degree to compute
    order : int
        order for which the Legendre functions should be computed
    colat : float, array_like(m,)
        co-latitude of evaluation points in radians

    Returns
    -------
    Pnm : array_like(m, max_degree + 1 - order)
        fully normalized associated Legendre functions for the given order
    """
    if order == 0:
        return legendre_polynomials(max_degree, colat)

    t = np.cos(np.atleast_1d(colat))
    s = np.sqrt(1 - t**2)
    coefficient_count = max_degree + 1 - order

    function_array = np.empty((t.size, coefficient_count))

    recursion_values = np.empty((t.size, 3))
    recursion_values[:, 0] = 1
    recursion_values[:, 1] = np.sqrt(3) * s

    for n in range(2, order + 1):
        recursion_values[:, 2] = np.sqrt((2 * n + 1) / (2 * n)) * s * recursion_values[:, 1]
        recursion_values = np.roll(recursion_values, -1, axis=1)

    function_array[:, 0] = recursion_values[:, 1]
    if coefficient_count > 1:
        function_array[:, 1] = np.sqrt(2 * order + 3) * t * function_array[:, 0]

    for n in range(order + 2, max_degree + 1):
        function_array[:, n - order] = np.sqrt((2 * n - 1) / (n - order) * (2 * n + 1) / (n + order)) * \
            t * function_array[:, n - 1 - order] - \
            np.sqrt((2 * n + 1) / (2 * n - 3) * (n - order - 1) / (n - order) * (n + order - 1) / (n + order)) * \
            function_array[:, n - 2 - order]

    return function_array


def legendre_polynomials(max_degree, colat):
    """
    Fully normalized Legendre polynomials.

    Parameters
    ----------
    max_degree : int
        maximum spherical harmonic degree to compute
    colat : float, array_like(m,)
        co-latitude of evaluation points in radians

    Returns
    -------
    Pn : array_like(m, max_degree + 1)
        Array containing the fully normalized Legendre polynomials. Pn[:, n] returns the
        Legendre polynomial of degree n and order m for all points.

    """
    if max_degree == 0:
        return np.ones((np.atleast_1d(colat).size, 1))

    t = np.cos(np.atleast_1d(colat))
    polynomial_array = np.empty((t.size, max_degree + 1))

    polynomial_array[:, 0] = 1  # initial values for recursion
    polynomial_array[:, 1] = np.sqrt(3) * t

    for n in range(2, max_degree + 1):
        polynomial_array[:, n] = np.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0)) / n * \
                                 t * polynomial_array[:, n - 1] - \
                                 np.sqrt((2.0 * n + 1.0) / (2.0 * n - 3.0)) * (n - 1.0) / n * \
                                 polynomial_array[:, n - 2]

    return polynomial_array


def legendre_summation(coefficients, colat):
    """
    Compute the linear combination of Legendre polynomials using the Clenshaw algorithm [1]_.

    Parameters
    ----------
    coefficients : array_like(m,)
        coefficients of the linear combination
    colat : float, array_like
        co-latitude of evaluation points in radians

    Returns
    -------
    sum : array_like
        evaluated linear combination (same shape as colat

     References
    ----------

    .. [1]  Clenshaw, C. W. (July 1955). "A note on the summation of Chebyshev series". Mathematical Tables and
            Other Aids to Computation. 9 (51): 118. doi:10.1090/S0025-5718-1955-0071856-0.

    """
    t = np.cos(colat)

    b2 = 0
    b1 = 0
    for k in np.arange(coefficients.size - 1, 0, -1):
        alpha = np.sqrt((2 * k + 1) * (2 * k + 3)) / (k + 1)
        beta = -np.sqrt((2 * k + 5) / (2 * k + 1)) * (k + 1) / (k + 2)

        bk = coefficients[k] + alpha * t * b1 + beta * b2
        b2 = b1
        b1 = bk

    return coefficients[0] + np.sqrt(3) * t * b1 - 0.5 * np.sqrt(5) * b2


def trigonometric_functions(nmax, lon):
    """
    Convenience function to compute the trigonometric functions (cosine, sine) for the use
    in spherical harmonics.

    Parameters
    ----------
    nmax : int
        maximum spherical harmonic degree to compute
    lon : float, array_like(m,)
       longitude of evaluation points in radians

    Returns
    -------
    cs : numpy.ndarray(m, nmax + 1, nmax + 1)
        Array containing the fully normalized spherical harmonics. cs[:, n, m] returns cos(m * lon)
        cs[:, m-1, n] (for m > 0) returns sin(m * lon).

    """
    longitude = np.atleast_1d(lon)
    cs_array = np.empty((longitude.size, nmax + 1, nmax + 1))
    cs_array[:, :, 0] = 1
    for m in range(1, nmax + 1):
        cs_array[:, m:, m] = np.cos(m * longitude)[:, np.newaxis]
        cs_array[:, m - 1, m:] = np.sin(m * longitude)[:, np.newaxis]

    return cs_array


def spherical_harmonics(nmax, colat, lon):
    """
    Fully normalized spherical harmonics.

    Parameters
    ----------
    nmax : int
       maximum spherical harmonic degree to compute
    colat : float, array_like(m,)
       co-latitude of evaluation points in radians
    lon : float, array_like(m,)
       longitude of evaluation points in radians

    Returns
    -------
    Ynm : array_like(m, nmax + 1, nmax + 1)
       Array containing the fully normalized spherical harmonics. Ynm[:, n, m] returns Cnm(colat, lon)
       Ynm[:, m-1, n] (for m > 0) returns Snm(colat, lon).

    """
    longitude = np.atleast_1d(lon)

    sh_array = legendre_functions(nmax, colat)
    for m in range(1, nmax + 1):
        sh_array[:, m:, m] *= np.cos(m * longitude)[:, np.newaxis]
        sh_array[:, m - 1, m:] *= np.sin(m * longitude)[:, np.newaxis]

    return sh_array


def ravel_coefficients(array, min_degree=0, max_degree=None):
    """
    Ravel a 2d or 3d array representing degree/order.

    Parameters
    ----------
    array : ndarray(k, m, m) or ndarray(m, m)
        coefficient array
    min_degree : int
        return coefficients starting with min_degree
    max_degree : int
        return coefficients up to (and including) max_degree

    Returns
    -------
    coeffs : ndarray(k, (max_degree + 1)**2 - min_degree**2) or ndarray((max_degree + 1)**2 - min_degree**2)
        ravelled array
    """
    if max_degree is None:
        max_degree = array.shape[-1] - 1

    coefficient_count = (max_degree + 1) * (max_degree + 1) - min_degree * min_degree

    if array.ndim == 2:
        x = np.zeros(coefficient_count, dtype=array.dtype)

        idx = 0
        for n in range(min_degree, array.shape[-1]):
            x[idx] = array[n, 0]
            idx += 1
            for m in range(1, n + 1):
                x[idx] = array[n, m]
                x[idx + 1] = array[m - 1, n]
                idx += 2

    elif array.ndim == 3:
        x = np.zeros((array.shape[0], coefficient_count), dtype=array.dtype)

        idx = 0
        for n in range(min_degree, array.shape[-1]):
            x[:, idx] = array[:, n, 0]
            idx += 1
            for m in range(1, n + 1):
                x[:, idx] = array[:, n, m]
                x[:, idx + 1] = array[:, m - 1, n]
                idx += 2

    else:
        raise ValueError('Only 2d or 3d spherical harmonic arrays can be unraveled.')

    return x


def unravel_coefficients(vector, min_degree=0, max_degree=None):
    """
    Unravel a 1d spherical harmonic coefficient vector into a 2d array

    Parameters
    ----------
    vector : ndarray(k,)
        coefficient vector
    min_degree : int
        return coefficients starting with min_degree
    max_degree : int
        return coefficients up to (and including) max_degree

    Returns
    -------
    array : ndarray(max_degree + 1, max_degree + 1)
        spherical harmonic coefficient vector as 2d array

    """
    if max_degree is None:
        max_degree = int(np.sqrt(vector.size + min_degree * min_degree) - 1)

    array = np.zeros((max_degree + 1, max_degree + 1), dtype=vector.dtype)

    idx = 0
    for n in range(min_degree, max_degree + 1):
        array[n, 0] = vector[idx]
        idx += 1
        for m in range(1, n + 1):
            array[n, m] = vector[idx]
            array[m - 1, n] = vector[idx + 1]
            idx += 2

    return array


def normal_gravity(r, colat, a=6378137.0, f=298.2572221010 ** -1, convergence_threshold=1e-9):
    """
    Normal gravity on the ellipsoid (GRS80).

    Parameters
    ----------
    r : float, array_like, shape(m, )
        radius of evaluation point(s) in meters
    colat : float, array_like, shape (m,)
        co-latitude of evaluation points in radians
    a : float
        semi-major axis of ellipsoid (Default: GRS80)
    f : float
        flattening of ellipsoid (Default: GRS80)
    convergence_threshold : float
        maximum absolute difference between latitude iterations in radians

    Returns
    -------
    g : float, array_like, shape(m,) (depending on types of r and colat)
        normal gravity at evaluation point(s) in [m/s**2]
    """
    ga = 9.7803267715
    gb = 9.8321863685
    m = 0.00344978600308

    z = np.cos(colat) * r
    p = np.abs(np.sin(colat) * r)

    b = a * (1 - f)
    e2 = (a / b - 1) * (a / b + 1)
    latitude = np.arctan2(z * (1 + e2), p)

    L = np.abs(latitude) < 60 / 180 * np.pi

    latitude_old = np.full(latitude.shape, np.inf)
    h = np.zeros(latitude.shape)

    while np.max(np.abs(latitude - latitude_old)) > convergence_threshold:
        latitude_old = latitude.copy()

        N = (a / b) * a / np.sqrt(1 + e2 * np.cos(latitude) ** 2)
        h[L] = p[L] / np.cos(latitude[L]) - N[L]
        h[~L] = z[~L] / np.sin(latitude[~L]) - N[~L] / (1 + e2)

        latitude = np.arctan2(z * (1 + e2), p * (1 + e2 * h / (N + h)))

    cos2 = np.cos(latitude) ** 2
    sin2 = np.sin(latitude) ** 2

    gamma0 = (a * ga * cos2 + b * gb * sin2) / np.sqrt(a ** 2 * cos2 + b ** 2 * sin2)
    return gamma0 - 2 * ga / a * (1 + f + m + (-3 * f + 5 * m / 2) * sin2) * h + 3 * ga / a ** 2 * h ** 2


def geocentric_radius(latitude, a=6378137.0, f=298.2572221010 ** -1):
    """
    Geocentric radius of a point on the ellipsoid.

    Parameters
    ----------
    latitude : float, array_like, shape(m, )
       latitude of evaluation point(s) in radians
    a : float
       semi-major axis of ellipsoid (Default: GRS80)
    f : float
       flattening of ellipsoid (Default: GRS80)

    Returns
    -------
    r : float, array_like, shape(m,) (depending on type latitude)
       geocentric radius of evaluation point(s) in [m]
    """
    e2 = f * (2 - f)
    nu = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)

    return nu * np.sqrt(np.cos(latitude) ** 2 + (1 - e2) ** 2 * np.sin(latitude) ** 2)


def colatitude(latitude, a=6378137.0, f=298.2572221010 ** -1):
    """
    Co-latitude of a point on the ellipsoid.

    Parameters
    ----------
    latitude : float, array_like, shape(m, )
      latitude of evaluation point(s) in radians
    a : float
      semi-major axis of ellipsoid (Default: GRS80)
    f : float
      flattening of ellipsoid (Default: GRS80)

    Returns
    -------
    psi : float, array_like, shape(m,) (depending on type latitude)
      colatitude of evaluation point(s) in [rad]
    """
    e2 = f * (2 - f)
    nu = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)

    return np.arccos(nu * (1 - e2) * np.sin(latitude) / geocentric_radius(latitude, a, f))


class TemporalBasisFunction(metaclass=abc.ABCMeta):
    """
    Base class for temporal basis functions, such as polynomials or oscillations. Derived classes must implement
    a method `design_matrix` which returns the design matrix of a least squares adjustment where the observations
    are a time series.
    """
    @abc.abstractmethod
    def design_matrix(self, epochs):
        pass


class Oscillation(TemporalBasisFunction):
    """
    Implements an oscillation represented by sine and cosine.

    .. math:: \Phi(t) = a \cdot \cos(\frac{2\pi}{T}(t-t_0)) + b \cdot \sin(\frac{2\pi}{T}(t-t_0))

    Parameters
    ----------
    period : float
        oscillation period in days
    reference_epoch : datetime object
        reference epoch of the oscillation
    """
    def __init__(self, period, reference_epoch=None):

        self.__period = period
        self.__reference_epoch = reference_epoch

    def design_matrix(self, epochs):
        """
        Returns the design matrix for the determination of the parameters a and b.

        Parameters
        ----------
        epochs : list of datetime objects
            time stamps of the observations

        Returns
        -------
        design_matrix : ndarray(n, 2)
            design matrix
        """
        t = np.array([grates.time.mjd(e) for e in epochs])
        if self.__reference_epoch is not None:
            t -= grates.time.mjd(self.__reference_epoch)

        dmatrix = np.empty((t.size, 2))
        dmatrix[:, 0] = np.cos(2*np.pi/self.__period * t)
        dmatrix[:, 1] = np.sin(2 * np.pi / self.__period * t)

        return dmatrix


class Polynomial(TemporalBasisFunction):
    """
    Implements a polynomial of degree p represented by real coefficients.

    .. math:: \Phi(t) = \sum_{k=0}^{p+1} a_k (t - t_0)^k

    Parameters
    ----------
    degree : int
        polynomial degree
    reference_epoch : datetime object
        reference epoch of the oscillation
    """
    def __init__(self, degree, reference_epoch=None):

        self.__degree = degree
        self.__reference_epoch = reference_epoch

    def design_matrix(self, epochs):
        """
        Returns the design matrix for the determination of the polynomial coefficients.

        Parameters
        ----------
        epochs : list of datetime objects
            time stamps of the observations

        Returns
        -------
        design_matrix : ndarray(n, p + 1)
            design matrix
        """
        t = np.array([grates.time.mjd(e) for e in epochs])
        if self.__reference_epoch is not None:
            t -= grates.time.mjd(self.__reference_epoch)

        dmatrix = np.empty((t.size, self.__degree + 1))
        dmatrix[:, 0] = 1
        for k in range(1, self.__degree + 1):
            dmatrix[:, k] = t**k

        return dmatrix


def load_love_numbers(max_degree=None, frame='CE'):
    """
    Load Love numbers computed by Wang et al. (2012) [1]_ for the elastic Earth model ak135 [2]_.
    from degree 0 to 46340.

    Parameters
    ----------
    max_degree : int
        maximum degree of the load love numbers to be returned (default: return all love numbers).
    frame : str
        frame of the load love numbers (CM, CE, CF).

    Returns
    -------
    k : array_like
        gravity change load love numbers
    h : array_like
        radial displacement load love numbers
    l : array_like
        horizontal displacement load love numbers

    References
    ----------

    .. [1] Hansheng Wang, Longwei Xiang, Lulu Jia, Liming Jiang, Zhiyong Wang, Bo Hu, and Peng Gao. 2012. Load Love
           numbers and Green's functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined
           crustal structure from Crust 2.0. Comput. Geosci. 49 (December, 2012), 190–199.
           DOI:https://doi.org/10.1016/j.cageo.2012.06.022

    .. [2] B. L. N. Kennett, E. R. Engdahl, R. Buland, Constraints on seismic velocities in the Earth from traveltimes,
           Geophysical Journal International, Volume 122, Issue 1, July 1995, Pages 108–124,
           https://doi.org/10.1111/j.1365-246X.1995.tb03540.x

    """
    if max_degree is not None and max_degree < 1:
        return np.zeros(1), np.zeros(1), np.zeros(1)

    hlk = np.vstack((np.zeros((1, 3)), np.loadtxt(pkg_resources.resource_filename('grates', 'data/ak135-LLNs-complete.dat.gz'),
                                                  skiprows=1, usecols=(1, 2, 3), max_rows=max_degree)))

    if frame.lower() == 'cm':
        hlk[1, :] -= 1
    elif frame.lower() == 'cf':
        hlk_ce = hlk[1, :].copy()
        hlk[1, 0] = (hlk_ce[0] - hlk_ce[1]) * 2 / 3
        hlk[1, 1] = (hlk_ce[0] - hlk_ce[1]) * -1 / 3
        hlk[1, 2] = (-1 / 3 * hlk_ce[0] - 2 / 3 * hlk_ce[1])
    elif frame.lower() == 'ce':
        pass
    else:
        raise ValueError('frame of load love numbers must be one of CM, CE, or CF (got <' + frame + '>)')

    return hlk[:, 2], hlk[:, 0], hlk[:, 1]


def kaula_curve(min_degree, max_degree, kaula_factor=1e-10, kaula_power=4.0):
    """
    Return a Kaula-type curve of the form :math:`\sigma_n^2 = f \cdot \frac{1}{n^p}` as a coefficient array.

    Parameters
    ----------
    min_degree : int
        minimum degree of the curve (degrees below min_degree are filled with zeros)
    max_degree : int
        maximum degree of the curve
    kaula_factor : float
        scale factor of Kaula curve
    kaula_power : float
        power of Kaula curve

    Returns
    -------
    anm : ndarray(max_degree + 1, max_degree + 1)
        Kaula curve as coefficient array
    """
    anm = np.zeros((max_degree + 1, max_degree + 1))
    for n in range(min_degree, max_degree + 1):
        row_index, col_index = grates.gravityfield.degree_indices(n)
        anm[row_index, col_index] = kaula_factor * np.power(float(n), -float(kaula_power))

    return anm
