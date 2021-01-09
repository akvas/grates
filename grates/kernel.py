# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Isotropic harmonic integral kernels.
"""

import numpy as np
import abc
import grates.utilities
import grates.gravityfield
import grates.grid


def get_kernel(kernel_name):
    """
    Return kernel coefficients.

    Parameters
    ----------
    kernel_name : string
        name of kernel, currently implemented: water height ('EWH', 'water_height'),
        ocean bottom pressure ('OBP', 'ocean_bottom_pressure'), potential ('potential'),
        geoid height ('geoid_height'), surface density ('surface_density')

    Returns
    -------
    kernel : Kernel subclass instance
        kernel associated with kernel_name

    Raises
    ------
    ValueError
        if an unrecognized kernel name is passed

    """
    if kernel_name.lower() in ['ewh', 'water_height']:
        ker = grates.kernel.WaterHeight()

    elif kernel_name.lower() in ['obp', 'ocean_bottom_pressure']:
        ker = grates.kernel.OceanBottomPressure()

    elif kernel_name.lower() in ['potential']:
        ker = grates.kernel.Potential()

    elif kernel_name.lower() in ['geoid', 'geoid_height']:
        ker = grates.kernel.GeoidHeight()

    elif kernel_name.lower() in ['surface_density']:
        ker = grates.kernel.SurfaceDensity()

    else:
        raise ValueError("Unrecognized kernel '{0:s}'.".format(kernel_name))

    return ker


class Kernel(metaclass=abc.ABCMeta):
    """
    Base interface for band-limited spherical harmonic kernels.

    Subclasses must implement a method `_coefficients` which depends on min_degree, max_degree, radius and
    co-latitude and returns kernel coefficients.

    Kernel coefficients transform the corresponding quantity (e.g. water height) into potential, the inverse
    coefficients transform potential coefficients into the corresponding quantity.
    """

    @abc.abstractmethod
    def _coefficients(self, min_degree, max_degree, r, colat):
        pass

    def coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """
        Kernel coefficients for degrees min_degree to max_degree.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to return
        max_degree : int
            maximum coefficient degree to return
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : ndarray(m, max_degree + 1 - min_degree)
            kernel coefficients for degree n for all evaluation points

        Raises
        ------
        ValueError:
            if r and colat cannot be sensibly broadcast
        """
        if np.isscalar(r) and np.isscalar(colat):
            radius, colatitude = r, colat
        elif np.isscalar(r) and isinstance(colat, np.ndarray):
            radius, colatitude = np.full(colat.shape, r), colat
        elif isinstance(r, np.ndarray) and np.isscalar(colat):
            radius, colatitude = r, np.full(r.shape, colat),
        elif isinstance(r, np.ndarray) and isinstance(colat, np.ndarray):
            if r.shape != colat.shape:
                raise ValueError('shape mismatch in radius and colatitude: objects cannot be broadcast to a single shape')
            else:
                radius, colatitude = r, colat
        else:
            raise ValueError('input must be either numeric scalar or ndarrays of matching or broadcastable dimensions')

        return self._coefficients(min_degree, max_degree, radius, colatitude)

    def inverse_coefficient(self, n, r=6378136.3, colat=0):
        """
        Return inverse kernel coefficient.

        Parameters
        ----------
        n : int
            degree of kernel coefficient
        r : float, ndarray(m,)
            evaluation radius
        colat : float, ndarray(m,)
            colatitude of evaluation points

        Returns
        -------
        inverse_coeff : ndarray(m,)
            inverse kernel coefficient for degree n
        """
        return self.coefficient(n, r, colat)**-1

    def coefficient(self, n, r=6378136.3, colat=0):
        """
        Return kernel coefficient for a specific degree.

        Parameters
        ----------
        n : int
            degree of kernel coefficient
        r : float, ndarray(m,)
            evaluation radius
        colat : float, ndarray(m,)
            colatitude of evaluation points

        Returns
        -------
        inverse_coeff : ndarray(m,)
            inverse kernel coefficient for degree n
        """
        return self.coefficients(n, n, r, colat).squeeze(axis=1)

    def inverse_coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """
        Inverse kernel coefficients for degrees min_degree to max_degree.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to return
        max_degree : int
            maximum coefficient degree to return
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : ndarray(m, max_degree + 1 - min_degree)
            inverse kernel coefficients for degrees min_degree to max_degree for all evaluation points
        """
        return 1.0 / self.coefficients(min_degree, max_degree, r, colat)

    def coefficient_array(self, min_degree, max_degree, r=6378136.3, colat=0):
        """
        Return kernel coefficients up to a given maximum degree as spherical harmonic coefficient array.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to return
        max_degree : int
            maximum coefficient degree to return
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : ndarray(m, max_degree + 1, max_degree + 1)
            kernel coefficients for degrees min_degree to max_degree for all evaluation points
        """
        count = max(np.asarray(r).size, np.asarray(colat).size)
        kn = self.coefficients(min_degree, max_degree, r, colat)

        kn_array = np.zeros((count, max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            kn_array[:, row_idx, col_idx] = kn[n - min_degree]

        return kn_array

    def inverse_coefficient_array(self, min_degree, max_degree, r=6378136.3, colat=0):
        """
        Return kernel coefficients up to a given maximum degree as spherical harmonic coefficient array.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to return
        max_degree : int
            maximum coefficient degree to return
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : ndarray(m, max_degree + 1, max_degree + 1)
            inverse kernel coefficients for degrees min_degree to max_degree for all evaluation points
        """
        count = max(np.asarray(r).size, np.asarray(colat).size)
        kn = self.inverse_coefficients(min_degree, max_degree, r, colat)

        kn_array = np.zeros((count, max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            kn_array[:, row_idx, col_idx] = kn[n - min_degree]

        return kn_array

    def evaluate(self, min_degree, max_degree, psi, r=6378136.3, colat=0):
        """
        Evaluate kernel in spatial domain.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to use
        max_degree : int
            maximum coefficient degree to use
        psi : ndarray(m,)
            spherical distance in radians
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kernel : ndarray(m,)
            kernel evaluated at the given spherical distance
        """
        kn = self.coefficients(min_degree, max_degree, r, colat) * np.sqrt(2 * np.arange(min_degree, max_degree + 1) + 1)

        return grates.utilities.legendre_summation(kn, psi)

    def evaluate_grid(self, min_degree, max_degree, source_longitude, source_latitude, eval_longitude, eval_latitude, r=6378136.3, colat=0):
        """
        Evaluate kernel on a regular grid.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to use
        max_degree : int
            maximum coefficient degree to use
        source_longitude : float
            longitude of source point in radians
        source_latitude : float
            latitude of source point in radians
        eval_longitude : ndarray(m,)
            longitude of evaluation meridians in radians
        eval_latitude : ndarray(n,)
            latitude of evaluation parallels in radians
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kernel : ndarray(m,)
            kernel evaluated at the given spherical distance
        """
        lon, lat = np.meshgrid(eval_longitude, eval_latitude)
        psi = grates.grid.spherical_distance(source_longitude, source_latitude, lon, lat, r=1)

        return self.evaluate(min_degree, max_degree, psi, r, colat)

    def modulation_transfer(self, min_degree, max_degree, max_psi=np.pi, nsteps=100):
        """
        Modulation transfer function for bandlimited isotropic kernels. Implemented after [1]_.

        Parameters
        ----------
        min_degree : int
            minimum coefficient degree to use
        max_degree : int
            maximum coefficient degree to use
        max_psi : float
            compute the MTR up to maximum spherical distance [radians] (default: pi)
        nsteps : int
            number of samples to create

        Returns
        -------
        psi : ndarray(m,)
            spherical distance in radians
        mtf : ndarray(m,)
            modulation transfer function

        References
        ----------

        .. [1] Vishwakarma, B.D.; Devaraju, B.; Sneeuw, N. What Is the Spatial Resolution of grace Satellite Products
               for Hydrology? Remote Sens. 2018, 10, 852.

        """
        psi = np.linspace(0, max_psi, nsteps)

        kn_ref = self.evaluate(min_degree, max_degree, psi)
        kn_ref = np.concatenate((kn_ref[1::-1], kn_ref))
        modulation = 2 * self.evaluate(min_degree, max_degree, psi * 0.5)

        mtf = np.zeros(psi.size)
        for k in range(psi.size):
            mtf[k] = max(1 - modulation[k]/(np.max(kn_ref[k:] + kn_ref[0:kn_ref.size - k])), 0)

        return psi, mtf


class WaterHeight(Kernel):
    """
    Implementation of the water height kernel. Applied to a sequence of potential coefficients, the result is
    equivalent water height in meters when propagated to space domain.

    Parameters
    ----------
    rho : float
        density of water in [kg/m**3]
    """
    def __init__(self, rho=1025):

        self.__rho = rho
        self.__love_numbers, _, _ = grates.data.load_love_numbers()

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        kn = (4 * np.pi * 6.673e-11 * self.__rho) * (1 + self.__love_numbers[min_degree:max_degree + 1]) / (2 * np.arange(min_degree, max_degree + 1, dtype=float) + 1)
        return (kn[:, np.newaxis] * r).T


class OceanBottomPressure(Kernel):
    """
    Implementation of the ocean bottom pressure kernel. Applied to a sequence of potential coefficients, the result
    is ocean bottom pressure in Pascal when propagated to space domain.
    """
    def __init__(self):

        self.__love_numbers, _, _ = grates.data.load_love_numbers()

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        kn = (4 * np.pi * 6.673e-11) * (1 + self.__love_numbers[min_degree:max_degree + 1]) / (2 * np.arange(min_degree, max_degree + 1, dtype=float) + 1)
        return (kn[:, np.newaxis] * (r / grates.gravityfield.GRS80.normal_gravity(r, colat))).T


class SurfaceDensity(Kernel):
    """
    Implementation of the surface density kernel.
    """
    def __init__(self):

        self.__love_numbers, _, _ = grates.data.load_love_numbers()

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        kn = (4 * np.pi * 6.673e-11) * (1 + self.__love_numbers[min_degree:max_degree + 1]) / (2 * np.arange(min_degree, max_degree + 1, dtype=float) + 1)
        return (kn[:, np.newaxis] * r).T


class Potential(Kernel):
    """
    Implementation of the Poisson kernel (disturbing potential).
    """
    def __init__(self):
        pass

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        count = max(np.asarray(r).size, np.asarray(colat).size)

        return np.ones((count, max_degree + 1 - min_degree))


class Gauss(Kernel):
    """
    Implementation of the Gauss kernel.
    """
    def __init__(self, radius):

        if radius < 0:
            raise ValueError('Gaussian filter radius must be positive (got {0:f})'.format(radius))

        nmax = 1024
        self.__radius = radius

        if self.__radius > 0:
            b = np.log(2.0) / (1 - np.cos(radius / 6378.1366))
            self.__wn = np.zeros(nmax + 1)
            self.__wn[0] = 1.0
            self.__wn[1] = (1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b)) - 1 / b
            for n in range(2, nmax + 1):
                self.__wn[n] = -(2 * n - 1) / b * self.__wn[n - 1] + self.__wn[n - 2]
                if self.__wn[n] < 1e-7:
                    break
        else:
            self.__wn = np.ones(nmax + 1)

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        local_nmax = self.__wn.size - 1
        if max_degree > local_nmax:
            if self.__radius > 0:
                wn = self.__wn.copy()
                self.__wn = np.empty(max_degree + 1)
                self.__wn[0:local_nmax + 1] = wn
                b = np.log(2.0) / (1 - np.cos(self.__radius / 6378.1363))
                for d in range(local_nmax + 1, max_degree + 1):
                    self.__wn[d] = -(2 * d - 1) / b * self.__wn[d - 1] + self.__wn[d - 2]
                    if self.__wn[d] < 1e-7:
                        break
            else:
                self.__wn = np.ones(max_degree + 1)

        count = max(np.asarray(r).size, np.asarray(colat).size)

        return np.tile(self.__wn[min_degree:max_degree + 1], (count, 1))


class GeoidHeight(Kernel):
    """
    Implementation of the geoid height kernel (disturbing potential divided by normal gravity).
    """
    def __init__(self):
        pass

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        return np.tile(grates.gravityfield.GRS80.normal_gravity(r, colat)[:, np.newaxis], (1, max_degree + 1 - min_degree))


class UpwardContinuation(Kernel):
    """
    Implementation of the upward continuation kernel.

    Parameters
    ----------
    R : float
        reference radius
    """
    def __init__(self, R=6.3781363000e+06):

        self.__R = R

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        return np.power(np.atleast_1d(self.__R / r)[:, np.newaxis], np.arange(min_degree, max_degree + 1, dtype=int) + 1)
