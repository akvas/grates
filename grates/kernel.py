# Copyright (c) 2018 Andreas Kvas
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
        name of kernel, currently implemented: water height ('ewh', 'water_height'),
        ocean bottom pressure ('obp', 'ocean_bottom_pressure')

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

    else:
        raise ValueError("Unrecognized kernel '{0:s}'.".format(kernel_name))

    return ker


class Kernel(metaclass=abc.ABCMeta):
    """
    Base interface for band-limited spherical harmonic kernels.

    Subclasses must implement a method `coefficient` which depends on degree, radius and
    co-latitude and returns kernel coefficients.

    Kernel coefficients transform the corresponding quantity (e.g. water height) into potential, the inverse
    coefficients transform potential coefficients into the corresponding quantity.
    """

    @abc.abstractmethod
    def coefficient(self, n, r, colat):
        pass

    def inverse_coefficient(self, n, r=6378136.6, colat=0):
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
        inverse_coeff : float, ndarray(m,)
            inverse kernel coefficient for degree n
        """
        return self.coefficient(n, r, colat)**-1

    def coefficients(self, min_degree, max_degree, r=6378136.6, colat=0):
        """Return kernel coefficients up to a given maximum degree."""
        return np.vstack([self.coefficient(n, r, colat) for n in range(min_degree, max_degree + 1)]).T

    def inverse_coefficients(self, min_degree, max_degree, r=6378136.6, colat=0):
        """Return inverse kernel coefficients up to a given maximum degree."""
        return np.vstack([self.inverse_coefficient(n, r, colat) for n in range(min_degree, max_degree + 1)]).T

    def coefficient_array(self, min_degree, max_degree, r=6378136.6, colat=0):
        """Return kernel coefficients up to a given maximum degree as spherical harmonic coefficient array."""
        count = max(np.asarray(r).size, np.asarray(colat).size)

        kn_array = np.zeros((count, max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            kn_array[:, row_idx, col_idx] = self.coefficient(n, r, colat)

        return kn_array

    def inverse_coefficient_array(self, min_degree, max_degree, r=6378136.6, colat=0):
        """Return inverse kernel coefficients up to a given maximum degree as spherical harmonic coefficient array."""
        count = max(np.asarray(r).size, np.asarray(colat).size)

        kn_array = np.zeros((count, max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            kn_array[:, row_idx, col_idx] = self.inverse_coefficient(n, r, colat)

        return kn_array

    def evaluate(self, nmax, psi, r=6378136.6, colat=0):
        """
        Evaluate kernel in spatial domain.

        Parameters
        ----------
        nmax : int
            maximum spherical harmonic degree
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
        kn = np.array([self.coefficient(n, r, colat)*np.sqrt(2*n + 1) for n in range(nmax + 1)])

        return grates.utilities.legendre_summation(kn, psi)

    def evaluate_grid(self, nmax, source_longitude, source_latitude, eval_longitude, eval_latitude,
                      r=6378136.6, colat=0):
        """
        Evaluate kernel on a regular grid.

        Parameters
        ----------
        nmax : int
            maximum spherical harmonic degree
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

        return self.evaluate(nmax, psi, r, colat)

    def modulation_transfer(self, nmax, max_psi=np.pi, nsteps=100):
        """
        Modulation transfer function for bandlimited isotropic kernels. Implemented after [1]_.

        Parameters
        ----------
        nmax : int
            maximum expansion degree of the kernel
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

        kn_ref = self.evaluate(nmax, psi)
        kn_ref = np.concatenate((kn_ref[1::-1], kn_ref))
        modulation = 2 * self.evaluate(nmax, psi * 0.5)

        mtf = np.zeros(psi.size)
        for k, p in enumerate(psi):
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
        self.__love_numbers, _, _ = grates.utilities.load_love_numbers()

    def coefficient(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        love_number = self.__love_numbers[n] if n < self.__love_numbers.size else 0
        kn = (2 * n + 1) / (1 + love_number) / (4 * np.pi * 6.673e-11 * self.__rho)

        return r/kn


class OceanBottomPressure(Kernel):
    """
    Implementation of the ocean bottom pressure kernel. Applied to a sequence of potential coefficients, the result
    is ocean bottom pressure in Pascal when propagated to space domain.
    """
    def __init__(self):

        self.__love_numbers, _, _ = grates.utilities.load_love_numbers()

    def coefficient(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        love_number = self.__love_numbers[n] if n < self.__love_numbers.size else 0
        kn = (2 * n + 1) / (1 + love_number) / (4 * np.pi * 6.673e-11)

        return r/(kn * grates.utilities.normal_gravity(r, colat))


class SurfaceDensity(Kernel):
    """
    Implementation of the surface density kernel.
    """
    def __init__(self):

        self.__love_numbers, _, _ = grates.utilities.load_love_numbers()

    def coefficient(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        love_number = self.__love_numbers[n] if n < self.__love_numbers.size else 0
        kn = (2 * n + 1) / (1 + love_number) / (4 * np.pi * 6.673e-11 * r)

        return 1/kn


class Potential(Kernel):
    """
    Implementation of the Poisson kernel (disturbing potential).
    """
    def __init__(self):
        pass

    def coefficient(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        count = max(np.asarray(r).size, np.asarray(colat).size)

        return np.ones(count)


class Gauss(Kernel):
    """
    Implementation of the Gauss kernel.
    """
    def __init__(self, radius):

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

    def coefficient(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        nmax = self.__wn.size - 1
        if n > nmax:
            if self.__radius > 0:
                wn = self.__wn.copy()
                self.__wn = np.empty(n + 1)
                self.__wn[0:nmax+1] = wn
                b = np.log(2.0) / (1 - np.cos(self.__radius / 6378.1366))
                for d in range(nmax + 1, n + 1):
                    self.__wn[d] = -(2 * n - 1) / b * self.__wn[d - 1] + self.__wn[d - 2]
                    if self.__wn[d] < 1e-7:
                        break
            else:
                self.__wn = np.ones(n + 1)

        count = max(np.asarray(r).size, np.asarray(colat).size)

        return np.full(count, self.__wn[n])


class GeoidHeight(Kernel):
    """
    Implementation of the geoid height kernel (disturbing potential divided by normal gravity).
    """
    def __init__(self):
        pass

    def coefficient(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """

        return grates.utilities.normal_gravity(r, colat)
