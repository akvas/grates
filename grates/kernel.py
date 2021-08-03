# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Harmonic integral kernels.
"""

import numpy as np
import abc
import grates.utilities
import grates.gravityfield
import grates.grid
import scipy.optimize
import functools


def get_kernel(kernel_name):
    """
    Return kernel coefficients.

    Parameters
    ----------
    kernel_name : string
        name of kernel, currently implemented: water height ('EWH', 'water_height'),
        ocean bottom pressure ('OBP', 'ocean_bottom_pressure'), potential ('potential'),
        geoid height ('geoid', 'geoid_height'), surface density ('surface_density'),
        gravity anomaly, ('anomaly', 'gravity_anomaly')

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

    elif kernel_name.lower() in ['anomaly', 'gravity_anomaly']:
        ker = grates.kernel.GravityAnomaly()

    else:
        raise ValueError("Unrecognized kernel '{0:s}'.".format(kernel_name))

    return ker


class IsotropicKernel(metaclass=abc.ABCMeta):
    """
    Base interface for band-limited isotropic harmonic kernels.

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
        kn = self.coefficient(n, r, colat)
        return np.zeros(kn.shape) if np.allclose(kn, 0.0) else 1.0 / kn

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
        kn = self.coefficients(min_degree, max_degree, r, colat)
        return np.vstack([np.zeros(kn.shape[0]) if np.allclose(kn[:, k], 0.0) else 1.0 / kn[:, k] for k in range(kn.shape[1])]).T

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
            kn_array[:, row_idx, col_idx] = kn[:, n - min_degree]

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
            kn_array[:, row_idx, col_idx] = kn[:, n - min_degree]

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
        kn = np.zeros(max_degree + 1)
        kn[min_degree:] = self.coefficients(min_degree, max_degree, r, colat)[0, :] * np.sqrt(2 * np.arange(min_degree, max_degree + 1) + 1)

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

    def spatial_resolution(self, min_degree, max_degree, R=6378136.3, threshold=1000):
        """
        Compute the spatial resolution of the kernel. Two Dirac pulses are shifted on the sphere until a local minimum in the connecting line between the two occurs.

        Parameters
        ----------
        min_degree : int
            minimum evaluation degree
        max_degree : int
            maximum evaluation degree
        R : float
            radius of the evaluation sphere in meters
        threshold : float
            algorithm stops once the search window is smaller than threshold (given in meters)

        Returns
        -------
        resolution : float
            distance in meters between two Dirac pulses when a local minimum occurs
        """
        def kernel_sum(psi0, psi):
            return self.evaluate(min_degree, max_degree, psi).squeeze() + self.evaluate(min_degree, max_degree, psi0 - psi).squeeze()

        def brute_force(min_psi, max_psi):
            if (max_psi - min_psi) * R < threshold:
                return max_psi * 0.5 + min_psi * 0.5

            psi0 = np.linspace(min_psi, max_psi, 3)
            for k in range(1, psi0.size):
                res = scipy.optimize.fminbound(functools.partial(kernel_sum, (psi0[k],)), 0, psi0[k])
                has_local_minimum = np.abs(res - psi0[k]) * R > threshold and np.abs(res) * R > threshold
                if has_local_minimum:
                    return brute_force(psi0[k - 1], psi0[k])

        return brute_force(0, np.pi) * R


class WaterHeight(IsotropicKernel):
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


class OceanBottomPressure(IsotropicKernel):
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


class SurfaceDensity(IsotropicKernel):
    """
    Implementation of the surface density kernel.
    """
    def __init__(self):

        self.__love_numbers, _, _ = grates.data.load_love_numbers()

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        kn = (4 * np.pi * 6.673e-11) * (1 + self.__love_numbers[min_degree:max_degree + 1]) / (2 * np.arange(min_degree, max_degree + 1, dtype=float) + 1)
        return (kn[:, np.newaxis] * r).T


class Potential(IsotropicKernel):
    """
    Implementation of the Poisson kernel (disturbing potential).
    """
    def __init__(self):
        pass

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        count = max(np.asarray(r).size, np.asarray(colat).size)

        return np.ones((count, max_degree + 1 - min_degree))


class GravityAnomaly(IsotropicKernel):
    """
    """
    def __init__(self):
        pass

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        kn = np.array([1 / (n - 1) if n != 1 else 0.0 for n in np.arange(min_degree, max_degree + 1, dtype=float)])
        return (kn[:, np.newaxis] * r).T


class Gauss(IsotropicKernel):
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


class GeoidHeight(IsotropicKernel):
    """
    Implementation of the geoid height kernel (disturbing potential divided by normal gravity).
    """
    def __init__(self):
        pass

    def _coefficients(self, min_degree, max_degree, r=6378136.3, colat=0):
        """Kernel coefficients for degrees min_degree to max_degree."""
        return np.tile(grates.gravityfield.GRS80.normal_gravity(r, colat)[:, np.newaxis], (1, max_degree + 1 - min_degree))


class UpwardContinuation(IsotropicKernel):
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


class AnisotropicKernel:
    """
    Representation of possibly anisotropic kernels in space domain.

    Parameters
    ----------
    K : ndarray
        kernel matrix (spherical harmonic mapping in degreewise order)
    min_degree : int
        minimum filter degree
    max_degree : int
        maximum filter degree
    """
    def __init__(self, K, min_degree, max_degree):

        self.__matrix = K.copy()
        self.__min_degree = min_degree
        self.__max_degree = max_degree

    def evaluate(self, source_longitude, source_latitude, eval_longitude, eval_latitude):
        """
        Evaluate the filter kernel in space domain.

        Parameters
        ----------
        source_longitude : float
            longitude of source point in radians
        source_latitude : float
            latitude of source point in radians
        eval_longitude : ndarray(m,)
            longitude of evaluation points in radians
        eval_latitude : ndarray(m,)
            latitude of evaluation points in radians

        Returns
        -------
        kernel : ndarray(m,)
            kernel values at the evaluation points
        """
        spherical_harmonics_source = grates.utilities.spherical_harmonics(self.__max_degree, np.pi * 0.5 - source_latitude, source_longitude)
        v1 = grates.utilities.ravel_coefficients(spherical_harmonics_source, self.__min_degree, self.__max_degree) @ self.__matrix

        spherical_harmonics_eval = grates.utilities.spherical_harmonics(self.__max_degree, np.pi * 0.5 - eval_latitude, eval_longitude)

        return np.atleast_1d((v1 @ grates.utilities.ravel_coefficients(spherical_harmonics_eval, self.__min_degree, self.__max_degree).T).squeeze())

    def evaluate_grid(self, source_longitude, source_latitude, eval_longitude, eval_latitude):
        """
        Evaluate the filter kernel on a longitude/latitude grid.

        Parameters
        ----------
        source_longitude : float
            longitude of source point in radians
        source_latitude : float
            latitude of source point in radians
        eval_longitude : ndarray(m,)
            longitude of evaluation meridians in radians
        eval_latitude : ndarray(n,)
            latitude of evaluation parallels in radians

        Returns
        -------
        grid : ndarray(m, n)
            kernel values on the grid
        """
        spherical_harmonics_source = grates.utilities.spherical_harmonics(self.__max_degree,
                                                                          np.pi * 0.5 - source_latitude,
                                                                          source_longitude)
        v1 = grates.utilities.ravel_coefficients(spherical_harmonics_source, self.__min_degree, self.__max_degree) @ self.__matrix

        pnm = grates.utilities.legendre_functions(self.__max_degree, np.pi * 0.5 - eval_latitude)
        cs = grates.utilities.trigonometric_functions(self.__max_degree, eval_longitude)

        grid = np.empty((eval_latitude.size, eval_longitude.size))
        for k in range(eval_latitude.size):
            grid[k, :] = (grates.utilities.ravel_coefficients(cs * pnm[k], self.__min_degree, self.__max_degree) @ v1.T).squeeze()

        return grid

    def modulation_transfer(self, psi, central_longitude=0, central_latitude=0, azimuth=0):
        """
        Modulation transfer function of anisotropic filter kernel. Two kernels are shifted on a great circle which
        passes through the evaluation point with a given azimuth. Inspired by [1]_

        Parameters
        ----------
        psi : float, ndarray(k,)
            spherical distance in radians for which to compute the modulation transfer function
        central_longitude : float
            longitude of evaluation point in radians
        central_latitude : float
            latitude of evaluation point in radians
        azimuth : float
            azimuth of great circle in the evaluation point in radians

        Returns
        -------
        mtf :  ndarray(nsteps,)
            modulation transfer function

        References
        ----------

        .. [1] Vishwakarma, B.D.; Devaraju, B.; Sneeuw, N. What Is the Spatial Resolution of grace Satellite Products
               for Hydrology? Remote Sens. 2018, 10, 852.

        """
        psi_array = np.atleast_1d(psi)
        theta0 = np.pi * 0.5 - (psi_array + central_latitude)
        x0 = np.vstack(
            (np.sin(theta0) * np.cos(central_longitude), np.sin(theta0) * np.sin(central_longitude), np.cos(theta0)))

        ux = x0[0, 0]
        uy = x0[1, 0]
        uz = x0[2, 0]

        ca = np.cos(azimuth)
        sa = np.sin(azimuth)

        rotation_matrix = np.array([[ca + ux**2 * (1 - ca), ux * uy * (1 - ca) - uz * sa, ux * uz * (1 - ca) + uy * sa],
                                    [uy * ux * (1 - ca) + uz * sa, ca + uy**2 * (1 - ca), uy * uz * (1 - ca) - ux * sa],
                                    [uz * ux * (1 - ca) - uy * sa, uz * uy * (1 - ca) + ux * sa, ca + uz**2 * (1 - ca)]])
        x = rotation_matrix @ x0
        lon = -np.arctan2(x[1, :], x[0, :])
        lat = np.pi * 0.5 - np.arctan2(np.sqrt(x[0, :]**2 + x[1, :]**2), x[2, :])

        kn1 = self.evaluate(lon[0], lat[0], lon, lat).flatten()

        mtf = np.zeros(psi.size)
        for k in range(0, psi_array.size):
            kn2 = self.evaluate(lon[k], lat[k], lon[0:k + 1], lat[0:k + 1]).flatten()

            kn = kn1[0:k + 1] + kn2
            edge_threshold = min(kn[0], kn[-1])
            mtf[k] = 0 if np.min(kn) >= edge_threshold else 1 - kn[int(kn.size // 2)] / np.max(kn)

        return mtf

    def spatial_resolution(self, central_longitude=0, central_latitude=0, direction='north_south', R=6378136.3, threshold=1000):
        """
        Compute the spatial resolution of the kernel. Two Dirac pulses are shifted on the sphere until a local minimum in the connecting line between the two occurs.
        If direction 'east_west' is chosen, the Dirac pulses are shifted along a circle of constant latitude. This will lead to convergence issues for points near the poles.

        Parameters
        ----------
        min_degree : int
            minimum evaluation degree
        max_degree : int
            maximum evaluation degree
        direction : str
            direction of evaluation ('north_south', 'east_west')
        R : float
            radius of the evaluation sphere in meters
        threshold : float
            algorithm stops once the search window is smaller than threshold (given in meters)

        Returns
        -------
        resolution : float
            distance in meters between two Dirac pulses when a local minimum occurs
        """
        if direction == 'north_south':
            cs = grates.utilities.trigonometric_functions(self.__max_degree, central_longitude)

            def kernel_sum(u12K, cs, theta):
                Ynm = grates.utilities.legendre_functions(self.__max_degree, theta) * cs
                return np.sum(u12K * grates.utilities.ravel_coefficients(Ynm, self.__min_degree, self.__max_degree))

            def brute_force(min_psi, max_psi):
                if (max_psi - min_psi) * R < threshold:
                    return max_psi * 0.5 + min_psi * 0.5

                psi0 = np.linspace(min_psi, max_psi, 3)
                for k in range(1, psi0.size):

                    theta1 = np.pi * 0.5 - central_latitude - psi0[k] * 0.5
                    theta2 = np.pi * 0.5 - central_latitude + psi0[k] * 0.5

                    Ynm = grates.utilities.spherical_harmonics(self.__max_degree, (theta1, theta2), central_longitude)
                    u12K = np.sum(grates.utilities.ravel_coefficients(Ynm, self.__min_degree, self.__max_degree), axis=0) @ self.__matrix

                    res = scipy.optimize.fminbound(functools.partial(kernel_sum, u12K, cs), theta1, theta2)
                    has_local_minimum = np.abs(res - theta1) * R > threshold and np.abs(res - theta2) * R > threshold
                    if has_local_minimum:
                        return brute_force(psi0[k - 1], psi0[k])

            return brute_force(0, np.pi) * R

        elif direction == 'east_west':
            pnm = grates.utilities.legendre_functions(self.__max_degree, np.pi * 0.5 - central_latitude)

            def kernel_sum(u12K, pnm, lon):
                Ynm = pnm * grates.utilities.trigonometric_functions(self.__max_degree, lon)
                return np.sum(u12K * grates.utilities.ravel_coefficients(Ynm, self.__min_degree, self.__max_degree))

            def brute_force(min_psi, max_psi):
                if (max_psi - min_psi) * R * np.cos(central_latitude) < threshold:
                    return max_psi * 0.5 + min_psi * 0.5

                psi0 = np.linspace(min_psi, max_psi, 3)
                for k in range(1, psi0.size):

                    lon1 = central_longitude - psi0[k] * 0.5
                    lon2 = central_longitude + psi0[k] * 0.5

                    Ynm = grates.utilities.spherical_harmonics(self.__max_degree, np.pi * 0.5 - central_latitude, (lon1, lon2))
                    u12K = np.sum(grates.utilities.ravel_coefficients(Ynm, self.__min_degree, self.__max_degree), axis=0) @ self.__matrix

                    res = scipy.optimize.fminbound(functools.partial(kernel_sum, u12K, pnm), lon1, lon2)
                    has_local_minimum = np.abs(res - lon1) * R * np.cos(central_latitude) > threshold and np.abs(res - lon2) * R * np.cos(central_latitude) > threshold
                    if has_local_minimum:
                        return brute_force(psi0[k - 1], psi0[k])

            return brute_force(0, np.pi) * np.cos(central_latitude) * R

        else:
            raise ValueError('Argument <direction> must be one of "north_south" or "east_west". Got {0}.'.format(direction))
