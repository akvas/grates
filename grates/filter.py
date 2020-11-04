# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Spatial filters for post-processing of potential coefficients.
"""

from grates.gravityfield import PotentialCoefficients
import grates.kernel
import grates.utilities
import pkg_resources
import numpy as np
import abc
import scipy.signal as sig
import scipy.linalg as la


class SpatialFilter(metaclass=abc.ABCMeta):
    """
    Base interface for spatial filters applied to a PotentialCoefficients instance. Derived classes must implement a
    `filter` method which takes a PotentialCoefficients instance as argument. The gravity field passed to this method
    should remain unchanged. Additionally, derived classes must implement a `matrix` method which returns the filter
    as a dense filter matrix.
    """
    @abc.abstractmethod
    def filter(self, gravityfield):
        pass

    @abc.abstractmethod
    def matrix(self, min_degree, max_degree):
        pass


class Gaussian(SpatialFilter):
    """
    Implements a Gaussian filter.

    Parameters
    ----------
    radius : float
        filter radius in kilometers
    """
    def __init__(self, radius):

        self.radius = radius

    def filter(self, gravityfield):
        """
        Apply the Gaussian filter to a PotentialCoefficients instance.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            gravity field to be filtered, remains unchanged

        Returns
        -------
        result : PotentialCoefficients instance
            filtered copy of input
        """
        if not isinstance(gravityfield, PotentialCoefficients):
            raise TypeError("Filter operation only implemented for instances of 'PotentialCoefficients'")

        nmax = gravityfield.max_degree()

        kn = grates.kernel.Gauss(self.radius)
        wn = np.zeros(nmax + 1)
        for n in range(nmax + 1):
            wn[n] = kn.coefficient(n)

        result = gravityfield.copy()
        for n in range(2, nmax+1):
            result.anm[n, 0:n + 1] *= wn[n]
            result.anm[0:n, n] *= wn[n]

        return result

    def matrix(self, min_degree, max_degree):
        """
        Gaussian filter as filter matrix.

        Parameters
        ----------
        min_degree : int
            minimum filter degree
        max_degree : int
            maximum filter degree

        Returns
        -------
        filter_matrix : ndarray((max_degree + 1)**2 - min_degree**2, (max_degree + 1)**2 - min_degree**2)
            2d ndarray representing the filter
        """
        kn = grates.kernel.Gauss(self.radius)
        filter_array = np.zeros((max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            filter_array[n, 0:n + 1] = kn.coefficient(n)
            filter_array[0:n, n] = kn.coefficient(n)

        return np.diag(grates.utilities.ravel_coefficients(filter_array, min_degree, max_degree))


class OrderWiseFilter(SpatialFilter):
    """
    Implements a spherical harmonic filter with a sparse filter matrix.
    The filter matrix only considers correlations between spherical harmonic coefficients with the same
    order and trigonometric function (sine/cosine). A popular realization of such a filter is
    the DDK filter by Kusche et al. (2009) [1]_.

    References
    ----------

    .. [1] Kusche, J., Schmidt, R., Petrovic, S. et al. Decorrelated GRACE time-variable gravity solutions by GFZ,
           and their validation using a hydrological model. J Geod 83, 903–913 (2009).
           https://doi.org/10.1007/s00190-009-0308-3

    """
    def __init__(self, orderwise_blocks):

        self.__array = orderwise_blocks
        self.__nmax = orderwise_blocks[0].shape[0]-1

    def filter(self, gravityfield):
        """
        Apply the filter to a PotentialCoefficients instance.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            gravity field to be filtered, remains unchanged

        Returns
        -------
        result : PotentialCoefficients instance
            filterd copy of input

        Raises
        ------
        ValueError
            if maximum spherical harmonic degree is greater than 120
        """
        if not isinstance(gravityfield, PotentialCoefficients):
            raise TypeError("Filter operation only implemented for instances of 'PotentialCoefficients'")

        nmax = gravityfield.max_degree()
        if nmax > self.__nmax:
            raise ValueError('DDK filter only implemented for a maximum degree of {1:d} (nmax={0:d} supplied).'
                             .format(nmax, self.__nmax))

        result = gravityfield.copy()

        result.anm[:, 0] = (self.__array[0][0:nmax + 1, 0:nmax + 1] @ gravityfield.anm[:, 0:1]).flatten()
        for m in range(1, nmax + 1):
            result.anm[m::, m] = (self.__array[2 * m - 1][0:nmax + 1 - m, 0:nmax + 1 - m] @
                                  gravityfield.anm[m::, m:m + 1]).flatten()
            result.anm[m - 1, m::] = (self.__array[2 * m][0:nmax + 1 - m, 0:nmax + 1 - m] @
                                      gravityfield.anm[m - 1:m, m::].T).flatten()

        result.anm[0:2, 0:2] = gravityfield.anm[0:2, 0:2].copy()

        return result

    def matrix(self, min_degree, max_degree):
        """
        Return dense filter matrix.

        Parameters
        ----------
        min_degree : int
            minimum filter degree
        max_degree : int
            maximum filter degree

        Returns
        -------
        filter_matrix : ndarray((max_degree + 1)**2 - min_degree**2, (max_degree + 1)**2 - min_degree**2)
            2d ndarray representing the filter
        """
        coefficient_count = (max_degree + 1) * (max_degree + 1)

        filter_matrix = np.zeros((coefficient_count, coefficient_count))
        degrees = np.arange(max_degree + 1, dtype=int)
        index = degrees ** 2

        filter_matrix[np.ix_(index, index)] = self.__array[0][0:max_degree + 1, 0:max_degree + 1]
        for m in range(1, max_degree + 1):
            filter_matrix[np.ix_(index[m:] + 2 * m - 1, index[m:] + 2 * m - 1)] = \
                self.__array[2 * m - 1][0:max_degree + 1 - m, 0:max_degree + 1 - m]
            filter_matrix[np.ix_(index[m:] + 2 * m, index[m:] + 2 * m)] = \
                self.__array[2 * m][0:max_degree + 1 - m, 0:max_degree + 1 - m]

        return filter_matrix[min_degree * min_degree:, min_degree * min_degree:]


class DDK(OrderWiseFilter):
    """
    Implements the DDK filter by Kusche et al. (2009) [1]_.

    Parameters
    ----------
    level : int
        DDK filter level (positive, non-zero)

    References
    ----------

    .. [1] Kusche, J., Schmidt, R., Petrovic, S. et al. Decorrelated GRACE time-variable gravity solutions by GFZ,
           and their validation using a hydrological model. J Geod 83, 903–913 (2009).
           https://doi.org/10.1007/s00190-009-0308-3

    """
    def __init__(self, level):

        if level < 1:
            raise ValueError('DDK level must be at least 1 (requested DDK{0:d}).'.format(level))

        normals = np.load(pkg_resources.resource_filename('grates', 'data/ddk_normals.npz'), allow_pickle=True)['arr_0']
        nmax = normals[0].shape[0]-1
        weights = 10**(15-level) * np.arange(nmax + 1, dtype=float) ** 4
        weights[0] = 1

        array = []
        for normals_block in normals:
            m = nmax + 1 - normals_block.shape[0]
            array.append(np.linalg.solve(normals_block + np.diag(weights[m:]), normals_block))

        super(DDK, self).__init__(array)


class BlockedVDK(OrderWiseFilter):
    """
    Implements a blocked version of the VDK filter. Instead of using the full normal equation matrix, the DDK filter
    correlation structure is used.

    Parameters
    ----------
    normal_equation_matrix : ndarray
        normal equation matrix in degree wise coefficient order
    min_degree : int
        minimum degree contained in the normal equation matrix
    max_degree : int
        maximum degree contained in the normal equation matrix
    kaula_scale : float
        scale factor for the Kaula regularization used (scale factor for degree wise weights)
    kaula_power : float
        power for the Kaula regularization used (scale factor for degree wise weights)

    References
    ----------

    .. [1] Kusche, J., Schmidt, R., Petrovic, S. et al. Decorrelated GRACE time-variable gravity solutions by GFZ,
           and their validation using a hydrological model. J Geod 83, 903–913 (2009).
           https://doi.org/10.1007/s00190-009-0308-3

    """
    def __init__(self, normal_equation_matrix, min_degree, max_degree, kaula_scale, kaula_power):

        parameter_count = normal_equation_matrix.shape[0]
        weights = kaula_scale * np.arange(max_degree + 1, dtype=float) ** kaula_power
        weights[0] = 1
        coefficient_meta = np.zeros((3, parameter_count), dtype=int)

        idx = 0
        for n in range(min_degree, max_degree + 1):
            coefficient_meta[1, idx] = n
            idx += 1
            for m in range(1, n + 1):
                coefficient_meta[1, idx] = n
                coefficient_meta[2, idx] = m

                coefficient_meta[0, idx + 1] = 1
                coefficient_meta[1, idx + 1] = n
                coefficient_meta[2, idx + 1] = m
                idx += 2

        index_array = coefficient_meta[2, :] == 0
        normals = [np.zeros((max_degree + 1, max_degree + 1))]
        normals[0][min_degree:, min_degree:] = normal_equation_matrix[np.ix_(index_array, index_array)]
        for m in range(1, max_degree + 1):
            index_array_cosine = np.logical_and(coefficient_meta[2, :] == m, coefficient_meta[0, :] == 0)
            index_array_sine = np.logical_and(coefficient_meta[2, :] == m, coefficient_meta[0, :] == 1)

            if m >= min_degree:
                normals.append(normal_equation_matrix[np.ix_(index_array_cosine, index_array_cosine)])
                normals.append(normal_equation_matrix[np.ix_(index_array_sine, index_array_sine)])
            else:
                coefficient_count = max_degree + 1 - m

                normals.append(np.zeros((coefficient_count, coefficient_count)))
                normals[-1][min_degree - m:, min_degree - m:] = normal_equation_matrix[np.ix_(index_array_cosine,
                                                                                              index_array_cosine)]
                normals.append(np.zeros((coefficient_count, coefficient_count)))
                normals[-1][min_degree - m:, min_degree - m:] = normal_equation_matrix[np.ix_(index_array_sine,
                                                                                             index_array_sine)]

        array = []
        for normals_block in normals:
            m = max_degree + 1 - normals_block.shape[0]
            array.append(np.linalg.solve(normals_block + np.diag(weights[m:]), normals_block))

        super(BlockedVDK, self).__init__(array)


class VDK(SpatialFilter):
    """
    Implementation of the VDK filter.

    Parameters
    ----------
    normal_equation_matrix : ndarray
        normal equation matrix in degree wise coefficient order
        min_degree : int
        minimum degree contained in the normal equation matrix
    max_degree : int
        maximum degree contained in the normal equation matrix
    kaula_scale : float
        scale factor for the Kaula regularization used (scale factor for degree wise weights)
    kaula_power : float
        power for the Kaula regularization used (scale factor for degree wise weights)
    """
    def __init__(self, normal_equation_matrix, min_degree, max_degree, kaula_scale, kaula_power):

        degree_weights = kaula_scale * np.arange(max_degree + 1, dtype=float) ** kaula_power
        degree_weights[0] = 1

        weights = np.full(int((max_degree + 1)**2 - min_degree**2), np.nan)
        idx = 0
        for n in range(min_degree, max_degree + 1):
            weights[idx] = degree_weights[n]
            idx += 1
            for m in range(1, n + 1):
                weights[idx] = degree_weights[n]
                weights[idx + 1] = degree_weights[n]
                idx += 2

        self.__W = np.linalg.solve(normal_equation_matrix + np.diag(weights), normal_equation_matrix)

        self.__nmin = min_degree
        self.__nmax = max_degree

    def filter(self, gravityfield):
        """
        Apply the filter to a PotentialCoefficients instance.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            gravity field to be filtered, remains unchanged

        Returns
        -------
        result : PotentialCoefficients instance
            filterd copy of input

        """
        result = gravityfield.copy()

        x = grates.utilities.ravel_coefficients(result.anm, self.__nmin, self.__nmax)[:, np.newaxis]
        x_filtered = (self.__W @ x).flatten()

        result.anm[self.__nmin:self.__nmax + 1, self.__nmin:self.__nmax + 1] = \
            grates.utilities.unravel_coefficients(x_filtered, self.__nmin, self.__nmax)[self.__nmin:self.__nmax + 1,
                                                                                        self.__nmin:self.__nmax + 1]

        return result

    def matrix(self, min_degree, max_degree):
        """
        Return dense filter matrix.

        Parameters
        ----------
        min_degree : int
            minimum filter degree
        max_degree : int
            maximum filter degree

        Returns
        -------
        filter_matrix : ndarray((max_degree + 1)**2 - min_degree**2, (max_degree + 1)**2 - min_degree**2)
            2d ndarray representing the filter
        """
        if self.__nmin == min_degree and self.__nmax == max_degree:
            return self.__W.copy()
        else:
            raise NotImplemented('generic min/max degrees not yet implemented')


class FilterKernel:
    """
    Kernel representation of possibly anisotropic filter in space domain.

    Parameters
    ----------
    filter : SpatialFilter instance or ndarray(max_degree + 1, max_degree + 1)
        filter matrix
    min_degree : int
        minimum filter degree
    max_degree : int
        maximum filter degree
    """
    def __init__(self, filter, min_degree, max_degree):

        self.__matrix = filter.matrix(min_degree, max_degree) if isinstance(filter, SpatialFilter) else filter
        self.__min_degree = min_degree
        self.__max_degree = max_degree

    def evaluate(self, source_longitude, source_latitude, eval_longitude, eval_latitude, kernel='potential'):
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
        kernel : str
            name of kernel of filter in and output (for example 'ewh' means that the filter is applied to a
            water height field)

        Returns
        -------
        kernel : ndarray(m,)
            kernel values at the evaluation points
        """
        kn = grates.kernel.get_kernel(kernel)

        inverse_coefficients = kn.inverse_coefficient_array(self.__max_degree)

        spherical_harmonics_source = grates.utilities.spherical_harmonics(self.__max_degree,
                                                                          np.pi * 0.5 - source_latitude,
                                                                          source_longitude)
        v1 = grates.utilities.ravel_coefficients(spherical_harmonics_source * inverse_coefficients,
                                self.__min_degree, self.__max_degree) @ self.__matrix

        coefficients = kn.coefficient_array(self.__max_degree)
        spherical_harmonics_eval = grates.utilities.spherical_harmonics(self.__max_degree, np.pi * 0.5 - eval_latitude,
                                                                     eval_longitude) * coefficients

        return np.atleast_1d((v1 @ grates.utilities.ravel_coefficients(spherical_harmonics_eval, self.__min_degree,
                                                                       self.__max_degree).T).squeeze())

    def evaluate_grid(self, source_longitude, source_latitude, eval_longitude, eval_latitude, kernel='potential'):
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
        kernel : str
            name of kernel of filter in and output (for example 'ewh' means that the filter is applied to a
            water height field)

        Returns
        -------
        grid : ndarray(m, n)
            kernel values on the grid
        """
        kn = grates.kernel.get_kernel(kernel)

        inverse_coefficients = kn.inverse_coefficient_array(self.__max_degree)

        spherical_harmonics_source = grates.utilities.spherical_harmonics(self.__max_degree,
                                                                          np.pi * 0.5 - source_latitude,
                                                                          source_longitude)
        v1 = grates.utilities.ravel_coefficients(spherical_harmonics_source * inverse_coefficients,
                                self.__min_degree, self.__max_degree) @ self.__matrix

        coefficients = kn.coefficient_array(self.__max_degree)
        pnm = grates.utilities.legendre_functions(self.__max_degree, np.pi * 0.5 - eval_latitude) * coefficients
        cs = grates.utilities.trigonometric_functions(self.__max_degree, eval_longitude)

        grid = np.empty((eval_latitude.size, eval_longitude.size))
        for k in range(eval_latitude.size):
            grid[k, :] = (grates.utilities.ravel_coefficients(cs * pnm[k], self.__min_degree,
                                                              self.__max_degree) @ v1.T).squeeze()

        return grid

    def modulation_transfer(self, psi, central_longitude=0, central_latitude=0, azimuth=0, kernel='potential'):
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
        kernel : str
                name of kernel of filter in and output (for example 'ewh' means that the filter is applied to a
                water height field)

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
        theta0 = np.pi*0.5 - (psi_array + central_latitude)
        x0 = np.vstack(
            (np.sin(theta0) * np.cos(central_longitude), np.sin(theta0) * np.sin(central_longitude), np.cos(theta0)))

        ux = x0[0, 0]
        uy = x0[1, 0]
        uz = x0[2, 0]

        ca = np.cos(azimuth)
        sa = np.sin(azimuth)

        rotation_matrix = np.array([[ca + ux**2*(1 - ca), ux*uy*(1-ca)-uz*sa, ux*uz*(1-ca) + uy*sa],
                                    [uy*ux*(1-ca)+uz*sa, ca + uy**2*(1-ca), uy*uz*(1-ca)-ux*sa],
                                    [uz*ux*(1-ca)-uy*sa, uz*uy*(1-ca)+ux*sa, ca+uz**2*(1-ca)]])
        x = rotation_matrix@x0
        lon = -np.arctan2(x[1, :], x[0, :])
        lat = np.pi*0.5 - np.arctan2(np.sqrt(x[0, :]**2 + x[1, :]**2), x[2, :])

        kn1 = self.evaluate(lon[0], lat[0], lon, lat, kernel=kernel).flatten()

        mtf = np.zeros(psi.size)
        for k in range(0, psi_array.size):
            kn2 = self.evaluate(lon[k], lat[k], lon[0:k+1], lat[0:k+1], kernel=kernel).flatten()

            kn = kn1[0:k+1] + kn2
            edge_threshold = min(kn[0], kn[-1])
            mtf[k] = 0 if np.min(kn) >= edge_threshold else 1 - kn[int(kn.size//2)]/np.max(kn)

        return mtf

    def spatial_resolution(self, central_longitude=0, central_latitude=0, azimuth=0, max_psi=np.pi, nsteps=100,
                           kernel='potential', mtf_threshold=1e-3):
        """
        Determine the spatial resolution of the filter kernel along a great circle segment in a given direction.
        The spatial resolution is determined on the basis of the modulation transfer function. Two Dirac impulses
        are placed on the sphere and filtered. The first impulse is placed on the central point, the second is
        shifted along a great circle segment in the givel direction. The two peaks can be resolved once a local minimum
        along the great circle between the two Dirac impulses is present.

        Parameters
        ----------
        central_longitude : float
            longitude of source point in radians
        central_latitude : float
            latitude of source point in radians
        azimuth : float, ndarray(m,)
            azimuth of great circle segment in radians
        max_psi : float
            maximum spherical distance in radians
        nsteps : int
            resolution of the points along the great circle segment
        kernel : str
            name of kernel of filter in and output (for example 'ewh' means that the filter is applied to a
            water height field)

        Returns
        -------
        spatial_resolution : ndarray(m,)
            spatial resolution along the given azimuth in radians
        """
        psi = np.linspace(0, max_psi, nsteps)
        theta0 = np.pi * 0.5 - (psi + central_latitude)
        x0 = np.vstack(
            (np.sin(theta0) * np.cos(central_longitude), np.sin(theta0) * np.sin(central_longitude), np.cos(theta0)))

        ux = x0[0, 0]
        uy = x0[1, 0]
        uz = x0[2, 0]

        azimuth_array = np.atleast_1d(azimuth)
        spatial_resolution = np.zeros(azimuth_array.size)

        search_factor = int(nsteps//10)

        for i in range(azimuth_array.size):
            ca = np.cos(azimuth_array[i])
            sa = np.sin(azimuth_array[i])

            rotation_matrix = np.array(
                [[ca + ux ** 2 * (1 - ca), ux * uy * (1 - ca) - uz * sa, ux * uz * (1 - ca) + uy * sa],
                 [uy * ux * (1 - ca) + uz * sa, ca + uy ** 2 * (1 - ca), uy * uz * (1 - ca) - ux * sa],
                 [uz * ux * (1 - ca) - uy * sa, uz * uy * (1 - ca) + ux * sa, ca + uz ** 2 * (1 - ca)]])
            x = rotation_matrix @ x0
            lon = -np.arctan2(x[1, :], x[0, :])
            lat = np.pi * 0.5 - np.arctan2(np.sqrt(x[0, :] ** 2 + x[1, :] ** 2), x[2, :])

            kn1 = self.evaluate(lon[0], lat[0], lon, lat, kernel=kernel).flatten()

            lower_bound = psi.size - search_factor
            upper_bound = psi.size
            for k in range(0, psi.size, search_factor):
                kn2 = self.evaluate(lon[k], lat[k], lon[0:k + 1:search_factor], lat[0:k + 1:search_factor],
                                    kernel=kernel).flatten()

                target_function = -kn1[0:k + 1:search_factor] - kn2
                peaks, _ = sig.find_peaks(target_function)
                if len(peaks) > 0:
                    upper_bound = k + 1
                    lower_bound = max(k - search_factor, 0)
                    break

            for k in range(lower_bound, upper_bound):
                kn2 = self.evaluate(lon[k], lat[k], lon[0:k + 1], lat[0:k + 1], kernel=kernel).flatten()
                target_function = -kn1[0:k + 1] - kn2
                peaks, _ = sig.find_peaks(target_function)
                if len(peaks) > 0:
                    spatial_resolution[i] = psi[k]
                    break


            #     edge_threshold = min(kn[0], kn[-1])
            #     mtf = 0 if np.min(kn) >= edge_threshold else 1 - kn[int(kn.size // 2)] / np.max(kn)
            #     if mtf > mtf_threshold:
            #         upper_bound = k + 1
            #         lower_bound = max(k - search_factor, 0)
            #         break
            #
            # for k in range(lower_bound, upper_bound):
            #     kn2 = self.evaluate(lon[k], lat[k], lon[0:k + 1], lat[0:k + 1], kernel=kernel).flatten()
            #     kn = kn1[0:k + 1] + kn2
            #     edge_threshold = min(kn[0], kn[-1])
            #     mtf = 0 if np.min(kn) >= edge_threshold else 1 - kn[int(kn.size // 2)] / np.max(kn)
            #     if mtf > mtf_threshold:
            #         spatial_resolution[i] = psi[k]
            #         break

        return spatial_resolution
