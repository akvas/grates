# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Spatial filters for post-processing of potential coefficients.
"""

import abc
import numpy as np
from grates.gravityfield import PotentialCoefficients
import grates.kernel
import grates.utilities


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

        nmax = gravityfield.max_degree

        kn = grates.kernel.Gauss(self.radius)
        wn = np.zeros(nmax + 1)
        for n in range(nmax + 1):
            wn[n] = kn.coefficient(n)

        result = gravityfield.copy()
        for n in range(2, nmax + 1):
            result.anm[grates.gravityfield.degree_indices(n)] *= wn[n]

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
            filter_array[grates.gravityfield.degree_indices(n)] = kn.coefficient(n)

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
        self.__nmax = orderwise_blocks[0].shape[0] - 1

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

        nmax = gravityfield.max_degree
        if nmax > self.__nmax:
            raise ValueError('DDK filter only implemented for a maximum degree of {1:d} (max_degree={0:d} supplied).'
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

        normals = DDK.__blocked_normals()
        nmax = normals[0].shape[0] - 1
        weights = 10**(15 - level) * np.arange(nmax + 1, dtype=float) ** 4
        weights[0] = 1

        array = []
        for normals_block in normals:
            m = nmax + 1 - normals_block.shape[0]
            array.append(np.linalg.solve(normals_block + np.diag(weights[m:]), normals_block))

        super(DDK, self).__init__(array)

    @staticmethod
    def __blocked_normals():
        """
        Return the orderwise normal equation blocks of the DDK normal equation matrix.

        Returns
        -------
        block_matrix : list of ndarrays
            orderwise matrix blocks (alternating cosine/sine per order, order 0 only contains cosine coefficients)
        """
        return grates.data.ddk_normal_blocks()

    @staticmethod
    def normal_equation_matrix():
        """
        Return the dense DDK normal equation matrix in degreewise ordering.

        Returns
        -------
        matrix : ndarray(n, n)
            dense DDK normal equation matrix
        """
        normals = DDK.__blocked_normals()
        max_degree = normals[0].shape[0] - 1

        coefficient_count = (max_degree + 1) * (max_degree + 1)

        normal_matrix = np.zeros((coefficient_count, coefficient_count))
        degrees = np.arange(max_degree + 1, dtype=int)
        index = degrees ** 2

        normal_matrix[np.ix_(index, index)] = normals[0][0:max_degree + 1, 0:max_degree + 1]
        for m in range(1, max_degree + 1):
            normal_matrix[np.ix_(index[m:] + 2 * m - 1, index[m:] + 2 * m - 1)] = \
                normals[2 * m - 1][0:max_degree + 1 - m, 0:max_degree + 1 - m]
            normal_matrix[np.ix_(index[m:] + 2 * m, index[m:] + 2 * m)] = \
                normals[2 * m][0:max_degree + 1 - m, 0:max_degree + 1 - m]

        return normal_matrix[4:, 4:]


class BlockedNormalsVDK(OrderWiseFilter):
    """
    Implements a blocked version of the VDK filter [1]_. Instead of using the full normal equation matrix, the DDK filter [2]_
    correlation structure is used. This means that only correlations between spherical harmonic coefficients with the same
    order and trigonometric function (sine/cosine) are considered.

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

    .. [1] Horvath, A., Murböck, M., Pail, R., & Horwath, M. (2018). Decorrelation of GRACE time variable gravity field
           solutions using full covariance information. Geosciences, 8(9), 323. https://doi.org/10.3390/geosciences8090323

    .. [2] Kusche, J., Schmidt, R., Petrovic, S. et al. Decorrelated GRACE time-variable gravity solutions by GFZ,
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

        super(BlockedNormalsVDK, self).__init__(array)


class GeneralMatrix(SpatialFilter):
    """
    Spherical harmonic filter defined by an arbitrary square matrix.

    Parameters
    ----------
    normal_equation_matrix : ndarray
        normal equation matrix in degree wise coefficient order
        min_degree : int
        minimum degree contained in the normal equation matrix
    min_degree : int
        minimum degree contained in the filter matrix
    max_degree : int
        maximum degree contained in the filter matrix
    """
    def __init__(self, matrix, min_degree, max_degree):

        if matrix.ndim > 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('filter matrix must be square (got {0})'.format(str(matrix.shape)))
        if (max_degree + 1) * (max_degree + 1) - min_degree * min_degree != matrix.shape[0]:
            raise ValueError('filter matrix dimensions do not correspond to min_degree and max_degree (got {0}, {1:d}, {2:d})'.format(str(matrix.shape), min_degree, max_degree))

        self.__W = matrix
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
        max_degree = min(result.max_degree, self.__nmax)

        x = grates.utilities.ravel_coefficients(gravityfield.anm, self.__nmin, self.__nmax)
        x_filtered = self.__W @ x

        result.anm = grates.utilities.unravel_coefficients(x_filtered, self.__nmin, max_degree)
        result.anm[0:self.__nmin, 0:self.__nmin] = gravityfield.anm[0:self.__nmin, 0:self.__nmin].copy()

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
            target_sequence = grates.gravityfield.CoefficientSequenceDegreeWise(min_degree, max_degree)
            source_sequence = grates.gravityfield.CoefficientSequenceDegreeWise(self.__nmin, self.__nmax)

            W = np.zeros((target_sequence.coefficient_count, target_sequence.coefficient_count))

            idx_source, idx_target = grates.gravityfield.CoefficientSequence.reorder_indices(source_sequence, target_sequence)

            W[np.ix_(idx_target, idx_target)] = self.__W[np.ix_(idx_source, idx_source)].copy()

            return W


class VDK(GeneralMatrix):
    """
    Implementation of the VDK filter [1]_.

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

    .. [1] Horvath, A., Murböck, M., Pail, R., & Horwath, M. (2018). Decorrelation of GRACE time variable gravity field
           solutions using full covariance information. Geosciences, 8(9), 323. https://doi.org/10.3390/geosciences8090323

    """
    def __init__(self, normal_equation_matrix, min_degree, max_degree, kaula_scale, kaula_power):

        coefficient_weights = np.empty((max_degree + 1, max_degree + 1))
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            coefficient_weights[row_idx, col_idx] = kaula_scale * float(n)**kaula_power

        NP = normal_equation_matrix.copy()
        NP.flat[::NP.shape[0] + 1] = np.diag(normal_equation_matrix) + grates.utilities.ravel_coefficients(coefficient_weights, min_degree, max_degree)

        super(VDK, self).__init__(np.linalg.solve(NP, normal_equation_matrix), min_degree, max_degree)

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
        max_degree = min(result.max_degree, self.__nmax)

        x = grates.utilities.ravel_coefficients(gravityfield.anm, self.__nmin, self.__nmax)[:, np.newaxis]
        x_filtered = (self.__W @ x).flatten()

        result.anm = grates.utilities.unravel_coefficients(x_filtered, self.__nmin, max_degree)
        result.anm[0:self.__nmin, 0:self.__nmin] = gravityfield.anm[0:self.__nmin, 0:self.__nmin].copy()

        return result


class FilterKernel(grates.kernel.AnisotropicKernel):
    """
    Kernel representation of possibly anisotropic filter in space domain.

    Parameters
    ----------
    spatial_filter : SpatialFilter instance or ndarray(max_degree + 1, max_degree + 1)
        filter matrix
    min_degree : int
        minimum filter degree
    max_degree : int
        maximum filter degree
    """
    def __init__(self, spatial_filter, min_degree, max_degree, input_kernel='potential'):

        K = spatial_filter.matrix(min_degree, max_degree) if isinstance(spatial_filter, SpatialFilter) else spatial_filter

        kernel_generator = grates.kernel.get_kernel(input_kernel)
        kn = kernel_generator.coefficient_array(min_degree, max_degree)
        kn_prime = kernel_generator.inverse_coefficient_array(min_degree, max_degree)

        K2 = (K * grates.utilities.ravel_coefficients(kn, min_degree, max_degree)[np.newaxis, :]) * grates.utilities.ravel_coefficients(kn_prime, min_degree, max_degree)[:, np.newaxis]

        super(FilterKernel, self).__init__(K2, min_degree, max_degree)
