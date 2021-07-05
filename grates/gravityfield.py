# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Representations of Earth's gravity field.
"""

import numpy as np
import grates.grid
import grates.kernel
import grates.utilities
import scipy.spatial


def degree_indices(n, max_order=None):
    """
    Return array indices for all coeffcients of degree n.

    The coefficients are ordered by trigonometric function (first all cosine coefficients then all sine coefficients)
    with increasing order.

    Parameters
    ----------
    n : int
        degree

    Returns
    -------
    rows : array_like(k,)
        row indices of all coefficients of degree n
    columns : array_like(k,)
        column indices of all coefficients of degree n

    """
    count = min(n, max_order) if max_order is not None else n

    rows = np.concatenate((np.full(count + 1, n, dtype=int), np.arange(count, dtype=int)))
    columns = np.concatenate((np.arange(count + 1, dtype=int), np.full(count, n, dtype=int)))

    return rows, columns


def order_indices(max_degree, m):
    """
    Return array indices for all coeffcients of order m.

    The coefficients are ordered by trigonometric function (first all cosine coefficients then all sine coefficients)
    with increasing degree.

    Parameters
    ----------
    max_degree : int
        maximum degree of target coefficient array

    m : int
        order

    Returns
    -------
    rows : array_like(k,)
        row indices of all coefficients of order m
    columns : array_like(k,)
        column indices of all coefficients of order m

    """
    rows = np.arange(m, max_degree + 1, dtype=int)
    columns = np.full(rows.size, m)

    if m > 0:
        rows = np.concatenate((rows, np.full(max_degree + 1 - m, m - 1)))
        columns = np.concatenate((columns, np.arange(m, max_degree + 1, dtype=int)))

    return rows, columns


class PotentialCoefficients:
    """
    Class representation of a set of potential coefficients.

    Parameters
    ----------
    GM : float
        geocentric gravitational constant
    R : float
        reference radius
    max_degree : int
        pre-allocate the coefficient array up to max_degree
    """
    def __init__(self, GM=3.9860044150e+14, R=6.3781363000e+06, max_degree=None):

        self.GM = GM
        self.R = R
        degree_count = 0 if max_degree is None else max_degree + 1
        self.anm = np.zeros((degree_count, degree_count))
        self.epoch = None

    def copy(self):
        """Return a deep copy of the PotentialCoefficients instance."""
        gf = PotentialCoefficients(self.GM, self.R)
        gf.anm = self.anm.copy()
        gf.epoch = self.epoch

        return gf

    def slice(self, min_degree=None, max_degree=None, min_order=None, max_order=None, step_degree=1, step_order=1):
        """
        Slice a PotentialCoefficients instance to a specific degree and order range. Return value is a new
        PotentialCoefficients instance, the original gravity field is unchanged.

        Parameters
        ----------
        min_degree : int
            minimum degree of sliced PotentialCoefficients (Default: 0)
        max_degree : int
            maximum degree of sliced PotentialCoefficients (Default: maximum degree if calling object)
        min_order : int
            minimum order of sliced PotentialCoefficients (Default: 0)
        max_order : int
            maximum order of sliced PotentialCoefficients (Default: max_degree)
        step_degree : int
            step between min_degree and max_degree (Default: 1)
        step_order : int
            step between min_order and max_order (Default: 1)

        Returns
        -------
        gravityfield : PotentialCoefficients
            new PotentialCoefficients instance with all coefficients outside of the passed degree and order ranges
            set to zero

        """
        min_degree = 0 if min_degree is None else min_degree
        max_degree = self.max_degree if max_degree is None else max_degree
        min_order = 0 if min_order is None else min_order
        max_order = max_degree if max_order is None else max_order

        idx_degree = np.isin(self.__degree_array(), range(min_degree, max_degree + 1, step_degree))
        idx_order = np.isin(self.__order_array(), range(min_order, max_order + 1, step_order))

        gf = PotentialCoefficients(self.GM, self.R)
        gf.anm = np.zeros(self.anm.shape)
        gf.anm[np.logical_and(idx_degree, idx_order)] = self.anm[np.logical_and(idx_degree, idx_order)].copy()
        gf.epoch = self.epoch

        gf.truncate(max_degree)

        return gf

    def append(self, trigonometric_function, degree, order, value):
        """Append a coefficient to a PotentialCoefficients instance."""
        if degree > self.max_degree:
            tmp = np.zeros((degree + 1, degree + 1))
            tmp[0:self.anm.shape[0], 0:self.anm.shape[1]] = self.anm.copy()
            self.anm = tmp

        if trigonometric_function in ('c', 'cos', 'cosine'):
            self.anm[degree, order] = value
        elif trigonometric_function in ('s', 'sin', 'sine') and order > 0:
            self.anm[order - 1, degree] = value

    def truncate(self, max_degree):
        """Truncate a PotentialCoefficients instance to a new maximum spherical harmonic degree."""
        if max_degree < self.max_degree:
            self.anm = self.anm[0:max_degree + 1, 0:max_degree + 1]

    def __degree_array(self):
        """Return degrees of all coefficients as numpy array"""
        da = np.zeros(self.anm.shape, dtype=int)
        for n in range(self.max_degree + 1):
            da[n, 0:n + 1] = n
            da[0:n, n] = n

        return da

    def __order_array(self):
        """Return orders of all coefficients as numpy array"""
        da = np.zeros(self.anm.shape, dtype=int)
        for m in range(1, self.max_degree + 1):
            da[m - 1, m::] = m
            da[m::, m] = m

        return da

    @property
    def max_degree(self):
        """Return maximum spherical harmonic degree of a PotentialCoefficients instance."""
        return self.anm.shape[0] - 1

    def __add__(self, other):
        """Coefficient-wise addition of two PotentialCoefficients instances."""
        if not isinstance(other, PotentialCoefficients):
            raise TypeError("unsupported operand type(s) for +: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        factor = (other.R / self.R) ** other.__degree_array() * (other.GM / self.GM)
        if self.max_degree >= other.max_degree:
            result = self.copy()
            result.anm[0:other.anm.shape[0], 0:other.anm.shape[1]] += (other.anm * factor)
        else:
            result = PotentialCoefficients(self.GM, self.R)
            result.anm = other.anm * factor
            result.anm[0:self.anm.shape[0], 0:self.anm.shape[1]] += self.anm
            result.epoch = self.epoch

        return result

    def __sub__(self, other):
        """Coefficient-wise subtraction of two PotentialCoefficients instances."""
        if not isinstance(other, PotentialCoefficients):
            raise TypeError("unsupported operand type(s) for -: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        return self + (other * -1)

    def __mul__(self, other):
        """Multiplication of a PotentialCoefficients instance with a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for *: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        result = self.copy()
        result.anm *= other

        return result

    def __truediv__(self, other):
        """Division of a PotentialCoefficients instance by a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for /: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        return self * (1.0 / other)

    def degree_amplitudes(self, max_order=None, kernel='potential'):
        """
        Compute degree amplitudes from potential coefficients.

        Parameters
        ----------
        kernel : string
            name of kernel for the degree amplitude computation
        max_order : int
            include only coefficients up to max_order (default: include all coefficients)

        Returns
        -------
        degrees : array_like shape (self.max_degree()+1,)
            integer sequence of degrees
        amplitudes : array_like shape (self.max_degree()+1,)
            computed degree amplitudes
        """
        degrees = np.arange(self.max_degree + 1)
        amplitudes = np.zeros(degrees.size)

        kernel = grates.kernel.get_kernel(kernel)

        for n in degrees:
            amplitudes[n] = np.sum(self.anm[grates.gravityfield.degree_indices(n, max_order=max_order)] ** 2) * \
                kernel.inverse_coefficient(n) ** 2

        return degrees, np.sqrt(amplitudes) * self.GM / self.R

    def coefficient_triangle(self, min_degree=2, max_degree=None):
        """
        Arrange spherical harmonic coefficients as triangle for visualization.

        Parameters
        ----------
        min_degree : int
            degrees below min_degree are masked out
        max_degree : int
            triangle is truncated at max_degree

        Returns
        -------
        triangle : masked_array shape (max_degree+1, 2*max_degree-1)

        """
        max_degree = self.max_degree if max_degree is None else max_degree

        triangle = np.hstack((np.rot90(self.anm, -1), self.anm))
        mask = np.hstack((np.rot90(np.tril(np.ones(self.anm.shape, dtype=bool)), -1), np.triu(np.ones(self.anm.shape, dtype=bool), 1)))
        mask[0:min_degree] = True

        return np.ma.masked_array(triangle, mask=mask)[0:max_degree + 1, :]

    def coefficient_amplitudes(self, kernel='potential'):
        """
        Compute the amplitude of spherical harmonic coefficients

        Parameters
        ----------
        kernel : str
            potential convert coefficients name to kernel before amplitude computation (default: potential)

        Returns
        -------
        amplitude : masked_array shape (max_degree+1, max_degree+1)
            amplitude of degree n and order m is stored in amplitude[n, m], unit depends on kernel

        """
        kernel = grates.kernel.get_kernel(kernel)
        anm_temp = np.zeros(self.anm.shape)
        for n in range(self.max_degree + 1):
            idx_row, idx_col = grates.gravityfield.degree_indices(n)
            anm_temp[idx_row, idx_col] = self.anm[idx_row, idx_col] * self.GM / self.R * kernel.inverse_coefficient(n)

        amp = np.zeros(self.anm.shape)
        amp[:, 0] = np.abs(anm_temp[:, 0])
        for m in range(1, self.max_degree + 1):
            amp[m:, m] = np.sqrt(anm_temp[m:, m]**2 + anm_temp[m - 1, m:]**2)

        mask = np.triu(np.ones(amp.shape, dtype=bool), 1)

        return np.ma.masked_array(amp, mask=mask)

    def coefficient_phases(self):
        """
        Compute the phase of spherical harmonic coefficients

        Returns
        -------
        phase : masked_array shape (max_degree+1, max_degree+1)
            phase (in radians) of degree n and order m is stored in phase[n, m]

        """
        phase = np.zeros(self.anm.shape)
        for m in range(1, self.max_degree + 1):
            phase[m:, m] = np.arctan2(self.anm[m - 1, m:], self.anm[m:, m])

        mask = np.triu(np.ones(self.anm.shape, dtype=bool), 1)

        return np.ma.masked_array(phase, mask=mask)

    def to_grid(self, grid=grates.grid.GeographicGrid(), kernel='ewh'):
        """
        Compute gridded values from a set of potential coefficients.

        Parameters
        ----------
        grid : instance of Grid subclass
            point distribution (Default: 0.5x0.5 degree geographic grid)
        kernel : string
            gravity field functional to be gridded (Default: equivalent water height). See Kernel for details.

        Returns
        -------
        output_grid : instance of type(grid)
            deep copy of the input grid with the gridded values
        """
        output_grid = grid.copy()
        output_grid.values = np.zeros(output_grid.point_count)

        grid_kernel = grates.kernel.get_kernel(kernel)

        try:
            colatitude = grates.utilities.colatitude(grid.parallels, grid.semimajor_axis, grid.flattening)
            radius = grates.utilities.geocentric_radius(grid.parallels, grid.semimajor_axis, grid.flattening)

            kn = grid_kernel.inverse_coefficients(0, self.max_degree, radius, colatitude) * np.power((self.R / radius)[:, np.newaxis], np.arange(self.max_degree + 1, dtype=int) + 1) * self.GM / self.R

            Pnm = grates.utilities.legendre_functions(self.max_degree, colatitude)
            Pnm[:, :, 0] *= kn
            for m in range(1, self.max_degree + 1):
                Pnm[:, m:, m] *= kn[:, m:]
                Pnm[:, m - 1, m:] *= kn[:, m:]
            Pnm *= self.anm[np.newaxis, :, :]

            cs = grates.utilities.trigonometric_functions(self.max_degree, grid.meridians)

            for k in range(self.max_degree + 1):
                output_grid.value_array += Pnm[:, k, :] @ cs[:, k, :].T

        except AttributeError:
            block_size = min(512, output_grid.point_count)
            block_index = [0]
            while block_index[-1] < output_grid.point_count:
                block_index.append(min(block_index[-1] + block_size, output_grid.point_count))

            for i1, i2 in zip(block_index[0:-1], block_index[1:]):
                colatitude = grates.utilities.colatitude(output_grid.latitude[i1:i2], output_grid.semimajor_axis, output_grid.flattening)
                radius = grates.utilities.geocentric_radius(output_grid.latitude[i1:i2], output_grid.semimajor_axis, output_grid.flattening)

                kn = grid_kernel.inverse_coefficients(0, self.max_degree, radius, colatitude) * np.power((self.R / radius)[:, np.newaxis], np.arange(self.max_degree + 1, dtype=int) + 1) * self.GM / self.R
                Ynm = grates.utilities.spherical_harmonics(self.max_degree, colatitude, output_grid.longitude[i1:i2])
                Ynm[:, :, 0] *= kn
                for m in range(1, self.max_degree + 1):
                    Ynm[:, m:, m] *= kn[:, m:]
                    Ynm[:, m - 1, m:] *= kn[:, m:]

                for k in range(self.max_degree + 1):
                    output_grid.values[i1:i2] += Ynm[:, k, :] @ self.anm[k, :]

        return output_grid

    @property
    def values(self):
        """
        Return a vector representation of the potential coefficients.

        Returns
        -------
        coeffs : ndarray((max_degree + 1)**2)
            ravelled coefficient array
        """
        return grates.utilities.ravel_coefficients(self.anm)

    @values.setter
    def values(self, val):
        """
        Assign potential coefficients values from a vector representation.

        Parameters
        ----------
        val : ndarray((max_degree + 1)**2) or None
            ravelled coefficient array
        """
        if val is None:
            self.value_array = None
        elif isinstance(val, np.ndarray):
            if val.ndim > 1:
                raise ValueError("unable to assign values of dimension {0:d} to gravity field".format(val.ndim))
            self.anm = grates.utilities.unravel_coefficients(val)
        else:
            raise ValueError("grid values must be either None or " + str(np.ndarray))

    def gravitational_acceleration(self, xyz):
        """
        Compute the gravitational acceleration of the gravity field at cartesion coordinate triples.

        Parameters
        ----------
        xyz : ndarray(m, 3)
            evaluation positions as cartesion coordinates

        Returns
        -------
        g : ndarray(m, 3)
            gravitational acceleration in m / s^2
        """
        r, colat, lon = grates.grid.cartesian2spherical(xyz)
        n = np.arange(self.max_degree + 1, dtype=float)

        g = np.empty((xyz.shape[0], 3))

        Pnm_co = grates.utilities.legendre_functions_per_order(self.max_degree + 1, 0, colat)
        Pnm_p1 = grates.utilities.legendre_functions_per_order(self.max_degree + 1, 1, colat)

        fnm_zero = np.sqrt((n + 1) * (n + 1)) * np.sqrt((2 * n + 1) / (2 * n + 3))
        fnm_plus = np.sqrt((n + 1) * (n + 2)) * np.sqrt((2 * n + 1) / (2 * n + 3)) * np.sqrt(2)

        Cnm_zero = Pnm_co[:, 1:] * fnm_zero
        Cnm_plus = (Pnm_p1 * np.cos(lon)[:, np.newaxis]) * fnm_plus
        Snm_plus = (Pnm_p1 * np.sin(lon)[:, np.newaxis]) * fnm_plus

        g[:, 0] = -(Cnm_plus * np.power(self.R / r[:, np.newaxis], n + 2)) @ self.anm[:, 0]
        g[:, 1] = -(Snm_plus * np.power(self.R / r[:, np.newaxis], n + 2)) @ self.anm[:, 0]
        g[:, 2] = -2 * (Cnm_zero * np.power(self.R / r[:, np.newaxis], n + 2)) @ self.anm[:, 0]
        for m in range(1, self.max_degree + 1):
            Pnm_m1 = Pnm_co
            Pnm_co = Pnm_p1
            Pnm_p1 = grates.utilities.legendre_functions_per_order(self.max_degree + 1, m + 1, colat)

            continuation = np.power(self.R / r[:, np.newaxis], n[m:] + 2)

            fnm_minus = np.sqrt((n[m:] - m + 1) * (n[m:] - m + 2)) * np.sqrt((2 * n[m:] + 1) / (2 * n[m:] + 3))
            if m == 1:
                fnm_minus *= np.sqrt(2)
            fnm_zero = np.sqrt((n[m:] - m + 1) * (n[m:] + m + 1)) * np.sqrt((2 * n[m:] + 1) / (2 * n[m:] + 3))
            fnm_plus = np.sqrt((n[m:] + m + 1) * (n[m:] + m + 2)) * np.sqrt((2 * n[m:] + 1) / (2 * n[m:] + 3))

            Cnm_minus = continuation * (Pnm_m1[:, 2:] * np.cos((m - 1) * lon)[:, np.newaxis]) * fnm_minus
            Snm_minus = continuation * (Pnm_m1[:, 2:] * np.sin((m - 1) * lon)[:, np.newaxis]) * fnm_minus

            Cnm_zero = continuation * (Pnm_co[:, 1:] * np.cos(m * lon)[:, np.newaxis]) * fnm_zero
            Snm_zero = continuation * (Pnm_co[:, 1:] * np.sin(m * lon)[:, np.newaxis]) * fnm_zero

            Cnm_plus = continuation * (Pnm_p1 * np.cos((m + 1) * lon)[:, np.newaxis]) * fnm_plus
            Snm_plus = continuation * (Pnm_p1 * np.sin((m + 1) * lon)[:, np.newaxis]) * fnm_plus

            g[:, 0] += (Cnm_minus - Cnm_plus) @ self.anm[m:, m] + (Snm_minus - Snm_plus) @ self.anm[m - 1, m:]
            g[:, 1] += (-Snm_minus - Snm_plus) @ self.anm[m:, m] + (Cnm_minus + Cnm_plus) @ self.anm[m - 1, m:]
            g[:, 2] += -2 * Cnm_zero @ self.anm[m:, m] - 2 * Snm_zero @ self.anm[m - 1, m:]

        return g * self.GM / (2 * self.R**2)


class SurfaceMasCons:

    def __init__(self, point_distribution, kernel):
        self.point_distribution = point_distribution
        if self.point_distribution.values is None:
            self.point_distribution.values = np.zeros(self.point_distribution.point_count)
        self.kernel = kernel
        self.epoch = None

    def copy(self):
        """Copy the SurfaceMasCons instance"""
        other = SurfaceMasCons(self.point_distribution.copy(), self.kernel)
        other.epoch = self.epoch
        return other

    def is_compatible(self, other):
        return self.point_distribution.is_compatible(other.point_distribution)

    @property
    def values(self):
        return self.point_distribution.values

    @values.setter
    def values(self, val):
        self.point_distribution.values = val

    def __add__(self, other):
        """Point-wise addition of two SurfaceMasCons instances."""
        if not isinstance(other, SurfaceMasCons):
            raise TypeError("unsupported operand type(s) for +: '" + str(type(self)) + "' and '" + str(type(other)) + "'")
        if not self.is_compatible(other):
            raise ValueError("point distributions of '" + str(type(self)) + "' instances are not compatible")

        result = self.copy()
        result.values += other.values

        return result

    def __sub__(self, other):
        """Point-wise subtraction of two SurfaceMasCons instances."""
        if not isinstance(other, SurfaceMasCons):
            raise TypeError("unsupported operand type(s) for -: '" + str(type(self)) + "' and '" + str(type(other)) + "'")
        if not self.is_compatible(other):
            raise ValueError("point distributions of '" + str(type(self)) + "' instances are not compatible")

        result = self.copy()
        result.values -= other.values

        return result

    def __mul__(self, other):
        """Multiplication of a SurfaceMasCons instance with a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for *: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        result = self.copy()
        result.values *= other

        return result

    def __truediv__(self, other):
        """Division of a SurfaceMasCons instance by a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for /: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        return self * (1.0 / other)

    def to_potential_coefficients(self, min_degree, max_degree, GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Perform spherical harmonic analysis of the mascon values.

        Parameters
        ----------
        min_degree : int
            minimum degree of the analysis
        max_degree : int
            maximum degree of the analysis
        GM : float
            geocentric gravitational constant
        R : reference radius

        Returns
        -------
        potential_coefficients : PotentialCoefficients
            result of the spherical harmonic analysis as potential coefficients
        """
        return self.point_distribution.to_potential_coefficients(min_degree, max_degree, self.kernel, GM, round)


class AnisotropicBasisFunctions:
    """
    Gravity field represented by anisotropic kernel basis functions.
    """
    def __init__(self, point_distribution, K, min_degree, max_degree, GM=3.9860044150e+14, R=6.3781363000e+06):

        self.__K = K.copy()
        self.point_distribution = point_distribution
        self.__min_degree = min_degree
        self.__max_degree = max_degree
        self.GM = GM
        self.R = R
        self.epoch = None
        self.values = np.zeros((self.point_distribution.size))

    @property
    def values(self):
        return self.point_distribution.values

    @values.setter
    def values(self, val):
        self.point_distribution.values = val

    def is_compatible(self, other):
        return self.point_distribution.is_compatible(other.point_distribution)

    def to_grid(self, grid=grates.grid.GeographicGrid(), kernel='ewh'):
        """
        Compute gridded values from anisotropic kernel functions.

        Parameters
        ----------
        grid : instance of Grid subclass
            point distribution (Default: 0.5x0.5 degree geographic grid)
        kernel : string
            gravity field functional to be gridded (Default: equivalent water height). See Kernel for details.

        Returns
        -------
        output_grid : instance of type(grid)
            deep copy of the input grid with the gridded values
        """
        kernel_function = grates.kernel.get_kernel(kernel)

        radius = grates.utilities.geocentric_radius(grid.parallels, grid.semimajor_axis, grid.flattening)
        colatitude = grates.utilities.colatitude(grid.parallels, grid.semimajor_axis, grid.flattening)
        kn = kernel_function.inverse_coefficients(0, self.__max_degree, radius, colatitude)

        continuation = np.power(self.R / radius[:, np.newaxis], np.arange(0, self.__max_degree + 1, dtype=float) + 1) * kn

        Pnm = grates.utilities.legendre_functions(self.__max_degree, colatitude)
        for n in range(self.__min_degree, self.__max_degree + 1):
            row, column = degree_indices(n)
            Pnm[:, row, column] *= continuation[:, n:n + 1]

        block_index = np.concatenate((np.arange(0, self.point_distribution.point_count, 512), np.atleast_1d(self.point_distribution.point_count)))

        output_values = np.zeros((grid.parallels.size, grid.meridians.size))
        for idx_start, idx_end in zip(block_index[0:-1], block_index[1:]):
            c = grates.utilities.colatitude(self.point_distribution.latitude[idx_start:idx_end], self.point_distribution.semimajor_axis, self.point_distribution.flattening)
            Ynm = grates.utilities.ravel_coefficients(grates.utilities.spherical_harmonics(self.__max_degree, c, self.point_distribution.longitude[idx_start:idx_end]), self.__min_degree, self.__max_degree).T
            K_tmp = self.__K @ (Ynm @ self.values[idx_start:idx_end])
            for k in range(grid.meridians.size):
                cs = grates.utilities.trigonometric_functions(self.__max_degree, grid.meridians[k])
                output_values[:, k] += grates.utilities.ravel_coefficients(Pnm * cs, self.__min_degree, self.__max_degree) @ K_tmp * self.GM / self.R

        output_grid = grid.copy()
        output_grid.value_array = output_values
        return output_grid


class RadialBasisFunctions:
    """
    Gravity field represented by radial basis functions.

    Parameters
    ----------
    point_distribution : grates.grid.Grid
        nodal points of splines as grates.grid.Grid instance
    K : 2d-ndarray
        kernel shape factors as coefficient array
    min_degree : int
        minimum degree of modelled frequency band
    max_degree : int
        maximum degree of modelled frequency band
    GM : float
        geocentric gravitational constant
    R : float
        reference radius
    """
    def __init__(self, point_distribution, K, min_degree, max_degree, GM=3.9860044150e+14, R=6.3781363000e+06):

        self.__K = K.copy()
        self.point_distribution = point_distribution.copy()
        self.__min_degree = min_degree
        self.__max_degree = max_degree
        self.GM = GM
        self.R = R
        self.epoch = None
        self.values = np.zeros((self.point_distribution.size))

    @property
    def values(self):
        return self.point_distribution.values

    @values.setter
    def values(self, val):
        self.point_distribution.values = val

    def is_compatible(self, other):
        return self.point_distribution.is_compatible(other.point_distribution)

    def to_potential_coefficients(self, blocking_factor=256):
        """
        Convert the radial basis functions to spherical harmonics.

        Parameters
        ----------
        blocking_factor : int
            block size into which to split the nodal points (this is only to keep memory consumption low)

        Returns
        -------
        coeffs : grates.gravityfield.PotentialCoefficients
            converted gravity field as PotentialCoefficients instance
        """
        coefficients = PotentialCoefficients(self.GM, self.R)
        coefficients.anm = np.zeros((self.__max_degree + 1, self.__max_degree + 1))
        coefficients.epoch = self.epoch

        start_index = 0
        while start_index < self.point_distribution.size:
            colatitude = grates.utilities.colatitude(self.point_distribution.latitude[start_index:min(start_index + blocking_factor, self.values.size)], self.point_distribution.semimajor_axis, self.point_distribution.flattening)
            radius = grates.utilities.geocentric_radius(self.point_distribution.latitude[start_index:min(start_index + blocking_factor, self.values.size)], self.point_distribution.semimajor_axis, self.point_distribution.flattening)

            Ynm = grates.utilities.spherical_harmonics(self.__max_degree, colatitude, self.point_distribution.longitude[start_index:min(start_index + blocking_factor, self.values.size)])
            kn = np.power((self.R / radius)[:, np.newaxis], np.arange(self.__max_degree + 1, dtype=int) + 1)
            Ynm[:, :, 0] *= kn
            for m in range(1, self.__max_degree + 1):
                Ynm[:, m:, m] *= kn[:, m:]
                Ynm[:, m - 1, m:] *= kn[:, m:]
            Ynm *= self.__K[np.newaxis, :, :]

            coefficients.anm += np.sum(Ynm * self.values[start_index:min(start_index + blocking_factor, self.values.size), np.newaxis, np.newaxis], axis=0)
            start_index += blocking_factor

        return coefficients

    def to_grid(self, grid=grates.grid.GeographicGrid(), kernel='ewh'):
        """
        Compute gridded values from radial basis functions.

        Parameters
        ----------
        grid : instance of Grid subclass
            point distribution (Default: 0.5x0.5 degree geographic grid)
        kernel : string
            gravity field functional to be gridded (Default: equivalent water height). See Kernel for details.

        Returns
        -------
        output_grid : instance of type(grid)
            deep copy of the input grid with the gridded values
        """
        return self.to_potential_coefficients().to_grid(grid, kernel)


class TimeVariableGravityField:
    """
    Compose a time variable gravity field from multiple constituents, for example trend, annual cycle and an irregular time series.
    All constituents should be of the same type, or at least summable, and implement an evaluate_at method.

    Parameters
    ----------
    constituents : list of gravityfield_like
        multiple time variable gravity fields
    """
    def __init__(self, constituents):

        self.constituents = constituents

    def evaluate_at(self, epoch):
        """
        Evaluate the time variable gravity field at a specfific epoch.

        Parameters
        ----------
        epoch : dt.datetime
            epoch where the gravity field is evaluated

        Returns
        -------
        gravity_field : gravityfield_like
            sum of all constituent evaluated at the epoch
        """
        return np.sum([c.evaluate_at(epoch) for c in self.constituents])


class TimeSeries:
    """
    Class representation of a gravity field time series.

    Parameters
    ----------
    data : list, tuple
        list of gravity fields of the same data type
    """
    def __init__(self, data):

        self.__data = data
        self.__dtype = type(self.__data[0])
        for d in self.__data:
            if not isinstance(d, self.__dtype):
                raise ValueError("Found inconsistent data types (" + self.__dtype + " and " + type(d) + ")")
            if d.epoch is None:
                raise ValueError("At least one data point has no valid time stamp")

        self.sort()

    def __len__(self):
        """Return length of the time series."""
        return len(self.__data)

    def __getitem__(self, index):
        """
        Return the gravity field at index.

        Parameters
        ----------
        index : int
            index of gravity field to be returned
        """
        return self.__data[index]

    def __setitem__(self, index, value):
        """
        Assign a new value to the gravity field at index. The list of gravity fields is sorted afterwards.

        Parameters
        ----------
        index : int
            index of gravity field to be set
        value : gravtiy field type
            new value at index. must be the same data type as the rest of the time series

        """
        if not isinstance(value, self.__dtype):
            raise ValueError("Inconsistent data types (" + self.__dtype + " and " + type(value) + ")")

        self.__data[index] = value
        self.sort()

    def copy(self):
        """Return a copy of the time series."""
        new_data = [d.copy() for d in self.__data]

        return TimeSeries(new_data)

    def __add__(self, other):
        """Element wise addition of two TimeSeries instances."""
        if len(self) != len(other):
            raise ValueError("Length of time series differs")

        new_data = []
        for k in range(len(self)):
            if self.__data[k].epoch != other[k].epoch:
                raise ValueError("Time stamps of elements differ")
            new_data.append(self.__data[k] + other[k])

        return TimeSeries(new_data)

    def __mul__(self, other):
        """Multiplication of a PotentialCoefficients instance with a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for *: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        return TimeSeries([d.copy() * other for d in self.__data])

    def __truediv__(self, other):
        """Multiplication of a PotentialCoefficients instance with a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for *: '" + str(type(self)) + "' and '" + str(type(other)) + "'")

        return self * (1.0 / other)

    def __sub__(self, other):
        """Element wise subtraction of two TimeSeries instances."""
        return self + (other * -1)

    def sort(self):
        """Sort data according to epochs of elements."""
        self.__data.sort(key=lambda d: d.epoch)

    def items(self):
        """Return generator for iterating over the time series."""
        for d in self.__data:
            yield d.epoch, d

    def interpolate_to(self, epoch):
        """
        Piecewise linear interpolation of the time series elements to an arbitrary epoch.

        Parameters
        ----------
        epoch : datetime.datetime
            epoch to be interpolated to

        Returns
        -------
        interp: gravity field type
            interpolated value

        Raises
        ------
        ValueError :
            if less than two data points are available or extrapolation is requested
        """
        t = np.array([d.epoch for d in self.__data])
        if t. size < 2:
            raise ValueError("at least two data points are required for interpolation (has {0})")
        if epoch < t[0] or epoch > t[-1]:
            raise ValueError("extrapolation is not supported (trying to extrapolate to " + str(epoch) + " from the interval " + str(t[0]) + ", " + str(t[-1]) + ")")

        idx = np.searchsorted(t, epoch)
        weight = (epoch - t[idx - 1]).total_seconds() / (t[idx] - t[idx - 1]).total_seconds()

        output = self.__data[idx - 1] * (1 - weight) + self.__data[idx] * weight
        output.epoch = epoch

        return output

    def evaluate_at(self, epoch):
        """
        Evaluate the time series at a specific epoch. This is a wrapper for interpolate_to.

        Parameters
        ----------
        epoch : datetime.datetime
            epoch to be interpolated to

        Returns
        -------
        interp: gravity field type
            interpolated value
        """
        return self.interpolate_to(epoch)

    def to_array(self):
        """
        Returns the time series as array.

        Returns
        -------
        time_series : ndarray(n, p)
            time series of n gravity field epochs with p parameters.
        """
        shape = len(self.__data), self.__data[0].values.size

        data_matrix = np.empty(shape)

        for k in range(data_matrix.shape[0]):
            data_matrix[k, :] = self.__data[k].values[0:shape[1]]

        return data_matrix

    def epochs(self):
        """
        Returns the time series time stamps as a list of datetime objects.

        Returns
        -------
        epochs : list of datetime objects
            time stamps as list of datetime objects
        """
        return [d.epoch for d in self.__data]

    def detrend(self, basis_functions):
        """
        Estimate and subtract a parametric model from the time series. This method modifies the
        TimeSeries in-place.

        Parameters
        ----------
        basis_functions : TemporalBasisFunction or list of TemporalBasisFunction instances
            parametric model as temporal basis functions
        """
        t = self.epochs()

        design_matrix = np.hstack([bf.design_matrix(t) for bf in basis_functions])
        observations = self.to_array()
        estimated_trend = np.linalg.pinv(design_matrix) @ observations
        observations -= design_matrix @ estimated_trend
        for k, d in enumerate(self.__data):
            d.values = observations[k, :]

        return estimated_trend

    def bin(self, bin_center_epochs, func=np.mean, no_data=np.nan):
        """
        Aggreate time series data in bins. Each epoch is sorted into a bin based on time difference to the bin center time stamps. `func` is then applied to each resulting
        set of data. If a set is empty `no_data` is assigned to the bin center.

        Parameters
        ----------
        bin_center_epochs: list of bin_center_epochs
            time stamps of the bin centers
        func: callable
            aggregate function (default: np.mean)
        no_data: float
            no data value

        Returns
        -------
        time_series: TimeSeries
            binned time series
        """
        t_tree = np.array([grates.time.mjd(e) for e in bin_center_epochs])[:, np.newaxis]
        t_query = np.array([grates.time.mjd(e) for e in self.epochs()])[:, np.newaxis]

        tree = scipy.spatial.KDTree(t_tree)
        _, indices = tree.query(t_query)

        data = []
        for k in range(t_tree.size):
            values = [self.__data[i] for i in np.where(np.array(indices) == k)[0]]
            data.append(func(values))
            data[-1].epoch = grates.time.datetime(t_tree[k, 0])

        return TimeSeries(data)

    def append(self, other):

        for _, d in other.items():
            self.__data.append(d)

        self.sort()

class Trend:
    """
    Linear gravity field trend.

    Parameters
    ----------
    gravity_field : gravityfield_like
        trend coefficients as gravity field
    reference_epoch : dt.datetime
        reference epoch of the trend coefficients
    time_scale : float
        time unit of the trend coefficients in days (365.25 corresponds to potential/year, 1.0 corresponds to potential/day and so on)
    """
    def __init__(self, gravity_field, reference_epoch, time_scale=365.25):

        self.__data = gravity_field.copy()
        self.__reference_epoch = reference_epoch
        self.__time_scale = time_scale

    def evaluate_at(self, epoch):
        """
        Evaluate the trend at a specific epoch.

        .. math:: V(t) = V \cdot (t - t_0)

        Parameters
        ----------
        epoch : dt.datetime
            epoch where the trend is evaluated

        Returns
        -------
        gravity_field : gravityfield_like
            trend evaluated at epoch t
        """
        dt = (epoch - self.__reference_epoch).total_seconds() / (86400 * self.__time_scale)

        output = self.__data * dt
        output.epoch = epoch

        return output


class Oscillation:
    """
    Sinosoidal oscilattion of gravity field values.

    Parameters
    ----------
    gravity_field_cosine : gravityfield_like
        cosine coefficients as gravity field
    gravity_field_sine : gravityfield_like
        sine coefficients as gravity field
    period : float
        oscillation period in days
    reference_epoch : dt.datetime
        reference epoch of the trend coefficients
    """
    def __init__(self, gravity_field_cosine, gravity_field_sine, period, reference_epoch):

        self.__data_cosine = gravity_field_cosine.copy()
        self.__data_sine = gravity_field_sine.copy()
        self.__reference_epoch = reference_epoch
        self.__period = period

    def evaluate_at(self, epoch):
        """
        Evaluate the oscillation at a specific epoch.

        .. math:: V(t) = V_c \cdot \cos 2\pi \frac{(t - t_0)}{T} + V_s \sin 2\pi \frac{(t - t_0)}{T}

        Parameters
        ----------
        epoch : dt.datetime
            epoch where the trend is evaluated

        Returns
        -------
        gravity_field : gravityfield_like
            oscillation evaluated at epoch t
        """
        dt = (epoch - self.__reference_epoch).total_seconds() / (86400 * self.__period)

        output = self.__data_cosine * np.cos(2 * np.pi * dt) + self.__data_sine * np.sin(2 * np.pi * dt)
        output.epoch = epoch

        return output


def gridded_rms(temporal_gravityfield, epochs, kernel='ewh', base_grid=grates.grid.GeographicGrid()):
    """
    Propagate a time variable gravity field to space domain an compute the RMS over all epochs.

    Parameters
    ----------
    temporal_gravityfield : time variable gravity field
        the time variable gravity field to be evaluated
    epoch : list of dt.datetime
        epochs at which the gravity field is evaluated
    kernel : str
        kernel of the grid values (default: equivalent water height)
    base_grid : grates.grid.Grid
        grid to which the gravity field is propagated

    Returns
    -------
    rms_grid : grates.grid.Grid
        gridded RMS values
    """
    rms_values = np.zeros(base_grid.point_count)

    for t in epochs:
        gf = temporal_gravityfield.evaluate_at(t)
        rms_values += gf.to_grid(base_grid, kernel=kernel).values**2

    rms_grid = base_grid.copy()
    rms_grid.values = np.sqrt(rms_values / len(epochs))

    return rms_grid


class CoefficientSequence:

    class Coefficient:

        __slots__ = ['degree', 'order', 'basis_function']

        degree: np.uint16
        order: np.uint16
        basis_function: np.int8

        def __init__(self, basis_function, n, m):

            self.degree = n
            self.order = m
            self.basis_function = basis_function

        def __str__(self):
            return 'Coefficient({0}, {1:d}, {2:d})'.format('c' if self.basis_function == 0 else 's', self.degree, self.order)

        def __repr__(self):
            return str(self)

        def __eq__(self, other):
            return self.basis_function == other.basis_function and self.degree == other.degree and self.order == other.order

    class ComparableCoefficientSequence:

        __slots__ = ['degree', 'order', 'basis_function']

        degree: np.uint16
        order: np.uint16
        basis_function: np.int8

        def __init__(self, coefficient):

            self.basis_function = coefficient.basis_function
            self.degree = coefficient.degree
            self.order = coefficient.order

        def __eq__(self, other):
            return self.basis_function == other.basis_function and self.degree == other.degree and self.order == other.order

    def __init__(self, coefficients):

        self.coefficients = tuple(coefficients)

    @property
    def coefficient_count(self):
        """Return the number of coefficients in the coefficient sequence."""
        return len(self.coefficients)

    def vector_indices(self, degree=None, order=None, cs=None):
        """
        Return the indices of a coefficient range represented by degree, order, and basis_function.

        Parameters
        ----------
        degree : int or None
            degree of coefficient (if None all degrees are returned).
        order : int or None
            order of coefficient (if None all orders are returned).
        basis_function : str or None
            basis_function of coefficient ('c' or 's', if None coefficients of both basis functions are returned).

        Returns
        -------
        indices : ndarray(m,)
            indices of the given coefficient range as integer ndarray
        """
        mask = np.ones(self.coefficient_count, dtype=bool)

        if degree is not None:
            mask = np.logical_and(mask, [c.degree == degree for c in self.coefficients])

        if order is not None:
            mask = np.logical_and(mask, [c.order == order for c in self.coefficients])

        if cs is not None:
            if cs in ('c', 'cos', 'cosine'):
                basis_function = 0
            elif cs in ('s', 'sin', 'sine'):
                basis_function = 1
            else:
                raise ValueError('basis function not recognized')

            mask = np.logical_and(mask, [c.basis_function == basis_function for c in self.coefficients])

        return np.where(mask)[0]

    @staticmethod
    def reorder_indices(source_sequence, target_sequence):
        """
        Generate an index vector to reorder coefficient given in a source sequence into target sequence.

        Parameters
        ----------
        source_sequence : CoefficientSequence
            source coefficient sequence
        target_sequence : CoefficientSequence
            target coeffcient sequence

        Returns
        -------
        source_indicies : ndarray(m,)
            indices of common parameters in source sequence
        target_indices : ndarray(m,)
            indices of common parameters in target sequence
        """
        c1 = [target_sequence.Comparable(c) for c in source_sequence.coefficients]
        c2 = [target_sequence.Comparable(c) for c in target_sequence.coefficients]

        _, ix1, ix2 = np.intersect1d(c1, c2, assume_unique=True, return_indices=True)

        return ix1, ix2


class CoefficientSequenceDegreeWise(CoefficientSequence):
    """
    Degreewise coefficient sequence. Coefficients are ordered by ascending degree and order with
    alternating cosine and sine coefficients.

    C00, C10, C11, S11, C20, C21, S21, C22, S22, ...

    Parameters
    ----------
    min_degree : int
        minimum degree in sequence
    max_degree : int
        maximum degree in sequence
    """

    class Comparable(CoefficientSequence.ComparableCoefficientSequence):

        def __init__(self, coefficient):
            super(CoefficientSequenceDegreeWise.Comparable, self).__init__(coefficient)

        def __lt__(self, other):

            if self.degree < other.degree:
                return True
            if self.degree == other.degree and self.order < other.order:
                return True
            if self.degree == other.degree and self.order == other.order and self.basis_function < other.basis_function:
                return True

            return False

    def __init__(self, min_degree, max_degree):

        coefficients = []
        for n in range(min_degree, max_degree + 1):
            coefficients.append(super(CoefficientSequenceDegreeWise, self).Coefficient(np.int8(0), n, 0))

            for m in range(1, n + 1):
                coefficients.append(super(CoefficientSequenceDegreeWise, self).Coefficient(np.int8(0), n, m))
                coefficients.append(super(CoefficientSequenceDegreeWise, self).Coefficient(np.int8(1), n, m))

        super(CoefficientSequenceDegreeWise, self).__init__(coefficients)


class CoefficientSequenceOrderWiseAlternating(CoefficientSequence):
    """
    Orderwise coefficient sequence. Coefficients are ordered by increasing order.
    For each order first cosine coefficients are ordered by increasing degree, with alternating
    cosine/sine coefficients per order.

    Parameters
    ----------
    min_degree : int
        minimum degree in sequence
    max_degree : int
        maximum degree in sequence
    """

    class Comparable(CoefficientSequence.ComparableCoefficientSequence):

        def __init__(self, coefficient):
            super(CoefficientSequenceOrderWiseAlternating.Comparable, self).__init__(coefficient)

        def __lt__(self, other):

            if self.order < other.order:
                return True
            if self.order == other.order and self.degree < other.degree:
                return True
            if self.degree == other.degree and self.order == other.order and self.basis_function < other.basis_function:
                return True

            return False

    def __init__(self, min_degree, max_degree):

        coefficients = []
        for n in range(min_degree, max_degree + 1):
            coefficients.append(super(CoefficientSequenceOrderWiseAlternating, self).Coefficient(np.int8(0), n, 0))

        for m in range(1, max_degree + 1):
            for n in range(max(min_degree, m), max_degree + 1):
                coefficients.append(super(CoefficientSequenceOrderWiseAlternating, self).Coefficient(np.int8(0), n, m))
                coefficients.append(super(CoefficientSequenceOrderWiseAlternating, self).Coefficient(np.int8(1), n, m))

        super(CoefficientSequenceOrderWiseAlternating, self).__init__(coefficients)


class CoefficientSequenceOrderWise(CoefficientSequence):
    """
    Orderwise coefficient sequence. Coefficients are ordered by increasing order.
    For each order first cosine coefficients are ordered by increasing degree, the sine
    coefficients are ordered by increasing degree.

    Parameters
    ----------
    min_degree : int
        minimum degree in sequence
    max_degree : int
        maximum degree in sequence
    """

    class Comparable(CoefficientSequence.ComparableCoefficientSequence):

        def __init__(self, coefficient):
            super(CoefficientSequenceOrderWise.Comparable, self).__init__(coefficient)

        def __lt__(self, other):

            if self.order < other.order:
                return True
            if self.order == other.order and self.basis_function < other.basis_function:
                return True
            if self.basis_function == other.basis_function and self.order == other.order and self.degree < other.degree:
                return True

            return False

    def __init__(self, min_degree, max_degree):

        coefficients = []
        for n in range(min_degree, max_degree + 1):
            coefficients.append(super(CoefficientSequenceOrderWise, self).Coefficient(np.int8(0), n, 0))

        for m in range(1, max_degree + 1):
            for n in range(max(min_degree, m), max_degree + 1):
                coefficients.append(super(CoefficientSequenceOrderWise, self).Coefficient(np.int8(0), n, m))
            for n in range(max(min_degree, m), max_degree + 1):
                coefficients.append(super(CoefficientSequenceOrderWise, self).Coefficient(np.int8(1), n, m))

        super(CoefficientSequenceOrderWise, self).__init__(coefficients)


class CoefficientSequenceFlatArray(CoefficientSequence):
    """
    Coefficient sequence of a flattened coefficient array.

    Parameters
    ----------
    max_degree : int
        maximum degree in sequence
    """

    class Comparable(CoefficientSequence.ComparableCoefficientSequence):

        def __init__(self, coefficient):
            super(CoefficientSequenceFlatArray.Comparable, self).__init__(coefficient)

        def __lt__(self, other):

            row_idx = self.degree if self.basis_function == 0 else self.order - 1
            other_row_idx = other.degree if other.basis_function == 0 else other.order - 1

            if row_idx < other_row_idx:
                return True

            col_idx = self.order if self.basis_function == 0 else self.degree
            other_col_idx = other.order if other.basis_function == 0 else other.degree

            if row_idx == other_row_idx and col_idx < other_col_idx:
                return True

            return False

    def __init__(self, max_degree):

        degrees = np.empty((max_degree + 1, max_degree + 1), dtype=int)
        orders = np.empty((max_degree + 1, max_degree + 1), dtype=int)
        basis_functions = np.zeros((max_degree + 1, max_degree + 1), dtype=np.int8)

        for n in range(max_degree + 1):
            degrees[degree_indices(n)] = n
            orders[order_indices(max_degree, n)] = n

        basis_functions[np.triu_indices(basis_functions.shape[0], 1)] = 1

        coefficients = []
        for bf, n, m in zip(basis_functions.flatten(), degrees.flatten(), orders.flatten()):
            coefficients.append(super(CoefficientSequenceFlatArray, self).Coefficient(bf, n, m))

        super(CoefficientSequenceFlatArray, self).__init__(coefficients)


class ReferenceField(PotentialCoefficients):
    """
    Class representation of a geodetic reference system that defines a reference ellipsoid and reference gravity field.

    Parameters
    ----------
    GM : float
        geocentric gravitational constant [m^3 / s^2]
    omega : float
        angular rotation speed of the earth [rad / s]
    a : float
        equatorial radius (semi-major axis) [m]
    f : float
        flattening of the ellipsoid (either f or J2 must be given)
    J2 : float
        dynamical form factor (either f or J2 must be given)

    Examples
    --------
    >>> WGS84 = ReferenceField(GM=3986004.418e8, omega=7292115.0e-11, a=6378137.0, f=1/298.257223563)

    >>> GRS80 = ReferenceField(GM=3986005e8, omega=7292115.0e-11, a=6378137.0, J2=108263e-8)

    """
    def __init__(self, GM, omega, a, f=None, J2=None):

        self.omega = omega

        if J2 is None:
            self.flattening = f
            e2 = f * (2 - f)
            e = np.sqrt(e2)
            e_prime = e / np.sqrt(1 - e**2)

            n = np.arange(1, 21, dtype=float)
            q0 = -2 * np.sum(np.power(-1, n) * n * np.power(e_prime, 2 * n + 1) / ((2 * n + 1) * (2 * n + 3)))
            self.J2 = (e2 - 4 / 15 * (omega**2 * a**3) / GM * e**3 / (2 * q0)) / 3

        elif f is None:
            self.J2 = J2
            e = 0.1
            e0 = np.inf

            n = np.arange(1, 21, dtype=float)
            while not np.isclose(e, e0, atol=1e-22, rtol=0):
                e0 = e
                e_prime = e / np.sqrt(1 - e**2)
                q0 = -2 * np.sum(np.power(-1, n) * n * np.power(e_prime, 2 * n + 1) / ((2 * n + 1) * (2 * n + 3)))
                e = np.sqrt(3 * J2 + 4 / 15 * (omega**2 * a**3) / GM * e**3 / (2 * q0))

            e2 = e**2
            self.flattening = 1 - np.sqrt(1 - e2)
        else:
            raise ValueError('either flattening f or dynamic force factor J2 must be given for a full definition of the reference field')

        coefficients = [1.0]
        n = 1
        while not np.isclose(coefficients[-1], 0, atol=1e-22, rtol=0):
            factor = 1 if n % 2 == 0 else -1
            c2n = factor * (3 * e2**n * (1 - n + 5 * n * self.J2 / e2) / ((2 * n + 1) * (2 * n + 3) * np.sqrt(4 * n + 1)))
            coefficients.append(c2n)
            n += 1

        max_degree = (len(coefficients) - 1) * 2
        anm = np.zeros((max_degree + 1, max_degree + 1))
        anm[0::2, 0] = coefficients

        super(ReferenceField, self).__init__(GM, a)
        self.anm = anm

    def normal_gravity(self, r, colat):
        """
        Normal gravity (graviational and centrifugal acceleration) of the reference field.

        Parameters
        ----------
        r : float, array_like, shape (m,)
            geocentric radius of evaluation points in radians
        colat : float, array_like, shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        g : float, ndarray(m,)
            normal gravity at evaluation point(s) in [m/s**2]
        """
        point_count = max(np.asarray(r).size, np.asarray(colat).size)
        xyz = np.zeros((point_count, 3))
        xyz[:, 0] = r * np.sin(colat)
        xyz[:, 2] = r * np.cos(colat)

        _, lat, _ = grates.grid.cartesian2geodetic(xyz, self.R, self.flattening)

        g = self.gravitational_acceleration(xyz)
        g[:, 0] += self.omega**2 * xyz[:, 0]

        return -np.cos(lat) * g[:, 0] - np.sin(lat) * g[:, 2]


WGS84 = ReferenceField(GM=3986004.418e8, omega=7292115.0e-11, a=6378137.0, f=1 / 298.257223563)
GRS80 = ReferenceField(GM=3986005e8, omega=7292115.0e-11, a=6378137.0, J2=108263e-8)
