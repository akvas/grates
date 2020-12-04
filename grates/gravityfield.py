# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Representations of Earth's gravity field.
"""

import numpy as np
import grates.grid
import grates.kernel
import grates.utilities


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
        kernel = grates.kernel.get_kernel(kernel)

        if isinstance(grid, grates.grid.RegularGrid):
            gridded_values = np.zeros((grid.parallels.size, grid.meridians.size))
            orders = np.arange(self.max_degree + 1)[:, np.newaxis]

            colat = grates.utilities.colatitude(grid.parallels, grid.semimajor_axis, grid.flattening)
            r = grates.utilities.geocentric_radius(grid.parallels, grid.semimajor_axis, grid.flattening)
            P = grates.utilities.legendre_functions(self.max_degree, colat)
            P *= self.anm

            for n in range(self.max_degree + 1):
                row_idx, col_idx = grates.gravityfield.degree_indices(n)
                continuation = np.power(self.R / r, n + 1)
                kn = kernel.inverse_coefficient(n, r, colat)

                CS = np.vstack((np.cos(orders[0:n + 1] * grid.meridians), np.sin(orders[1:n + 1] * grid.meridians)))
                gridded_values += (P[:, row_idx, col_idx] * (continuation * kn)[:, np.newaxis]) @ CS

            output_grid = grid.copy()
            output_grid.values = gridded_values * (self.GM / self.R)
            output_grid.epoch = self.epoch
        else:
            raise NotImplementedError('Propagation to arbitrary point distributions is not yet implemented.')

        return output_grid

    def vector(self):
        """
        Return a vector representation of the potential coefficients.

        Returns
        -------
        coeffs : ndarray((max_degree + 1)**2)
            ravelled coefficient array
        """
        return grates.utilities.ravel_coefficients(self.anm)

    def update_from_vector(self, x):
        """
        Set the coefficient array from a vector.

        Parameters
        ----------
        x : ndarray((max_degree + 1)**2)
            coefficients in vector representation
        """
        self.anm = grates.utilities.unravel_coefficients(x)

    def gravity(self, xyz):
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

        return self.__data[idx - 1] * (1 - weight) + self.__data[idx] * weight

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
        shape = len(self.__data), self.__data[0].vector().size

        data_matrix = np.empty(shape)

        for k in range(data_matrix.shape[0]):
            data_matrix[k, :] = self.__data[k].vector()[0:shape[1]]

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
            d.update_from_vector(observations[k, :])


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

        return self.__data * dt


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

        return self.__data_cosine * np.cos(2 * np.pi * dt) + self.__data_sine * np.sin(2 * np.pi * dt)


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
    """
    Class representation of the ordering of spherical harmonic coefficients.

    Parameters
    ----------
    degrees : array_like(coefficient_count,)
        spherical harmonic degree of coefficients
    orders : array_like(coefficient_count,)
        spherical harmonic order of coefficients
    basis_functions : array_like(coeffient_count,) containing 'c' or 's'
        whether the coefficient is a cosine (value 'c') coefficient or sine (value 's') coefficient
    """
    def __init__(self, degrees, orders, basis_functions):

        self.degrees = np.asarray(degrees)
        self.orders = np.asarray(orders)
        self.basis_functions = np.asarray(basis_functions)

    @property
    def coefficient_count(self):
        """Return the number of coefficients in the coefficient sequence."""
        return self.degrees.size

    def vector_indices(self, degree=None, order=None, basis_function=None):
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
            mask = np.logical_and(mask, self.degrees == degree)

        if order is not None:
            mask = np.logical_and(mask, self.orders == order)

        if basis_function is not None:
            mask = np.logical_and(mask, self.basis_functions == basis_function)

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
        source_indices = []
        target_indices = []
        for k in range(target_sequence.coefficient_count):
            other_idx = source_sequence.vector_indices(target_sequence.degrees[k], target_sequence.orders[k], target_sequence.basis_functions[k])
            if len(other_idx) > 0:
                source_indices.append(other_idx[0])
                target_indices.append(k)

        return np.array(source_indices), np.array(target_indices)


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
    def __init__(self, min_degree, max_degree):

        coefficient_count = (max_degree + 1) * (max_degree + 1) - min_degree * min_degree

        degrees = np.empty(coefficient_count, dtype=int)
        orders = np.zeros(coefficient_count, dtype=int)
        basis_functions = np.full(coefficient_count, 'c', dtype=str)

        index = 0
        for n in range(min_degree, max_degree + 1):
            degrees[index] = n
            index += 1

            for m in range(1, n + 1):
                degrees[index:index + 2] = n
                orders[index:index + 2] = m
                basis_functions[index + 1] = 's'
                index += 2

        super(CoefficientSequenceDegreeWise, self).__init__(degrees, orders, basis_functions)


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
    def __init__(self, min_degree, max_degree):

        coefficient_count = (max_degree + 1) * (max_degree + 1) - min_degree * min_degree

        degrees = np.empty(coefficient_count, dtype=int)
        orders = np.zeros(coefficient_count, dtype=int)
        basis_functions = np.full(coefficient_count, 'c', dtype=str)

        index = 0
        for n in range(min_degree, max_degree + 1):
            degrees[index] = n
            index += 1

        for m in range(1, max_degree + 1):
            for n in range(max(min_degree, m), max_degree + 1):
                degrees[index] = n
                orders[index] = m
                index += 1

            for n in range(max(min_degree, m), max_degree + 1):
                degrees[index] = n
                orders[index] = m
                basis_functions[index] = 's'
                index += 1

        super(CoefficientSequenceOrderWise, self).__init__(degrees, orders, basis_functions)


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
    def __init__(self, min_degree, max_degree):

        coefficient_count = (max_degree + 1) * (max_degree + 1) - min_degree * min_degree

        degrees = np.empty(coefficient_count, dtype=int)
        orders = np.zeros(coefficient_count, dtype=int)
        basis_functions = np.full(coefficient_count, 'c', dtype=str)

        index = 0
        for n in range(min_degree, max_degree + 1):
            degrees[index] = n
            index += 1

        for m in range(1, max_degree + 1):
            for n in range(max(min_degree, m), max_degree + 1):
                degrees[index] = n
                orders[index] = m
                index += 1

                degrees[index] = n
                orders[index] = m
                basis_functions[index] = 's'
                index += 1

        super(CoefficientSequenceOrderWiseAlternating, self).__init__(degrees, orders, basis_functions)


class CoefficientSequenceFlatArray(CoefficientSequence):
    """
    Coefficient sequence of a flattened coefficient array.

    Parameters
    ----------
    max_degree : int
        maximum degree in sequence
    """
    def __init__(self, max_degree):

        degrees = np.empty((max_degree + 1, max_degree + 1), dtype=int)
        orders = np.empty((max_degree + 1, max_degree + 1), dtype=int)
        basis_functions = np.empty((max_degree + 1, max_degree + 1), dtype=str)

        for n in range(max_degree + 1):
            degrees[degree_indices(n)] = n
            orders[order_indices(max_degree, n)] = n

        basis_functions[np.tril_indices(basis_functions.shape[0])] = 'c'
        basis_functions[np.triu_indices(basis_functions.shape[0], 1)] = 's'

        super(CoefficientSequenceFlatArray, self).__init__(degrees.flatten(), orders.flatten(), basis_functions.flatten())
