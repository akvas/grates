# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Experimental features with frequent interface changes.
"""

import abc
import numpy as np
import scipy.integrate
import scipy.linalg as la
import scipy.signal as sig
import grates.kernel
import grates.utilities
import netCDF4
import grates.grid


class Topography(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def cross_section(self, latitude):
        pass


class TopographyNetCDF(Topography):

    def __init__(self, netcdf_file, variable_longitude='lon', variable_latitude='lat', variable_elevation='elevation'):

        if isinstance(netcdf_file, netCDF4.Dataset):
            self.dataset = netcdf_file
        elif isinstance(netcdf_file, str):
            self.dataset = netCDF4.Dataset(netcdf_file)
        else:
            raise TypeError("Argument netcdf_file must either be a netCDF4.Dataset or str instance")

        self.lon_var = variable_longitude
        self.lat_var = variable_latitude
        self.elev_var = variable_elevation

    def cross_section(self, latitude):

        latitude_index = np.searchsorted(self.dataset[self.lat_var][:], latitude)
        longitudes = self.dataset[self.lon_var][:]
        z = self.dataset[self.elev_var][latitude_index, :]

        return longitudes,  z, np.gradient(z, longitudes)


class TopographyGridded(Topography):

    def __init__(self, longitude, latitude, elevation, basin=None, a=6378137.0, f=298.2572221010**-1):

        self.__longitude = longitude
        self.__latitude = latitude
        self.__elevation = elevation
        self.__a = a
        self.__f = f
        self.__basin = basin

    def cross_section(self, latitude):

        latitude_index = np.searchsorted(self.__latitude, latitude)

        if self.__basin is not None:
            mask = self.__basin.contains_points(self.__longitude, latitude)
        else:
            mask = np.ones(self.__longitude.size, dtype=bool)

        z = self.__elevation[latitude_index, :]
        dz = np.gradient(z, self.__longitude)

        return self.__longitude[mask], z[mask], dz[mask]


class Transport(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute(self, latitude, depth_bounds, data):
        pass


class Spectral(Transport):

    def __init__(self, topography, seawater_density=1025, earthrotation=7.29211585531e-5):

        self.__topography = topography
        self.__density = seawater_density
        self.__earthrotation = earthrotation

    def coefficient_factors(self, latitudes, depth_bounds, nmax, GM=3.9860044150e+14, R=6.3781363000e+06):

        latitudes = np.atleast_1d(latitudes)
        orders = np.arange(nmax + 1, dtype=float)[:, np.newaxis]
        obp_kernel = grates.kernel.OceanBottomPressure()

        colatitude = grates.utilities.colatitude(latitudes)
        radius = grates.utilities.geocentric_radius(latitudes)

        legendre_array = grates.utilities.legendre_functions(nmax, colatitude)

        coefficient_factor = np.empty((latitudes.size, nmax + 1, nmax + 1))
        for k, latitude in enumerate(latitudes):
            lon, z, dz = self.__topography.cross_section(latitude)
            depth_mask = np.logical_or(z < depth_bounds[0], z > depth_bounds[1])
            dz[depth_mask] = 0

            factors_cosine = scipy.integrate.trapz(np.cos(orders * lon) * dz, lon)
            factors_sine = scipy.integrate.trapz(np.sin(orders * lon) * dz, lon)

            coefficient_factor[k, :, :] = legendre_array[k, :, :] * GM / R / (
                    2 * self.__density * self.__earthrotation * np.sin(latitude))

            continuation = np.power(R / radius[k], range(nmax + 1))
            for n in range(1, nmax + 1):
                row_idx, col_idx = grates.gravityfield.degree_indices(n)

                coefficient_factor[k, row_idx, col_idx] *= obp_kernel.inverse_coefficient(n) * continuation[n] * \
                                                           np.concatenate((factors_cosine[0:n + 1], factors_sine[1:n + 1]))

            coefficient_factor[k, :, 0] = 0

        return coefficient_factor

    def compute(self, latitudes, depth_bounds, data):

        latitudes = np.atleast_1d(latitudes)
        factors = self.coefficient_factors(latitudes, depth_bounds, data[0].max_degree(), data[0].GM, data[0].R)

        transport_series = np.zeros((len(data), latitudes.size))
        epochs = []

        for k, coeffs in enumerate(data):
            epochs.append(coeffs.epoch)

            transport_series[k, :] = np.sum(factors*coeffs.anm, axis=(1, 2))

        return epochs, transport_series


class Spatial(Transport):

    def __init__(self, topography, seawater_density=1025, earthrotation=7.29211585531e-5):

        self.__topography = topography
        self.__density = seawater_density
        self.__earthrotation = earthrotation

    def compute(self, latitudes, depth_bounds, data):

        latitudes = np.atleast_1d(latitudes)

        for k, latitude in enumerate(latitudes):
            lon, z, dz = self.__topography.cross_section(latitude)
            depth_mask = np.logical_or(z < depth_bounds[0], z > depth_bounds[1])
            dz[depth_mask] = 0


def stream_function(transport, latitudes, depth_layers, data):

    latitudes = np.atleast_1d(latitudes)

    sf = np.zeros((len(data), depth_layers.size, latitudes.size))
    for k in range(depth_layers.size):
        _, psi = transport.compute(latitudes, (depth_layers[k], 0), data)
        sf[:, k, :] = psi

    t = [d.epoch for d in data]
    return t, sf


def lsa_psd(x, y, nperseg=256, window='boxcar'):

    nperseg = min(nperseg, x.size)

    dx = float(np.median(np.diff(x)))
    interval_bounds = [0]
    while interval_bounds[-1] < x.size:
        interval_bounds.append(min(interval_bounds[-1] + nperseg, x.size))
    interval_bounds.append(x.size)

    frequencies = np.fft.rfftfreq(nperseg, dx)
    is_even = (nperseg % 2) == 0
    loop_count = frequencies.size - 2 if is_even else frequencies.size - 1

    N = [np.zeros((1, 1))]
    n = [np.zeros((1, 1))]
    for k in range(loop_count):
        N.append(np.zeros((2, 2)))
        n.append(np.zeros((2, 1)))
    if is_even:
        N.append(np.zeros((1, 1)))
        n.append(np.zeros((1, 1)))

    for idx_start, idx_end in zip(interval_bounds[0:-1], interval_bounds[1:]):
        interval_length = idx_end - idx_start
        if interval_length < 3:
            continue

        w = sig.get_window(window, interval_length)[:, np.newaxis]

        t = x[idx_start:idx_end]
        l = y[idx_start:idx_end, np.newaxis]*w

        A = np.ones((interval_length, 1))*w
        N[0] += A.T@A
        n[0] += A.T@l

        for k in range(1, loop_count + 1):
            A = np.vstack((np.cos(2*np.pi*frequencies[k]*t),
                           np.sin(2*np.pi*frequencies[k]*t))).T*w
            N[k] += A.T @ A
            n[k] += A.T @ l

        if is_even:
            A = np.ones((interval_length, 1))*w
            A[1::2, 0] = -1
            N[-1] += A.T @ A
            n[-1] += A.T @ l

    x_hat = []
    for k in range(len(N)):
        x_hat.append(np.linalg.solve(N[k], n[k]))

    psd = np.zeros(frequencies.size)
    for idx_start, idx_end in zip(interval_bounds[0:-1], interval_bounds[1:]):
        interval_length = idx_end - idx_start
        if interval_length < 3:
            continue

        t = x[idx_start:idx_end]

        A = np.ones((interval_length, 1))
        l_hat = A@x_hat[0]
        psd[0] = np.sum(l_hat**2)

        for k in range(1, loop_count + 1):
            A = np.vstack((np.cos(2*np.pi*frequencies[k]*t),
                           np.sin(2*np.pi*frequencies[k]*t))).T
            l_hat = A @ x_hat[k]
            psd[k] = np.sum(l_hat ** 2)

        if is_even:
            A = np.ones((interval_length, 1))
            A[1::2, 0] = -1
            l_hat = A @ x_hat[-1]
            psd[-1] = np.sum(l_hat ** 2)

    return frequencies, psd/dx*np.sqrt(2)


def vce_psd(x, y, nperseg=256, initial_variance=1, max_iter=5, detrend=False, window='boxcar'):

    nperseg = min(nperseg, x.size)

    dx = np.median(np.diff(x))
    interval_bounds = [0]
    segment_length = 0
    for k in range(1, x.size):
        segment_length += 1
        if x[k]-x[k-1] > dx*1.5 or segment_length == nperseg:
            interval_bounds.append(min(interval_bounds[-1] + segment_length, x.size))
            segment_length = 0
    interval_bounds.append(x.size)

    dummy = np.array(interval_bounds)
    nperseg = np.max(dummy[1:]-dummy[0:-1])

    dct_matrix = np.fromfunction(lambda i, j: 2*np.cos(np.pi*i*j/(nperseg - 1)), (nperseg, nperseg))
    dct_matrix[:, (0, -1)] *= 0.5
    dct_matrix *= 1.0/np.sqrt(2*(nperseg-1))

    initial_covariance = np.zeros(nperseg)
    initial_covariance[0] = initial_variance
    variance_components = (dct_matrix@initial_covariance[:, np.newaxis]).squeeze()

    for iteration in range(max_iter):
        covariance_function = (dct_matrix@variance_components[:, np.newaxis]).squeeze()
        covariance_matrix = la.toeplitz(covariance_function, covariance_function)

        square_sum = np.zeros(covariance_function.size)
        redundancy = np.zeros(covariance_function.size)

        for idx_start, idx_end in zip(interval_bounds[0:-1], interval_bounds[1:]):
            interval_length = idx_end - idx_start
            if interval_length < 2:
                continue

            projection_matrix = np.linalg.inv(covariance_matrix[0:interval_length, 0:interval_length])
            e = y[idx_start:idx_end, np.newaxis]

            residuals = (projection_matrix @ e).squeeze()

            for k in range(residuals.size):
                square_sum[k] += np.sum(residuals[0:residuals.size-k]*residuals[k:])
                redundancy[k] += np.sum(np.diag(projection_matrix, k))

        redundancy[1:] *= 2
        square_sum[1:] *= 2

        ePe = (square_sum[np.newaxis, :] @ dct_matrix).squeeze()
        r = (redundancy[np.newaxis, :] @ dct_matrix).squeeze()

        variance_components *= ePe/r

    return np.linspace(0, 0.5/dx, variance_components.size),  variance_components*dx*np.sqrt(2*(nperseg-1)), dummy[1:]-dummy[0:-1], dct_matrix


def legendre_matern(sigma0, alpha, nu, psi, max_degree=1024, min_degree=2):

    n = np.arange(max_degree + 1)

    coefficients = sigma0**2 * (alpha**2 + n**2)**-(nu + 0.5)
    coefficients[0:min_degree] = 0

    return grates.utilities.legendre_summation(coefficients, psi)







