# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Meridional transport from satellite gravimetry.
"""

import abc
import numpy as np
import scipy.integrate
import grates.kernel
import grates.utilities
import grates.grid


class Bathymetry(metaclass=abc.ABCMeta):
    """
    Base class for discrete ocean bathymetry. Derived classes must implement a cross_section method
    which returns a 1d array given a latiude.
    """
    @abc.abstractmethod
    def cross_section(self, latitude):
        pass


class BathymetryGridded(Bathymetry):
    """
    Bathymetry from an existing dataset given on a regular grid (defined by meridians and parallels).

    Parameters
    ----------
    longitude : ndarray(m,)
        longitude of meridians in radians
    latitude : ndarray(n,)
        latitude of parallels in radians
    elevation : ndarray(n, m)
        elevation (points below the ocean surface are negative) in meters
    basin : grates.grid.Basin
        restrict cross sections to a specific basin outline
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    def __init__(self, longitude, latitude, elevation, basin=None, a=6378137.0, f=298.2572221010**-1):

        self.__longitude = np.asarray(longitude)
        self.__latitude = np.asarray(latitude)
        self.__elevation = np.asarray(elevation)
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
    """
    Base class for meridional transport. Derived classes must implement a compute method which depends on
    a 1d latitude array, a 1d depth_bounds array, and gravity field time series.
    """
    @abc.abstractmethod
    def compute(self, latitude, depth_bounds, data):
        pass


class Spectral(Transport):
    """
    Compute meridional transport from gravity fields given in spectral domain (potential coefficients).

    """
    def __init__(self, topography, seawater_density=1025, earthrotation=7.29211585531e-5):

        self.__topography = topography
        self.__density = seawater_density
        self.__earthrotation = earthrotation

    def coefficient_factors(self, latitudes, depth_bounds, max_degree, GM=3.9860044150e+14, R=6.3781363000e+06):

        latitudes = np.atleast_1d(latitudes)
        orders = np.arange(max_degree + 1, dtype=float)[:, np.newaxis]
        obp_kernel = grates.kernel.OceanBottomPressure()

        colatitude = grates.utilities.colatitude(latitudes)
        radius = grates.utilities.geocentric_radius(latitudes)

        legendre_array = grates.utilities.legendre_functions(max_degree, colatitude)

        coefficient_factor = np.empty((latitudes.size, max_degree + 1, max_degree + 1))
        for k, latitude in enumerate(latitudes):
            lon, z, dz = self.__topography.cross_section(latitude)
            depth_mask = np.logical_or(z < depth_bounds[0], z > depth_bounds[1])
            dz[depth_mask] = 0

            factors_cosine = scipy.integrate.trapz(np.cos(orders * lon) * dz, lon)
            factors_sine = scipy.integrate.trapz(np.sin(orders * lon) * dz, lon)

            coefficient_factor[k, :, :] = legendre_array[k, :, :] * GM / R / (
                    2 * self.__density * self.__earthrotation * np.sin(latitude))

            continuation = np.power(R / radius[k], range(max_degree + 1))
            for n in range(1, max_degree + 1):
                row_idx, col_idx = grates.gravityfield.degree_indices(n)

                coefficient_factor[k, row_idx, col_idx] *= obp_kernel.inverse_coefficient(n) * continuation[n] * \
                                                           np.concatenate((factors_cosine[0:n + 1], factors_sine[1:n + 1]))

            coefficient_factor[k, :, 0] = 0

        return coefficient_factor

    def compute(self, latitudes, depth_bounds, data):

        latitudes = np.atleast_1d(latitudes)
        factors = self.coefficient_factors(latitudes, depth_bounds, data[0].max_degree, data[0].GM, data[0].R)

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