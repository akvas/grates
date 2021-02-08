# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Integrated transport from satellite gravimetry.
"""

import abc
import numpy as np
import scipy.integrate
import scipy.interpolate
import grates.kernel
import grates.utilities
import grates.grid


class Bathymetry(metaclass=abc.ABCMeta):
    """
    Base class for discrete ocean bathymetry. Derived classes must implement a cross_section method
    which returns a 1d array given central longitude and latiude, azimuth and a sampling. Cross sections
    are constructed along lines of constant azimuth (loxodromes).
    """
    @abc.abstractmethod
    def cross_section(self, central_longitude, central_latitude, azimuth, sampling):
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
        elevation = np.asarray(elevation)
        self.__a = a
        self.__f = f
        self.__basin = basin
        self.__elevation = scipy.interpolate.RegularGridInterpolator((self.__latitude, self.__longitude), elevation)

    def cross_section(self, central_longitude, central_latitude, azimuth, sampling):
        """
        Construct a cross section given longitude, latitude of the central point and the directional azimuth (0: south to north, pi/2: west to east).
        Cross sections are constructed along lines of constant azimuth (loxodromes) by bilinearly interpolating the gridded bathymetry to the points
        along the cross section.

        Parameters
        ----------
        central_longitude : float
            longitude of central point in radians
        central_latitude : float
            latitude of central point in radians
        azimuth : float
            directional azimuth in radians (0: south to north, pi/2: west to east)
        sampling : float
            sampling along the loxodrome in meters (note: should be should small enough to capure all features in the input bathymetry)

        Returns
        -------
        cs : CrossSection
            class representation of the cross section
        """
        def generate_points(central_longitude, central_latitude, azimuth, sampling):

            if np.isclose(np.cos(azimuth), 0, rtol=0, atol=1e-15):
                r1 = np.arange(0, np.pi * self.__a * np.cos(central_latitude), sampling)
                r = np.concatenate((-r1[::-1], r1[1:]))

                lon = np.mod(r / (self.__a * np.cos(central_latitude)) + central_longitude + np.pi, 2 * np.pi) - np.pi
                lat = np.full(lon.shape, central_latitude)
            else:
                max_distance = self.__a * np.pi

                r1 = np.arange(0, max_distance, sampling)
                r = np.concatenate((-r1[::-1], r1[1:]))

                lat = r / self.__a * np.cos(azimuth) + central_latitude
                lat[lat > 0.5 * np.pi] = np.pi - lat[lat > 0.5 * np.pi]
                lat[lat < -0.5 * np.pi] = -lat[lat < -0.5 * np.pi] - np.pi
                lon = central_longitude + np.tan(azimuth) * np.log(np.tan(lat * 0.5 + np.pi * 0.25) / np.tan(central_latitude * 0.5 + np.pi * 0.25))

            in_bounds = np.logical_and(np.logical_and(lon >= np.min(self.__longitude), lon <= np.max(self.__longitude)),
                                       np.logical_and(lat >= np.min(self.__latitude), lat <= np.max(self.__latitude)))
            lon = lon[in_bounds]
            lat = lat[in_bounds]
            r = r[in_bounds]

            return np.vstack((lat, lon)).T, r

        points_sample, r_sample = generate_points(central_longitude, central_latitude, azimuth, sampling)
        z = self.__elevation(points_sample, method='linear')
        dz = np.gradient(z, r_sample)

        if self.__basin is not None:
            mask = self.__basin.contains_points(points_sample[:, 1], points_sample[:, 0])
        else:
            mask = np.ones(points_sample.shape[0], dtype=bool)

        return CrossSection(points_sample[mask, 1], points_sample[mask, 0], r_sample[mask], z[mask], dz[mask])


class CrossSection:
    """
    Class representation of a bathymetry cross section.
    """
    def __init__(self, longitude, latitude, path, z, dz):

        self.longitude = longitude
        self.latitude = latitude
        self.path = path
        self.z = z
        self.dz = dz

    @property
    def is_parallel(self):
        return np.allclose(self.latitude, np.median(self.latitude))

    @property
    def is_meridian(self):
        return np.allclose(self.longitude, np.median(self.longitude))


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

    Parameters
    ----------
    cross_section : CrossSection
        cross section topography
    seawater_density : float
        average seawater density [kg / m^3]
    earthrotation : float
        average earth rotation velocity [rad / s]
    """
    def __init__(self, cross_section, seawater_density=1025, earthrotation=7.29211585531e-5):

        self.__cross_section = cross_section
        self.__density = seawater_density
        self.__earthrotation = earthrotation

    def coefficient_factors(self, depth_bounds, max_degree, GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Compute the coefficientwise factors for the linear operator to convert potential coefficients into transport.

        Parameters
        ----------
        depth_bounds : array_like(m + 1)
            boundaries of the m depth layers in ascending order
        max_degree : int
            maximum spherical harmonic degree
        GM : float
            geocentric gravitational constant
        R : float
            reference radius
        """
        obp_kernel = grates.kernel.OceanBottomPressure()

        colatitude = grates.utilities.colatitude(self.__cross_section.latitude)
        radius = grates.utilities.geocentric_radius(self.__cross_section.latitude)

        coriolis_density = 2 * self.__earthrotation * np.sin(self.__cross_section.latitude) * self.__density
        spherical_harmonics = grates.utilities.spherical_harmonics(max_degree, colatitude, self.__cross_section.longitude)
        kn = obp_kernel.inverse_coefficients(0, max_degree, radius, colatitude) / coriolis_density[:, np.newaxis] * np.power(R / radius[:, np.newaxis], range(max_degree + 1)) * GM / R

        for n in range(max_degree + 1):
            rows, columns = grates.gravityfield.degree_indices(n)
            spherical_harmonics[:, rows, columns] *= kn[:, n:n + 1]

        path, z, dz = self.__cross_section.path, self.__cross_section.z, self.__cross_section.dz.copy()

        coefficient_factors = []
        for lower_bound, upper_bound in zip(depth_bounds[0:-1], depth_bounds[1:]):
            outside_depth_layer = np.logical_or(z < lower_bound, z > upper_bound)
            dz[outside_depth_layer] = 0
            coefficient_factors.append(scipy.integrate.trapz(spherical_harmonics * dz[:, np.newaxis, np.newaxis], path, axis=0))
            if self.__cross_section.is_parallel:
                coefficient_factors[-1][:, 0] = 0

        return coefficient_factors

    def compute(self, depth_bounds, data):
        """
        Compute transport in multiple depth bounds for a time variable gravity field.

        Parameters
        ----------
        depth_bounds : array_like(m + 1)
            boundaries of the m depth layers in ascending order
        data : grates.gravityfield.TimeSeries
            time series of potential coefficients

        Returns
        -------
        epochs : list of datetime
            time stamps of k computed epochs
        transport_series : ndarray(k, m)
            time series of transport estimates for m depth layers
        """
        factors = self.coefficient_factors(depth_bounds, data[0].max_degree, data[0].GM, data[0].R)

        transport_series = np.zeros((len(data), len(depth_bounds) - 1))
        epochs = []

        for k, coeffs in enumerate(data):
            epochs.append(coeffs.epoch)
            for l in range(len(factors)):
                transport_series[k, l] = np.sum(factors[l] * coeffs.anm)

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
