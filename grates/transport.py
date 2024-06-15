# Copyright (c) 2020-2024 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Integrated meridional transport from satellite gravimetry.
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

    def mean_coriolis_parameter(self, earthrotation=7.29211585531e-5):
        return 2 * earthrotation * np.sin(np.median(self.latitude))


class Transport(metaclass=abc.ABCMeta):
    """
    Base class for (meridional) transport. Derived classes must implement a compute method which depends on
    a 1d latitude array, a 1d depth_bounds array, and gravity field time series.
    """
    @abc.abstractmethod
    def compute(self, depth_bounds, data, **kwargs):
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
        r"""
        Compute the coefficientwise factors for the linear operator to convert potential coefficients into transport.

        Starting from the transport integral in space domain

        .. math::

            \psi(\varphi) = \frac{1}{\rho_0(\varphi) f(\varphi)} \int_{x_1}^{x_2} OBP(\varphi,x) \tilde{t}'(\varphi,x) dx,

        we can expand :math:`OBP(\varphi,x)` into a series of spherical harmonics,

        .. math::

            OBP(\varphi,x(\lambda)) = \frac{GM}{R} \sum_n k_n \left( \frac{R}{r(\varphi)} \right)^{n+1} \sum_m P_{nm}(\cos \vartheta(\varphi)) (c_{nm} \cos m\lambda + s_{nm} \sin m\lambda).

        Since only the spatial basis functions depend on the geohraphic coordinates, they can be integrated beforehand and do not need to be evaluated for each time step.


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

        coriolis_density = self.__cross_section.mean_coriolis_parameters(self.__earthrotation) * self.__density
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

    def compute(self, depth_bounds, data, **kwargs):
        """
        Compute transport in multiple depth bounds from a time variable gravity field.

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
    r"""
    Compute meridional transport from gravity fields given in space domain via ocean bottom pressure (OBP) grids.

    The fundamental principle behind this application is given by the integral

    .. math::
        :name: eq:geostrophic-flow

        \psi = \frac{1}{\rho_0 f}\int_{z_\text{min}}^{z_\text{max}} p(\lambda_e(z)) - p(\lambda_w(z))dz

    which yields the transport :math:`\psi` through a depth layer bounded by :math:`(z_\text{min}, z_\text{max})`.
    The integrand :math:`p(\lambda_e(z)) - p(\lambda_w(z))` is the pressure difference between eastern and western
    longitudinal extent at depth :math:`z` described by :math:`\lambda_e(z)` and :math:`\lambda_w(z)` respectively.
    Since the points :math:`(\lambda_e(z), z)` and :math:`(\lambda_w(z), z)` are located at the basin slope,
    the corresponding pressure values are OBP and thus observable by GRACE/GRACE-FO.
    The seawater density :math:`\rho_0` is assumed constant in this simplification, while the Coriolis factor
    :math:`f` will naturally be constant within a longitudinal cross section and will be approximated by the median latitude in arbitrarily oriented cross sections.

    While :ref:`Eq. 1 <eq:geostrophic-flow>` is widely used for the application at hand, its evaluation
    is cumbersome in practice.
    First, OBP variations are typically given as regular or irregular longitude/latitude grids, but
    the integration is performed in depth direction, rather than horizontally.
    Second, intervening topography like the Mid-Atlantic Ridge cannot be easily treated and necessitates
    a split of the cross section.
    To make transport computation from OBP data more straightforward, we decided to reformulate
    this fundamental equation using Green's theorem.
    In its general form, Green's theorem relates the integral over a region :math:`R`, with a line integral
    over its boundary `C`.
    For two generic functions :math:`P` and `Q` in the :math:`x, z`-plane where the ocean basin cross section is defined
    (:math:`x` is the horizontal axis, :math:`z` is the vertical axis), the theorem states that

    .. math::
        :name: eq:greens-theorem

        \iint_R \left(\frac{\partial P}{\partial x} - \frac{\partial Q}{\partial z}\right) dx dz = \int_C P \; dz + \int_C Q \; dx.

    If we set :math:`P = p` (pressure) and :math:`Q = 0`, :ref:`Eq. 2 <eq:greens-theorem>` reduces to

    .. math::
        :name: eq:greens-theorem-obp

        \iint_R \frac{\partial p}{\partial x} dx dz = \int_C p \; dz

    and can be readily applied to the formulation for transport computed from the geostrophic meridional
    velocity :math:`v = \frac{1}{\rho_0 f} \frac{\partial p}{\partial x}`, with

    .. math::
        :name: eq:transport-line-integral

        \psi = \iint_R v(x, z) dx dz =
        \frac{1}{\rho_0 f}\iint_D\frac{\partial p}{\partial x} dx dz =
        \frac{1}{\rho_0 f} \int_C p \; dz.

    In the problem at hand, the surface boundary :math:`C`, consists of a horizontal line :math:`T = (x, z_\text{max})` (i.e., the sea surface or the upper boundary of the depth layer)
    and a curve :math:`B`, which represents either the ocean floor topography :math:`t` or the lower bound of the depth layer.

    For convenience, we introduce the synthetic topography :math:`\tilde{t}(x) = \text{max}\{z_\text{min}, t(x)\}`,
    thus :math:`B = (x, \tilde{t}(x))`.
    With these definitions we can simplify the line integral in :ref:`Eq. 3 <eq:transport-line-integral>`
    by splitting the integration range into :math:`B` and :math:`T` and expressing the integration in terms of :math:`x`,
    which yields

    .. math::
        :name: eq:greens-theorem-obp-final

        \int_C p \; dz =  \int_B p \; dz  + \underbrace{\int_T p \; dz}_{=0} =
        \int_{x_1}^{x_2} p(x, \tilde{t}(x)) dz = \int_{x_1}^{x_2} p(x, \tilde{t}(x)) \tilde{t}'(x) dx.

    It is easy to see that the integral over :math:`T` is zero since :math:`dz = 0` for the upper boundary.
    The change in :math:`z`-direction along :math:`B` is governed by the (synthetic) topography :math:$
    `\tilde{t}` and
    can be expressed as :math:`dz =\tilde{t}'(x) dx`.
    In regions where :math:`B` follows the ocean floor, :math:`p` is OBP so :ref:`Eq. 4 <eq:greens-theorem-obp-final>`
    can be evaluated with GRACEO/GRACE-FO data.
    Where :math:`B` follows the layer depth bound :math:`z_\text{min}`, this is not the case,
    however, there :math:`\tilde{t}'(x)` is zero, so these values do not affect the computed transport.
    Consequently, the meridional transport can be computed by simply integrating
    the GRACE/GRACE-FO OBP values multiplied with the synthetic topography,
    from the cross section bounds :math:`x_1` to :math:`x_2`,

    .. math::

        \psi(\varphi) = \frac{1}{\rho_0(\varphi) f(\varphi)} \int_{x_1}^{x_2} OBP(\varphi,x) \tilde{t}'(\varphi,x) dx.

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

    def compute(self, depth_bounds, data, epochs=None, longitude=None, latitude=None):
        """
        Compute transport in multiple depth bounds from a time series of ocean bottom pressure (OBP) grids.

        Parameters
        ----------
        depth_bounds : array_like(m + 1)
            boundaries of the m depth layers in ascending order
        data : array_like(n_times, n_lat, n_lon)
            time series of potential coefficients

        Returns
        -------
        epochs : list of datetime
            time stamps of k computed epochs
        transport_series : ndarray(k, m)
            time series of transport estimates for m depth layers
        """
        path, z, dz = self.__cross_section.path, self.__cross_section.z, self.__cross_section.dz.copy()
        points_sample = np.vstack((self.__cross_section.latitude, self.__cross_section.longitude)).T

        transport_series = np.zeros((data.shape[0], len(depth_bounds) - 1))

        for k in range(data.shape[0]):
            obp_interp = scipy.interpolate.RegularGridInterpolator((latitude, longitude), data[k])
            obp_values = obp_interp(points_sample, method='linear')
            for l in range(len(depth_bounds) - 1):
                depth_mask = np.logical_or(z < depth_bounds[l], z > depth_bounds[l + 1])
                dzl = dz.copy()
                dzl[depth_mask] = 0
                transport_series[k, l] = scipy.integrate.trapz(obp_values * dz, path)

        return epochs, transport_series
