# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Point distributions on the ellipsoid.
"""

import abc
import numpy as np
from scipy.special import roots_legendre
import scipy.spatial
import scipy.integrate
import scipy.optimize
import healpy
import grates.utilities
import grates.kernel
import grates.gravityfield
import grates.data


class SurfaceElement(metaclass=abc.ABCMeta):
    """
    Base interface for different shapes of surface tiles.
    """
    @property
    @abc.abstractmethod
    def longitude(self):
        pass

    @property
    @abc.abstractmethod
    def latitude(self):
        pass


class RectangularSurfaceElement(SurfaceElement):
    """
    Rectangular surface element defined by lower left corner coordinates, width and height.

    Parameters
    ----------
    x : float
        longitude of lower left point in radians
    y : float
        latitude of lower left point in radians
    width : float
        width in radians
    height : float
        width in radians
    """
    __slots__ = ['x', 'y', 'width', 'height']

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def longitude(self):
        return np.array((self.x, self.x + self.width, self.x + self.width, self.x))

    @property
    def latitude(self):
        return np.array((self.y, self.y, self.y + self.height, self.y + self.height))


class PolygonSurfaceElement(SurfaceElement):
    """
    Surface element of arbitrary shape, defined by a sequence of grid points.

    Parameters
    ----------
    x : ndarray(vertex_count)
        longitude of polygon vertices in radians
    y : ndarray(vertex_count
        latitude of polygon vertices in radians
    """
    __slots__ = ['xy']

    def __init__(self, x, y):
        self.xy = np.vstack((x, y)).T

    @property
    def longitude(self):
        return self.xy[:, 0]

    @property
    def latitude(self):
        return self.xy[:, 1]


class Grid(metaclass=abc.ABCMeta):
    """
    Base interface for point collections.

    Subclasses must implement a deep copy, getter for radius and colatitude as well as a method which returns
    whether the grid is regular (e.g. equiangular geographic coordinates) or an arbitrary point distribution.
    """

    @abc.abstractmethod
    def copy(self):
        pass

    @property
    @abc.abstractmethod
    def semimajor_axis(self):
        pass

    @property
    @abc.abstractmethod
    def flattening(self):
        pass

    @property
    @abc.abstractmethod
    def longitude(self):
        pass

    @property
    @abc.abstractmethod
    def latitude(self):
        pass

    @property
    @abc.abstractmethod
    def area(self):
        pass

    @abc.abstractmethod
    def values(self):
        pass

    @abc.abstractmethod
    def point_count(self):
        pass

    @property
    def size(self):
        return self.point_count

    @abc.abstractmethod
    def voronoi_cells(self):
        pass

    @property
    def colatitude(self):
        return grates.utilities.colatitude(self.latitude, self.semimajor_axis, self.flattening)

    def is_compatible(self, other):
        """
        Checks whether the point distributions of two grids agree.

        Returns
        -------
        compatible : bool
            longitude and latitude of both grids are numerically equal
        """
        if self.point_count == other.point_count:
            return np.allclose(self.longitude, other.longitude) and np.allclose(self.latitude, other.latitude)

        return False

    def cartesian_coordinates(self):
        """
        Compute and return the grid points in cartesian coordinates.

        Returns
        -------
        cartesian_coordinates : ndarray(point_count, 3)
            ndarray containing the cartesian coordinates of the grid points (x, y, z).
        """
        return geodetic2cartesian(self.longitude, self.latitude, h=0, a=self.semimajor_axis, f=self.flattening)

    def mean(self, mask=None):
        """
        Compute the weighted average of grid points, potentially with a mask. The individual points are weighted
        by their area elements.

        Parameters
        ----------
        mask : array_like(point_count), None
            boolean array with the same shape as the value array. If None, all points are averaged.

        Returns
        -------
        mean : float
            weighted mean over all grid points in mask

        See Also
        --------
        grates.grid.GeographicGrid.create_mask : member function which creates masks from polygons

        """
        if mask is None:
            mask = np.ones(self.point_count, dtype=bool)

        areas = self.area
        if areas is not None:
            return np.sum(areas[mask] * self.values[mask]) / np.sum(areas[mask])
        else:
            return np.mean(self.values[mask])

    def rms(self, mask=None):
        """
        Compute the weighted RMS of grid points, potentially with a mask. The individual points are weighted
        by their area elements.

        Parameters
        ----------
        mask : array_like(point_count), None
            boolean array with the same shape as the value array. If None, all points are averaged.

        Returns
        -------
        rms : float
            weighted rms over all grid points in mask

        See Also
        --------
        grates.grid.GeographicGrid.create_mask : member function which creates masks from polygons

        """
        if mask is None:
            mask = np.ones(self.point_count, dtype=bool)

        areas = self.area
        if areas is not None:
            return np.sqrt(np.sum(areas[mask] * self.values[mask]**2) / np.sum(areas[mask]))
        else:
            return np.sqrt(np.mean(self.values[mask]**2))

    def std(self, mask=None):
        """
        Compute the weighted standard deviation of grid points, potentially with a mask. The individual points are weighted
        by their area elements.

        Parameters
        ----------
        mask : array_like(point_count), None
            boolean array with the same shape as the value array. If None, all points are averaged.

        Returns
        -------
        std : float
            weighted standard deviation over all grid points in mask

        See Also
        --------
        grates.grid.GeographicGrid.create_mask : member function which creates masks from polygons

        """
        if mask is None:
            mask = np.ones(self.point_count, dtype=bool)

        values = self.values[mask] - self.mean(mask)
        areas = self.area
        if areas is not None:
            return np.sqrt(np.sum(areas[mask] * values**2) / np.sum(areas[mask]))
        else:
            return np.sqrt(np.mean(values**2))

    def create_mask(self, basin, buffer=None):
        """
        Create a mask (boolean array) for the Geographic grid instance based on a polygon.

        Parameters
        ----------
        basin : Basin
            Basin instance or list of polygons (2d arrays with longitude and latitude in columns 0 and 1).
        buffer : float
            buffer around the basin polygons in meters (default: no buffer)

        Returns
        -------
        mask : array_like(m,n)
            boolean array of size(nlons, nlats), True for points inside the polygon, False for points outside.
        """
        if isinstance(basin, grates.grid.Basin):
            return basin.contains_points(self.longitude, self.latitude, buffer)
        else:
            return grates.grid.Basin(basin).contains_points(self.longitude, self.latitude, buffer)

    def distance_matrix(self):
        """
        Compute the spherical distance between all grid points.

        Returns
        -------
        psi : ndarray(m, m)
            spherical distance between all m grid points in radians
        """
        point_count = self.point_count
        psi = np.empty((point_count, point_count))

        lons, lats = self.longitude, self.latitude

        for k in range(point_count):
            psi[k, k:] = spherical_distance(lons[k], lats[k], lons[k:], lats[k:], r=1)
            psi[k + 1:, k] = psi[k, k + 1:]

        return psi

    def subset(self, mask):
        """
        Subset grid based on basin polygons.

        Parameters
        ----------
        mask : array_like(point_count), None
            boolean array with the same shape as the value array. If None, all points are averaged.

        Returns
        -------
        grid : IrregularGrid instance
            subset of grid points as IrregularGrid instance
        """
        lons, lats, areas = self.longitude, self.latitude, self.area

        remaining_longitude = lons[mask]
        remaining_latitude = lats[mask]
        remaining_areas = areas[mask] if areas is not None else None

        grid = IrregularGrid(remaining_longitude, remaining_latitude, remaining_areas, self.semimajor_axis, self.flattening)
        try:
            return grid.to_regular()
        except ValueError:
            return grid

    def nn_index(self, lon, lat):
        """
        Compute the nearest grid point of each point in the sample (lon, lat) and return it as as list of
        indices for each grid point. The nearest neighbour is computed based on the 3D euclidean distance.

        Parameters
        ----------
        lon: ndarray(m,)
            longitude of sample points in radians
        lat: ndarray(,m)
            latitude of sample points in radians

        Returns
        -------
        index : list of index arrays
            indices of sample points for each grid point (index[k] contains the indices of all points in the sample
            to which the k-th point is the nearest neighbour)
        """
        tree = scipy.spatial.cKDTree(self.cartesian_coordinates())
        sample_coordinates = IrregularGrid(lon, lat, a=self.semimajor_axis, f=self.flattening).cartesian_coordinates()

        _, index_3d = tree.query(sample_coordinates[::1, :])

        point_index = [None] * self.point_count
        for k in range(len(point_index)):
            point_index[k] = np.nonzero(k == index_3d)[0]

        return point_index

    def point_neighbours(self, level=1):
        """
        Return the indices of the neighbours of each point as a list of arrays.

        Parameters
        ----------
        level : int
            level of neighbourhood (level 1 only returns the direct neighbours, level 2 additionally the neighbours of the level 1 neighbours and so on)

        Returns
        -------
        neighbours : list of index arrays
            indices of neighbours for each grid point (neighbours[k] contains the indices of all neighbours of the k-th point)
        """
        X = self.cartesian_coordinates()
        hull = scipy.spatial.ConvexHull(X)

        neighbours = [set() for _ in range(self.size)]
        for simplex in hull.simplices:
            neighbours[simplex[0]].update(simplex[1:])
            neighbours[simplex[1]].update(simplex[0::2])
            neighbours[simplex[2]].update(simplex[0:2])

        for _ in range(1, level):
            new_neighbours = [set() for _ in range(self.size)]
            for k in range(len(neighbours)):
                for ni in neighbours[k]:
                    new_neighbours[k].add(ni)
                    new_neighbours[k].update(neighbours[ni])
            neighbours = new_neighbours

        neighbours_sorted = [None for _ in range(self.size)]

        lons = self.longitude
        lats = self.latitude
        for k in range(self.size):
            unsorted_indices = tuple(neighbours[k])

            d = X[unsorted_indices, :] - X[k, :]

            R = np.vstack( ( (-np.sin(lons[k]), np.cos(lons[k]), 0),
                             (-np.sin(lats[k]) * np.cos(lons[k]), -np.sin(lats[k]) * np.sin(lons[k]), np.cos(lats[k]) ) ) )

            xy = R @ d.T

            idx = np.lexsort((xy[0, :], xy[1, :]))
            neighbours_sorted[k] = np.array(unsorted_indices)[idx]

        return neighbours_sorted

    @abc.abstractmethod
    def synthesis_matrix_per_order(self, m, min_degree, max_degree, kernel, GM, R):
        pass

    def synthesis_matrix(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Generates the linear operator which transform spherical harmonic coefficients into gridded values.

        Parameters
        ----------
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the output functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        A : ndarray(m, n)
            matrix that relates n spherical harmonic coefficients to m grid points
        """
        target_sequence = grates.gravityfield.CoefficientSequenceDegreeWise(min_degree, max_degree)

        A = np.empty((self.point_count, target_sequence.coefficient_count))

        A[:, target_sequence.vector_indices(order=0)] = self.synthesis_matrix_per_order(0, min_degree, max_degree, kernel, GM, R)
        for m in range(1, max_degree + 1):
            idx = np.concatenate((target_sequence.vector_indices(order=m, cs='c'), target_sequence.vector_indices(order=m, cs='s')))
            A[:, idx] = np.hstack(self.synthesis_matrix_per_order(m, min_degree, max_degree, kernel, GM, R))

        return A

    @abc.abstractmethod
    def analysis_matrix(self, min_degree, max_degree, kernel, GM, R):
        pass

    def window_matrix(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Create a window matrix for spherical harmonic coefficients. The grid values are interpreted as a window function and
        should be in the range [0, 1].

        Parameters
        ----------
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the output functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        W : ndarray(n, n)
            matrix that windows n spherical harmonic coefficients in degreewise order
        """
        A = self.analysis_matrix(min_degree, max_degree, kernel, GM, R)
        A *= self.values

        return A @ self.synthesis_matrix(min_degree, max_degree, kernel, GM, R)

    def to_potential_coefficients(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Perform spherical harmonic analysis of the grid values.

        Parameters
        ----------
        min_degree : int
            minimum degree of the analysis
        max_degree : int
            maximum degree of the analysis
        kernel : str
            name of the grid value kernel
        GM : float
            geocentric gravitational constant
        R : reference radius

        Returns
        -------
        potential_coefficients : PotentialCoefficients
            result of the spherical harmonic analysis as potential coefficients
        """
        if self.values is None:
            raise ValueError('grid has no values to propagate to potential coefficients')

        A = self.analysis_matrix(min_degree, max_degree, kernel, GM, R)
        x = A @ self.values

        coeffs = grates.gravityfield.PotentialCoefficients(GM, R)
        coeffs.anm = grates.utilities.unravel_coefficients(x, min_degree, max_degree)

        return coeffs


class RegularGrid(Grid):
    """
    Base class for regular, global point distributions on the ellipsoid, for example a geographic grid. The points of a
    regular grid are characterized by the location of parallel circles and meridians rather than longitude/latitude
    pairs.

    Parameters
    ----------
    meridians : ndarray(n,)
        longitude of meridians in radians
    parallels : ndarray(m,)
        latitude of parallel circles in radians
    area_elements : None or ndarray(m, n)
        area element of each grid point
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    def __init__(self, meridians, parallels, area_elements=None, a=6378137.0, f=298.2572221010**-1):

        self.parallels = parallels
        self.meridians = meridians

        self.__a = a
        self.__f = f

        lon_edges = np.concatenate(([-np.pi], self.meridians[0:-1] + 0.5 * np.diff(self.meridians), [np.pi]))
        lat_edges = np.concatenate(([0.5 * np.pi], self.parallels[0:-1] + 0.5 * np.diff(self.parallels), [-0.5 * np.pi]))

        self.__areas = 2.0 * (np.sin(np.abs(np.diff(lat_edges)) * 0.5) * np.cos(self.parallels))[:, np.newaxis] * np.diff(lon_edges) if area_elements is None else area_elements
        self.value_array = None
        self.epoch = None

    def copy(self):

        grid = RegularGrid(self.meridians.copy(), self.parallels.copy(), self.__areas.copy(), self.semimajor_axis, self.flattening)
        if self.value_array is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid

    def to_regular(self, threshold=1e-6):
        """
        Try to coerce the grid into a regular sampling given by meridians and parallels.

        Parameters
        ----------
        threshold : float
            distance in which two points are considered equal, given in meters on the sphere

        Returns
        -------
        grid : RegularGrid
            regular representation of the irregular grid

        Raises
        ------
        ValueError:
            if the grid cannot be represented by parallels and meridians
        """
        if threshold <= 0:
            raise ValueError('threshold should be positive (got {0:e})'.format(threshold))

        return self.copy()

    @property
    def semimajor_axis(self):
        return self.__a

    @property
    def flattening(self):
        return self.__f

    @property
    def point_count(self):
        return self.parallels.size * self.meridians.size

    @property
    def longitude(self):
        lon = np.empty(self.parallels.size * self.meridians.size)
        for k in range(self.parallels.size):
            lon[k * self.meridians.size: (k + 1) * self.meridians.size] = self.meridians

        return lon

    @property
    def latitude(self):
        lat = np.empty(self.parallels.size * self.meridians.size)
        for k in range(self.parallels.size):
            lat[k * self.meridians.size: (k + 1) * self.meridians.size] = self.parallels[k]

        return lat

    @property
    def area(self):
        return self.__areas.ravel()

    @property
    def values(self):
        if self.value_array is not None:
            return self.value_array.ravel()

    @values.setter
    def values(self, val):
        if val is None:
            self.value_array = None
        elif isinstance(val, np.ndarray):
            if val.ndim > 1:
                raise ValueError("unable to assign values of dimension {0:d} to grid".format(val.ndim))
            if val.size != self.point_count:
                raise ValueError("unable to assign values of size {0:d} to grid with {1:d} points".format(val.size, self.point_count))
            self.value_array = np.reshape(val, (self.parallels.size, self.meridians.size))
        else:
            raise ValueError("grid values must be either None or " + str(np.ndarray))

    def synthesis_matrix_per_order(self, m, min_degree, max_degree, kernel, GM, R):
        """
        Generates the linear operator which transform spherical harmonic coefficients of order m and trigonometric function basis_function into gridded values.

        Parameters
        ----------
        m : int
            order for which the synthesis matrix should be assembled
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the output functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        A : ndarray(p, n), tuple of ndarray(p, n)
            matrix that relates n spherical harmonic coefficients of order m and to p grid points (for orders > 0 a tuple
            where the first element is the matrix corresponding to the cosine coefficients and the second element corresponds to
            the sine coefficients)
        """
        colat = grates.utilities.colatitude(self.parallels, self.semimajor_axis, self.flattening)
        r = grates.utilities.geocentric_radius(self.parallels, self.semimajor_axis, self.flattening)

        grid_kernel = grates.kernel.get_kernel(kernel)
        kn = grid_kernel.inverse_coefficients(0, max_degree, r, colat) * np.power((R / r)[:, np.newaxis], np.arange(max_degree + 1, dtype=int) + 1) * GM / R

        Pnm = (grates.utilities.legendre_functions_per_order(max_degree, m, colat) * kn[:, m:])[:, max(min_degree - m, 0):]
        if m == 0:
            return np.vstack([np.tile(p, (self.meridians.size, 1)) for p in Pnm])
        else:
            return np.vstack([p * np.cos(m * self.meridians[:, np.newaxis]) for p in Pnm]), np.vstack([p * np.sin(m * self.meridians[:, np.newaxis]) for p in Pnm])

    def __analysis_matrix_per_order(self, m, min_degree, max_degree, kernel, GM, R):
        """
        Generates the linear operator which converts gridded values into spherical harmonic coefficients of specific order and
        trigonometric basis function.

        Parameters
        ----------
        m : int
            order for which the synthesis matrix should be assembled
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the input functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        F : ndarray(n, m)
            matrix that relates m grid points to n spherical harmonic coefficients
        """
        if m == 0:
            Ak = self.synthesis_matrix_per_order(m, min_degree, max_degree, kernel, GM, R)
            return np.linalg.solve((Ak * self.area[:, np.newaxis]).T @ Ak, (Ak * self.area[:, np.newaxis]).T)
        else:
            Ak_cnm, Ak_snm = self.synthesis_matrix_per_order(m, min_degree, max_degree, kernel, GM, R)
            return (np.linalg.solve((Ak_cnm * self.area[:, np.newaxis]).T @ Ak_cnm, (Ak_cnm * self.area[:, np.newaxis]).T),
                    np.linalg.solve((Ak_snm * self.area[:, np.newaxis]).T @ Ak_snm, (Ak_snm * self.area[:, np.newaxis]).T))

    def analysis_matrix(self, min_degree, max_degree, kernel, GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Generates the linear operator which converts gridded values into spherical harmonic coefficients.
        This function exploits the regular point distribution and determines the coefficients order-by-order.

        Parameters
        ----------
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the input functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        F : ndarray(n, m)
            matrix that relates m grid points to n spherical harmonic coefficients
        """
        target_sequence = grates.gravityfield.CoefficientSequenceDegreeWise(min_degree, max_degree)

        A = np.empty((target_sequence.coefficient_count, self.point_count))

        A[target_sequence.vector_indices(order=0), :] = self.__analysis_matrix_per_order(0, min_degree, max_degree, kernel, GM, R)
        for m in range(1, max_degree + 1):
            idx = np.concatenate((target_sequence.vector_indices(order=m, cs='c'), target_sequence.vector_indices(order=m, cs='s')))
            A[idx, :] = np.vstack(self.__analysis_matrix_per_order(m, min_degree, max_degree, kernel, GM, R))

        return A

    def voronoi_cells(self):
        """
        Compute the global Voronoi diagram of the grid points. For regular grids, the Voronoi cells are assumed
        to be rectangles centered at each grid point.

        Returns
        -------
        cells : list of RectangularSurfaceElement instances
            Voronoi cell for each grid point as RectangularSurfaceElement instance
        """
        lon_edges = np.concatenate(([-np.pi], self.meridians[0:-1] + 0.5 * np.diff(self.meridians), [np.pi]))
        lat_edges = np.concatenate(([0.5 * np.pi], self.parallels[0:-1] + 0.5 * np.diff(self.parallels), [-0.5 * np.pi]))

        cells = []
        for parallel_index in range(self.parallels.size):
            for meridian_index in range(self.meridians.size):
                cells.append(RectangularSurfaceElement(lon_edges[meridian_index], lat_edges[parallel_index + 1], lon_edges[meridian_index + 1] - lon_edges[meridian_index],
                                                       lat_edges[parallel_index] - lat_edges[parallel_index + 1]))
        return cells

    def to_potential_coefficients(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Perform spherical harmonic analysis of the grid values.

        Parameters
        ----------
        min_degree : int
            minimum degree of the analysis
        max_degree : int
            maximum degree of the analysis
        kernel : str
            name of the grid value kernel
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        potential_coefficients : PotentialCoefficients
            result of the spherical harmonic analysis as potential coefficients
        """
        if self.values is None:
            raise ValueError('grid has no values to propagate to potential coefficients')
        anm = np.zeros((max_degree + 1, max_degree + 1))
        values = self.values

        matrix_cnm = self.__analysis_matrix_per_order(0, min_degree, max_degree, kernel, GM, R)
        anm[min_degree:, 0] = matrix_cnm @ values
        for m in range(1, max_degree + 1):
            matrix_cnm, matrix_snm = self.__analysis_matrix_per_order(m, min_degree, max_degree, kernel, GM, R)
            idx_start = max(m, min_degree)
            anm[idx_start:, m] = matrix_cnm @ values
            anm[m - 1, idx_start:] = matrix_snm @ values

        coeffs = grates.gravityfield.PotentialCoefficients(GM, R)
        coeffs.anm = anm

        return coeffs

    def covariance_propagation(self, covariance_matrix, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Propagate a spherical harmonic covariance matrix to gridded values. Only the main diagonal of the grid covariance matrix is preserved.
        This method sets the grid values to the gridded standard deviations.

        Parameters
        ----------
        covariance_matrix : ndarray(m, m)
            2d ndarray representing the spherical harmonic covariance matrix given in degreewise order
        min_degree : int
            minimum degree of the analysis
        max_degree : int
            maximum degree of the analysis
        kernel : str
            name of the grid value kernel
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        standard_deviation : ndarray(n,)
            1d-ndarray containing the standard deviations associated with the grid points
        """
        grid_covariance = np.zeros(self.point_count)

        colat = grates.utilities.colatitude(self.parallels, self.semimajor_axis, self.flattening)
        r = grates.utilities.geocentric_radius(self.parallels, self.semimajor_axis, self.flattening)

        grid_kernel = grates.kernel.get_kernel(kernel)
        kn = grid_kernel.inverse_coefficients(0, max_degree, r, colat) * np.power((R / r)[:, np.newaxis], np.arange(max_degree + 1, dtype=int) + 1) * GM / R

        Pnm = grates.utilities.legendre_functions(max_degree, colat)
        Pnm[:, :, 0] *= kn
        for m in range(1, max_degree + 1):
            Pnm[:, m:, m] *= kn[:, m:]
            Pnm[:, m - 1, m:] *= kn[:, m:]
        Pnm = grates.utilities.ravel_coefficients(Pnm, min_degree, max_degree)
        cs = grates.utilities.ravel_coefficients(grates.utilities.trigonometric_functions(max_degree, self.meridians), min_degree, max_degree)

        for k in range(self.parallels.size):
            F = cs * Pnm[k:k + 1, :]
            grid_covariance[k * self.meridians.size:(k + 1) * self.meridians.size] = np.diag(F @ covariance_matrix @ F.T)

        self.values = np.sqrt(grid_covariance)

        return np.sqrt(grid_covariance)


class IrregularGrid(Grid):
    """
    Base class for irregular point distributions on the ellipsoid. The points of an irregular grid are characterized
    by longitude/latitude pairs which cannot be represented by just parallels and meridians.
    """
    def __init__(self, longitude, latitude, area_element=None, a=6378137.0, f=298.2572221010**-1):

        self.__lons = longitude
        self.__lats = latitude
        self.__areas = np.full(self.__lons.size, 4 * np.pi / self.__lons.size) if area_element is None else area_element
        self.__a = a
        self.__f = f

        self.__values = None
        self.epoch = None

    def copy(self):

        grid = IrregularGrid(self.__lons.copy(), self.__lats.copy(), self.__areas.copy(), self.semimajor_axis, self.flattening)
        if self.__values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid

    def to_regular(self, threshold=1e-6):
        """
        Try to coerce the grid into a regular sampling given by meridians and parallels.

        Parameters
        ----------
        threshold : float
            distance in which two points are considered equal, given in meters on the ellipsoid

        Returns
        -------
        grid : RegularGrid
            regular representation of the irregular grid

        Raises
        ------
        ValueError:
            if the grid cannot be represented by parallels and meridians
        """
        if threshold <= 0:
            raise ValueError('threshold should be positive (got {0:e})'.format(threshold))

        threshold /= self.semimajor_axis

        sorted_longitude = np.sort(self.longitude)
        meridians = []
        search_idx = 0
        while search_idx < sorted_longitude.size and len(meridians) < self.point_count:
            meridians.append(sorted_longitude[search_idx])
            search_idx += np.searchsorted(sorted_longitude[search_idx + 1:], sorted_longitude[search_idx] + threshold) + 1

        sorted_latitude = np.sort(self.latitude)
        parallels = []
        search_idx = 0
        while search_idx < sorted_latitude.size and len(parallels) < self.point_count:
            parallels.append(sorted_latitude[search_idx])
            search_idx += np.searchsorted(sorted_latitude[search_idx + 1:], sorted_latitude[search_idx] + threshold) + 1

        if len(meridians) * len(parallels) != self.point_count:
            raise ValueError('grid cannot be coerced to a regular sampling')

        grid = RegularGrid(np.array(meridians), np.array(parallels[::-1]), a=self.semimajor_axis, f=self.flattening)
        if self.values is not None:
            tree = scipy.spatial.cKDTree(np.vstack((self.longitude, self.latitude)).T)
            _, index = tree.query(np.vstack((grid.longitude, grid.latitude)).T)
            grid.values = self.values[index]

        return grid

    @property
    def semimajor_axis(self):
        return self.__a

    @property
    def flattening(self):
        return self.__f

    @property
    def longitude(self):
        return self.__lons

    @property
    def latitude(self):
        return self.__lats

    @property
    def area(self):
        return self.__areas

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, val):
        if val is None:
            self.__values = None
        elif isinstance(val, np.ndarray):
            if val.ndim > 1:
                raise ValueError("unable to assign values of dimension {0:d} to grid".format(val.ndim))
            if val.size != self.point_count:
                raise ValueError("unable to assign values of size {0:d} to grid with {1:d} points".format(val.size, self.point_count))
            self.__values = val
        else:
            raise ValueError("grid values must be either None or " + str(np.ndarray))

    @property
    def point_count(self):
        return self.__lons.size

    def synthesis_matrix_per_order(self, m, min_degree, max_degree, kernel, GM, R):
        """
        Generates the linear operator which transform spherical harmonic coefficients of order m and trigonometric function basis_function into gridded values.

        Parameters
        ----------
        m : int
            order for which the synthesis matrix should be assembled
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the output functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        A : ndarray(p, n)
            matrix that relates spherical harmonic coefficients  of order m and trigonometric function basis_function to p grid points
        """
        colat = grates.utilities.colatitude(self.latitude, self.semimajor_axis, self.flattening)
        r = grates.utilities.geocentric_radius(self.latitude, self.semimajor_axis, self.flattening)

        grid_kernel = grates.kernel.get_kernel(kernel)
        kn = grid_kernel.inverse_coefficients(0, max_degree, r, colat) * np.power((R / r)[:, np.newaxis], np.arange(max_degree + 1, dtype=int) + 1) * GM / R

        Pnm = (grates.utilities.legendre_functions_per_order(max_degree, m, colat) * kn[:, m:])[:, max(min_degree - m, 0):]
        if m == 0:
            return Pnm
        else:
            return Pnm * np.cos(m * self.longitude[:, np.newaxis]), Pnm * np.sin(m * self.longitude[:, np.newaxis])

    def analysis_matrix(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Spherical harmonic analysis matrix.

        Parameters
        ----------
        min_degree : int
            minimum spherical harmonic degree
        max_degree : int
            maximum spherical harmonic degree
        kernel : str
            name of the kernel which represents the input functional
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        F : ndarray(n, m)
            matrix that relates m grid points to n spherical harmonic coefficients
        """
        A = self.synthesis_matrix(min_degree, max_degree, kernel, GM, R) * np.sqrt(self.area)[:, np.newaxis]

        return np.linalg.solve(A.T @ A, A.T * np.sqrt(self.area))

    def voronoi_cells(self):
        """
        Compute the global spherical Voronoi diagram of the grid points. Before computing the surface tiles,
        the grid points are mapped onto the unit sphere. Then, the resulting polygons are projected onto
        the ellipsoid.

        Returns
        -------
        cells : list of SurfaceElement instances
            Voronoi cell for each grid point as surface element instance
        """
        e = np.sqrt(self.flattening * (2 - self.flattening))
        if self.flattening == 0.0:
            sin_lat = np.sin(self.latitude)
        else:
            q = (1 - e**2) * np.sin(self.latitude) / (1 - e**2 * np.sin(self.latitude)**2) - (1 - e**2) / (2 * e) * np.log((1 - e * np.sin(self.latitude)) / (1 + e * np.sin(self.latitude)))
            q0 = (1 - e**2) / (1 - e**2) - (1 - e**2) / (2 * e) * np.log((1 - e) / (1 + e))

            sin_lat = q / q0

        sin_lat[sin_lat > 1.0] = 1
        sin_lat[sin_lat < -1.0] = -1
        X = np.empty((self.size, 3))
        X[:, 0] = np.cos(self.longitude) * np.sqrt(1 - sin_lat**2)
        X[:, 1] = np.sin(self.longitude) * np.sqrt(1 - sin_lat**2)
        X[:, 2] = sin_lat

        sv = scipy.spatial.SphericalVoronoi(X, radius=1)

        _, theta, vertex_lon = cartesian2spherical(sv.vertices)
        vertex_lat = authalic2geodetic(np.pi * 0.5 - theta, self.flattening)

        cells = []
        for region in sv.regions:
            points = sv.vertices[region]
            central_point = np.mean(points, axis=0)

            e = np.cross(central_point, [0, 0, 1])
            e /= np.sqrt(np.sum(e**2))
            n = np.cross(e, central_point)

            azimuth = np.arctan2((points @ e[:, np.newaxis]).flatten(), (points @ n[:, np.newaxis]).flatten())
            idx = np.argsort(-azimuth)

            lon = vertex_lon[region]
            if np.ptp(lon) > np.pi:
                lon = np.mod(lon, 2 * np.pi)
            lat = vertex_lat[region]

            cells.append(PolygonSurfaceElement(lon[idx], lat[idx]))
        return cells

    def covariance_propagation(self, covariance_matrix, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Propagate a spherical harmonic covariance matrix to gridded values. Only the main diagonal of the grid covariance matrix is preserved.
        This method sets the grid values to the gridded standard deviations.

        Parameters
        ----------
        covariance_matrix : ndarray(m, m)
            2d ndarray representing the spherical harmonic covariance matrix given in degreewise order
        min_degree : int
            minimum degree of the analysis
        max_degree : int
            maximum degree of the analysis
        kernel : str
            name of the grid value kernel
        GM : float
            geocentric gravitational constant
        R : float
            reference radius

        Returns
        -------
        standard_deviation : ndarray(n,)
            1d-ndarray containing the standard deviations associated with the grid points
        """
        grid_covariance = np.zeros(self.point_count)
        blocking_factor = 256
        blocks = [0]
        while blocks[-1] < self.point_count:
            blocks.append(min(blocks[-1] + blocking_factor, self.point_count))

        grid_kernel = grates.kernel.get_kernel(kernel)
        for k in range(len(blocks) - 1):
            colat = grates.utilities.colatitude(self.latitude[blocks[k]:blocks[k + 1]], self.semimajor_axis, self.flattening)
            r = grates.utilities.geocentric_radius(self.latitude[blocks[k]:blocks[k + 1]], self.semimajor_axis, self.flattening)

            kn = grid_kernel.inverse_coefficients(0, max_degree, r, colat) * np.power((R / r)[:, np.newaxis], np.arange(max_degree + 1, dtype=int) + 1) * GM / R

            Ynm = grates.utilities.spherical_harmonics(max_degree, colat, self.longitude[blocks[k]:blocks[k + 1]])
            Ynm[:, :, 0] *= kn
            for m in range(1, max_degree + 1):
                Ynm[:, m:, m] *= kn[:, m:]
                Ynm[:, m - 1, m:] *= kn[:, m:]

            F = grates.utilities.ravel_coefficients(Ynm, min_degree, max_degree)
            grid_covariance[blocks[k]:blocks[k + 1]] = np.diag(F @ covariance_matrix @ F.T)

        self.values = np.sqrt(grid_covariance)

        return np.sqrt(grid_covariance)


class GeographicGrid(RegularGrid):
    """
    Class representation of a global geographic grid defined by step size in longitude and latitude.

    The resulting point coordinates are center points of area elements (pixels). This means that for
    `dlon=dlat=1` the lower left coordinate will be (-179.5, -89.5) and the upper right (179.5, 89.5) degrees.

    Parameters
    ----------
    dlon : float
        longitudinal step size in degrees
    dlat : float
        latitudinal step size in degrees
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    def __init__(self, dlon=0.5, dlat=0.5, a=6378137.0, f=298.2572221010**-1):

        self.__dlon = dlon
        self.__dlat = dlat

        nlons = 360 / self.__dlon
        nlats = 180 / self.__dlat

        meridians = np.linspace(-np.pi + dlon / 180 * np.pi * 0.5, np.pi - dlon / 180 * np.pi * 0.5, int(nlons))
        parallels = -np.linspace(-np.pi * 0.5 + dlat / 180 * np.pi * 0.5, np.pi * 0.5 - dlat / 180 * np.pi * 0.5, int(nlats))
        areas = np.tile(2.0 * dlon / 180 * np.pi * np.sin(dlat * 0.5 / 180 * np.pi) * np.cos(parallels)[:, np.newaxis], (1, meridians.size))

        super(GeographicGrid, self).__init__(meridians, parallels, areas, a, f)

    def copy(self):

        grid = GeographicGrid(self.__dlon, self.__dlat, self.semimajor_axis, self.flattening)
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class GaussGrid(RegularGrid):
    """
    Class representation of a Gaussian grid. The Gaussian grid is similar to an equi-angular geographic grid. The main
    difference between the two point distributions are the locations of the parallel circles. In a Gaussian grid,
    they are located at the roots of a Legendre polynomial of degree parallel_count. The grid is first created
    on the unit sphere and then projected onto the ellipsoid.

    Parameters
    ----------
    parallel_count : int
        number of parallel circles
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    def __init__(self, parallel_count, a=6378137.0, f=298.2572221010**-1):

        zeros, weights, _ = roots_legendre(parallel_count, mu=True)

        dlon = np.pi / parallel_count
        meridians = np.linspace(-np.pi + dlon * 0.5, np.pi - dlon * 0.5, 2 * parallel_count)

        cosine_theta = -zeros
        sine_theta = np.sqrt(1 - cosine_theta**2)

        parallels = np.arctan2(cosine_theta, (1 - f)**2 * sine_theta)

        areas = np.tile(dlon * weights[:, np.newaxis], (1, meridians.size))

        super(GaussGrid, self).__init__(meridians, parallels, areas, a, f)

    def copy(self):
        """Deep copy of a GaussGrid instance."""
        grid = GaussGrid(self.parallels.size, self.semimajor_axis, self.flattening)
        if self.value_array is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class ReuterGrid(IrregularGrid):
    """
    Class representation of a Reuter grid. The grid is created on the unit sphere and then mapped onto the ellipsoidal surface
    according to latitude_mapping.

    Parameters
    ----------
    level : int
        Reuter grid level
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    latitude_mapping : str
        One of ('authalic', 'geocentric', 'conformal'). Method on which the unit sphere is mapped onto the ellipsoid.
    """
    def __init__(self, level, a=6378137.0, f=298.2572221010**-1, latitude_mapping='geocentric'):

        dlat = np.pi / level

        self.__parallels = np.empty(level + 1)
        self.__longitudes = np.empty(self.__parallels.size, dtype=object)

        self.__parallels[0] = 0.5 * np.pi
        self.__longitudes[0] = np.zeros(1)

        for k in range(1, level):

            theta = k * dlat
            self.__parallels[k] = np.pi * 0.5 - theta

            point_count = int(2 * np.pi / np.arccos((np.cos(dlat) - np.cos(theta)**2) / (np.sin(theta)**2)))
            self.__longitudes[k] = np.empty(point_count)
            for i in range(point_count):
                self.__longitudes[k][i] = np.mod((i + 1.5) * 2 * np.pi / point_count + np.pi, 2 * np.pi) - np.pi

        self.__parallels[-1] = -0.5 * np.pi
        self.__longitudes[-1] = np.zeros(1)
        self.__areas = np.empty(self.__parallels.size)
        self.__areas[0] = 2 * np.pi * (1 - np.cos(dlat * 0.5))
        self.__areas[-1] = 2 * np.pi * (1 - np.cos(dlat * 0.5))
        for k in range(1, self.__areas.size - 1):
            self.__areas[k] = 4 * np.pi / self.__longitudes[k].size * np.sin(0.5 * dlat) * np.cos(self.__parallels[k])

        if latitude_mapping.lower() == 'authalic':
            self.__parallels = authalic2geodetic(self.__parallels, f)
        elif latitude_mapping.lower() == 'geocentric':
            self.__parallels = geocentric2geodetic(self.__parallels, f)
        elif latitude_mapping.lower() == 'conformal':
            self.__parallels = conformal2geodetic(self.__parallels, f)
        else:
            raise ValueError('Unknown latitude mapping "{0}".'.format(latitude_mapping))

        lons = []
        lats = []
        areas = []
        for k in range(self.__parallels.size):
            lons.append(self.__longitudes[k])
            lats.append(np.full(self.__longitudes[k].size, self.__parallels[k]))
            areas.append(np.full(self.__longitudes[k].size, self.__areas[k]))

        super(ReuterGrid, self).__init__(np.concatenate(lons), np.concatenate(lats), np.concatenate(areas), a, f)
        self.__level = level

    def copy(self):
        """Deep copy of a ReuterGrid instance."""
        grid = ReuterGrid(self.__level, self.semimajor_axis, self.flattening)
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class GeodesicGrid(IrregularGrid):
    """
    Implementation of a Geodesic grid based on the icosahedron.

    Parameters
    ----------
    level : int
        subdivision level
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    latitude_mapping : str
        One of ('authalic', 'geocentric', 'conformal'). Method on which the unit sphere is mapped onto the ellipsoid.
    """
    def __init__(self, level, a=6378137.0, f=298.2572221010**-1, latitude_mapping='geocentric'):

        ratio = np.pi * 0.5 - np.arccos(
            (np.cos(72 * np.pi / 180) + np.cos(72 * np.pi / 180) * np.cos(72 * np.pi / 180)) / (np.sin(72 * np.pi / 180) * np.sin(72 * np.pi / 180)))

        vertex_lons = np.array([0, 0, 72, 144, 216, 288, 36, 108, 180, 252, 324, 0]) * np.pi / 180
        vertex_lats = np.empty(vertex_lons.size)
        vertex_lats[0:6] = ratio
        vertex_lats[6:] = -ratio
        vertex_lats[0] = 0.5 * np.pi
        vertex_lats[-1] = - 0.5 * np.pi

        vertices = np.vstack((np.cos(vertex_lons) * np.cos(vertex_lats), np.sin(vertex_lons) * np.cos(vertex_lats), np.sin(vertex_lats))).T

        points_cartesian = [np.array(p) / np.sqrt(np.sum(np.asarray(p)**2)) for p in vertices]

        triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [2, 1, 6], [3, 2, 7], [4, 3, 8],
                              [5, 4, 9], [1, 5, 10], [6, 7, 2], [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 6, 1],
                              [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]])

        edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 1],
                          [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 2], [7, 3], [8, 4], [9, 5], [10, 1],
                          [6, 7], [7, 8], [8, 9], [9, 10], [10, 6], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10]])

        def normalize(v):
            return v / np.sqrt(np.sum(v**2))

        def subdivide_edge(p1, p2, level):

            step_angle = np.arccos(np.inner(p1, p2)) / (level + 1)
            vec = normalize(np.cross(np.cross(p1, p2), p1))

            return [np.cos((i + 1) * step_angle) * p1 + np.sin((i + 1) * step_angle) * vec for i in range(level)]

        def subdivide_triangle(p1, p2, p3, level):

            edge12 = subdivide_edge(p1, p2, level)
            edge23 = subdivide_edge(p2, p3, level)
            edge31 = subdivide_edge(p3, p1, level)

            points = []
            for i in range(1, level):
                for k in range(i):

                    e13 = np.cross(edge12[i], edge31[level - 1 - i])
                    e12 = np.cross(edge12[i - 1 - k], edge23[level - i + k])
                    e23 = np.cross(edge23[k], edge31[level - 1 - k])

                    v1 = np.cross(e13, e12)
                    v2 = np.cross(e23, e13)
                    v3 = np.cross(e23, e12)

                    points.append(-normalize(normalize(v1) + normalize(v2) + normalize(v3)))

            return points

        for k in range(edges.shape[0]):
            points_cartesian.extend(subdivide_edge(points_cartesian[edges[k, 0]], points_cartesian[edges[k, 1]], level))

        for k in range(triangles.shape[0]):
            points_cartesian.extend(subdivide_triangle(points_cartesian[triangles[k, 0]],
                                                       points_cartesian[triangles[k, 1]],
                                                       points_cartesian[triangles[k, 2]], level))

        xyz = np.asarray(points_cartesian)
        lons = np.arctan2(xyz[:, 1], xyz[:, 0])
        lats = np.arctan2(xyz[:, 2], np.sqrt(1 - xyz[:, 2] ** 2))

        if latitude_mapping.lower() == 'authalic':
            lats = authalic2geodetic(lats, f)
        elif latitude_mapping.lower() == 'geocentric':
            lats = geocentric2geodetic(lats, f)
        elif latitude_mapping.lower() == 'conformal':
            lats = conformal2geodetic(lats, f)
        else:
            raise ValueError('Unknown latitude mapping "{0}".'.format(latitude_mapping))

        idx = np.lexsort((lons, -lats))
        super(GeodesicGrid, self).__init__(lons[idx], lats[idx], np.full(lats.size, 4 * np.pi / lats.size), a, f)
        self.__level = level

    def copy(self):
        """Deep copy of a GeodesicGrid instance."""
        grid = GeodesicGrid(self.__level, self.semimajor_axis, self.flattening)
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class SpiralGrid(IrregularGrid):
    """
    Implementation of a spiral grid [1]_.

    References
    ----------
    .. [1] Hüttig, C., and Stemmer, K. (2008), The spiral grid: A new approach to discretize the sphere
           and its application to mantle convection, Geochem. Geophys. Geosyst., 9, Q02018, doi:10.1029/2007GC001581.

    Parameters
    ----------
    level : int
        subdivision level
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    latitude_mapping : str
        One of ('authalic', 'geocentric', 'conformal'). Method on which the unit sphere is mapped onto the ellipsoid.
    """
    def __init__(self, resolution, a=6378137.0, f=298.2572221010**-1, latitude_mapping='geocentric'):

        def intfun(a, R, c):
            return R * np.sqrt(1 + c**2 * np.sin(a)**2)

        def optfun(x, sk, R, c):
            return np.abs(sk - scipy.integrate.quad(intfun, 0, x, args=(R, c))[0])

        R = a
        c = R * np.pi / resolution * 2
        S = scipy.integrate.quad(intfun, 0, np.pi, args=(R, c))[0]
        P = np.ceil(S / resolution) + 1
        s = S / P
        point_count = int(P) + 1

        colat = np.empty(point_count)
        colat[0] = 0
        for k, sk in enumerate(np.arange(s, S, s)):
            res = scipy.optimize.minimize_scalar(optfun, args=(sk, R, c))
            colat[k + 1] = res.x
        colat[-1] = np.pi

        lons = np.arctan2(np.sin(c * colat), np.cos(c * colat))
        if latitude_mapping.lower() == 'authalic':
            lats = grates.grid.authalic2geodetic(np.pi * 0.5 - colat, f)
        elif latitude_mapping.lower() == 'geocentric':
            lats = grates.grid.geocentric2geodetic(np.pi * 0.5 - colat, f)
        elif latitude_mapping.lower() == 'conformal':
            lats = grates.grid.conformal2geodetic(np.pi * 0.5 - colat, f)
        else:
            raise ValueError('Unknown latitude mapping "{0}".'.format(latitude_mapping))

        super(SpiralGrid, self).__init__(lons, lats, np.full(lats.size, 4 * np.pi / lats.size), a, f)
        self.__resolution = resolution


class HEALPix(IrregularGrid):
    """
    Class representation of the HEALPix grid [1]_.

    References
    ----------
    .. [1] HEALPIX - a Framework for High Resolution Discretization, and Fast Analysis of Data Distributed on the Sphere.
           By K.M. Górski, Eric Hivon, A.J. Banday, B.D. Wandelt, F.K. Hansen, M. Reinecke, M. Bartelmann, 2005, ApJ 622, 759

    Parameters
    ----------
    nside : int
        NSIDE parameter of the HEALPix sphere
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    latitude_mapping : str
        One of ('authalic', 'geocentric', 'conformal'). Method on which the unit sphere is mapped onto the ellipsoid.
    """
    def __init__(self, nside, a=6378137.0, f=298.2572221010**-1, latitude_mapping='geocentric'):

        point_count = healpy.nside2npix(nside)

        colat, lons = healpy.pix2ang(nside, range(point_count))
        if latitude_mapping.lower() == 'authalic':
            lats = grates.grid.authalic2geodetic(np.pi * 0.5 - colat, f)
        elif latitude_mapping.lower() == 'geocentric':
            lats = grates.grid.geocentric2geodetic(np.pi * 0.5 - colat, f)
        elif latitude_mapping.lower() == 'conformal':
            lats = grates.grid.conformal2geodetic(np.pi * 0.5 - colat, f)
        else:
            raise ValueError('Unknown latitude mapping "{0}".'.format(latitude_mapping))

        super(HEALPix, self).__init__(lons, lats, np.full(lats.size, 4 * np.pi / lats.size), a, f)
        self.__nside = nside


class GreatCircleSegment(IrregularGrid):
    """
    Class representation of points along a great circle. The points are first created along the
    great circle of a sphere and then projected onto the ellipsoid.

    Parameters
    ----------
    central_longitude : float
        longitude of source point in radians
    central_latitude : float
        latitude of source points in radians
    azimuth : float
        azimuth of the great circle in radians
    point_count : int
        number of points of the segment
    max_psi : float
        maximum spherical distance in radians
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    def __init__(self, central_longitude, central_latitude, azimuth, point_count=100, max_psi=np.pi,
                 a=6378137.0, f=298.2572221010**-1):

        psi_array = np.linspace(0, max_psi, point_count)
        theta0 = grates.utilities.colatitude(central_latitude, a, f) - psi_array
        x0 = np.vstack(
            (np.sin(theta0) * np.cos(central_longitude), np.sin(theta0) * np.sin(central_longitude), np.cos(theta0)))

        ux = x0[0, 0]
        uy = x0[1, 0]
        uz = x0[2, 0]

        ca = np.cos(azimuth)
        sa = np.sin(azimuth)

        rotation_matrix = np.array(
            [[ca + ux ** 2 * (1 - ca), ux * uy * (1 - ca) - uz * sa, ux * uz * (1 - ca) + uy * sa],
             [uy * ux * (1 - ca) + uz * sa, ca + uy ** 2 * (1 - ca), uy * uz * (1 - ca) - ux * sa],
             [uz * ux * (1 - ca) - uy * sa, uz * uy * (1 - ca) + ux * sa, ca + uz ** 2 * (1 - ca)]])
        x = rotation_matrix.T @ x0

        lons = np.arctan2(x[1, :], x[0, :])
        lats = np.arctan2(x[2, :], (1 - f) ** 2 * np.sqrt(x[0, :] ** 2 + x[1, :] ** 2))

        super(GreatCircleSegment, self).__init__(lons, lats, None, a, f)
        self.__central_longitude = central_longitude
        self.__central_latitude = central_latitude
        self.__azimuth = azimuth
        self.__nsteps = point_count
        self.__max_psi = max_psi

    def copy(self):
        """Deep copy of a GreatCircleSegment instance."""
        grid = GreatCircleSegment(self.__central_longitude, self.__central_latitude, self.__azimuth, self.__nsteps,
                                  self.__max_psi, self.semimajor_axis, self.flattening)
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class CSRMasconGridRL06(IrregularGrid):
    """
    The grid on which the CSR RL06 mascons are estimated. It is based on a geodesic grid of level 37.
    Voronoi cells are split along coast lines resulting in a total of 42107 points. The grid is given on the WGS84 ellipsoid.
    """
    def __init__(self):

        lon, lat, area, self.__polygon_points, self.__point_to_vertex, self.__polygon_index, self.ocean_mask = grates.data.csr_rl06_mascon_grid()
        super(CSRMasconGridRL06, self).__init__(lon, lat, area, a=6378137.0, f=298.257223563**-1)

    def copy(self):
        """Deep copy of a CSRMasconGridRL06 instance."""
        grid = CSRMasconGridRL06()
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch
        return grid

    def voronoi_cells(self):
        """
        Construct the Voronoi diagram of the grid points.

        Returns
        -------
        cells : list of SurfaceElement instances
            Voronoi cell for each grid point as surface element instance
        """
        vertices = self.__polygon_points[self.__point_to_vertex]

        cells = []
        for k in range(self.__polygon_index.size - 1):
            cells.append(PolygonSurfaceElement(vertices[self.__polygon_index[k]:self.__polygon_index[k + 1], 0], vertices[self.__polygon_index[k]:self.__polygon_index[k + 1], 1]))
        return cells


class JPLMasconGridRL06(ReuterGrid):
    """
    The grid on which the JPL RL06 mascons are estimated. It is based on a Reuter grid of level 60.
    The surface elements are not Voronoi cells but rectangles in geographic coordinates. The grid is given on the sphere.
    """
    def __init__(self):
        super(JPLMasconGridRL06, self).__init__(60, a=6378136.3, f=0)

        dlat = np.pi / self._ReuterGrid__level

        self.__surface_elements = [RectangularSurfaceElement(self._ReuterGrid__longitudes[0][0] - np.pi, self._ReuterGrid__parallels[0] - dlat * 0.5, 2 * np.pi, dlat)]
        for k in range(1, self._ReuterGrid__level):
            point_count = self._ReuterGrid__longitudes[k].size
            for i in range(point_count):
                self.__surface_elements.append(RectangularSurfaceElement(self._ReuterGrid__longitudes[k][i] - np.pi / point_count, self._ReuterGrid__parallels[k] - dlat * 0.5, 2 * np.pi / point_count, dlat))
        self.__surface_elements.append(RectangularSurfaceElement(self._ReuterGrid__longitudes[-1][0] - np.pi, self._ReuterGrid__parallels[-1] - dlat * 0.5, 2 * np.pi, dlat))

    def voronoi_cells(self):
        """
        Construct the Voronoi diagram of the grid points.

        Returns
        -------
        cells : list of SurfaceElement instances
            Voronoi cell for each grid point as surface element instance
        """
        return self.__surface_elements

    def copy(self):
        """Deep copy of a JPLRL06MasconGrid instance."""
        grid = JPLMasconGridRL06()
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch
        return grid


class GSFCMasconGridRL06(IrregularGrid):
    """
    The grid on which the GSFC RL06 mascons are estimated. The surface elements are not Voronoi cells but rectangles in geographic coordinates.
    The grid is given on the sphere.
    """
    def __init__(self):

        lon, lat, area, mascon_width, mascon_height = grates.data.gsfc_rl06_mascon_grid()
        super(GSFCMasconGridRL06, self).__init__(lon, lat, area, a=6378136.3, f=0)

        lower_coordinate = lat - mascon_height * 0.5
        upper_coordinate = lat + mascon_height * 0.5
        mascon_height[lower_coordinate < -np.pi * 0.5] *= 0.5
        mascon_height[upper_coordinate > np.pi * 0.5] *= 0.5
        self.__surface_elements = []
        for k in range(lon.size):
            self.__surface_elements.append(RectangularSurfaceElement(lon[k] - mascon_width[k] * 0.5, lat[k] - mascon_height[k] * 0.5, mascon_width[k], mascon_height[k]))

    def voronoi_cells(self):
        """
        Construct the Voronoi diagram of the grid points.

        Returns
        -------
        cells : list of SurfaceElement instances
            Voronoi cell for each grid point as surface element instance
        """
        return self.__surface_elements

    def copy(self):
        """Deep copy of a GSFCMasconGridRL06 instance."""
        grid = GSFCMasconGridRL06()
        if self.values is not None:
            grid.values = self.values.copy()
        grid.epoch = self.epoch
        return grid


class Basin:
    """
    Simple class representation of an area enclosed by a polygon boundary, potentially with holes. No sanity checking
    for potential geometry errors is performed. Polygon edges are treated as great circle segments.

    Parameters
    ----------
    polygons : ndarray(k, 2) or  list of ndarray(k, 2)
        Coordinates defining the basin. Can be either a single two-column ndarray with longitude/latitude pairs for
        rows, or a list of ndarrays in the same format. Longitude/latitude should be given in radians.
    """
    def __init__(self, polygons):

        if isinstance(polygons, np.ndarray):
            self.__polygons = polygons,
        else:
            self.__polygons = polygons

    def bounding_box(self):
        """
        Returns the bound box (min_lon, min_lat, max_lon, max_lat) of the basin.

        Returns
        -------
        lon_min: float
            minimum longitude in radians
        lat_min: float
            minimum latitude in radians
        lon_max: float
            maximum longitude in radians
        lat_max: float
            maximum latitude in radians
        """
        lons = np.concatenate([p[:, 0] for p in self.__polygons])
        lats = np.concatenate([p[:, 1] for p in self.__polygons])

        return np.min(lons), np.min(lats), np.max(lons), np.max(lats)

    def contains_points(self, lon, lat, buffer=None):
        """
        Method to check whether points are within the basin bounds.

        Parameters
        ----------
        lon : float, ndarray(m,), ndarray(m,n)
            longitude of points to be tested (should be given in radians)
        lat : float, ndarray(m,), ndarray(m,n)
            latitude of points to be tested (should be given in radians)
        buffer : float
            buffer around basin polygons in meters (default: no buffer)

        Returns
        -------
        contains_ponts : ndarray of bools (shape depends on input)
            boolean array indicating whether the passed points are in the basin bounds
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        count = np.zeros(lon.shape if lat.size == 1 else lat.shape, dtype=int)
        for polygon in self.__polygons:
            count += spherical_pip(polygon, lon, lat)
        is_inside_polygon = np.mod(count, 2).astype(bool)

        if buffer is not None:
            is_inside_buffer = np.zeros(count.shape, dtype=bool)
            for polygon in self.__polygons:
                is_inside_buffer = np.logical_or(is_inside_buffer, spherical_pib(polygon, lon, lat, np.abs(buffer)))
            is_inside_polygon[is_inside_buffer] = buffer > 0

        return is_inside_polygon

    @staticmethod
    def from_extent(lon_min, lat_min, lon_max, lat_max):
        """
        Create basin polygon from  lower left and upper right points.

        Parameters
        ----------

        lon_min: float
            minimum longitude in radians
        lat_min: float
            minimum latitude in radians
        lon_max: float
            maximum longitude in radians
        lat_max: float
            maximum latitude in radians
        """
        poly = np.empty((4, 2))
        poly[0, :] = (lon_min, lat_min)
        poly[1, :] = (lon_min, lat_max)
        poly[2, :] = (lon_max, lat_max)
        poly[3, :] = (lon_max, lat_min)

        return Basin(poly)


def winding_number(polygon, x, y):
    """
    Winding number algorithm for point in polygon tests.

    Parameters
    ----------
    polygon : ndarray(k, 2)
        two-column ndarray with longitude/latitude pairs defining the polygon
    x : ndarray(m,), ndarray(m,n)
        x-coordinates of points to be tested
    y : ndarray(m,), ndarray(m,n)
        y-coordinates of points to be tested

    Returns
    -------
    contains : ndarray(m,), ndarray(m,n)
        boolean array indicating which point is contained in the polygon
    """
    coords = polygon
    if np.any(polygon[0] != polygon[-1]):
        coords = np.append(polygon, polygon[0][np.newaxis, :], axis=0)

    wn = np.zeros(x.shape if y.size == 1 else y.shape, dtype=int)

    for p0, p1 in zip(coords[0:-1], coords[1:]):
        l1 = p0[1] <= y
        l2 = p1[1] > y

        loc_to_edge = (p1[0] - p0[0]) * (y - p0[1]) - (x - p0[0]) * (p1[1] - p0[1])

        wn[np.logical_and(np.logical_and(l1, l2), loc_to_edge > 0)] += 1
        wn[np.logical_and(np.logical_and(~l1, ~l2), loc_to_edge < 0)] -= 1

    return wn != 0


def spherical_pip(polygon, lon, lat, a=6378137.0, f=298.2572221010**-1):
    """
    Point-in-polygon test for geographic coordinates. Both polygon vertices and test points are projected onto the
    unit sphere before evaluation. Polygon edges are treated as great circle segments.

    The algorithm computes great circle intersections between the polygon edges and great circle segments from
    a point known to be outside of the polygon to the evaluation points. We chose the antipode of the (cartesian)
    barycentrum of the polygon vertices to be outside the polygon. Since "inside" and "outside" are not uniquely defined
    on the sphere, this implicitly requires that polygons are confined to one hemisphere (relative to their
    barycentrum).

    To speed up computation, first all points outside an enclosing spherical cap are discarded.

    Parameters
    ----------
    polygon : ndarray(k, 2)
        two-column ndarray with longitude/latitude pairs in radians defining the polygon
    lon : ndarray(m,)
        longitude of points to be tested in radians
    lat : ndarray(m,)
        latitude of points to be tested in radians
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid

    Returns
    -------
    contains : ndarray(m,)
        boolean array indicating which point is contained in the polygon
    """
    cartesian_coords = geodetic2cartesian(polygon[:, 0], polygon[:, 1], h=0, a=a, f=f)
    cartesian_coords /= np.sqrt(np.sum(cartesian_coords**2, axis=1))[:, np.newaxis]

    antipode = -np.mean(cartesian_coords, axis=0)
    antipode /= np.sqrt(np.sum(antipode**2))

    spherical_cap = -cartesian_coords @ antipode[:, np.newaxis]
    min_cos_angle = np.min(spherical_cap, axis=0)

    cartesian_coords = np.append(cartesian_coords, cartesian_coords[0][np.newaxis, :], axis=0)

    xyz = geodetic2cartesian(lon, lat, h=0, a=a, f=f)
    xyz /= np.sqrt(np.sum(xyz**2, axis=1))[:, np.newaxis]

    inside_polygon = (-xyz @ antipode[:, np.newaxis]).flatten() >= min_cos_angle
    p = np.cross(xyz[inside_polygon, :], antipode)
    xyz_cross_p = np.cross(xyz[inside_polygon, :], p)
    antipode_cross_p = np.cross(antipode, p)

    crossing_count = np.zeros(p.shape[0], dtype=int)
    for b0, b1 in zip(cartesian_coords[1:], cartesian_coords[0:-1]):
        q = np.cross(b0, b1)

        t = np.cross(p, q)
        norm_t = np.sqrt(np.sum(t**2, axis=1))
        remaining_points = norm_t > 0
        if not np.any(remaining_points):
            continue
        t[remaining_points, :] /= norm_t[remaining_points, np.newaxis]

        s1 = np.sum(xyz_cross_p * t, axis=1)
        s2 = np.sum(antipode_cross_p * t, axis=1)
        s3 = np.sum(np.cross(b0, q) * t, axis=1)
        s4 = np.sum(np.cross(b1, q) * t, axis=1)

        is_crossing = np.logical_or((np.sign(-s1) + np.sign(s2) + np.sign(-s3) + np.sign(s4)) == -4,
                                    (np.sign(-s1) + np.sign(s2) + np.sign(-s3) + np.sign(s4)) == 4)
        crossing_count[is_crossing] += 1

    mask = inside_polygon.copy()
    mask[inside_polygon] = np.mod(crossing_count, 2).astype(bool)

    return mask


def spherical_pib(polygon, lon, lat, buffer, a=6378137.0, f=298.2572221010**-1):
    """
    Tests whether the points lon/lat are within a certain distance of the polygon edges. Distances are computed
    on the sphere.

    Parameters
    ---------
    polygon : ndarray(k, 2)
        two-column ndarray with longitude/latitude pairs in radians defining the polygon
    lon : ndarray(m,)
        longitude of points to be tested in radians
    lat : ndarray(m,)
        latitude of points to be tested in radians
    buffer : float
        buffer around the polygon edges in meters
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    cartesian_coords = geodetic2cartesian(polygon[:, 0], polygon[:, 1], h=0, a=a, f=f)
    cartesian_coords /= np.sqrt(np.sum(cartesian_coords ** 2, axis=1))[:, np.newaxis]

    antipode = -np.mean(cartesian_coords, axis=0)
    antipode /= np.sqrt(np.sum(antipode ** 2))

    xyz = geodetic2cartesian(lon, lat, h=0, a=a, f=f)
    xyz /= np.sqrt(np.sum(xyz ** 2, axis=1))[:, np.newaxis]

    spherical_cap = -cartesian_coords @ antipode[:, np.newaxis]
    min_cos_angle = np.cos(np.arccos(np.min(spherical_cap, axis=0)) + buffer / a)

    inside_cap = (-xyz @ antipode[:, np.newaxis]).flatten() >= min_cos_angle
    remaining_index = np.where(inside_cap)[0]
    inside_buffer = np.zeros(xyz.shape[0], dtype=bool)

    cartesian_coords = np.append(cartesian_coords, cartesian_coords[0][np.newaxis, :], axis=0)
    for b0, b1 in zip(cartesian_coords[1:], cartesian_coords[0:-1]):

        within_vertex = np.cos(buffer / a) <= (xyz[remaining_index, :] @ b0[:, np.newaxis]).flatten()
        inside_buffer[remaining_index] = within_vertex
        remaining_index = remaining_index[~within_vertex]

        within_vertex = np.cos(buffer / a) <= (xyz[remaining_index, :] @ b1[:, np.newaxis]).flatten()
        inside_buffer[remaining_index] = within_vertex
        remaining_index = remaining_index[~within_vertex]

        n = np.cross(b0, b1)
        norm_n = np.sqrt(np.sum(n ** 2))
        if norm_n == 0.0:
            continue
        n /= norm_n

        s = xyz[remaining_index, :] @ n[:, np.newaxis]
        p = xyz[remaining_index, :] - s * n
        p /= np.sqrt(np.sum(p ** 2, axis=1))[:, np.newaxis]

        within_edge = np.logical_and(np.logical_and(np.inner(np.cross(b0, p), np.cross(b0, b1)) >= 0,
                                     np.inner(np.cross(b1, p), np.cross(b1, b0)) >= 0),
                                     np.cos(buffer / a) <= np.sum(p * xyz[remaining_index, :], axis=1))
        inside_buffer[remaining_index] = within_edge
        remaining_index = remaining_index[~within_edge]

    return inside_buffer


def spherical_distance(lon1, lat1, lon2, lat2, r=6378136.3):
    """
    Compute the spherical distance between points (lon1, lat1) and (lon2, lat2) on a sphere with
    radius r.

    Parameters
    ----------
    lon1 : float, array_like(m,), array_like(m,n)
        longitude of source points in radians
    lat1 : float, array_like(m,), array_like(m,n)
        latitude of source points in radians
    lon2 : float, array_like(m,), array_like(m,n)
        longitude of target points in radians
    lat2 : float, array_like(m,), array_like(m,n)
        latitude of target points in radians
    r : float
        radius of the sphere in meters

    Returns
    -------
    d : ndarray(m,), ndarray(m,n)
        spherical distance between points (lon1, lat1) and (lon2, lat2) in meters
    """
    return np.arctan2(np.sqrt((np.cos(lat2) * np.sin(lon2 - lon1))**2 + (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))**2),
                      np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)) * r


def geodetic2cartesian(lon, lat, h=0, a=6378137.0, f=298.2572221010**-1):
    """
    Compute 3D cartesian coordinates from ellipsoidal (geographic) longitude, latitude and height.

    Parameters
    ----------
    lon : float, ndarray(m,)
        geographic longitude in radians
    lat : float, ndarray(m,)
        geographic latitude in radians
    h : float, ndarray(m,)
        ellipsoidal height in meters (default: 0)
    a : float
        semi-major axis of ellipsoid in meters
    f : float
        flattening of ellipsoid

    Returns
    -------
    xyz : ndarray(m,3(
        3D cartesian coordinages
    """
    if f == 0.0:
        return spherical2cartesian(a + h, np.pi * 0.5 - lat, lon)

    e2 = 2 * f - f ** 2
    radius_of_curvature = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    return np.vstack(((radius_of_curvature + h) * np.cos(lat) * np.cos(lon),
                      (radius_of_curvature + h) * np.cos(lat) * np.sin(lon),
                      ((1 - e2) * radius_of_curvature + h) * np.sin(lat))).T


def cartesian2geodetic(xyz, a=6378137.0, f=298.2572221010**-1, max_iter=10, threshold=1e-6):
    """
    Compute geodetic longitude, latitude and height from 3D cartesian coordinates.
    This function iteratively solves Bowring's irrational geodetic-latitude equation [1]_. It is accurate to
    the micrometer level in the height component.

    References
    ----------

    .. [1] B. R. Bowring (1976) TRANSFORMATION FROM SPATIAL TO GEOGRAPHICAL COORDINATES, Survey Review, 23:181, 323-327,
           DOI: 10.1179/sre.1976.23.181.323

    Parameters
    ----------
    xyz : ndarray(m, 3)
        3D cartesian coordinages
    a : float
        semi-major axis of ellipsoid in meters
    f : float
        flattening of ellipsoid
    max_iter : int
        maximum number of iterations
    threshold : float
        iteration threshold for ellipsoidal height in meters (default: micrometer)

    Returns
    -------
    lon : ndarray(m,)
        geographic longitude in radians
    lat : ndarray(m,)
        geographic latitude in radians
    h : ndarray(m,)
        ellipsoidal height in meters
    """
    if f == 0.0:
        r, colat, lon = cartesian2spherical(xyz)
        return lon, np.pi * 0.5 - colat, r - a

    e2 = 2 * f - f**2

    p2 = xyz[:, 0]**2 + xyz[:, 1]**2

    h0 = 0
    k = (1 - e2)**-1
    for _ in range(max_iter):
        c = np.power(p2 + (1 - e2) * xyz[:, -1]**2 * k**2, 1.5) / (a * e2)
        k = 1 + (p2 + (1 - e2) * xyz[:, -1]**2 * k**3) / (c - p2)
        h = (k**-1 - (1 - e2)) * np.sqrt(p2 + xyz[:, -1]**2 * k**2) / e2
        if np.max(np.abs(h - h0)) < threshold:
            break
        h0 = h

    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arctan2(k * xyz[:, -1], np.sqrt(p2))

    return lon, lat, h


def cartesian2spherical(xyz):
    """
    Convert a cartesion coordinate tripe to spherical coordniates (r, colatitude, longitude).

    Parameters
    ----------
    xyz : ndarray(m, 3)
        cartesian coordinate triple for m points

    Returns
    -------
    r : ndarray(m,)
        geocentric radius
    colatitude : ndarray(m,)
        colatitude (polar angle) in radians
    longitude : ndarray(m,)
        longitude in radians
    """
    r = np.sqrt(np.sum(xyz**2, axis=1))
    colatitude = np.arctan2(np.sqrt(np.sum(xyz[:, 0:2]**2, axis=1)), xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    return r, colatitude, lon


def spherical2cartesian(r, colatitude, lon):
    """Spherical coordinates to cartesian coordinates."""

    xyz = np.empty((lon.size, 3))
    xyz[:, 0] = r * np.sin(colatitude) * np.cos(lon)
    xyz[:, 1] = r * np.sin(colatitude) * np.sin(lon)
    xyz[:, 2] = r * np.cos(colatitude)

    return xyz


def authalic_radius(a=6378137.0, f=298.2572221010**-1):
    """Radius of the authalic sphere corresponding to the ellipsoid (a, f)."""
    e = np.sqrt(f * (2 - f))
    q0 = (1 - e**2) / (1 - e**2) - (1 - e**2) / (2 * e) * np.log((1 - e) / (1 + e))

    return a * np.sqrt(q0 * 0.5)


def geodetic2authalic(latitude, f=298.2572221010**-1):
    """Convert geodetic latitude to authalic latitude."""
    if f == 0.0:
        return latitude

    e = np.sqrt(f * (2 - f))

    q = (1 - e**2) * np.sin(latitude) / (1 - e**2 * np.sin(latitude)**2) - (1 - e**2) / (2 * e) * np.log((1 - e * np.sin(latitude)) / (1 + e * np.sin(latitude)))
    q0 = (1 - e**2) / (1 - e**2) - (1 - e**2) / (2 * e) * np.log((1 - e) / (1 + e))

    return np.arcsin(q / q0)


def authalic2geodetic(beta, f=298.2572221010**-1):
    """Convert authalic latitude to geodetic latitude."""
    e2 = f * (2 - f)

    return beta + (1 / 3 * e2 + 31 / 180 * e2**2 + 517 / 5040 * e2**3 + 120389 / 181400 * e2**4 + 1362254 / 29937600 * e2**5) * np.sin(2 * beta) + \
                  (23 / 360 * e2**2 + 251 / 3780 * e2**3 + 102287 / 1814400 * e2**4 + 450739 / 997920 * e2**5) * np.sin(4 * beta) + \
                  (761 / 45360 * e2**3 + 47561 / 1814400 * e2**4 + 434501 / 14968800 * e2**5) * np.sin(6 * beta) + \
                  (6059 / 1209600 * e2**4 + 625511 / 59875200 * e2**5) * np.sin(8 * beta) + \
                  (48017 / 29937600 * e2**5) * np.sin(10 * beta)


def geocentric2geodetic(beta, f=298.2572221010**-1):
    """Convert geocentric to geodetic latitude."""
    return np.arctan2(np.sin(beta), np.cos(beta) * (1 - f)**2)


def geodetic2geocentric(latitude, f=298.2572221010**-1):
    """Convert geodetic to geocentric latitude."""
    return np.arctan2((1 - f)**2 * np.sin(latitude), np.cos(latitude))


def geodetic2conformal(latitude, f=298.2572221010**-1):
    """Convert geodetic to conformal latitude."""
    e = np.sqrt(f * (2 - f))

    return 2 * np.arctan2(np.sqrt((1 + np.sin(latitude)) * (1 - e * np.sin(latitude))**e), np.sqrt((1 - np.sin(latitude)) * (1 + e * np.sin(latitude))**e)) - np.pi * 0.5


def conformal2geodetic(beta, f=298.2572221010**-1):
    """Convert conformal to geodetic latitude."""
    e = np.sqrt(f * (2 - f))

    return beta + (1 / 2 * e**2 + 5 / 24 * e**4 + 1 / 12 * e**6 + 13 / 360 * e**8) * np.sin(2 * beta) + \
                  (7 / 48 * e**4 + 29 / 240 * e**6 + 811 / 11520 * e**8) * np.sin(4 * beta) + \
                  (7 / 120 * e**6 + 81 / 1120 * e**8) * np.sin(6 * beta) + (4279 / 161280 * e**8) * np.sin(8 * beta)
