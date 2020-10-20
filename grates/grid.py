# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Point distributions on the ellipsoid.
"""

import numpy as np
import abc
import grates.utilities
import grates.kernel
import grates.gravityfield
from scipy.special import roots_legendre
import scipy.spatial


class SurfaceElement(metaclass=abc.ABCMeta):
    """
    Base interface for different shapes of surface tiles.
    """
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


class Grid(metaclass=abc.ABCMeta):
    """
    Base interface for point collections.

    Subclasses must implement a deep copy, getter for radius and colatitude as well as a method which returns
    whether the grid is regular (e.g. equiangular geographic coordinates) or an arbitrary point distribution.
    """
    __slots__ = ['semimajor_axis', 'flattening', 'values']

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def is_regular(self):
        pass

    @abc.abstractmethod
    def longitude(self):
        pass

    @abc.abstractmethod
    def latitude(self):
        pass

    @abc.abstractmethod
    def area(self):
        pass

    @abc.abstractmethod
    def point_count(self):
        pass

    @abc.abstractmethod
    def voronoi_cells(self):
        pass

    def cartesian_coordinates(self):
        """
        Compute and return the grid points in cartesian coordinates.

        Returns
        -------
        cartesian_coordinates : ndarray(point_count, 3)
            ndarray containing the cartesian coordinates of the grid points (x, y, z).
        """
        return ellipsoidal2cartesian(self.longitude(), self.latitude(), h=0, a=self.semimajor_axis(),
                                     f=self.flattening())

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
            mask = np.ones(self.point_count(), dtype=bool)

        areas = self.area()
        if areas is not None:
            return np.sum(areas*self.values[mask])/np.sum(areas)
        else:
            return np.mean(self.values[mask])

    def create_mask(self, basin):
        """
        Create a mask (boolean array) for the Geographic grid instance based on a polygon.

        Parameters
        ----------
        basin : Basin
            Basin instance.

        Returns
        -------
        mask : array_like(m,n)
            boolean array of size(nlons, nlats), True for points inside the polygon, False for points outside.
        """
        return basin.contains_points(self.longitude(), self.latitude())

    def distance_matrix(self):
        """
        Compute the spherical distance between all grid points.

        Returns
        -------
        psi : ndarray(m, m)
            spherical distance between all m grid points in radians
        """
        point_count = self.point_count()
        psi = np.empty((point_count, point_count))

        lons, lats = self.longitude(), self.latitude()

        for k in range(point_count):
            psi[k, k:] = spherical_distance(lons[k], lats[k], lons[k:], lats[k:], r=1)
            psi[k + 1:, k] = psi[k, k + 1:]

        return psi

    def subset(self, mask, invert_mask=False):
        """
        Subset grid based on basin polygons.

        Parameters
        ----------
        mask : array_like(point_count), None
            boolean array with the same shape as the value array. If None, all points are averaged.
        invert_mask : bool
            if True the subset is created from all points outside the basin

        Returns
        -------
        grid : IrregularGrid instance
            subset of grid points as IrregularGrid instance
        """
        lons, lats, areas = self.longitude(), self.latitude(), self.area()

        if invert_mask:
            grid_mask = ~mask
        else:
            grid_mask = mask

        remaining_longitude = lons[grid_mask]
        remaining_latitude = lats[grid_mask]
        remaining_areas = areas[grid_mask] if areas is not None else None

        return IrregularGrid(remaining_longitude, remaining_latitude, remaining_areas,
                             self.semimajor_axis, self.flattening)

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

        point_index = [None] * self.point_count()
        for k in range(len(point_index)):
            point_index[k] = np.nonzero(k == index_3d)[0]

        return point_index


class RegularGrid(Grid):
    """
    Base class for regular, global point distributions on the ellipsoid, for example a geographic grid. The points of a
    regular grid are characterized by the location of parallel circles and meridians rather than longitude/latitude
    pairs.
    """
    __slots__ = ['lons', 'lats', 'areas']

    def copy(self):
        pass

    def is_regular(self):
        return True

    def point_count(self):
        return self.lats.size*self.lons.size

    def longitude(self):
        lon = np.empty(self.lats.size * self.lons.size)
        for k in range(self.lats.size):
            lon[k * self.lons.size: (k + 1) * self.lons.size] = self.lons

        return lon

    def latitude(self):
        lat = np.empty(self.lats.size * self.lons.size)
        for k in range(self.lats.size):
            lat[k * self.lons.size: (k + 1) * self.lons.size] = self.lats[k]

        return lat

    def area(self):
        areas = np.empty(self.lats.size * self.lons.size)
        for k in range(self.lats.size):
            areas[k * self.lons.size: (k + 1) * self.lons.size] = self.areas[k]

        return areas

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
        point_count = self.lons.size * self.lats.size
        coefficient_count = (max_degree + 1) * (max_degree + 1) - min_degree * min_degree

        colat = grates.utilities.colatitude(self.lats, self.semimajor_axis, self.flattening)

        P = grates.utilities.legendre_functions(max_degree, colat)
        grid_kernel = grates.kernel.get_kernel(kernel)

        A = np.empty((point_count, coefficient_count))

        column_index = 0
        r = grates.utilities.geocentric_radius(self.lats, self.semimajor_axis, self.flattening)
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            continuation = np.power(R / r, n + 1)
            kn = grid_kernel.inverse_coefficient(n, r, colat)

            Pnm = P[:, row_idx, col_idx].T * continuation * kn * GM/R

            for k in range(self.lats.size):
                A[k * self.lons.size: (k + 1) * self.lons.size, column_index] = Pnm[0, k]
            column_index += 1

            for m in range(1, n + 1):
                cosml = np.cos(m * self.lons)
                for k in range(self.lats.size):
                    A[k * self.lons.size: (k + 1) * self.lons.size, column_index] = Pnm[m, k] * cosml

                sinml = np.sin(m * self.lons)
                for k in range(self.lats.size):
                    A[k * self.lons.size: (k + 1) * self.lons.size, column_index + 1] = Pnm[m, k] * sinml
                column_index += 2

        return A

    def analysis_matrix(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
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
        orders = np.empty((max_degree + 1)*(max_degree + 1) - min_degree*min_degree)
        is_cosine = np.zeros(orders.size, dtype=bool)
        index = 0
        for n in range(min_degree, max_degree + 1):
            orders[index] = 0
            is_cosine[index] = True
            index += 1
            for m in range(1, n + 1):
                orders[index] = m
                is_cosine[index] = True
                orders[index + 1] = m
                index += 2

        areas = np.empty(self.lats.size*self.lons.size)
        for k in range(self.lats.size):
            areas[k * self.lons.size: (k + 1) * self.lons.size] = self.areas[k]

        A = self.synthesis_matrix(min_degree, max_degree, kernel, GM, R)

        column_index = np.logical_and(orders == 0, is_cosine)
        Ak = A[:, column_index]
        A[:, column_index] = np.linalg.solve((Ak*areas[:, np.newaxis]).T@Ak, (Ak*areas[:, np.newaxis]).T).T
        for m in range(1, max_degree + 1):
            column_index = np.logical_and(orders == m, is_cosine)
            Ak = A[:, column_index]
            A[:, column_index] = np.linalg.solve((Ak*areas[:, np.newaxis]).T @ Ak, (Ak * areas[:, np.newaxis]).T).T

            column_index = np.logical_and(orders == m, ~is_cosine)
            Ak = A[:, column_index]
            A[:, column_index] = np.linalg.solve((Ak*areas[:, np.newaxis]).T @ Ak, (Ak * areas[:, np.newaxis]).T).T

        return A.T

    def voronoi_cells(self):
        """
        Compute the global Voronoi diagram of the grid points. For regular grids, the Voronoi cells are assumed
        to be rectangles centered at each grid point.

        Returns
        -------
        cells : list of RectangularSurfaceElement instances
            Voronoi cell for each grid point as RectangularSurfaceElement instance
        """
        dlon = np.diff(self.lons)
        lon_edges = np.concatenate(([-np.pi], self.lons[0:-1] + 0.5 * dlon, [np.pi]))

        dlat = np.diff(self.lats)
        lat_edges = np.concatenate(([0.5 * np.pi], self.lats[0:-1] + 0.5 * dlat, [-0.5 * np.pi]))

        cells = []
        for l in range(self.lats.size):
            for k in range(self.lons.size):
                cells.append(RectangularSurfaceElement(lon_edges[k], lat_edges[l+1], lon_edges[k + 1] - lon_edges[k],
                                                       lat_edges[l] - lat_edges[l + 1]))
        return cells


class IrregularGrid(Grid):
    """
    Base class for irregular point distributions on the ellipsoid. The points of an irregular grid are characterized
    by longitude/latitude pairs which cannot be represented by just parallels and meridians.
    """
    def __init__(self, longitude, latitude, area=None, a=6378137.0, f=298.2572221010**-1):

        self.lons = longitude
        self.lats = latitude
        self.areas = area
        self.semimajor_axis = a
        self.flattening = f

    def copy(self):
        pass

    def is_regular(self):
        return False

    def longitude(self):
        return self.lons

    def latitude(self):
        return self.lats

    def area(self):
        return self.areas

    def point_count(self):
        return self.lons.size

    def synthesis_matrix(self, min_degree, max_degree, kernel='potential', GM=3.9860044150e+14, R=6.3781363000e+06):
        """
        Spherical harmonic synthesis matrix.

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
        colat = grates.utilities.colatitude(self.lats, self.semimajor_axis, self.flattening)
        r = grates.utilities.geocentric_radius(self.lats, self.semimajor_axis, self.flattening)
        Ynm = grates.utilities.spherical_harmonics(max_degree, colat, self.lons)

        grid_kernel = grates.kernel.get_kernel(kernel)
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            continuation = np.power(R / r, n + 1) * GM / R
            Ynm[:, row_idx, col_idx] *= (continuation * grid_kernel.inverse_coefficient(n, r, colat))[:, np.newaxis]

        return grates.utilities.ravel_coefficients(Ynm, min_degree, max_degree)

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
        colat = grates.utilities.colatitude(self.lats, self.semimajor_axis, self.flattening)
        r = grates.utilities.geocentric_radius(self.lats, self.semimajor_axis, self.flattening)
        Ynm = grates.utilities.spherical_harmonics(max_degree, colat, self.lons)

        grid_kernel = grates.kernel.get_kernel(kernel)
        for n in range(min_degree, max_degree + 1):
            row_idx, col_idx = grates.gravityfield.degree_indices(n)
            continuation = np.power(r / R, n + 1) * R / GM / (4*np.pi) * self.areas
            Ynm[:, row_idx, col_idx] *= (continuation * grid_kernel.coefficient(n, r, colat))[:, np.newaxis]

        return grates.utilities.ravel_coefficients(Ynm, min_degree, max_degree).T

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
        X = self.cartesian_coordinates()
        norm = np.sqrt(np.sum(X**2, axis=1))
        X /= norm[:, np.newaxis]

        sv = scipy.spatial.SphericalVoronoi(X, radius=1)
        vertex_lon = np.arctan2(sv.vertices[:, 1], sv.vertices[:, 0])
        vertex_lat = np.arctan2(sv.vertices[:, 2], (1-self.flattening)**2 * np.sqrt(1 - sv.vertices[:, 2]**2))

        cells = []
        for region in sv.regions:
            points = sv.vertices[region]
            central_point = np.mean(points, axis=0)

            e = np.cross(central_point, [0, 0, 1])
            e /= np.sqrt(np.sum(e**2))
            n = np.cross(e, central_point)

            azimuth = np.arctan2((points@e[:, np.newaxis]).flatten(), (points@n[:, np.newaxis]).flatten())
            idx = np.argsort(-azimuth)

            lon = vertex_lon[region]
            if np.ptp(lon) > np.pi:
                lon = np.mod(lon, 2 * np.pi)
            lat = vertex_lat[region]

            cells.append(PolygonSurfaceElement(lon[idx], lat[idx]))
        return cells


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

        nlons = 360 / dlon
        nlats = 180 / dlat

        self.lons = np.linspace(-np.pi+dlon/180*np.pi * 0.5, np.pi-dlon/180*np.pi*0.5, int(nlons))
        self.lats = -np.linspace(-np.pi*0.5 + dlat/180*np.pi * 0.5, np.pi*0.5 - dlat/180*np.pi*0.5, int(nlats))

        self.semimajor_axis = a
        self.flattening = f
        self.values = np.empty((0, 0))
        self.epoch = None
        self.areas = 2.0*dlon/180*np.pi*np.sin(dlat*0.5/180*np.pi)*np.cos(self.lats)

    def copy(self):
        """Deep copy of GeographicGrid instance."""
        grid = GeographicGrid(a=self.semimajor_axis, f=self.flattening)
        grid.lons = self.lons.copy()
        grid.lats = self.lats.copy()
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

        zeros, weights, mu = roots_legendre(parallel_count, mu=True)

        dlon = np.pi/parallel_count
        self.lons = np.linspace(-np.pi + dlon*0.5, np.pi - dlon*0.5,  2 * parallel_count)

        cosine_theta = -zeros
        sine_theta = np.sqrt(1 - cosine_theta**2)

        self.lats = np.arctan2(cosine_theta, (1-f)**2*sine_theta)
        self.semimajor_axis = a
        self.flattening = f
        self.areas = dlon * weights
        self.epoch = None

    def copy(self):
        """Deep copy of a GaussGrid instance."""
        grid = GaussGrid(self.lats.size, self.semimajor_axis, self.flattening)
        grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class ReuterGrid(IrregularGrid):
    """
    Class representation of a Reuter grid.

    Parameters
    ----------
    level : int
        Reuter grid level
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """
    def __init__(self, level, a=6378137.0, f=298.2572221010**-1):

        dlat = np.pi/level

        self.__parallels = np.empty(level + 1)
        self.__longitudes = np.empty(self.__parallels.size, dtype=object)

        self.__parallels[0] = 0.5 * np.pi
        self.__longitudes[0] = np.zeros(1)

        for k in range(1, level):

            theta = k * dlat
            self.__parallels[k] = np.arctan2(np.cos(theta), (1-f)**2*np.sin(theta))

            point_count = int(2*np.pi/np.arccos((np.cos(dlat)-np.cos(theta)**2)/(np.sin(theta)**2)))
            self.__longitudes[k] = np.empty(point_count)
            for i in range(point_count):
                self.__longitudes[k][i] = np.mod((i+1.5) * 2*np.pi/point_count+np.pi, 2*np.pi)-np.pi

        self.__parallels[-1] = -0.5 * np.pi
        self.__longitudes[-1] = np.zeros(1)
        self.__areas = np.empty(self.__parallels.size)
        self.__areas[0] = 2 * np.pi * (1 - np.cos(dlat * 0.5))
        self.__areas[-1] = 2 * np.pi * (1 - np.cos(dlat * 0.5))
        for k in range(1, self.__areas.size - 1):
            self.__areas[k] = 4 * np.pi / self.__longitudes[k].size * np.sin(0.5 * dlat) * np.cos(self.__parallels[k])

        lons = []
        lats = []
        areas = []
        for k in range(self.__parallels.size):
            lons.append(self.__longitudes[k])
            lats.append(np.full(self.__longitudes[k].size, self.__parallels[k]))
            areas.append(np.full(self.__longitudes[k].size, self.__areas[k]))

        super(ReuterGrid, self).__init__(np.concatenate(lons), np.concatenate(lats), np.concatenate(areas), a, f)

        self.values = np.empty(self.lons.shape)
        self.epoch = None
        self.__level = level

    def copy(self):
        """Deep copy of a ReuterGrid instance."""
        grid = ReuterGrid(self.__level, self.semimajor_axis, self.flattening)
        grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


class GeodesicGrid(IrregularGrid):
    """
    Implementation of a Geodesic grid based on the icosahedron.
    """
    def __init__(self, level, a=6378137.0, f=298.2572221010**-1):

        ratio = np.pi * 0.5 - np.arccos(
            (np.cos(72 * np.pi/180) + np.cos(72 * np.pi/180) * np.cos(72 * np.pi/180)) /
            (np.sin(72 * np.pi/180) * np.sin(72 * np.pi/180)))

        vertex_lons = np.array([0, 0, 72, 144, 216, 288, 36, 108, 180, 252, 324, 0]) * np.pi/180
        vertex_lats = np.empty(vertex_lons.size)
        vertex_lats[0:6] = ratio
        vertex_lats[6:] = -ratio
        vertex_lats[0] = 0.5 * np.pi
        vertex_lats[-1] = - 0.5 * np.pi

        vertices = np.vstack((np.cos(vertex_lons)*np.cos(vertex_lats), np.sin(vertex_lons)*np.cos(vertex_lats),
                              np.sin(vertex_lats))).T

        points_cartesian = [np.array(p)/np.sqrt(np.sum(np.asarray(p)**2)) for p in vertices]

        triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [2, 1, 6], [3, 2, 7], [4, 3, 8],
                              [5, 4, 9], [1, 5, 10], [6, 7, 2], [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 6, 1],
                              [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]])

        edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5], [5, 1],
                          [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [6, 2], [7, 3], [8, 4], [9, 5], [10, 1],
                          [6, 7], [7, 8], [8, 9], [9, 10], [10, 6], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10]])

        def normalize(v):
            return v / np.sqrt(np.sum(v**2))

        def subdivide_edge(p1, p2, level):

            step_angle = np.arccos(np.inner(p1, p2))/(level + 1)
            vec = normalize(np.cross(np.cross(p1, p2), p1))

            return [np.cos((i + 1) * step_angle) * p1 + np.sin((i + 1)*step_angle) * vec for i in range(level)]

        def subdivide_triangle(p1, p2, p3, level):

            edge12 = subdivide_edge(p1, p2, level)
            edge23 = subdivide_edge(p2, p3, level)
            edge31 = subdivide_edge(p3, p1, level)

            points = []
            for i in range(1, level):
                for k in range(i):

                    e13 = np.cross(edge12[i], edge31[level-1-i])
                    e12 = np.cross(edge12[i-1-k], edge23[level-i+k])
                    e23 = np.cross(edge23[k], edge31[level-1-k])

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
        lats = np.arctan2(xyz[:, 2], (1 - f) ** 2 * np.sqrt(1 - xyz[:, 2] ** 2))

        super(GeodesicGrid, self).__init__(lons, lats, np.full(lats.size, 4 * np.pi / lats.size), a, f)

        self.values = np.empty(self.lons.shape)
        self.epoch = None
        self.__level = level

    def copy(self):
        """Deep copy of a GeodesicGrid instance."""
        grid = GeodesicGrid(self.__level, self.semimajor_axis, self.flattening)
        grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid


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

        self.values = np.empty(self.lons.shape)
        self.epoch = None
        self.__central_longitude = central_longitude
        self.__central_latitude = central_latitude
        self.__azimuth = azimuth
        self.__nsteps = point_count
        self.__max_psi = max_psi

    def copy(self):
        """Deep copy of a GreatCircleSegment instance."""
        grid = GreatCircleSegment(self.__central_longitude, self.__central_latitude, self.__azimuth, self.__nsteps,
                                  self.__max_psi, self.semimajor_axis, self.flattening)
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

        is_inside = np.zeros(lon.shape if lat.size == 1 else lat.shape, dtype=bool)

        if buffer is not None:
            for polygon in self.__polygons:
                is_inside = np.logical_or(is_inside, spherical_pib(polygon, lon, lat, buffer))

        count = np.zeros(is_inside.shape, dtype=int)
        for polygon in self.__polygons:
            count += spherical_pip(polygon, lon, lat)

        return np.logical_or(np.mod(count, 2).astype(bool), is_inside)


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
    cartesian_coords = ellipsoidal2cartesian(polygon[:, 0], polygon[:, 1], h=0, a=a, f=f)
    cartesian_coords /= np.sqrt(np.sum(cartesian_coords**2, axis=1))[:, np.newaxis]

    antipode = -np.mean(cartesian_coords, axis=0)
    antipode /= np.sqrt(np.sum(antipode**2))

    spherical_cap = -cartesian_coords @ antipode[:, np.newaxis]
    min_cos_angle = np.min(spherical_cap, axis=0)

    cartesian_coords = np.append(cartesian_coords,  cartesian_coords[0][np.newaxis, :], axis=0)

    xyz = ellipsoidal2cartesian(lon, lat, h=0, a=a, f=f)
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

        is_crossing = np.logical_or((np.sign(-s1)+np.sign(s2)+np.sign(-s3)+np.sign(s4)) == -4,
                                    (np.sign(-s1)+np.sign(s2)+np.sign(-s3)+np.sign(s4)) == 4)
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
    cartesian_coords = ellipsoidal2cartesian(polygon[:, 0], polygon[:, 1], h=0, a=a, f=f)
    cartesian_coords /= np.sqrt(np.sum(cartesian_coords ** 2, axis=1))[:, np.newaxis]

    antipode = -np.mean(cartesian_coords, axis=0)
    antipode /= np.sqrt(np.sum(antipode ** 2))

    xyz = ellipsoidal2cartesian(lon, lat, h=0, a=a, f=f)
    xyz /= np.sqrt(np.sum(xyz ** 2, axis=1))[:, np.newaxis]

    spherical_cap = -cartesian_coords @ antipode[:, np.newaxis]
    min_cos_angle = np.cos(np.arccos(np.min(spherical_cap, axis=0)) + buffer/a)

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
                                     np.cos(buffer / a) <= np.sum(p*xyz[remaining_index, :], axis=1))
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
    return np.arctan2(np.sqrt((np.cos(lat2)*np.sin(lon2 - lon1))**2 +
                              (np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1))**2),
                      np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))*r


def ellipsoidal_distance(lon1, lat1, lon2, lat2, a=6378137.0, f=298.2572221010**-1):
    """
    Compute the distance between points (lon1, lat1) and (lon2, lat2) on an ellipsoid with
    semi-major axis a and flattening f.
    
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
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid

    Returns
    -------
    d : ndarray(m,), ndarray(m,n)
        ellipsoidal distance between points (lon1, lat1) and (lon2, lat2) in meters

    Notes
    -----
    This function uses the approximation formula by Lambert [1]_ and gives meter level accuracy.

    References
    ----------
    .. [1] Lambert, W. D (1942). "The distance between two widely separated points
           on the surface of the earth". J. Washington Academy of Sciences. 32 (5): 125–130.

    """
    beta1 = np.arctan((1 - f) * np.tan(lat1))
    beta2 = np.arctan((1 - f) * np.tan(lat2))

    sigma = spherical_distance(lon1, beta1, lon2, beta2, r=1)
    L = sigma != 0.0

    P = (beta1 + beta2) * 0.5
    Q = (beta2 - beta1) * 0.5

    X = (sigma[L] - np.sin(sigma[L])) * ((np.sin(P[L]) * np.cos(Q[L])) / np.cos(sigma[L]*0.5))**2
    Y = (sigma[L] + np.sin(sigma[L])) * ((np.cos(P[L]) * np.sin(Q[L])) / np.sin(sigma[L]*0.5))**2

    sigma[L] -= 0.5 * f * (X + Y)
    return a * sigma


def ellipsoidal2cartesian(lon, lat, h=0, a=6378137.0, f=298.2572221010**-1):
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
    e2 = 2 * f - f ** 2
    radius_of_curvature = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    return np.vstack(((radius_of_curvature + h) * np.cos(lat) * np.cos(lon),
                      (radius_of_curvature + h) * np.cos(lat) * np.sin(lon),
                      ((1 - e2) * radius_of_curvature + h) * np.sin(lat))).T
