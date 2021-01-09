# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

import grates
import pickle
import numpy as np
import pytest
import os
from grates.testing import TestCase


class TestGridConstructors(TestCase):

    __file_name = 'test_grid_constructors.dat'

    @staticmethod
    def create_data_dictionary():

        test_data = {'grid_geograph': grates.grid.GeographicGrid(),
                     'grid_geograph_sphere': grates.grid.GeographicGrid(a=6378136.3, f=0),
                     'grid_gauss': grates.grid.GaussGrid(91),
                     'grid_gauss_sphere': grates.grid.GaussGrid(91, a=6378136.3, f=0),
                     'geodesic_grid': grates.grid.GeodesicGrid(13),
                     'geodesic_grid_sphere': grates.grid.GeodesicGrid(13, a=6378136.3, f=0),
                     'reuter_grid': grates.grid.ReuterGrid(37),
                     'reuter_grid_sphere': grates.grid.ReuterGrid(37, a=6378136.3, f=0),
                     'great_circle_segment': grates.grid.GreatCircleSegment(1, 1, 1),
                     'great_circle_segment_sphere': grates.grid.GreatCircleSegment(1, 1, 1, a=6378136.3, f=0)}

        return test_data

    def generate_data(self):

        test_data = TestGridConstructors.create_data_dictionary()
        with open(self.__file_name, 'wb+') as f:
            pickle.dump(test_data, f)

    def delete_data(self):
        try:
            os.remove(self.__file_name)
        except FileNotFoundError:
            pass

    def test_grid_constructors(self):

        if not os.path.isfile(self.__file_name):
            pytest.skip('test data {0} not available'.format(self.__file_name))

        with open(self.__file_name, 'rb') as f:
            test_data = pickle.load(f)
            comparison = TestGridConstructors.create_data_dictionary()

            for name, grid in test_data.items():
                grid_comparison = comparison[name]

                np.testing.assert_array_equal(grid.longitude, grid_comparison.longitude)
                np.testing.assert_array_equal(grid.latitude, grid_comparison.latitude)


class TestGridSphericalDistance(TestCase):

    __file_name = 'test_spherical_harmonic_distance_data.dat'

    def generate_data(self):

        grid = grates.grid.GeographicGrid(dlon=10, dlat=10)
        S1 = grid.distance_matrix()

        S2 = np.zeros(S1.shape)
        lons, lats = grid.longitude, grid.latitude

        for k in range(grid.point_count):
            S2[k, k:] = grates.grid.spherical_distance(lons[k], lats[k], lons[k:], lats[k:], r=1)
            S2[k + 1:, k] = S2[k, k + 1:]

        test_data = {'input': {'lons': lons, 'lats': lats, 'grid': grid}, 'output': {'S1': S1, 'S2': S2}}

        with open(self.__file_name, 'wb+') as f:
            pickle.dump(test_data, f)

    def delete_data(self):
        os.remove(self.__file_name)

    def test_sphericla_distance_data(self):

        if not os.path.isfile(self.__file_name):
            pytest.skip('test data {0} not available'.format(self.__file_name))

        with open(self.__file_name, 'rb') as f:
            test_data = pickle.load(f)

            S1 = test_data['input']['grid'].distance_matrix()
            np.testing.assert_array_equal(S1, test_data['output']['S1'])
            np.testing.assert_array_equal(S1, test_data['output']['S2'])

    def test_distance_interface(self):

        d = grates.grid.spherical_distance(2, 2, 1, 1)
        assert np.isscalar(d)

        d = grates.grid.spherical_distance(2, 2, np.ones(5), np.ones(5))
        assert d.shape == (5,)

        d = grates.grid.spherical_distance(np.ones(5), np.ones(5), 2, 2)
        assert d.shape == (5,)

        d = grates.grid.spherical_distance(np.ones(5) + 1, np.ones(5) + 1, np.ones(5), np.ones(5))
        assert d.shape == (5,)


test_cases = [TestGridSphericalDistance(), TestGridConstructors()]
