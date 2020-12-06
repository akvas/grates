# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

import grates
import pickle
import numpy as np
import pytest
import os
from grates.testing import TestCase


class TestReferenceField(TestCase):

    __file_name = 'test_reference_field_data.dat'

    def generate_data(self):

        colat = np.linspace(0, np.pi, 10)
        r = np.random.randn(colat.size) * 1000 + 6378136.3

        g_wgs84 = grates.gravityfield.WGS84.normal_gravity(r, colat)
        g_grs80 = grates.gravityfield.GRS80.normal_gravity(r, colat)

        test_data = {'input': {'colat': colat, 'r': r}, 'output': {'g_wgs84': g_wgs84, 'g_grs80': g_grs80}}

        with open(self.__file_name, 'wb+') as f:
            pickle.dump(test_data, f)

    def delete_data(self):

        os.remove(self.__file_name)

    def test_reference_field_data(self):

        if not os.path.isfile(self.__file_name):
            pytest.skip('test data {0} not available'.format(self.__file_name))

        with open(self.__file_name, 'rb') as f:
            test_data = pickle.load(f)

            g_wgs84 = grates.gravityfield.WGS84.normal_gravity(test_data['input']['r'], test_data['input']['colat'])
            g_grs80 = grates.gravityfield.GRS80.normal_gravity(test_data['input']['r'], test_data['input']['colat'])

            np.testing.assert_array_equal(g_grs80, test_data['output']['g_grs80'])
            np.testing.assert_array_equal(g_wgs84, test_data['output']['g_wgs84'])

    def test_reference_field_constructor(self):

        reffield_J2 = grates.gravityfield.ReferenceField(GM=3986005e8, omega=7292115.0e-11, a=6378137.0, J2=108263e-8)
        reffield_f = grates.gravityfield.ReferenceField(GM=3986005e8, omega=7292115.0e-11, a=6378137.0, f=reffield_J2.flattening)

        assert np.isclose(reffield_J2.J2, reffield_f.J2, rtol=1e-14, atol=0)

        reffield_f = grates.gravityfield.ReferenceField(GM=3986004.418e8, omega=7292115.0e-11, a=6378137.0, f=1 / 298.257223563)
        reffield_J2 = grates.gravityfield.ReferenceField(GM=3986004.418e8, omega=7292115.0e-11, a=6378137.0, J2=reffield_f.J2)

        assert np.isclose(reffield_J2.flattening, reffield_f.flattening, rtol=1e-14, atol=0)

    def test_reference_field_interface(self):

        reffield_f = grates.gravityfield.ReferenceField(GM=3986004.418e8, omega=7292115.0e-11, a=6378137.0, f=1 / 298.257223563)

        g = reffield_f.normal_gravity(1, 1)
        assert g.shape == (1,)

        g = reffield_f.normal_gravity(np.ones(5), 1)
        assert g.shape == (5,)

        g = reffield_f.normal_gravity(1, np.ones(5))
        assert g.shape == (5,)

        g = reffield_f.normal_gravity(np.ones(5), np.ones(5))
        assert g.shape == (5,)

        with pytest.raises(ValueError):
            reffield_f.normal_gravity(np.ones(5), np.ones(6))

    def test_reference_field_normal_gravity(self):

        g = grates.gravityfield.GRS80.normal_gravity(6378137.0, 0.5 * np.pi)  # equator
        assert np.isclose(g, 9.7803267715, rtol=1e-11, atol=0)

        g = grates.gravityfield.GRS80.normal_gravity(6356752.3141, np.pi)  # pole
        assert np.isclose(g, 9.8321863685, rtol=1e-9, atol=0)


test_classes = [TestReferenceField()]
