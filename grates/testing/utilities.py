# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

import numpy as np
import grates
import pytest
from grates.testing import TestCase
import pickle
import os


class TestLegendreFunctions(TestCase):

    __file_name = 'test_legendre_functions_data.dat'

    def generate_data(self):

        max_degree = 5
        psi = np.linspace(0, np.pi, 10) + np.random.randn(10)
        lon = np.linspace(-np.pi, np.pi, 10) + np.random.randn(10)
        coefficients = np.random.randn(27)
        test_order = 5

        Pnm = grates.utilities.legendre_functions(max_degree, psi)
        Pn = grates.utilities.legendre_polynomials(max_degree, psi)
        Pnm_orderwise = grates.utilities.legendre_functions_per_order(max_degree, test_order, psi)
        Ynm = grates.utilities.spherical_harmonics(max_degree, psi, lon)
        lsum = grates.utilities.legendre_summation(coefficients, psi)

        test_data = {'input': {'colat': psi, 'max_degree': max_degree, 'coefficients': coefficients, 'test_order': test_order, 'lon': lon},
                     'output': {'Pnm': Pnm, 'Pn': Pn, 'Pnm_orderwise': Pnm_orderwise, 'lsum': lsum, 'Ynm': Ynm}}

        with open(self.__file_name, 'wb+') as f:
            pickle.dump(test_data, f)

    def delete_data(self):
        try:
            os.remove(self.__file_name)
        except FileNotFoundError:
            pass

    def test_legendre_functions_data(self):

        if not os.path.isfile(self.__file_name):
            pytest.skip('test data {0} not available'.format(self.__file_name))

        with open(self.__file_name, 'rb') as f:
            test_data = pickle.load(f)

            Pnm = grates.utilities.legendre_functions(test_data['input']['max_degree'], test_data['input']['colat'])
            Pn = grates.utilities.legendre_polynomials(test_data['input']['max_degree'], test_data['input']['colat'])
            Pnm_orderwise = grates.utilities.legendre_functions_per_order(test_data['input']['max_degree'], test_data['input']['test_order'], test_data['input']['colat'])
            Ynm = grates.utilities.spherical_harmonics(test_data['input']['max_degree'], test_data['input']['colat'], test_data['input']['lon'])
            lsum = grates.utilities.legendre_summation(test_data['input']['coefficients'], test_data['input']['colat'])

            np.testing.assert_array_equal(Pnm, test_data['output']['Pnm'])
            np.testing.assert_array_equal(Pn, test_data['output']['Pn'])
            np.testing.assert_array_equal(Pnm_orderwise, test_data['output']['Pnm_orderwise'])
            np.testing.assert_array_equal(Ynm, test_data['output']['Ynm'])
            np.testing.assert_array_equal(lsum, test_data['output']['lsum'])

    def test_legendre_functions_interface(self):

        max_degree = 5

        res = grates.utilities.legendre_functions(max_degree, 1)
        assert res.shape == (1, max_degree + 1, max_degree + 1)

        res = grates.utilities.legendre_functions(max_degree, np.random.randn(5))
        assert res.shape == (5, max_degree + 1, max_degree + 1)

        res = grates.utilities.legendre_functions(max_degree, [1, 2, 3])
        assert res.shape == (3, max_degree + 1, max_degree + 1)

    def test_legendre_functions_per_order_interface(self):

        max_degree = 5
        order = 3

        res = grates.utilities.legendre_functions_per_order(max_degree, order, 1)
        assert res.shape == (1, max_degree + 1 - order)

        res = grates.utilities.legendre_functions_per_order(max_degree, order, np.random.randn(5))
        assert res.shape == (5, max_degree + 1 - order)

        res = grates.utilities.legendre_functions_per_order(max_degree, 0, np.random.randn(5))
        assert res.shape == (5, max_degree + 1)

        res = grates.utilities.legendre_functions_per_order(max_degree, order, [1, 2, 3])
        assert res.shape == (3, max_degree + 1 - order)

        with pytest.raises(ValueError):
            grates.utilities.legendre_functions_per_order(max_degree, max_degree + 1, [1, 2, 3])

    def test_legendre_summation_interface(self):

        res = grates.utilities.legendre_summation(np.random.randn(5), np.random.randn(3))
        assert res.shape == (3,)

        res = grates.utilities.legendre_summation(np.random.randn(5), 1)
        assert res.shape == (1,)

        res = grates.utilities.legendre_summation(np.random.randn(5), [1, 2])
        assert res.shape == (2,)

    def test_trigonimetric_functions_interface(self):

        max_degree = 5

        res = grates.utilities.trigonometric_functions(max_degree, 1)
        assert res.shape == (1, max_degree + 1, max_degree + 1)

        res = grates.utilities.trigonometric_functions(max_degree, np.random.randn(5))
        assert res.shape == (5, max_degree + 1, max_degree + 1)

    def test_spherical_harmonics_interface(self):

        max_degree = 5

        res = grates.utilities.spherical_harmonics(max_degree, 1, 1)
        assert res.shape == (1, max_degree + 1, max_degree + 1)

        res = grates.utilities.spherical_harmonics(max_degree, 1, np.random.randn(5))
        assert res.shape == (5, max_degree + 1, max_degree + 1)

        res = grates.utilities.spherical_harmonics(max_degree, np.random.randn(5), 1)
        assert res.shape == (5, max_degree + 1, max_degree + 1)

        res = grates.utilities.spherical_harmonics(max_degree, np.random.randn(5), np.random.randn(5))
        assert res.shape == (5, max_degree + 1, max_degree + 1)

        with pytest.raises(ValueError):
            grates.utilities.spherical_harmonics(max_degree, np.random.randn(5), np.random.randn(6))


class TestCoefficientRavelling(TestCase):

    __file_name = 'test_ravel_coefficients.dat'

    def generate_data(self):

        anm = np.random.randn(6, 6)
        x1 = grates.utilities.ravel_coefficients(anm, 2, 4)
        x2 = grates.utilities.ravel_coefficients(anm)
        test_data = {'input': {'anm': anm, 'min_degree': 2, 'max_degree': 4}, 'output': {'x1': x1, 'x2': x2}}

        with open(self.__file_name, 'wb+') as f:
            pickle.dump(test_data, f)

    def delete_data(self):
        try:
            os.remove(self.__file_name)
        except FileNotFoundError:
            pass

    def test_ravel_coefficients_data(self):

        if not os.path.isfile(self.__file_name):
            pytest.skip('test data {0} not available'.format(self.__file_name))

        with open(self.__file_name, 'rb') as f:
            test_data = pickle.load(f)

            x1 = grates.utilities.ravel_coefficients(test_data['input']['anm'], test_data['input']['min_degree'], test_data['input']['max_degree'])
            np.testing.assert_array_equal(x1, test_data['output']['x1'])
            x2 = grates.utilities.ravel_coefficients(test_data['input']['anm'])
            np.testing.assert_array_equal(x2, test_data['output']['x2'])

    def test_ravel_coefficients(self):

        anm = np.random.randn(6, 6)

        x = grates.utilities.ravel_coefficients(anm)
        assert x.shape == (anm.size,)

        x = grates.utilities.ravel_coefficients(anm, 2, 5)
        assert x.shape == (32,)

        x = grates.utilities.ravel_coefficients(anm, 0, 5)
        assert x.shape == (anm.size,)

        x = grates.utilities.ravel_coefficients(anm, 0, 6)
        assert x.shape == (49,)

        anm = np.random.randn(3, 6, 6)

        x = grates.utilities.ravel_coefficients(anm)
        assert x.shape == (3, 36)

        x = grates.utilities.ravel_coefficients(anm, 2, 5)
        assert x.shape == (3, 32)

        x = grates.utilities.ravel_coefficients(anm, 0, 5)
        assert x.shape == (3, 36)

        x = grates.utilities.ravel_coefficients(anm, 0, 6)
        assert x.shape == (3, 49)

        with pytest.raises(ValueError):
            grates.utilities.ravel_coefficients(np.random.randn(1, 1, 1, 1), 0, 6)

    def test_unravel_coefficients(self):

        x = np.random.randn(36)

        anm = grates.utilities.unravel_coefficients(x)
        assert anm.shape == (6, 6)

        anm = grates.utilities.unravel_coefficients(x[4:], 2, 5)
        assert anm.shape == (6, 6)

        x = np.random.randn(3, 36)
        anm = grates.utilities.unravel_coefficients(x)
        assert anm.shape == (3, 6, 6)

        anm = grates.utilities.unravel_coefficients(x[:, 4:], 2, 5)
        assert anm.shape == (3, 6, 6)

        with pytest.raises(ValueError):
            grates.utilities.unravel_coefficients(np.random.randn(1, 1, 1, 1), 2, 5)

    def test_forward_backward(self):

        anm = np.random.randn(6, 6)

        x = grates.utilities.ravel_coefficients(anm)
        np.testing.assert_array_equal(anm, grates.utilities.unravel_coefficients(x))

        # x = grates.utilities.ravel_coefficients(anm, 2, 4)
        # np.testing.assert_array_equal(anm, grates.utilities.unravel_coefficients(x, 2, 4))


test_cases = [TestLegendreFunctions(), TestCoefficientRavelling()]
