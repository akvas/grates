import numpy as np
import grates
import pytest


def test_legendre_functions():

    max_degree = 5

    res = grates.utilities.legendre_functions(max_degree, 1)
    assert res.shape == (1, max_degree + 1, max_degree + 1)

    res = grates.utilities.legendre_functions(max_degree, np.random.randn(5))
    assert res.shape == (5, max_degree + 1, max_degree + 1)

    res = grates.utilities.legendre_functions(max_degree, [1, 2, 3])
    assert res.shape == (3, max_degree + 1, max_degree + 1)


def test_legendre_functions_per_order():

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


def test_legendre_summation():

    res = grates.utilities.legendre_summation(np.random.randn(5), np.random.randn(3))
    assert res.shape == (3,)

    res = grates.utilities.legendre_summation(np.random.randn(5), 1)
    assert res.shape == (1,)

    res = grates.utilities.legendre_summation(np.random.randn(5), [1, 2])
    assert res.shape == (2,)


def test_trigonimetric_functions():

    max_degree = 5

    res = grates.utilities.trigonometric_functions(max_degree, 1)
    assert res.shape == (1, max_degree + 1, max_degree + 1)

    res = grates.utilities.trigonometric_functions(max_degree, np.random.randn(5))
    assert res.shape == (5, max_degree + 1, max_degree + 1)


def test_spherical_harmonics():

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


def test_ravel_coefficients():

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


def test_unravel_coefficients():

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


def test_normal_gravity():

    res = grates.utilities.normal_gravity(6378136.3, 0.1)
    assert res.shape == (1,)

    res = grates.utilities.normal_gravity(6378136.3, np.random.randn(5))
    assert res.shape == (5,)

    res = grates.utilities.normal_gravity(6378136.3, [1, 2, 3])
    assert res.shape == (3,)
