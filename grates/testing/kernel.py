# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

import grates
import pytest
import numpy as np


def test_gauss_radius():

    with pytest.raises(ValueError):
        grates.kernel.Gauss(-1)

    kn = grates.kernel.Gauss(0)
    np.testing.assert_equal(1, kn.coefficients(2, 5))


def test_gauss_recursion():

    kn = grates.kernel.Gauss(300)

    c1 = kn.coefficients(2, 200)
    c2 = kn.coefficients(2, 2000)
    np.testing.assert_array_almost_equal(c1, c2[:, 0:c1.size], decimal=14)


def test_kernel_coefficients_shape():

    test_objects = [grates.kernel.SurfaceDensity(), grates.kernel.OceanBottomPressure(), grates.kernel.Potential(), grates.kernel.UpwardContinuation(),
                    grates.kernel.WaterHeight(), grates.kernel.GeoidHeight(), grates.kernel.Gauss(300)]

    for kn in test_objects:
        min_degree = 2
        max_degree = 5

        c = kn.coefficients(min_degree, max_degree, 1, 1)
        assert c.shape == (1, max_degree + 1 - min_degree)

        c = kn.coefficients(min_degree, max_degree, np.ones(5), 1)
        assert c.shape == (5, max_degree + 1 - min_degree)

        c = kn.coefficients(min_degree, max_degree, 1, np.ones(5))
        assert c.shape == (5, max_degree + 1 - min_degree)

        c = kn.coefficients(min_degree, max_degree, np.ones(5), np.ones(5))
        assert c.shape == (5, max_degree + 1 - min_degree)

        with pytest.raises(ValueError):
            c = kn.coefficients(min_degree, max_degree, np.ones(5), np.ones(6))


def test_kernel_coefficient_shape():

    test_objects = [grates.kernel.SurfaceDensity(), grates.kernel.OceanBottomPressure(), grates.kernel.Potential(), grates.kernel.UpwardContinuation(),
                    grates.kernel.WaterHeight(), grates.kernel.GeoidHeight(), grates.kernel.Gauss(300)]

    for kn in test_objects:
        degree = 5

        c = kn.coefficient(degree, 1, 1)
        assert c.shape == (1,)

        c = kn.coefficient(degree, np.ones(5), 1)
        assert c.shape == (5,)

        c = kn.coefficient(degree, 1, np.ones(5))
        assert c.shape == (5,)

        c = kn.coefficient(degree, np.ones(5), np.ones(5))
        assert c.shape == (5,)

        with pytest.raises(ValueError):
            c = kn.coefficient(degree, np.ones(5), np.ones(6))


