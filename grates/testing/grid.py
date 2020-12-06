# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

import grates
import pickle
import numpy as np
import pytest
import os
from grates.testing import TestCase


class TestGrid(TestCase):

    def generate_data(self):
        pass

    def delete_data(self):
        pass

    def test_distance_interface(self):

        d = grates.grid.spherical_distance(2, 2, 1, 1)
        assert np.isscalar(d)

        d = grates.grid.spherical_distance(2, 2, np.ones(5), np.ones(5))
        assert d.shape == (5,)

        d = grates.grid.spherical_distance(np.ones(5), np.ones(5), 2, 2)
        assert d.shape == (5,)

        d = grates.grid.spherical_distance(np.ones(5) + 1, np.ones(5) + 1, np.ones(5), np.ones(5))
        assert d.shape == (5,)
