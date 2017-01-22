# -*- coding: utf-8 -*-

import cPickle
# cloud pickle  is necessary for lamda functions
import cloudpickle

import unittest
import numpy as np

from evolutionary_algorithms.gp import GP

min_X = np.asarray([
    [2, 1],
    [3, 2],
    [1, 1]])
min_Y = np.asarray([0.2, 0.4, 0.7])


class TestGP(unittest.TestCase):
    """Test cases for RBMs."""

    def test_simple(self):
        constants = [-10, -5, 5, 10]
        num_features = 2
        population_size = 25
        gp = GP(num_features, constants, population_size)

        gp.evolve(min_X, min_Y, 25)

        cloudpickle.dump(gp, open('gp.p', 'wb'))
        gp = cPickle.load(open('gp.p', 'rb'))
