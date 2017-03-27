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

    def test_ist_stock(self):
        base = '/Users/paulenglert/Development/DataScience/ML/tests/'
        constants = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

        # read data
        train = np.loadtxt(base + 'data/istanbul_stock_train1.txt',
                           delimiter="\t",
                           skiprows=2)
        test = np.loadtxt(base + 'data/istanbul_stock_test1.txt',
                          delimiter="\t",
                          skiprows=2)
        num_features = train.shape[1] - 1

        GP.tournament_size = 4
        GP.maximum_initial_depth = 6
        GP.log_file_path = base + 'results/'
        GP.log_stdout = True
        GP.log_verbose = True
        gp = GP(num_features, constants, 200)

        gp.evolve(
            train[:, :-1], train[:, -1],
            test[:, :-1], test[:, -1],
            100)
