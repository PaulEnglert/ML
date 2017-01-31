# -*- coding: utf-8 -*-

import cPickle
# cloud pickle  is necessary for lamda functions
import cloudpickle

import unittest
import numpy as np

from evolutionary_algorithms.rnngp import RNNGP

min_X = np.asarray([
    [2, 1],
    [3, 2],
    [1, 1]])
min_Y = np.asarray([0.2, 0.4, 0.7])
min_X2 = np.asarray([
    [3, 1],
    [2, 2],
    [3, 1]])
min_Y2 = np.asarray([0.5, 0.6, 0.9])


class TestRNNGP(unittest.TestCase):
    """Test cases for RBMs."""

    def test_simple(self):
        constants = [-10, -5, 5, 10]
        num_features = 2
        population_size = 10
        rnngp = RNNGP(num_features, constants, population_size)

        rnngp.evolve(min_X, min_Y, valX=min_X2, valY=min_Y2, generations=50)

        cloudpickle.dump(rnngp, open('rnngp.p', 'wb'))
        rnngp = cPickle.load(open('rnngp.p', 'rb'))

    def test_ist_stock(self):
        base = '/Users/paulenglert/Development/DataScience/ML/tests/'
        constants = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

        # read data
        train = np.loadtxt(base + 'data/istanbul_stock_train1.txt',
                           delimiter="\t",
                           skiprows=2)
        # get validation data
        num_val = int(np.floor(train.shape[0] * 0.2))
        np.random.shuffle(train)
        val = train[0:num_val]
        train = train[num_val:]
        test = np.loadtxt(base + 'data/istanbul_stock_test1.txt',
                          delimiter="\t",
                          skiprows=2)
        num_features = train.shape[1] - 1

        RNNGP.tournament_size = 4
        RNNGP.max_initial_depth = 6
        RNNGP.log_file_path = base + 'results/'
        RNNGP.log_stdout = True
        RNNGP.gp_config['skip_generations'] = 50
        rnngp = RNNGP(num_features, constants, 200)

        rnngp.evolve(
            train[:, :-1], train[:, -1],
            val[:, :-1], val[:, -1],
            test[:, :-1], test[:, -1],
            500)

        cloudpickle.dump(rnngp, open('rnngpis.p', 'wb'))
