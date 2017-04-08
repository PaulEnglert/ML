# -*- coding: utf-8 -*-

import cPickle
# cloud pickle  is necessary for lamda functions
import cloudpickle
import logging as _l
import unittest
import numpy as np

from evolutionary_algorithms.rgp import RGP

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


class TestRGP(unittest.TestCase):
    """Test cases for RBMs."""

    def test_simple(self):
        constants = [-10, -5, 5, 10]
        num_features = 2
        population_size = 10
        rgp = RGP(num_features, constants, population_size)

        rgp.evolve(min_X, min_Y, valX=min_X2, valY=min_Y2, generations=1)

        cloudpickle.dump(rgp, open('RGP.p', 'wb'))
        rgp = cPickle.load(open('RGP.p', 'rb'))

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

        RGP.tournament_size = 4
        RGP.max_initial_depth = 6
        RGP.log_file_path = base + 'results/'
        RGP.log_stdout = True
        RGP.debug = False
        RGP.timeit = True
        RGP.collect_footprints = (False, 10, 4)
        RGP.gp_config['skip_generations'] = 10
        RGP.repulse_by = 'semantics'
        RGP.gp_config['validation_elite_size'] = 5
        RGP.r_general['blame_full_program'] = False
        rgp = RGP(num_features, constants, 200)

        _l.getLogger('rgp.main').info("Train_file = {0}".format(
            base + 'data/istanbul_stock_train1.txt'))
        _l.getLogger('rgp.main').info("Test_file = {0}".format(
            base + 'data/istanbul_stock_test1.txt'))
        _l.getLogger('rgp.main').info("-------------------------")

        rgp.evolve(
            train[:, :-1], train[:, -1],
            val[:, :-1], val[:, -1],
            test[:, :-1], test[:, -1],
            150)

        cloudpickle.dump(rgp, open('RGPis.p', 'wb'))

    def test_ist_stock_model_overfitting(self):
        base = '/Users/paulenglert/Development/DataScience/ML/tests/'
        constants = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        population_size = 200
        generations = 150

        # read data
        train = np.loadtxt(base + 'data/istanbul_stock_train1.txt',
                           delimiter="\t",
                           skiprows=2)
        # get validation data
        num_val = int(np.floor(train.shape[0] * 0.05))
        np.random.shuffle(train)
        val = train[0:num_val]
        train = train[num_val:]
        test = np.loadtxt(base + 'data/istanbul_stock_test1.txt',
                          delimiter="\t",
                          skiprows=2)
        num_features = train.shape[1] - 1

        RGP.tournament_size = 4
        RGP.max_initial_depth = 6
        RGP.log_file_path = base + 'results/'
        RGP.log_stdout = True
        RGP.debug = False
        RGP.simulate_repulsing = True
        RGP.timeit = False
        RGP.collect_footprints = (False, 10, 4)
        RGP.collect_accuracies = True
        RGP.gp_config['skip_generations'] = 10
        rgp = RGP(num_features, constants, population_size)

        _l.getLogger('rgp.main').info("Train_file = {0}".format(
            base + 'data/istanbul_stock_train1.txt'))
        _l.getLogger('rgp.main').info("Test_file = {0}".format(
            base + 'data/istanbul_stock_test1.txt'))
        _l.getLogger('rgp.main').info("-------------------------")

        rgp.use_predicted_overfitting(base + 'data/overfitting_predictor_3_300.model',
                                      base + 'data/severity_predictor_3_300.model',
                                      3, 300)

        rgp.evolve(
            train[:, :-1], train[:, -1],
            val[:, :-1], val[:, -1],
            testX=test[:, :-1], testY=test[:, -1],
            generations=generations)

        cloudpickle.dump(rgp, open('RGPis.p', 'wb'))

    def test_multiple_ist_stock(self):
        self.test_ist_stock()
        self.test_ist_stock()
        self.test_ist_stock()
        self.test_ist_stock()
        self.test_ist_stock()
