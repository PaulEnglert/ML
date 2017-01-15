# -*- coding: utf-8 -*-

import unittest
import numpy as np
from utilities.data_utils import make_batches

from neural_networks.perceptrons.slp import SLP, f_step


class TestSLP(unittest.TestCase):
	"""Test cases for SLPs."""

	def test_simple(self):
		num_features = 2
		num_neurons = 1

		trainX = np.asarray([
			[0.5, 2],
			[0.6, 5],
			[-0.6, 0.1],
			[-0.8, 2],
			[-0.3, -1],
			[-1, -0.1],
			[3, -2.6],
			[2.4, -0.34]
		])

		trainY = np.asarray([1,1,1,1,-1,-1,-1,-1])

		network = SLP(num_features, num_neurons, activation=f_step)

		epochs = 20
		network.train(make_batches(trainX, 2), make_batches(trainY, 2), epochs=epochs)

		testX = np.asarray([[-5,-2], [5,2]])
		testY = np.asarray([-1, 1])
		print 'Predicting\n {0}\n with target {1}:\n {2}'.format(testX, testY, network.predict(testX))
