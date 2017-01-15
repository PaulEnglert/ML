# -*- coding: utf-8 -*-

import unittest
import numpy as np

from neural_networks.boltzmann_machines.rbm import RBM


class TestRBM(unittest.TestCase):
	"""Test cases for RBMs."""

	def test_simple(self):
		num_inputs = 6
		num_hidden_units = 2
		learning_rate = 0.5
		trainX = np.asarray([
			[1,1,1,0,0,0],
			[1,0,1,0,0,0],
			[1,1,1,0,0,0],
			[0,0,1,1,1,0], 
			[0,0,1,1,0,0],
			[0,0,1,1,1,0],
		])
		trainY = np.asarray([1,1,1,0,0,0])

		network = RBM(num_inputs, num_hidden_units, learning_rate, debug=True)

		network.train(500, trainX, num_cd_steps=5, no_decay=True)

		network.label_units(trainX, trainY)
		network.print_labelling()

		prediction = network.predict(np.asarray([[0,0,0,1,1,0]]))
		network.print_prediction(prediction)

		n = 10
		dreamed = network.daydream(n)
		print('\nDaydreaming for '+str(n)+' gibbs steps:')
		print(dreamed)
