# -*- coding: utf-8 -*-

import unittest
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


from utilities.data_utils import make_batches, transform_target

from neural_networks.perceptrons.slp import SLP, f_step
from neural_networks.perceptrons.mlp import MLP
from neural_networks.perceptrons.mlp_textbook import MLP as MLP2

trainX_simple_linear = np.asarray([
	[0.5, 2],
	[0.6, 5],
	[-0.6, 0.1],
	[-0.8, 2],
	[-0.3, -1],
	[-1, -0.1],
	[3, -2.6],
	[2.4, -0.34]
])
trainY_simple_linear = np.asarray([1,1,1,1,0,0,0,0])

trainX_xor = np.asarray([[0,1],[0,0],[1,0],[1,1]])
trainY_xor = np.asarray([1,0,1,0])

class TestSLP(unittest.TestCase):
	"""Test cases for SLPs."""

	def test_simple(self):
		num_features = 2
		num_neurons = 1
		trainY = np.asarray([1,1,1,1,-1,-1,-1,-1])

		network = SLP(num_features, num_neurons, activation=f_step)

		epochs = 20
		network.train(make_batches(trainX_simple_linear, 2), make_batches(trainY, 2), epochs=epochs)

		testX = np.asarray([[-5,-2], [5,2]])
		testY = np.asarray([-1, 1])
		print 'Predicting\n {0}\n with target {1}:\n {2}'.format(testX, testY, network.predict(testX))


class TestMLP(unittest.TestCase):
	"""Test cases for MLPs."""

		
	def test_simple_mnist(self):

		print "(downloading data...)"
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.033, train_size=0.067, random_state=42)

		trainX = make_batches(trainX, 100)
		trainY = make_batches(transform_target(trainY), 100)
		testX_b = make_batches(testX, 100)
		testY_b = make_batches(transform_target(testY), 100)

		num_features = trainX[0].shape[1]
		num_outputs = 10
		hidden_layer_sizes=[300]
		network = MLP(num_features, hidden_layer_sizes, num_outputs, learning_rate=0.05, learning_rate_decay=1.05)

		network.print_structure()
		network.train(trainX, trainY, testX=testX_b,testY=testY_b,epochs=50)

		for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
			# classify the digit
			pred = network.predict(np.atleast_2d(testX[i]))
			# show the image and prediction
			print "Actual digit is {0}, predicted {1}".format(testY[i], np.argmax(pred))
		
