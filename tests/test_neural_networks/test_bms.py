# -*- coding: utf-8 -*-

import unittest
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from utilities.data_utils import create_association, make_batches

from neural_networks.boltzmann_machines.generative_rbm import GenRBM


movieDataX = np.asarray([
	[1,1,1,0,0,0],
	[1,0,1,0,0,0],
	[1,1,1,0,0,0],
	[0,0,1,1,1,0], 
	[0,0,1,1,0,0],
	[0,0,1,1,1,0],
])
movieDataY = np.asarray([1,1,1,0,0,0])


class TestGenRBM(unittest.TestCase):
	"""Test cases for RBMs."""

	def test_states_simple(self):
		num_visible = 6
		num_hidden = 2
		learning_rate = 0.5
		epochs = 10

		network = GenRBM(num_visible, num_hidden)
		network.train(movieDataX, epochs=epochs, learning_rate = learning_rate)

		print network.sample_hidden(movieDataX)

	def test_probs_simple(self):
		num_visible = 6
		num_hidden = 2
		learning_rate = 0.1
		epochs = 1500

		network = GenRBM(num_visible, num_hidden)
		network.train(movieDataX, epochs=epochs, learning_rate = learning_rate)

		print np.argmax(network.sample_hidden(movieDataX), axis=1)

		print network.dream(5, x=np.asarray([1,0,1,1,0,1]))

	def test_probs_mnist(self):
		print "(downloading data...)"
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.033, train_size=0.067, random_state=42)

		num_visible = trainX.shape[1]
		num_hidden = np.unique(trainY).shape[0]
		network = GenRBM(num_visible, num_hidden)

		network.train(
			make_batches(trainX, 100, keep_last=True), 
			epochs=50, 
			learning_rate=0.01,
			learning_rate_decay=1.2,
			lambda_1=0,
			lambda_2=0.001)

		# associations = create_association(network.sample_hidden(trainX), trainY, probabilities=False)

		# for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
		# 	# classify the digit
		# 	pred = network.sample_hidden(np.atleast_2d(testX[i]), use_softmax=True)
		# 	labels = [l[0] for l in associations.items() if l[1] == np.argmax(pred)]
		# 	# show the image and prediction
		# 	print "Actual digit is {0}, predicted {1}".format(testY[i], labels)
