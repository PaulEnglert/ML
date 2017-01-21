# -*- coding: utf-8 -*-

import unittest
import numpy as np

import cv2
import itertools as IT

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
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33, train_size=0.67, random_state=42)

		num_visible = trainX.shape[1]
		num_hidden = np.unique(trainY).shape[0]*10
		network = GenRBM(num_visible, num_hidden)

		network.train(
			make_batches(trainX.copy(), 100, keep_last=True), 
			epochs=100, 
			learning_rate=0.01,
			learning_rate_decay=1.2,
			lambda_1=0,
			lambda_2=0.001)

		np.savetxt('genrbm_weights_best.out', network.W_best, delimiter=";")
		np.savetxt('genrbm_weights_last.out', network.W, delimiter=";")

		# associations = create_association(network.sample_hidden(trainX), trainY, probabilities=False)
		# pred = network.sample_hidden(np.atleast_2d(testX[i]), use_softmax=True)
		# labels = [l[0] for l in associations.items() if l[1] == np.argmax(pred)]
		# print "Actual digit is {0}, predicted {1}".format(testY[i], labels)



	def test_pretrained_mnist(self):
		print "(downloading data...)"
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33, train_size=0.67, random_state=42)

		weights = np.loadtxt('genrbm_weights_last.out', delimiter=";")
		network = GenRBM(weights.shape[0]-1, weights.shape[1]-1, weights=weights)

		# VISUALIZE ABSTRACTIONS
		print "Generating Abstractions"
		h_states=[]
		v_sample=[]
		for u in range(weights.shape[1]):
			h_states.append([1 if u == i else 0 for i in range(weights.shape[1])])
			v_sample.append((network.sample_visible(np.asarray([h_states[u]])) * 255).reshape((28, 28)).astype("uint8"))
		ncols=10
		nrows=weights.shape[1]/ncols
		display = np.empty((28*nrows, 28*ncols), dtype=np.uint8)
		for i, j in IT.product(range(nrows), range(ncols)):
			arr = v_sample[i*ncols+j]
			x, y = i*28, j*28
			display[x:x+28, y:y+28] = arr
		cv2.imshow("Abstractions", display)
		cv2.waitKey(0)

		# VISUALIZE DREAMING
		print "Dreaming results"
		for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
			imOrg = (trainX[i] * 255).reshape((28, 28)).astype("uint8")
			dreamed = network.dream(25, trainX[i])
			comb = np.hstack((
				imOrg,
				(dreamed[5] * 255).reshape((28, 28)).astype("uint8"),
				(dreamed[10] * 255).reshape((28, 28)).astype("uint8"),
				(dreamed[15] * 255).reshape((28, 28)).astype("uint8"),
				(dreamed[20] * 255).reshape((28, 28)).astype("uint8"),
				(dreamed[25] * 255).reshape((28, 28)).astype("uint8"),
				))
			cv2.imshow("Original vs. Dreamed", comb)
			cv2.waitKey(0)
