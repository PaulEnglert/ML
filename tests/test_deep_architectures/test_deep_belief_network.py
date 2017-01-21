# -*- coding: utf-8 -*-

import cPickle

import unittest
import numpy as np

import cv2
import itertools as IT

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from utilities.data_utils import transform_target

from deep_architectures.deep_belief_network import DBN

class TestDBN(unittest.TestCase):
	"""Test cases for RBMs."""

	def test_simple_mnist(self):
		print "(downloading data...)"
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33, train_size=0.67, random_state=42)

		network = DBN([784,300,150,50,10])

		network.pre_train(trainX,
					epochs_per_layer=50, learning_rate = 0.01, 
					learning_rate_decay=1.2, lambda_1=0, lambda_2=0.001 )

		network.fine_tune(trainX, transform_target(trainY), epochs=25, 
							learning_rate = 0.01, learning_rate_decay=1.2)

		cPickle.dump(network, open('dbn.p', 'wb'))

	def test_pretrained_mnist(self):
		print "(downloading data...)"
		dataset = datasets.fetch_mldata("MNIST Original")
		(trainX, testX, trainY, testY) = train_test_split(
			dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33, train_size=0.67, random_state=42)

		print "Loading Model"
		network = cPickle.load(open('dbn.p'))

		predictions = np.argmax(network.mlp.predict(testX), axis=1)
		print classification_report(testY, predictions)
		# Y_t = transform_target(testY.copy())
		# for i in np.random.choice(np.arange(0, len(Y_t)), size = (10,)):
		# 	pred = network.mlp.predict(np.atleast_2d(testX[i]))
		# 	print "Actual digit is {0}, predicted {1}".format(testY[i], np.argmax(pred))

		# Display Abstractions
		print "Generating Abstractions"
		for l, layer in enumerate(network.rbm_stack):
			h_states=[]
			v_sample=[]
			for u in range(layer.W.shape[1]):
				h_states.append([1 if u == i else 0 for i in range(layer.W.shape[1])])
				sample = layer.sample_visible(np.asarray([h_states[u]]))
				for pl in reversed(range(l)):
					sample = network.rbm_stack[pl].sample_visible(np.asarray(sample))
				v_sample.append((sample * 255).reshape((28, 28)).astype("uint8"))
			ncols=10
			nrows=layer.W.shape[1]/ncols
			display = np.empty((28*nrows, 28*ncols), dtype=np.uint8)
			for i, j in IT.product(range(nrows), range(ncols)):
				arr = v_sample[i*ncols+j]
				x, y = i*28, j*28
				display[x:x+28, y:y+28] = arr
			cv2.imshow("Abstractions of Layer {0}".format(l), display)
		cv2.waitKey(0)