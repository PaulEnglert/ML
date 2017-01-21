# -*- coding: utf-8 -*-

import numpy as np

from neural_networks.boltzmann_machines.generative_rbm import GenRBM
from neural_networks.perceptrons.mlp import MLP

from utilities.data_utils import make_batches

class DBN():
	''' Deep belief network aka stacked boltzmann machines'''

	def __init__(self, layer_definitions):
		self.num_layers=len(layer_definitions)

		# build stack of RBMs for pretraining
		self.rbm_stack=[]
		for l in range(self.num_layers-1):
			self.rbm_stack.append(GenRBM(layer_definitions[l],layer_definitions[l+1]))
			
		# build MLP used for fine tuning
		print 'Initializing MLP with a configuration of {0}, {1}, {2}'.format(layer_definitions[0],[l for l in layer_definitions[1:-1]],layer_definitions[-1])
		self.mlp = MLP(layer_definitions[0], [l+1 for l in layer_definitions[1:-1]], layer_definitions[-1])

	def pre_train(self, trainX, epochs_per_layer=5,	learning_rate = 0.01, learning_rate_decay=1, 
					lambda_1=0, lambda_2=0):

		X = trainX.copy()
		for l in range(self.num_layers-1):
			print 'Training GenRBM {0}'.format(l)
			batches = make_batches(X.copy(), 100, keep_last=True)
			self.rbm_stack[l].train(batches, epochs=epochs_per_layer,
				learning_rate=learning_rate,
				learning_rate_decay=learning_rate_decay,
				lambda_1=lambda_1,
				lambda_2=lambda_2)							# train layer with X
			X = self.rbm_stack[l].sample_hidden(X)			# set X to sampled output of the layer

	def fine_tune(self, trainX, trainY, epochs=10, learning_rate=0.01, learning_rate_decay=1):
		print 'Fine Tuning GenRB as MLP'
		self.mlp.set_weights(self.__convert_weights(self.rbm_stack))
		self.mlp.train(	make_batches(trainX.copy(), 10, keep_last=False),
					 	make_batches(trainY.copy(), 10, keep_last=False),
					 	epochs=epochs, 
						learning_rate =learning_rate, 
						learning_rate_decay=learning_rate_decay) # train mlp on the weights of the rbm stack

	def __convert_weights(self, stack, use_best=False):
		weights = []
		for s in stack:
			# get weights of botzmann machine
			w = (s.W_best if use_best else s.W)
			# move first row to last and cut first column
			weights.append(w[[i for i in range(1, w.shape[0])]+[0],1:]) # normalize to [-1,1]
		return weights