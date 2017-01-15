# -*- coding: utf-8 -*-

import numpy as np
from random import random

"""Activation Functions"""
def f_sigmoidal(X, deriv=False):
	if not deriv:
		return 1 / ( 1 + np.exp( -1 * X) )
	else:
		return f_sigmoidal(X)*(1-f_sigmoidal(X))

def f_softmax(X):
	Z = np.sum(np.exp(X), axis=1)
	Z = Z.reshape(Z.shape[0], 1)
	return np.exp(X) / Z

def f_tanh(X, deriv=False):
	if not deriv:
		return np.tanh(X)
	else: 
		return 1 - f_tanh(X)**2

def f_gaussian(value):
	return exp( -1 * pow( value, 2 ) )

def f_hyperbolic(value):
		return ( exp( value ) - exp( -1 * value ) ) / ( exp( value ) + exp( -1 * value ) )

class NaiveMLP:
	class Neuron:
		count = 1

		def __init__(self, num_inputs, learning_rate, use_alpha, activation, **kwargs): 
			self.id = NaiveMLP.Neuron.count
			NaiveMLP.Neuron.count = NaiveMLP.Neuron.count + 1

			self.learning_rate = learning_rate
			self.output = None
			self.wsi = None

			self.input_weights = []
			for x in xrange(0, num_inputs):
				self.input_weights.append(random()-0.5)
			self.theta_weight = random()-0.5

			self.activation = activation

			self.delta = None
			self.last_input = None
			self.layer = kwargs.get('layer',0)
			self.use_alpha = use_alpha

		def compute_wsi(self, input_data, weights = None, theta_weight = None):
			if weights is None or theta_weight is None:
				self.wsi = self.theta_weight
				for index, value in enumerate(input_data):
					self.wsi = self.wsi + ( value * self.input_weights[index] )
			else:
				self.wsi = theta_weight
				for index, value in enumerate(input_data):
					self.wsi = self.wsi + ( value * weights[index] )

		def compute_output(self, input_data):
			self.compute_wsi(input_data)
			self.output = self.activation(self.wsi)
			self.last_input = input_data

		def update_weights(self):
			delta_w = -1*self.learning_rate*self.delta
			self.theta_weight = self.theta_weight + delta_w*1 + (random() if self.use_alpha else 0)*delta_w*1
			for index, x in enumerate(self.input_weights):
				self.input_weights[index] = self.input_weights[index] + (delta_w * self.last_input[index]) + (random() if self.use_alpha else 0)*delta_w*self.last_input[index]

		def update_dr_output_error(self, target):
			self.delta = (self.output - target) * self.output * (1 - self.output)
		
		def update_dr_hidden_error(self, ws_nextlayer):
			self.delta = (1 - self.output) * self.output * ws_nextlayer

	def __init__(self, num_inputs, learning_rate, use_alpha):
		self.num_inputs = num_inputs
		self.learning_rate = learning_rate
		self.neurons = []
		self.use_alpha = use_alpha
	
	def get_neuron(self, id):
		for n in self.neurons:
			if n.id == id: 
				return n
		return None

	def get_layer_neurons(self, layer):
		ns = []
		for n in self.neurons:
			if n.layer == layer: 
				ns.append(n)
		return ns

	def is_output_neuron(self, id):
		ons = self.get_layer_neurons(self.num_layers-1)
		for n in ons:
			if n.id == id:
				return True
		return False

	def build(self, layers, **kwargs):# layers should be list [num_input_units, num_hiddenunits-layer1, num_hiddenunits-layer2[, ...], num_output_units]
		self.num_layers = len(layers)
		if self.num_layers > 2:
			print 'WARNING: Using more than one hidden layer leads to the "vanishing gradient phenomenon", which will result in useless training!'
		self.neurons = []
		for l_index, count_in_layer in enumerate(layers):
			for n in xrange(0, count_in_layer):
				kwargs['layer'] = l_index
				n = NaiveMLP.Neuron((self.num_inputs if l_index == 0 else layers[l_index-1]), self.learning_rate, self.use_alpha, f_sigmoidal, **kwargs)
				self.neurons.append(n)
			print str(l_index)+': '+';'.join([str(n.id)+' (' +str(len(n.input_weights))+')' for n in self.get_layer_neurons(l_index)])
		print 'Built network with '+str(self.num_layers) + ' layers.'


	def train(self, trainX, trainY, num_iterations):
		print 'Learning for '+str(num_iterations) + ' iterations.'
		for i in xrange(0, num_iterations):
			print 'Iteration: '+str(i)
			for features, target in zip(trainX, trainY):
				# calculate all outputs
				predicted = self.predict(features.tolist())
						
				for l_index in reversed(range(0, self.num_layers)):
					# update deltas
					for n_index, neur in enumerate(self.get_layer_neurons(l_index)):
						if l_index == self.num_layers-1:
							neur.update_dr_output_error(target.tolist())
						else:
							ws_nextlayer = 0
							for n in self.get_layer_neurons(l_index+1):
								ws_nextlayer = ws_nextlayer + n.delta*n.input_weights[n_index]
							neur.update_dr_hidden_error(ws_nextlayer)
						neur.update_weights()

	def predict(self, features):
		outputs = []
		for l_index in xrange(0, self.num_layers):
			next_outputs = []
			for neur in self.get_layer_neurons(l_index):
				neur.compute_output((features if l_index == 0 else outputs))
				next_outputs.append(neur.output)
			outputs = next_outputs
		return outputs[0]
		
