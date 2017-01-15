# -*- coding: utf-8 -*-

import numpy as np

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


class MLP():
	'''Multi layer perceptron, for binary and multinominal classification.'''

	count = 0
	class Layer():
		def __init__(self, inputs, units, activation=f_sigmoidal, is_output=False):
			self.id = MLP.count
			MLP.count += 1

			self.num_inputs = inputs
			self.num_units = units
			self.is_output = is_output
			self.activation = activation

			self.weights = np.random.rand(self.num_inputs, units)*0.1
			self.weights[-1] = 0

		def propagate(self, data):
			self.activations = np.dot( data, self.weights )
			self.output = self.activation(self.activations)
			if not self.is_output:
				self.derivatives = self.activation(self.activations, deriv=True)
			return self.output

		def print_structure(self):
			print '{0} (input {1}, units {2})'.format(self.id, self.num_inputs, self.num_units)

		def print_data(self):
			print '{0}:\n{1}'.format(self.id, self.weights)

	def __init__(self, num_features, hidden_layer_sizes, num_outputs, learning_rate = 0.1, 
						learning_rate_decay=1):
		self.num_features = num_features
		self.num_outputs = num_outputs
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay
		self.num_layers = len(hidden_layer_sizes)+1


		self.layers = []
		self.layers.append(MLP.Layer(num_features+1, hidden_layer_sizes[0]))				 							# input layer
		for l in range(1,self.num_layers-1):
			self.layers.append(MLP.Layer(self.layers[-1].num_units+1, hidden_layer_sizes[l]))							# hidden layer
		self.layers.append(MLP.Layer(self.layers[-1].num_units+1, num_outputs, activation=f_softmax, is_output=True))	# output layer
	
	def train(self, trainX, trainY, testX=None, testY=None, epochs=500):


		for epoch in range(epochs):
			log_str = '[{0:4}]'.format(epoch)

			for batchX, batchY in zip(trainX, trainY):
				prediction = self.__propagate_forward(batchX)
				self.__propagate_backward(batchX, prediction, batchY)
				self.__update_weights(batchX)
			self.learning_rate /= self.learning_rate_decay

			# evaluate
			error = 0
			for batchX, batchY in zip(trainX, trainY):
				error += self.evaluate(batchX, batchY)
			log_str += ' training_error={0}'.format(np.sum(error)/len(trainX))

			if testX is not None and testY is not None:
				error = 0
				for batchX, batchY in zip(testX, testY):
					error += self.evaluate(batchX, batchY)
				log_str += ', test_error={0}'.format(np.sum(error)/len(testX))

			print log_str

	def __propagate_forward(self, data):
		output = data
		for l in range(self.num_layers):
			output = np.append(output, np.ones((data.shape[0], 1)), axis=1)
			output = self.layers[l].propagate(output)
		return output

	def __propagate_backward(self, X, prediction, target):
		if target.ndim == 1:
			target = np.atleast_2d(target).T
		elif target.ndim > 2:
			raise Exception('Target vector has to many dimensions: {0}. Expected 1 or 2'.format(target.ndim))
		
		for l in reversed(range(self.num_layers)):
			layer = self.layers[l]
			if self.layers[l].is_output:
				self.layers[l].delta = prediction-target #
			else:
				self.layers[l].delta = np.dot(self.layers[l+1].delta, np.transpose(self.layers[l+1].weights[0:-1,:]) ) * self.layers[l].derivatives # ignore biases here thats why the last rows are cut off
	
	def __update_weights(self, X):
		for l in range(self.num_layers):
			if l == 0:
				data = np.append(X, np.ones((X.shape[0], 1)), axis=1)  												# use input data
			else:
				data = np.append(self.layers[l-1].output, np.ones((self.layers[l-1].output.shape[0], 1)), axis=1) 	# get output of previous layer
			self.layers[l].weights += -self.learning_rate * ( np.dot(data.T, self.layers[l].delta) ) #


	def evaluate(self, dataX, dataY):
		error = ( dataY - self.predict(dataX) )**2 / dataY.shape[1] if dataY.ndim>1 else 1 							# normalize to number of patterns
		return error / len(dataX) 																					# normalize to number of observations

	def predict(self, data):
		return self.__propagate_forward(data)


	def print_structure(self):
		for layer in self.layers:
			layer.print_structure()

