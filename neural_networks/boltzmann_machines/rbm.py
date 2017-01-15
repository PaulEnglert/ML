# -*- coding: utf-8 -*-

import numpy as np


"""
Representation of a restricted boltzmann machine
"""
class RBM:

	def __init__(self, num_inputs, num_hidden_neurons, learning_rate, **kwargs):
		self.num_visible = num_inputs
		self.num_hidden = num_hidden_neurons
		self.learning_rate = learning_rate
		self.is_constructed = False
		self.is_trained = False
		self.log = kwargs.get('log', True)

	def construct(self, weights=None):
		# build weight matrix or use parameter
		if weights is None:
			self.weights = np.random.randn(self.num_visible, self.num_hidden)*0.1
			# add bias weights column
			self.weights = np.insert(self.weights, 0, 0, axis=1)
			# add bias row
			self.weights = np.insert(self.weights, 0, 0, axis=0)
		else:
			self.weights = weights
		if self.log:
			self.print_weights()
		self.is_constructed = True


	def train(self, num_epochs, trainX, trainY = None, testX = None, testY = None, use_states=False, log_epochs=False, num_cd_steps=1, no_decay=False, exit_on_error_increase=False):
		if not self.is_constructed:
			self.construct()

		# prepare data
		data = trainX.copy()
		data = np.insert(data, 0, 1, axis=1) # add column of bias states, all -> 1
		last_error=np.NaN
		# run 
		for epoch in range(num_epochs):
			# forward (positive) move: compute the states& probabilities of the hidden units, based on the real data
			hid_aes = np.dot(data, self.weights)
			hid_probs = self._logistic_function(hid_aes)
			hid_states  = hid_probs > np.random.randn(data.shape[0], self.num_hidden+1)
			# calculate association -> meaning calculate which units are on together and which are not (probability based, could also be states instead)
			if use_states:
				pos_association = np.dot(np.transpose(data.astype(np.bool)), hid_states)
			else:
				pos_association = np.dot(np.transpose(data), hid_probs) # vis_probs is not known yet, we'll take the states, which are equal to the dataset plus the bias
			
			for cd in range(num_cd_steps):
				# backward (negative) move: compute the states & probabilities of the visible units based on the hidden states
				vis_aes = np.dot(hid_states, np.transpose(self.weights))
				vis_probs = self._logistic_function(vis_aes)
				vis_probs[:,0] = 1 # fix bias unit

				# forward (negative) move: recompute the states & probabilities of the hidden units, this time based on the expected outcome of the reverse step before
				hid_aes = np.dot(vis_probs, self.weights)
				hid_probs = self._logistic_function(hid_aes)
				hid_states  = hid_probs > np.random.randn(data.shape[0], self.num_hidden+1)
			
			# calculate association after n alternating gibbs sampling steps
			if use_states:
				vis_states  = vis_probs > np.random.randn(data.shape[0], self.num_visible+1)
				neg_association = np.dot(np.transpose(vis_states), hid_states)
			else:
				neg_association = np.dot(np.transpose(vis_probs), hid_probs)

			# measure for preupdate exit
			error = np.sum((data - vis_probs) ** 2) / data.shape[0]
			if error > last_error and exit_on_error_increase:
				last_error = error
				print('	Exit due to increased error: '+str(last_error))
				break

			# update weights
			self.weights = self.weights + self.learning_rate*((pos_association - neg_association) / data.shape[0])
			# decay learning rate
			if not no_decay:
				self.learning_rate = self.learning_rate/2

			# finish epoch
			last_error = error
			if self.log:
				print('Epoch '+str(epoch))
				print('	100%:\n	error='+str(last_error)+'\n	...squared difference of data to expected visible state (probability)')
			if not self.log and log_epochs:
				print('	Epoch '+str(epoch)+' ('+str(num_cd_steps)+' CD steps): err='+str(last_error))
		
		if self.log:
			self.print_weights()
		self.is_trained = True
		return last_error


	def label_units(self, trainX, trainY):
		labels = np.unique(trainY)
		output = self.predict(trainX)
		# compute the average probabilities of a hidden unit for each class
		output = np.insert(output, 0, trainY, axis=1) # add labels in first column
		self.labelling=[]
		for l in labels:
			self.labelling.append([l]+[np.average(output[np.where(output[...,0]==l)][...,u]) for u in range(1,self.num_hidden+1)])
		return self.labelling

	def predict(self, d, return_states=False): # forward move in network
		if not self.is_trained:
			raise Exception('Network is not trained yet.')
		data = np.insert(d.copy(), 0, 1, axis=1) # add column of bias states, all -> 1
		hid_aes = np.dot(data, self.weights)
		hid_probs = self._logistic_function(hid_aes)
		if return_states:
			hid_states  = hid_probs > np.random.randn(data.shape[0], self.num_hidden+1)
			return hid_states[:,1:]
		else:
			return hid_probs[:,1:]

	def daydream(self, num_steps):
		if not self.is_trained:
			raise Exception('Network is not trained yet.')
		random_data = np.ones((num_steps, self.num_visible+1)) # create random data container
		random_data[0,1:] = np.random.rand(self.num_visible) # add random data to first row, the next rows will be populated in each iteration
		for step in range(num_steps):
			if step == num_steps-1:
				break
			# calculate hidden units states
			hid_aes = np.dot(random_data, self.weights)
			hid_probs = self._logistic_function(hid_aes)
			hid_probs[:,0] = 1 # fix bias unit
			hid_states  = hid_probs > np.random.randn(random_data.shape[0], self.num_hidden+1)
			# calculate visible units states
			vis_aes = np.dot(hid_states, np.transpose(self.weights))
			vis_probs = self._logistic_function(vis_aes)
			vis_probs[:,0] = 1 # fix bias unit
			vis_states  = vis_probs > np.random.randn(random_data.shape[0], self.num_visible+1)
			# set the newly calculated visible states as the next data row in the random_data
			random_data[step+1,:] = vis_states[step]
		return random_data[:,1:]

	# Utility functions

	def _logistic_function(self, x):
		return 1 / (1 + np.exp(-x))

	def print_weights(self):
		print('')
		header = '/;BIAS;'
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+(';' if i < self.num_hidden-1 else '')
		print(header)
		row_labels = ['BIAS']+['INP_'+str(i) for i in range(self.num_visible)]
		for row_label, row in zip(row_labels, self.weights):
			print '%s;%s' % (row_label, ';'.join('%03s' % i for i in row))

	def print_labelling(self):
		print('')
		header = 'label;'
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+(';' if i < self.num_hidden-1 else '')
		print(header)
		for row in self.labelling:
			print(';'.join(str(i) for i in row))

	def print_prediction(self, prediction):
		print('')
		header = ''
		for i in range(self.num_hidden):
			header = header + 'HID_'+str(i)+(';' if i < self.num_hidden-1 else '')
		print(header)
		for row in prediction:
			print(';'.join(str(i) for i in row))
