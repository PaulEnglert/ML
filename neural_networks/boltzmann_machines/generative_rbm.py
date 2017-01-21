# -*- coding: utf-8 -*-

import numpy as np


def f_sigmoidal(X, deriv=False):
	if not deriv:
		return 1 / ( 1 + np.exp( -1 * X) )
	else:
		return f_sigmoidal(X)*(1-f_sigmoidal(X))

def f_softmax(X):
	Z = np.sum(np.exp(X), axis=1)
	Z = Z.reshape(Z.shape[0], 1)
	return np.exp(X) / Z

"""
Representation of a restricted boltzmann machine
"""
class GenRBM:

	def __init__(self, num_visible_units, num_hidden_units, weights=None):
		self.W_best = None
		self.Epoch_best = 0
		self.e_best = np.Inf
		self.W = np.random.randn(num_visible_units+1, num_hidden_units+1)*0.1
		self.W[:,0] = 0 #bias column
		self.W[0,:] = 0 #bias row
		if weights is not None:
			self.W = weights
			self.W_best = weights
			print 'Using provided weight matrix.'
		print('Initialized Generative RBM with {0} visible and {1} hidden units.'.format(num_visible_units, num_hidden_units))

	def train(self, trainX, epochs = 10, learning_rate = 0.05, learning_rate_decay=1, lambda_1=0, lambda_2=0):
		for epoch in range(epochs):
			log_str = '[{0:4}] '.format(epoch)
			error = 0
			for batchX in trainX:
				X = np.insert(batchX, 0, 1, axis=1)
				# positive phase (sample hidden from the visible)
				P_h = f_sigmoidal(X.dot(self.W))
				H = P_h > np.random.randn(*P_h.shape)
				# compute the probability that unit i and j are on together
				p_plus = X.T.dot(P_h)
				# negative phase
				P_v = f_sigmoidal(H.dot(self.W.T))
				P_v[:,0] = 1
				# resample hidden from equilibrial visible
				P_h = f_sigmoidal(P_v.dot(self.W))
				# compute the probability that unit i and j are on together
				p_minus = P_v.T.dot(P_h)
				# update weights
				self.W += learning_rate * (( p_plus - p_minus )/X.shape[0])
				error += np.sum((X - P_v) ** 2) / X.shape[0]
			# penalize weights
			self.W -= learning_rate * ( np.sum(np.absolute(self.W))*lambda_1 + np.sum(self.W**2)*lambda_2 ) 
			# update learning rate
			learning_rate /= learning_rate_decay
			if self.e_best > error/len(trainX):
				self.e_best = error/len(trainX)
				self.W_best = self.W.copy()
				self.Epoch_best = epoch

			# log progress
			print log_str + ' reconstruction error={0}'.format(error/len(trainX))

		print 'Best reconstruction error={0} in epoch {1}.'.format(self.e_best, self.Epoch_best)

	def sample_hidden(self, V, use_softmax=False, use_best=True):
		if V.shape[1] == self.W.shape[0]-1:
			V = np.insert(V, 0, 1, axis=1) # add bias unit, if the input vector is an actual feature vector
		if use_softmax:
			return f_softmax(np.dot(V, self.W_best if use_best else self.W))[:,1:] 
		return f_sigmoidal(np.dot(V, self.W_best if use_best else self.W))[:,1:] 

	def sample_visible(self, H, use_softmax=False, use_best=True):
		if H.shape[1] == self.W.shape[1]-1:
			H = np.insert(H, 0, 1, axis=1) # add bias unit, if the input vector is an actual feature vector
		if use_softmax:
			return f_softmax(np.dot(H, self.W_best.T if use_best else self.W.T))[:,1:] 
		return f_sigmoidal(np.dot(H, self.W_best.T if use_best else self.W.T))[:,1:] 

	def dream(self, epochs, x=None, probabilities=True, use_best=True):
		if x is None:
			x = np.random.randn(self.W.shape[0]) > 0.5
			x[0] = 1
		elif x.shape[0] == self.W.shape[0]-1:
			x = np.insert(x, 0, 1)
		X = np.zeros((epochs+1, self.W.shape[0]))
		X[0,:] = x
		# run alternating gibbs steps
		for epoch in range(epochs):
			P_h = f_sigmoidal(X.dot(self.W_best if use_best else self.W))
			P_v = f_sigmoidal((P_h > np.random.randn(*P_h.shape)).dot((self.W_best if use_best else self.W).T))
			P_v[:,0] = 1
			X[epoch+1, :] = P_v[epoch,:] if probabilities else P_v[epoch,:] > np.random.randn(P_v.shape[1]) # set next row of X to output of this gibbs step
		return X[:,1:]

