# -*- coding: utf-8 -*-

import numpy as np


def f_step(x):
    X = x.copy()
    mask = X > 0
    X[mask] = 1
    X[~mask] = -1
    return X


def f_sigmoidal(x):
    return 1 / (1 + np.exp(-1 * x))


def f_softplus(X, deriv=False):  # relu
    if not deriv:
        return np.log(1 + np.exp(X))
    else:
        return f_sigmoidal(X)


class SLP():
    '''Single layer perceptron for binary classification.'''

    def __init__(self, num_features, num_neurons, activation=f_sigmoidal):
        ''' initialize the network based on the number of features in
            the data set and the number of desired output neurons
            (typically number of classes)
        '''
        self.num_neurons = num_neurons
        self.num_features = num_features
        self.activation = activation

        self.weights = np.random.randn(num_features + 1, num_neurons) * 0.1
        self.weights[-1] = 0  # set weights from bias to neurons to 0 initially

        print 'Initialized Single Layer Perceptron with {0} neuron(s)'.format(
            self.num_neurons)

    def train(self, trainX, trainY, epochs=500, testX=None, testY=None,
              exit_on_zero_error=True, learning_rate=0.1,
              learning_rate_decay=1):
        print 'Starting Training of Network'

        for epoch in range(epochs):
            log_str = '[{0:4}]'.format(epoch)

            for batchX, batchY in zip(trainX, trainY):
                output = self.predict(batchX)
                if batchY.ndim == 1:
                    batchY = np.transpose(np.atleast_2d(batchY))
                delta = batchY - output
                self.weights += learning_rate * np.dot(np.transpose(
                    np.append(batchX, np.ones((batchX.shape[0], 1)), axis=1)),
                    delta)

            t_error = self.evaluate(trainX, trainY)
            log_str += 'training_error={0}'.format(t_error)

            if testX is not None and testY is not None:
                log_str += 'test_error={0}'.format(self.evaluate(testX, testY))

            learning_rate /= learning_rate_decay
            print log_str
            # early exit condition
            if exit_on_zero_error and t_error == 0:
                print 'Exiting due to zero error reached'
                break

    def evaluate(self, dataX, dataY):
        error = 0
        for batchX, batchY in zip(dataX, dataY):
            if batchY.ndim == 1:
                batchY = np.transpose(np.atleast_2d(batchY))
            output = self.predict(batchX)
            # normalize the error by the number of patterns, i.e. neurons
            error += np.sum(batchY - output) / self.num_neurons
        return error

    def predict(self, dataX):
        dataX = np.append(dataX, np.ones((dataX.shape[0], 1)), axis=1)
        return self.activation(np.dot(dataX, self.weights))
