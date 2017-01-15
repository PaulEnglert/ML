# -*- coding: utf-8 -*-

import numpy as np

def make_batches(data, size, keep_last = False):
	batches = []
	batch = []
	for i in range(len(data)):
		batch.append(data[i])
		if len(batch) == size or i == len(data)-1:
			batches.append(np.asarray(batch))
			batch = []
	if batches[-1].shape[0] < size and not keep_last:
		return batches[0:-2]
	return batches

def transform_target(target):
	# assuming that target is a 1d vector of labels (e.g. [0..4])
	# transform the target, such that each neuron can be represented with one label
	transformed = []
	transformation = {}
	labels = np.unique(target)
	for l in range(len(labels)):
		nt = [(1 if u == l else 0) for u in range(len(labels))]
		transformation[labels[l]] = nt
	for t in target:
		transformed.append(transformation[t]) 
	return np.asarray(transformed)