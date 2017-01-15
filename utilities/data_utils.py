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