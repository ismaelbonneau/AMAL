import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import models

from sklearn.utils import shuffle

temperatures_csv = pd.read_csv("data/tempAMAL_train.csv")
print("Nb exemples: {}, cities: {}".format(temperatures_csv.shape[0], temperatures_csv.shape[1]))

def sample_sequence(df, city, seqlen):
	start = np.random.randint(0, df.shape[0] - seqlen)
	return np.array(df[city][start:start + seqlen])

def sample_sequences(df, seq_len, nbcities, nbsequences, normalize=True):
	""" """

	# prendre une longueur de séquence de taille fixe pour le moment
	cities = np.random.choice(df.columns[1:], nbcities, replace=False)

	sequences = []
	labels = []

	for i, city in enumerate(cities):
		j = 0
		while j < nbsequences:
			s = sample_sequence(df, city, seq_len)
			if not np.isnan(s).any():
				sequences.append(s)
				labels.append(i)
				j += 1
        
	sequences = np.array(sequences)
	labels = np.array(labels)
	if normalize:
		sequences /= np.max(sequences)
	# shuffle lines
	sequences, labels = shuffle(sequences, labels, random_state=1997)
	sequences = np.expand_dims(sequences, axis=2) # pour créer une dimension supplémentaire de taille 1

	assert sequences.shape[0] == labels.shape[0]

	return np.swapaxes(sequences,0,1), labels, cities

class Dataset_RNN(Dataset):
    """Dataset avec la taille de batch en axis 1 au lieu de 0.
        Pas mal pour notre RNN"""
	def __init__(self, x, y):
		super(Dataset_RNN, self).__init__()
		self.labels = torch.from_numpy(y)
		self.data = torch.from_numpy(x).float()
	def __getitem__(self, index):
		return self.data[:,index,:], self.labels[index]
	def __len__(self):
		return len(self.labels)