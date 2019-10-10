import numpy as np
import torch
import torch.nn as nn



class RNN(torch.nn.Module):
	"""docstring for RNN"""
	def __init__(self, dim, latent):
		super(RNN, self).__init__()

		self.dim = dim
		self.latent = latent

		self.Wx = torch.nn.Linear(dim, latent)
		self.Wh = torch.nn.Linear(latent, latent)

	def forward(self, x, h=None):
		""" x: sequence_length x batch x dim 
			h: batch x latent
			returns: length x batch x latent Tensor"""

		historique = []
		# pour chaque instant de la sequence:
		if h is None:
			ht = torch.zeros(x.size()[1], self.latent)
			historique.append(ht.numpy())
		for i, xt in enumerate(x):
			# ht: (batch x latent)
			ht = self.one_step(xt, ht)
			if i > 0:
				historique.append(ht) # Ne pas enregistrer les h0

		return torch.Tensor(historique)

	def one_step(self, x, h):
		""" x: batch x dim 
			h: batch x latent
			returns: batch x latent Tensor """
		return torch.nn.functional.leaky_relu(torch.add(self.Wx(x), self.Wh(h)))
		

class Decoder(torch.nn.Module):
	""" simple Neural Net """
	def __init__(self, inSize, outSize, layers=[]):
		super(Decoder, self).__init__()
		self.layers = nn.ModuleList([])
		for x in layers:
			self.layers.append(nn.Linear(inSize, x))
			inSize = x
		self.layers.append(nn.Linear(inSize, outSize))

	def forward(self, x):
		x = self.layers[0](x)
		for i in range(1, len(self.layers)):
			x = torch.nn.functional.leaky_relu(x)
			x = self.layers[i](x)
		return x

class ManyToOneRNN(torch.nn.Module):
	""" """
	def __init__(self, dim, latent, nbClass):
		super(ManyToOneRNN, self).__init__()
		self.rnn = RNN(dim, latent)
		self.decoder = Decoder(latent, nbClass)

	def forward(self, x):
		""" """
		hT = self.rnn(x)[-1]
		return self.decoder(hT)