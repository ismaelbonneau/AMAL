import torch
import torch.nn as nn

import numpy as np


class GRU(nn.Module):
	"""docstring for GRU"""
	def __init__(self, input_dim, latent_dim):
		super(GRU, self).__init__()

		self.W_update = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Sigmoid())
		self.W_reset = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Sigmoid())

		self.W = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Tanh())

	def one_step(self, x, h):

		zt = self.W_update(torch.cat((x, h), 0))
		rt = self.W_reset(torch.cat((x, h), 0))
		ht = (1 - zt) * h + zt * self.tanh(self.W(torch.cat((x, rt * h), 0)))

		return ht

	def forward(self, x, h=None):

		historique = []
		# pour chaque instant de la sequence:
		if h is None:
			ht = torch.zeros(x.size()[1], self.latent)
		for xt in x:
			# ht: (batch x latent)
			ht = self.one_step(xt, ht)
			historique.append(ht)
		return historique


class LSTM(nn.Module):
	""" doctring for LSTM """
	def __init__(self, input_dim, latent_dim):
		super(LSTM, self).__init__()

		self.W_forget = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Sigmoid())
		self.W_input = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Sigmoid())
		self.W_output = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Sigmoid())

		self.W_update = nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), nn.Tanh()) # Wc for internal memory update

	def one_step(self, x, h, c):

		ft = self.W_forget(torch.cat((x, h), 0))
		it = self.W_input(torch.cat((x, h), 0))
		Ct = ft * c + it * self.W_update(torch.cat((x, h), 0))
		ot = self.W_output(torch.cat((x, h), 0))

		return ht, Ct

	def forward(self, x, h=None):

		historique = []
		# pour chaque instant de la sequence:
		if h is None:
			ht = torch.zeros(x.size()[1], self.latent)
		Ct = torch.zeros(x.size()[1], self.latent)
		for xt in x:
			# ht: (batch x latent)
			ht, Ct = self.one_step(xt, ht, Ct)
			historique.append(ht)
		return historique


class RNN(torch.nn.Module):
    """docstring for RNN"""
    def __init__(self, input_dim, latent_dim):
        super(RNN, self).__init__()
        self.dim = dim
        self.latent = latent
        self.dropout = Dropout(0.1)
        self.Wx = torch.nn.Linear(dim, latent)
        self.Wh = torch.nn.Linear(latent, latent)

    def forward(self, x, h=None):
        """ x: sequence_length x batch x dim 
            h: batch x latent
            returns: length x batch x latent Tensor"""
        historique = []
        if h is None:
            ht = torch.zeros(x.size()[1], self.latent)
        for i, xt in enumerate(x):
            # ht: (batch x latent)
            ht = self.one_step(xt, ht)
            historique.append(ht)
        return historique

    def one_step(self, x, h):
        """ x: batch x dim 
            h: batch x latent
            returns: batch x latent Tensor """
        combined = self.Wx(x) + self.Wh(h)
        return torch.nn.functional.leaky_relu(combined)