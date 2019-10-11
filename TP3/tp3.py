import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import normalize

def afficher(x):
    plt.axis("off")
    plt.imshow(x.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.show()

def comparer(original, reconstructed):
    fig, axs = plt.subplots(1, 2)
	axs[0].axis("off")
	axs[0].imshow(original.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
	axs[1].axis("off")
	axs[1].imshow(reconstructed.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
	plt.show()

# Dataset d'images MNIST

class Dataset_MNIST(Dataset):
	"""docstring for Dataset_Arouf"""
	def __init__(self, x, y):
		super(Dataset_MNIST, self).__init__()
		self.labels = torch.from_numpy(y)
		self.data = torch.from_numpy(x).float()
	def __getitem__(self, index):
		return self.data[index], self.labels[index]
	def __len__(self):
		return len(self.labels)


######## Partie Auto Encoder ########

class TiedAutoEncoder(nn.Module):
	""" tied AutoEncoder: encoder and decoder share weights"""
	def __init__(self, input_dim, latent):
		super(TiedAutoEncoder, self).__init__()
		self.W = torch.nn.Parameter(torch.randn(input_dim, latent), requires_grad = True)
		self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)
		self.b2 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

	def encode(self, x):
		return nn.functional.relu(x @ self.W + self.b1)

	def decode(self, x):
		return torch.sigmoid(x @ self.W.t() + self.b2)

	def forward(self, x):
		return self.decode(self.encode(x))

class AutoEncoder(nn.Module):
	""" classical & simple AutoEncoder """
	def __init__(self, input_dim, latent):
		super(AutoEncoder, self).__init__()
		self.encoder = torch.nn.Linear(input_dim, latent)
		self.decoder = torch.nn.Linear(latent, input_dim)

	def encode(self, x):
		return nn.functional.relu(self.encoder(x))

	def decode(self, x):
		return torch.sigmoid(self.decoder(x))

	def forward(self, x):
		return self.decode(self.encode(x))


if __name__ == '__main__':

	from torch.utils.tensorboard import SummaryWriter
	import torchvision

	seed = 1997
	torch.manual_seed(seed)

	X_train, y_train = loadlocal_mnist(images_path='/home/ismael/Documents/master/AMAL/TP3/mnist/train-images.idx3-ubyte',
	 labels_path='/home/ismael/Documents/master/AMAL/TP3/mnist/train-labels.idx1-ubyte')

	X_test, y_test = loadlocal_mnist(images_path='/home/ismael/Documents/master/AMAL/TP3/mnist/t10k-images.idx3-ubyte',
	 labels_path='/home/ismael/Documents/master/AMAL/TP3/mnist/t10k-labels.idx1-ubyte')

	# images sous forme de vecteurs
	print("train shapes: ", X_train.shape, y_train.shape)
	print("test shapes: ", X_test.shape, y_test.shape)

	assert X_train.shape[0] == y_train.shape[0]
	assert X_test.shape[0] == y_test.shape[0]

	X_train = X_train.astype(float)
	X_test = X_test.astype(float)

	X_train /= 255.
	X_test /= 255.

	#X_train = normalize(X_train, axis=0) # normalise chaque feature
	#X_test = normalize(X_test, axis=0) # normalise chaque feature

	train_dataset = Dataset_MNIST(X_train, y_train)
	trainloader = DataLoader(dataset=train_dataset, batch_size=40, shuffle=True)

	test_dataset = Dataset_MNIST(X_test, y_test)
	testloader = DataLoader(dataset=test_dataset, batch_size=40, shuffle=False)

	latent_dim = 100 # encoder en dimension 100

	quelbail = {}

	for latent_dim in [10, 20, 30, 40, 50, 100]:
		print("latent dim: ", latent_dim)

		model = AutoEncoder(X_train.shape[1], latent_dim) 

		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

		writer = SummaryWriter()

		trainloss = []
		valloss = []

		epochs = 30

		for e in range(epochs):

			epoch_losses = []
			# par batch
			for image_batch, label_batch in trainloader:
				model.train()
				optimizer.zero_grad()

				reconstructed = model(image_batch)
				loss = criterion(image_batch, reconstructed)

				loss.backward()
				optimizer.step()

				epoch_losses.append(loss.data)

			val_losses = []
			with torch.no_grad():
				for image_batch, label_batch in testloader:
					model.eval()

					reconstructed = model(image_batch)
					loss = criterion(image_batch, reconstructed)

					val_losses.append(loss.data)
			
			#writer.add_scalar('Loss/train', epoch_losses[-1], e)
			#writer.add_scalar('Loss/test', val_losses[-1], e)
			trainloss.append(np.array(epoch_losses).mean())
			valloss.append(np.array(val_losses).mean())

			if e % 10 == 0:
				print("epoch {}: train {} test {}".format(e, np.array(epoch_losses).mean(),  np.array(val_losses).mean()))

		quelbail[latent_dim] = (trainloss, valloss)

		writer.close()

	pickle.dump(quelbail, open("chakalito.pkl", "wb"))

	model = TiedAutoEncoder(X_train.shape[1], 150) 

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

	writer = SummaryWriter()

	epochs = 60
	for e in range(epochs):
		for image_batch, label_batch in trainloader:
			model.train()
			optimizer.zero_grad()
			reconstructed = model(image_batch)
			loss = criterion(image_batch, reconstructed)
			loss.backward()
			optimizer.step()

	# enregistrer le modèle









