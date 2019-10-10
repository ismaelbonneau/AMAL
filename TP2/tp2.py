####### Ismael Bonneau et Issam Benamara

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

seed = 1997
torch.manual_seed(seed)

######## Chargement des données ########

with open("housing.data", "r") as f:
	lines = f.readlines()

	X = np.zeros((len(lines), 6), dtype='float32')
	Y = np.zeros(len(lines), dtype='float32')

	j = 0
	for line in lines:
		for i, bail in enumerate([0, 7, 8, 10, 11, 12]): #voir housing.names
			val = line.strip().split()[bail]
			X[j][i] = float(val)
		Y[j] = float(line.strip().split()[13])
		j += 1

######## Normalisation des features ########

Y /= Y.max()
X = normalize(X, axis=0) # normalise chaque feature

############################################


X_train = X[:400] 
X_test = X[400:] # environ 20%

Y_train = Y[:400].reshape(-1, 1) 
Y_test = Y[400:].reshape(-1, 1)  # environ 20%

class Dataset_Arouf(Dataset):
	"""docstring for Dataset_Arouf"""
	def __init__(self, x, y):
		super(Dataset_Arouf, self).__init__()
		self.labels = torch.from_numpy(y)
		self.data = torch.from_numpy(x)
	def __getitem__(self, index):
		return self.data[index], self.labels[index]
	def __len__(self):
		return len(self.labels)

class LinearRegressor(torch.nn.Module):
	"""docstring for LinearRegressor"""
	def __init__(self, inputSize):
		super(LinearRegressor, self).__init__()
		#self.linear = torch.nn.Linear(inputSize, 1)
		self.W = torch.nn.Parameter(torch.randn(inputSize, 1), requires_grad=True)
		self.b = torch.nn.Parameter(torch.randn(1), requires_grad=True)
  
	def forward(self, x): 
		#y_pred = self.linear(x) 
		y_pred = x @ self.W + self.b
		return y_pred 

class Perceval1(torch.nn.Module):
	"""docstring for Perceval
		version sans conteneur Sequential"""
	def __init__(self, inputSize):
		super(Perceval, self).__init__()
		self.linear1 = torch.nn.Linear(inputSize, 16)
		self.linear2 = torch.nn.Linear(16, 1)
		self.activation = torch.nn.Tanh()

	def forward(self, x):
		return self.linear2(self.activation(self.linear1(x)))

class Perceval(torch.nn.Module):
	"""docstring for Perceval"""
	def __init__(self, inputSize):
		super(Perceval, self).__init__()
		self.mlp = torch.nn.Sequential(torch.nn.Linear(inputSize, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
	def forward(self, x):
		return self.mlp(x)


train_dataset = Dataset_Arouf(X_train, Y_train)
trainloader = DataLoader(dataset=train_dataset, batch_size=40, shuffle=True)

############ Version sans OPTIMIZER ############

print("--- Version SANS optimizer ---\n\n")

model = LinearRegressor(6)
lossfn = torch.nn.MSELoss()

for e in range(epochs):

	ohplai = []
	for x_batch, y_batch in trainloader:

		model.train()
		# forward
		mult = model(x_batch)
		loss = lossfn(mult, y_batch)

		ohplai.append(loss.item())
		loss.backward()

		with torch.no_grad():
			model.W -= learningRate * model.W.grad
			model.b -= learningRate * model.b.grad
		model.W.grad.zero_()
		model.b.grad.zero_()

	with torch.no_grad():
		# compute test error

		model.eval()

		arouf = model(torch.from_numpy(X_test))
		loss_arouf = lossfn(arouf, torch.from_numpy(Y_test))

	if (e % 10) == 0:
		print("epoch %d " % e , "train MSE: ", np.array(ohplai).mean(), "val MSE: ", loss_arouf.item())


############ Version avec OPTIMIZER ############
print("\n\n--- Version AVEC optimizer ---\n\n")

model = LinearRegressor(6)
lossfn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

train_losses = []
val_losses = []

for e in range(epochs):

	ohplai = []
	for x_batch, y_batch in trainloader:

		model.train()
		# forward
		mult = model(x_batch)
		loss = lossfn(mult, y_batch)

		ohplai.append(loss.item())
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	train_losses.append(np.array(ohplai).mean())

	with torch.no_grad():
		# compute test error

		model.eval()

		arouf = model(torch.from_numpy(X_test))
		loss_arouf = lossfn(arouf, torch.from_numpy(Y_test))

		val_losses.append(loss_arouf.item())

	if (e % 10) == 0:
		print("epoch %d " % e , "train MSE: ", train_losses[-1], "val MSE: ", loss_arouf.item())

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

plt.title("regression Boston Housing")
plt.plot(range(epochs), train_losses, label="train loss")
plt.plot(range(epochs), val_losses, label="val loss")
plt.legend()
plt.savefig("boston.png")


############ Version avec NN ############
print("\n\n--- Version avec NN + optimiseur ---\n\n")

epochs = 100
learningRate = 0.001

model = Perceval(6)
lossfn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

train_losses = []
val_losses = []

for e in range(epochs):

	ohplai = []
	for x_batch, y_batch in trainloader:

		model.train()
		# forward
		mult = model(x_batch)
		loss = lossfn(mult, y_batch)

		ohplai.append(loss.item())
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	train_losses.append(np.array(ohplai).mean())

	with torch.no_grad():
		# compute validation error

		model.eval()

		arouf = model(torch.from_numpy(X_test))
		loss_arouf = lossfn(arouf, torch.from_numpy(Y_test))

		val_losses.append(loss_arouf.item())

	if (e % 10) == 0:
		print("epoch %d " % e , "train MSE: ", train_losses[-1], "val MSE: ", loss_arouf.item())

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

plt.title("regression Boston Housing")
plt.plot(range(epochs), train_losses, label="train loss")
plt.plot(range(epochs), val_losses, label="val loss")
plt.legend()
plt.savefig("boston_perceval.png")