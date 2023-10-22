import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare the data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100 , noise=20 , n_features=1 , random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape
#PREPARE the model
input_size = n_features
output_size = 1
model = nn.Linear(input_size , output_size)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters() , lr=0.01)

epochs = 100
for epoch in range(epochs):
    #forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    #backward pass
    loss.backward()

    #update w
    optimizer.step()

    #zero w
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch:{epoch+1} , loss = {loss.item()}')
    
predicted = model(X).detach()
plt.plot(X_numpy, Y_numpy, "ro")
plt.plot(X_numpy, predicted, 'b')
plt.show()