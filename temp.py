import torch
import numpy as np
import torchvision.datasets
import torch.utils.data
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.autograd import Function
from FrankWolfOptimizer import FrankWolfOptimizer

image_data_train = torchvision.datasets.MNIST(root='./',train=True,download=True,transform=ToTensor())
image_data_test = torchvision.datasets.MNIST(root='./',train=False,download=True,transform=ToTensor())

data_loader_train = torch.utils.data.DataLoader(image_data_train,
                                          batch_size=8,
                                          shuffle=True,
                                          )

data_loader_test = torch.utils.data.DataLoader(image_data_test,
                                          batch_size=8,
                                          shuffle=True,
                                          )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

inpsiz = 784
hidensiz = 512
numclases = 10
numepchs = 4
bachsiz = 100
l_r = 0.001


class NeuralNetwork(nn.Module):
    def __init__(self, inpsiz, hidensiz, numclases):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


modl = NeuralNetwork(inpsiz, hidensiz, numclases)
# loss = MultiClassHingeLoss()

loss = nn.CrossEntropyLoss()
# create DFW optimizer with learning rate of 0.1
# optim = DFW(modl.parameters(), eta=0.1)
optim = FrankWolfOptimizer(modl.parameters(), lr=1e-3, batch_size=7500, max_iter=30)


# mini training on batches for normal pytorch mnist setup
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    loss_list = []

    # optimizer.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        # one X is a batch of 8 instances
        print('batch number: ', batch)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss_value = loss_fn(pred, y)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        # loss_list.append(loss)


    # call frankwolfe method to compute combined gradient
    optimizer.frank_wolf_solver()


    # loss_tensor = torch.Tensor(loss_list)

    # loss_tensor.backward()


        # print(loss)
        # print(loss.T)
        # loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data_loader_train, modl, loss, optim)
    test(data_loader_test, modl, loss)
print("Done!")