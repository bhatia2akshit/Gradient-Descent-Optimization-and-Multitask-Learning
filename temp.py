import torch
import torch.utils.data
import torchvision
from torchvision.transforms import ToTensor
import torch.nn as nn
import math
from FrankWolfOptimizer_new import FrankWolfOptimizer

image_data_train = torchvision.datasets.MNIST(root='./',train=True,download=True,transform=ToTensor())
image_data_test = torchvision.datasets.MNIST(root='./',train=False,download=True,transform=ToTensor())

image_data_train_reshaped=image_data_train.data.reshape(image_data_train.data.shape[0],28*28)
image_data_test_reshaped=image_data_test.data.reshape(image_data_test.data.shape[0],28*28)

batch_size = 1024
num_batches = math.ceil(len(image_data_train)/batch_size)

data_loader_train = torch.utils.data.DataLoader(image_data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

data_loader_test = torch.utils.data.DataLoader(image_data_test,
                                          batch_size=batch_size,
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
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
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

optim = FrankWolfOptimizer(modl.parameters(), lr=1e-3, batch_size=num_batches, max_iter=30)  # ?? look ath this bro!


# mini training on batches for normal pytorch mnist setup
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    if len(optimizer.task_grads) > 0:
        # optimizer.task_theta = []
        optimizer.task_grads = []

    # optimizer.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        # one X is a batch of 8 instances
        # print('batch number: ', batch)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X, y = X.to(device), y.to(device)
        X = X.reshape(X.shape[0], 28*28)
        # Compute prediction error
        pred = model(X)

        loss_value = loss_fn(pred, y)
        loss_value.backward()
        optimizer.collect_grads()  # it appends gradients to theta parameter but doesn't call frankwolf method.
        optimizer.zero_grad()
    # call frankwolf method to compute combined gradient
    optimizer.step()


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
    # print(f"Epoch {t+1}\n-------------------------------")
    train(data_loader_train, modl, loss, optim)
    test(data_loader_test, modl, loss)

    print('an epoch is completed')
# print("Done!")