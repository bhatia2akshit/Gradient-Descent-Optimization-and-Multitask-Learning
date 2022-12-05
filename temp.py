
import torch.utils.data
import torchvision
from torchvision.transforms import ToTensor
import torch.nn as nn
import math
# from FrankWolfOptimizer_new import FrankWolfOptimizer
from FrankWolfOptimizer import FrankWolfOptimizer
from Trainer_Tester import NeuralNetwork, train, test

device = "cuda" if torch.cuda.is_available() else "cpu"

image_data_train = torchvision.datasets.MNIST(root='./',train=True,download=True,transform=ToTensor())
image_data_test = torchvision.datasets.MNIST(root='./',train=False,download=True,transform=ToTensor())

image_data_train_reshaped = image_data_train.data.reshape(image_data_train.data.shape[0],28*28)
image_data_test_reshaped = image_data_test.data.reshape(image_data_test.data.shape[0],28*28)

batch_size = 1024
batch_count = math.ceil(len(image_data_train)/batch_size)

data_loader_train = torch.utils.data.DataLoader(image_data_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

data_loader_test = torch.utils.data.DataLoader(image_data_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

def main():
    modl = NeuralNetwork()
    loss = nn.CrossEntropyLoss()
    optim = FrankWolfOptimizer(modl.parameters(), device, lr=0.01, batch_count=batch_count, batch_size=batch_size,
                               max_iter=100)
    epochs = 200
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train(data_loader_train, modl, loss, optim, device)
        test(data_loader_test, modl, loss, device)

        print(f'The epoch {t} is finished')

main()