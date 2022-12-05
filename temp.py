
import torch.utils.data
import torchvision
from torchvision.transforms import ToTensor
import torch.nn as nn
import math
import pickle
from FrankWolfOptimizer import FrankWolfOptimizer
from trainer_tester import TrainTester

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


class NeuralNetwork(nn.Module):
    def __init__(self):
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

def main():
    modl = NeuralNetwork()
    loss = nn.CrossEntropyLoss()
    optim = FrankWolfOptimizer(modl.parameters(), device, lr=0.01, batch_count=batch_count, batch_size=batch_size,
                               max_iter=100)
    train_test = TrainTester(device)
    epochs = 1

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_test.train(data_loader_train, modl, loss, optim)
        train_test.test(data_loader_test, modl, loss)

        print(f'The epoch {t+1} is finished')
    dict_stats = {'accuracy_train': train_test.train_accuracy_list, 'loss_train': train_test.train_loss_list,
                  'accuracy_test': train_test.test_accuracy_list, 'loss_test': train_test.test_loss_list}
    pickle.dump(dict_stats, open('./stats_saved.pkl','wb'))
main()