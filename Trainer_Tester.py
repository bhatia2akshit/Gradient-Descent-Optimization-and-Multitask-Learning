import torch.nn as nn
import torch


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


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)  # total number of instances among all batches.
    batch_count = len(dataloader)  # number of batches.
    model.train()
    model.to(device)
    if str(optimizer.__class__) == "<class 'FrankWolfOptimizer.FrankWolfOptimizer'>" and \
            len(optimizer.grads_all_tasks) > 0:
        optimizer.task_theta = []
        optimizer.grads_all_tasks = []
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # one X is a batch of 8 instances
        X, y = X.to(device), y.to(device)
        X = X.reshape(X.shape[0], 28*28)
        # Compute prediction error
        pred = model(X)
        loss_value = loss_fn(pred, y)
        current_loss = loss_value.item()
        train_loss += current_loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss /= batch_count
        correct /= size
        loss_value.backward()
        optimizer.collect_grads()  # it appends gradients to theta parameter but doesn't call frankwolf method.
        optimizer.zero_grad()


    optimizer.step()
    print('epoch training ends')


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)  # total number of instances among all batches.
    batch_count = len(dataloader)  # number of batches.
    model.eval()
    model.to(device)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            current_loss = loss_fn(pred, y).item()
            test_loss += current_loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= batch_count
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
