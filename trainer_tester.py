import torch.nn as nn
import torch


class TrainTester:
    def __init__(self, device):
        self.train_loss_list, self.test_loss_list, self.train_accuracy_list, self.test_accuracy_list = [], [], [], []
        self.device = device

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)  # total number of instances among all batches.
        batch_count = len(dataloader)  # number of batches.
        model.train()
        model.to(self.device)
        if str(optimizer.__class__) == "<class 'FrankWolfOptimizer.FrankWolfOptimizer'>" and \
                len(optimizer.grads_all_tasks) > 0:
            optimizer.task_theta = []
            optimizer.grads_all_tasks = []
        train_loss, correct = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            # one X is a batch of 8 instances
            X, y = X.to(self.device), y.to(self.device)
            X = X.reshape(X.shape[0], 28*28)
            # Compute prediction error
            pred = model(X)
            loss_value = loss_fn(pred, y)
            current_loss = loss_value.item()
            # train_loss_list.append(current_loss)
            train_loss += current_loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_value.backward()
            if str(optimizer.__class__) == "<class 'FrankWolfOptimizer.FrankWolfOptimizer'>":
                optimizer.collect_grads()  # it appends gradients to theta parameter but doesn't call frankwolf method.
            optimizer.zero_grad()

        train_loss /= batch_count  # average loss per epoch.
        correct /= size  # average accuracy per epoch.
        self.train_accuracy_list.append(correct*100)
        self.train_loss_list.append(train_loss)
        print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

        optimizer.step()

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)  # total number of instances among all batches.
        batch_count = len(dataloader)  # number of batches.
        model.eval()
        model.to(self.device)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                current_loss = loss_fn(pred, y).item()
                test_loss += current_loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= batch_count  # average test loss per epoch.
        self.test_loss_list.append(test_loss)
        correct /= size  # average test accuracy per epoch.
        self.test_accuracy_list.append(correct*100)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
