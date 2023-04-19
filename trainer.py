import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import permute

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class Trainer():
    """General class for training models.

    The __init__ method takes in the model and the train/test datasets.

    Args:
        model: PyTorch neural network module
        train_dataset: Training dataset
        test_dataset: Test dataset
        device: Device to run calculations on

    Attributes:
        train_split: Training split (80% of dataset)
        val_split: Validation split (20% of dataset)
    """
    def __init__(self, train_dataset, test_dataset, device='cpu'):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.split_data()

    def split_data(self):
        """Splits train dataset into train and val datasets"""
        self.train_split, self.val_split = random_split(self.train_dataset,
                                                        [0.8, 0.2])

    def train_ssl(self, online, target, optimizer, sampler=None, epochs=1, batch_size=1,
                  leave=False):
        """Training loop.

        Args:
            model: model to train
            optimizer: optimizer to use
            epochs: number of epochs to train
            batch_size: size of the mini-batch
            leave: option to leave visual training progress bar
        """
        shuffle = True if sampler is None else False
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  sampler=sampler, shuffle=shuffle)
        online, target = online.to(self.device), target.to(self.device)
        log = {'train_acc': [],
               'train_loss': [],
               'val_acc': [],
               'val_loss': [],
               'test_acc': [],
               'test_loss': [],
                }
        ema = EMA(0.99)
        for epoch in tqdm(range(1, epochs+1), leave=leave):
            online.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                X1, X2 = permute(X), permute(X)
                o1, o2 = online(X1, pred=True), online(X2, pred=True)
                with torch.no_grad():
                    t1, t2 = target(X1, pred=False), target(X2, pred=False)
                    t1.detach(); t2.detach()
                loss1 = self._ssl_loss(o1, t2)
                loss2 = self._ssl_loss(o2, t1)
                loss = (loss1 + loss2).mean()
                loss.backward()
                optimizer.step()
                update_moving_average(ema, target, online)
        return log, target

    def _ssl_loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def train_classifier(self, model, optimizer, criterion, ssl=None,
                         sampler=None, epochs=1, batch_size=1, leave=False):
        shuffle = True if sampler is None else False
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  sampler=sampler, shuffle=shuffle)
        model = model.to(self.device)
        log = {'train_acc': [],
               'train_loss': [],
               'val_acc': [],
               'val_loss': [],
               'test_acc': [],
               'test_loss': [],
                }
        for epoch in tqdm(range(1, epochs+1), leave=leave):
            model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                if ssl is not None:
                    with torch.no_grad():
                        X = ssl.repr(X)
                optimizer.zero_grad()
                y_ = model(X).squeeze()
                if type(criterion) is nn.BCEWithLogitsLoss:
                    y = F.one_hot((y > 0).long(), num_classes=2)
                loss = criterion(y_, y.float())
                loss.backward()
                optimizer.step()
            l, a = self._eval(model, criterion, self.train_split, ssl=ssl)
            log['train_loss'].append(l)
            log['train_acc'].append(a)
            l, a = self._eval(model, criterion, self.val_split, ssl=ssl)
            log['val_loss'].append(l)
            log['val_acc'].append(a)
        l, a = self._eval(model, criterion, self.test_dataset, ssl=ssl)
        log['test_loss'].append(l)
        log['test_acc'].append(a)
        return log, model

    def _eval(self, model, criterion, dataset, ssl=None):
        loader = DataLoader(dataset, batch_size=len(dataset))
        model = model.to(self.device)
        model.eval()
        with torch.inference_mode():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                if ssl is not None:
                    with torch.no_grad():
                        X = ssl.repr(X)
                y_ = model(X).squeeze()
                if type(criterion) is nn.BCEWithLogitsLoss:
                    y = F.one_hot((y > 0).long(), num_classes=2)
                loss = criterion(y_, y.float())
                acc = torch.sum(torch.sign(y_) == y) / len(y)
        return loss.item(), acc.item()
