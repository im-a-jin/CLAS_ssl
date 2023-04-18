import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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

    def train_ssl(self, model, optimizer, epochs=1, batch_size=1, leave=False):
        """Training loop.

        Args:
            model: model to train
            optimizer: optimizer to use
            epochs: number of epochs to train
            batch_size: size of the mini-batch
            leave: option to leave visual training progress bar
        """
        train_loader = DataLoader(self.train_split, batch_size=batch_size,
                                  shuffle=True)
        model = model.to(self.device)
        # TODO: add per epoch logging
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
                optimizer.zero_grad()
                X1, X2 = X, model.transform(X)
                z1, z2 = model(X1), model(X2)
                # SSL loss criterion (cosine similarity loss?)
                loss = nn.CosineSimilarity(z1, z2)
                loss.backward()
                optimizer.step()

    def train_classifier(self, model, optimizer, criterion, epochs=1,
                         batch_size=1, leave=False):
        train_loader = DataLoader(self.train_split, batch_size=batch_size,
                                  shuffle=True)
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
                optimizer.zero_grad()
                y_ = model(X).squeeze()
                if type(criterion) is nn.BCEWithLogitsLoss:
                    y = F.one_hot((y > 0).long(), num_classes=2)
                loss = criterion(y_, y.float())
                loss.backward()
                optimizer.step()
            l, a = self._eval(model, criterion, self.train_split)
            log['train_loss'].append(l)
            log['train_acc'].append(a)
            l, a = self._eval(model, criterion, self.val_split)
            log['val_loss'].append(l)
            log['val_acc'].append(a)
        l, a = self._eval(model, criterion, self.test_dataset)
        log['test_loss'].append(l)
        log['test_acc'].append(a)
        return log, model

    def _eval(self, model, criterion, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        model = model.to(self.device)
        model.eval()
        with torch.inference_mode():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                y_ = model(X).squeeze()
                if type(criterion) is nn.BCEWithLogitsLoss:
                    y = F.one_hot((y > 0).long(), num_classes=2)
                loss = criterion(y_, y.float())
                acc = torch.sum(torch.sign(y_) == y) / len(y)
        return loss.item(), acc.item()
