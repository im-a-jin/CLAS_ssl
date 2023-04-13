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

    def train(self, model, optimizer, epochs=1, batch_size=1, leave=False):
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
