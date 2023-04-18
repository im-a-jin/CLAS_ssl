import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class PPGDataset(Dataset):
    """Dataset class of CLAS PPG signals.

    The __init__ method takes in the filepath of the data file.

    Args:
        filepath: path to data

    Attributes:
        data: ppg signal
        classes: cognitive load label
    """
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.data = torch.FloatTensor(data[0])
        self.classes = torch.LongTensor(data[1])

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.classes)

    def __getitem__(self, index):
        """Gets a (data, class) tuple from the dataset"""
        return self.data[index], self.classes[index]

