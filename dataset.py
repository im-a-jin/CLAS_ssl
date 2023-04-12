import numpy as np
import torch
from torch.utils.data import Dataset

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
        amat = np.loadtxt(filepath)
        # TODO: modify amat indices to match data format
        self.data = torch.FloatTensor(amat[:, :-1])
        self.classes = torch.LongTensor(amat[:, -1])

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.classes)

    def __getitem__(self, index):
        """Gets a (data, class) tuple from the dataset"""
        return self.data[index], self.classes[index]

