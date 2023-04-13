import numpy as np
import torch
import torch.nn as nn

class SSLModel(nn.Module):
    """Self-supervised model for ppg signals.

    The __init__ method takes in the input signal and hidden dimensions.

    Args:
        sig_dim: input signal dimension
        latent_dim: hidden dimension(s)

    Note:
        Model input and structure subject to change
    """
    def __init__(self, sig_dim, latent_dim):
        super(SSLModel, self).__init__()

        # TODO: define model layers (conv1d layers)
        # - follow https://arxiv.org/pdf/2109.07839.pdf
        # see https://pytorch.org/docs/stable/nn.html for layers/losses/etc.

    def transform(self, x):
        """Augments the input x (i.e horizontal flip, permutation, etc.)"""
        # TODO: add various transforms
        return x

    def forward(self, x):
        """Forward pass of the neural network"""
        raise NotImplementedError


class Classifier(nn.Module):
    """Classifier head for SSLModel.
    
    The __init__ method takes in the input, hidden, and output dimensions.

    Args:
        input_dim: input dimension
        hidden_dim: hidden dimension
        output_dim: output dimension (1 or -1)?
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass for the classifier"""
        return self.linear2(self.relu(self.linear1(x)))
