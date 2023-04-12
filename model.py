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
        super(Model, self).__init__()

        # TODO: define model layers (conv1d layers)
        # see https://pytorch.org/docs/stable/nn.html for layers/losses/etc.

    def transform(self, x):
        """Augments the input x (i.e horizontal flip, permutation, etc.)"""
        return x

    def forward(self, x):
        """Forward pass of the neural network"""
        raise NotImplementedError
