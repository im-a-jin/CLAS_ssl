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
    def __init__(self, in_dim, chs, ks, out_dim=1):
        super(SSLModel, self).__init__()
        layers, in_ch, L = [], 1, in_dim
        for i in range(len(chs)):
            layers.append(nn.Conv1d(in_ch, chs[i], ks[i]))
            layers.append(nn.BatchNorm1d(chs[i]))
            layers.append(nn.ReLU())
            in_ch = chs[i]
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(out_dim)
        self.pred = nn.Linear(chs[-1], chs[-1])

    def repr(self, x):
        x = x.unsqueeze(1)
        return self.avgpool(self.layers(x)).squeeze()

    def forward(self, x, pred=True):
        """Forward pass of the neural network"""
        x = x.unsqueeze(1)
        z = self.avgpool(self.layers(x)).squeeze()
        if pred:
            z = self.pred(z)
        return z


class Classifier(nn.Module):
    """Classifier head for SSLModel.
    
    The __init__ method takes in the input, hidden, and output dimensions.

    Args:
        input_dim: input dimension
        hidden_dim: hidden dimension
        output_dim: output dimension (1 or -1)?
    """
    def __init__(self, in_dim, dims, out_dim=1):
        super(Classifier, self).__init__()
        layers = [nn.Linear(in_dim, dims[0]), nn.ReLU()]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.pred = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        """Forward pass for the classifier"""
        return self.pred(self.layers(x))


class ConvClassifier(nn.Module):
    def __init__(self, in_dim, chs, ks, out_dim=1):
        super(ConvClassifier, self).__init__()
        layers, in_ch, L = [], 1, in_dim
        for i in range(len(chs)):
            layers.append(nn.Conv1d(in_ch, chs[i], ks[i]))
            layers.append(nn.BatchNorm1d(chs[i]))
            layers.append(nn.ReLU())
            in_ch = chs[i]
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.pred = nn.Sequential(
                nn.Linear(chs[-1], 8),
                nn.ReLU(),
                nn.Linear(8, out_dim,)
                )

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.layers(x)
        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        return self.pred(z)
