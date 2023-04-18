import torch
import numpy as np
from dataset import *
from model import *
from trainer import *
from utils import *

# Load data
trainpath = "signals/train_6.mat"
testpath = "signals/test_6.mat"
traindata = PPGDataset(trainpath)
testdata = PPGDataset(testpath)

# Initialize model
sig_dim = 1280
latent_dim = 0
ssl = SSLModel(sig_dim, latent_dim)

# Initialize trainer and optimizer
trainer = Trainer(train_data, test_data)

# Run ssl trainer
epochs = 100
batch_size = 32
adam = torch.optim.Adam(model.parameters(), lr=0.001)
trainer.train_ssl(model, sgd, epochs=100, batch_size=batch_size)

# Run classifier trainer
epochs = 100
batch_size = 32
adam = torch.optim.Adam(model.parameters(), lr=0.001)
trainer.train_classifier(model,)
