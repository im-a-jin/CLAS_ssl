import torch
import numpy as np
from dataset import *
from model import *
from trainer import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

# Load data
trainpath = "signals/train_6.mat.pkl"
testpath = "signals/test_6.mat.pkl"
train = PPGDataset(trainpath)
test = PPGDataset(testpath)

model = ConvClassifier(1280, [4, 16], [256, 64], out_dim=2)
trainer = Trainer(train, test)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

weights = [sum(train.classes == -1)/len(train), sum(train.classes == 1)/len(train)]
sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)

log, m = trainer.train_classifier(model, optim, criterion, sampler=sampler, epochs=100, batch_size=512)

print(log['test_acc'])

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# MSE Loss
# y_pred = torch.sign(m(test.data)).detach().numpy()
# y_true = test.classes.numpy()
# BCE Loss
y_pred = torch.argmax(m(test.data), dim=1).detach().numpy()
y_true = test.classes.numpy() > 0
cm = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, normalize='all'))
cm.plot()
