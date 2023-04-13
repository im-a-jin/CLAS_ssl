import torch
import numpy as np

# Load data
train_path = ""
test_path = ""
train_data = PPGDataset(train_data)
test_data = PPGDataset(test_data)

# Initialize model
sig_dim = 0
latent_dim = 0
model = SSLModel(sig_dim, latent_dim)

# Initialize trainer and optimizer
trainer = Trainer(train_data, test_data)
sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Run trainer
epochs = 100
batch_size = 32
trainer.train(model, sgd, epochs=100, batch_size=batch_size)
