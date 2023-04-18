import scipy.io as io
import numpy as np
import pickle
import os

datadir = 'signals/'
labels = {0: -1, 1: 1, 2: 1}

def label_data(filename, split=50):
    mat = io.loadmat(os.path.join(datadir, filename))
    data = mat['allSegmentsParticpant'].squeeze()
    train_x, train_y = [], []
    for i in range(split):
        d = data[i]
        for j, s in enumerate(d):
            train_x.append(s)
            train_y.append(labels[j]*np.ones(d[j].shape[1]))
    test_x, test_y = [], []
    for i in range(split, len(data)):
        d = data[i]
        for j, s in enumerate(d):
            test_x.append(s)
            test_y.append(labels[j]*np.ones(d[j].shape[1]))
    train_x = np.concatenate(train_x, axis=1).T
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x, axis=1).T
    test_y = np.concatenate(test_y)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    trainpath = os.path.join(datadir, f"train_{filename}")
    testpath = os.path.join(datadir, f"test_{filename}")
    with open(trainpath, 'wb') as f:
        pickle.dump((train_x, train_y), f)
    with open(testpath, 'wb') as f:
        pickle.dump((test_x, test_y), f)
