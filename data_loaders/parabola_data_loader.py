import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_parabola_dataLoader(data_dir='./datasets/parabola',
                            type='SGD', config=None,
                            batch_size=None):

    # load the dataset
    X_train = np.load(data_dir + '/inputs.npy').T
    y_train = np.load(data_dir + '/targets.npy').reshape(-1, )
    X_test = np.load(data_dir + '/validation_inputs.npy').T
    y_test = np.load(data_dir + '/validation_targets.npy').reshape(-1, )
    init_weights = np.load(data_dir + '/init_weights.npy')

    X_test = X_test[:200]
    y_test = y_test[:200]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    data_train = TensorDataset(X_train, y_train)
    data_train_loader = DataLoader(dataset=data_train, batch_size=batch_size,
                                   shuffle=True)
    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test,
                                  batch_size=batch_size,
                                  shuffle=False)

    return data_train_loader, data_test_loader, None
