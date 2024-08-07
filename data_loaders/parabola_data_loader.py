import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

#from base import TwoBatchDataLoader


def get_parabola_dataLoader(ratio=0.2,
                            data_dir='./datasets/parabola',
                            type='Full-GD'):

    # load the dataset
    X_train = np.load(data_dir + '/inputs.npy').T
    y_train = np.load(data_dir + '/targets.npy').reshape(-1, )
    X_test = np.load(data_dir + '/validation_inputs.npy').T
    y_test = np.load(data_dir + '/validation_targets.npy').reshape(-1, )

    X_test = X_test[:200]
    y_test = y_test[:200]
    init_weights = np.load(data_dir + '/init_weights.npy')

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    data_train = TensorDataset(X_train, y_train)
    if type == 'Full-GD':
        data_train_loader = TwoBatchDataLoader(dataset=data_train,
                                               ratio=1.0,
                                               shuffle=True)
    elif type == 'Two-Batch':
        data_train_loader = TwoBatchDataLoader(dataset=data_train,
                                               ratio=ratio,
                                               shuffle=True)
    else:
        NotImplementedError('ERROR: data type not supported!')

    data_test = TensorDataset(X_test, y_test)
    data_test_loader = DataLoader(dataset=data_test,
                                  batch_size=X_test.shape[0],
                                  shuffle=False)

    return data_train_loader, data_test_loader