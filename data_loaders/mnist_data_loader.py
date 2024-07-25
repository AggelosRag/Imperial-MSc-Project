import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from base import TwoBatchTripletDataLoader, TwoBatchDataLoader


def get_mnist_dataLoader(data_dir='./datasets/parabola',
                        type='Full-GD',
                        batch_size=None):

    # Download training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    train_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=False,
                                  download=True,
                                  transform=transform)

    dict_of_lists = {6: [], 8: [], 9: []}
    for i, (_, label) in enumerate(train_dataset):
        if label in dict_of_lists.keys():
            dict_of_lists[label].append(
                train_dataset.data[i].reshape(1, 28, 28))

    for key in dict_of_lists.keys():
        dict_of_lists[key] = np.vstack(dict_of_lists[key]).reshape(-1, 1,
                                                                   28, 28)
        if key == 8:
            X = torch.cat((torch.tensor(dict_of_lists[6]),
                           torch.tensor(dict_of_lists[8])))
        elif key > 8:
            X = torch.cat((X, torch.tensor(dict_of_lists[key])))

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    # for i in range(4,10):
    for i in [6, 8, 9]:
        # targets += list(
        #     zip(thickness[i], width[i], slant[i], height[i]))
        targets += list(
            zip(thickness[i], width[i], length[i]))
        # targets += list(
        # zip(thickness[i], area[i], length[i],
        #                     width[i], height[i], slant[i]))
        if i == 6:
            k = 0
        elif i == 8:
            k = 1
        else:
            k = 2
        # labels.append([(i-4) for j in range(len(targets) - digits_size)])
        labels.append([k for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)

    def assign_bins(data, bin_edges):
        return np.digitize(data, bins=bin_edges, right=True)

    # Convert bin numbers to one-hot encoded values
    def one_hot_encode(bin_numbers, num_bins):
        return np.eye(num_bins)[bin_numbers - 1]

    bins_data_all = []
    for i in range(targets.shape[1]):
        # Combine the two lists
        combined_data = list(targets[:, i])

        # Sort the combined data
        combined_sorted = np.sort(combined_data)

        # Determine the number of data points per bin
        num_bins = 4
        bin_size = len(combined_sorted) // num_bins

        # Calculate bin edges
        bin_edges = [combined_sorted[i * bin_size] for i in
                     range(1, num_bins)] + [
                        combined_sorted[-1]]
        bin_edges = [-np.inf] + bin_edges

        # Assign bins to the original data lists
        bins_data = assign_bins(targets[:, i], bin_edges)

        # do one-hot encoding in the bins
        bins_data = one_hot_encode(bins_data, num_bins)

        # flatten the matrix
        bins_data_all.append(bins_data)

    # stack in the second dimension
    C = np.stack(bins_data_all, axis=1).reshape(-1, 12)

    y = np.array([item for sublist in labels for item in sublist])

    # Create synthetic dataset
    np.random.seed(42)
    num_classes = 3

    # Creating continuous concept targets (e.g., 5 concepts)

    # Standardize the data
    # scaler_X = StandardScaler()
    # scaler_C = StandardScaler()
    #
    # X = scaler_X.fit_transform(X)
    # C = scaler_C.fit_transform(C)

    # Split the data
    X_train, X_val, C_train, C_val, y_train, y_val = train_test_split(X, C, y,
                                                                      test_size=0.5,
                                                                      random_state=42)
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train[:8858], dtype=torch.float32)
    C_train = torch.tensor(C_train[:8858], dtype=torch.float32)
    y_train = torch.tensor(y_train[:8858], dtype=torch.long)

    X_val = torch.tensor(X_val[:8858], dtype=torch.float32)
    C_val = torch.tensor(C_val[:8858], dtype=torch.float32)
    y_val = torch.tensor(y_val[:8858], dtype=torch.long)

    # plot a bar plot with the number of concepts equal to 1 per class
    # for i in range(3):
    #     print(f'Class {i}')
    #     class_digit = C_train[y_train == i]
    #     for j in range(12):
    #         print(f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

    # Create DataLoader
    train_dataset = TensorDataset(X_train, C_train, y_train)
    val_dataset = TensorDataset(X_val, C_val, y_val)

    if type == 'Full-GD':
        data_train_loader = TwoBatchTripletDataLoader(dataset=train_dataset,
                                                      batch_size=X_train.shape[0],
                                                      shuffle=True)
        data_test_loader = TwoBatchTripletDataLoader(dataset=val_dataset,
                                                     batch_size=X_val.shape[0],
                                                     shuffle=False)
    elif type == 'Two-Batch':
        data_train_loader = TwoBatchTripletDataLoader(dataset=train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True)
        data_test_loader = TwoBatchTripletDataLoader(dataset=val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False)
    elif type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        data_test_loader = DataLoader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    else:
        NotImplementedError('ERROR: data type not supported!')

    return data_train_loader, data_test_loader


def get_mnist_cy_dataLoader(ratio=0.2,
                           data_dir='./datasets/parabola',
                           type='Full-GD',
                           batch_size=None):

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    # for i in range(4,10):
    for i in [6, 8, 9]:
        # targets += list(
        #     zip(thickness[i], width[i], slant[i], height[i]))
        targets += list(
            zip(thickness[i], width[i], length[i]))
        # targets += list(
        # zip(thickness[i], area[i], length[i],
        #                     width[i], height[i], slant[i]))
        if i == 6:
            k = 0
        elif i == 8:
            k = 1
        else:
            k = 2
        # labels.append([(i-4) for j in range(len(targets) - digits_size)])
        labels.append([k for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)

    def assign_bins(data, bin_edges):
        return np.digitize(data, bins=bin_edges, right=True)

    # Convert bin numbers to one-hot encoded values
    def one_hot_encode(bin_numbers, num_bins):
        return np.eye(num_bins)[bin_numbers - 1]

    bins_data_all = []
    for i in range(targets.shape[1]):
        # Combine the two lists
        combined_data = list(targets[:, i])

        # Sort the combined data
        combined_sorted = np.sort(combined_data)

        # Determine the number of data points per bin
        num_bins = 4
        bin_size = len(combined_sorted) // num_bins

        # Calculate bin edges
        bin_edges = [combined_sorted[i * bin_size] for i in
                     range(1, num_bins)] + [
                        combined_sorted[-1]]
        bin_edges = [-np.inf] + bin_edges

        # Assign bins to the original data lists
        bins_data = assign_bins(targets[:, i], bin_edges)

        # do one-hot encoding in the bins
        bins_data = one_hot_encode(bins_data, num_bins)

        # flatten the matrix
        bins_data_all.append(bins_data)

    # stack in the second dimension
    C = np.stack(bins_data_all, axis=1).reshape(-1, 12)

    y = np.array([item for sublist in labels for item in sublist])

    # Create synthetic dataset
    np.random.seed(42)
    num_classes = 3

    # Creating continuous concept targets (e.g., 5 concepts)

    # Standardize the data
    # scaler_X = StandardScaler()
    # scaler_C = StandardScaler()
    #
    # X = scaler_X.fit_transform(X)
    # C = scaler_C.fit_transform(C)

    # Split the data
    C_train, C_val, y_train, y_val = train_test_split(
        C,y, test_size=0.5,random_state=42
    )

    # Convert to PyTorch tensors
    C_train = torch.tensor(C_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    C_val = torch.tensor(C_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(C_train, y_train)
    val_dataset = TensorDataset(C_val, y_val)

    if type == 'Full-GD':
        data_train_loader = TwoBatchDataLoader(dataset=train_dataset,
                                               ratio=1.0,
                                               shuffle=True)
    elif type == 'Two-Batch':
        data_train_loader = TwoBatchDataLoader(dataset=train_dataset,
                                               ratio=ratio,
                                               shuffle=True)
    elif type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
    else:
        NotImplementedError('ERROR: data type not supported!')

    if batch_size is None:
        batch_size = C_val.shape[0]
    else:
        batch_size = batch_size

    data_test_loader = DataLoader(dataset=val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    return data_train_loader, data_test_loader

def collate_fn(batch):
    data, concepts, labels, indices = zip(*batch)
    data = torch.stack(data)
    concepts = torch.stack(concepts)
    labels = torch.stack(labels)
    indices = torch.tensor(indices)
    return data, concepts, labels, indices