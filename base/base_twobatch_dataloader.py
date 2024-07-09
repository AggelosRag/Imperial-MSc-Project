import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class TwoBatchDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.data_indices = np.arange(self.num_samples)
        super(TwoBatchDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data_indices)

        self.batches = [self.data_indices[i:i + self.batch_size] for i in range(0, self.num_samples, self.batch_size)]
        return self._iter_batches()

    def _iter_batches(self):
        for batch_indices in self.batches:
            batch_set = Subset(self.dataset, batch_indices)
            rest_indices = np.setdiff1d(self.data_indices, batch_indices)
            rest_set = Subset(self.dataset, rest_indices)

            batch_loader = DataLoader(batch_set, batch_size=len(batch_set), shuffle=False)
            if len(rest_set) == 0:
                rest_loader = None
                rest_data = None
                rest_labels = None
            else:
                rest_loader = DataLoader(rest_set, batch_size=len(rest_set), shuffle=False)
                rest_data, rest_labels = next(iter(rest_loader))

            batch_data, batch_labels = next(iter(batch_loader))
            yield (batch_data, batch_labels), (rest_data, rest_labels)


class TwoBatchTripletDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        super(TwoBatchTripletDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        data_indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(data_indices)

        self.batches = [data_indices[i:i + self.batch_size] for i in range(0, self.num_samples, self.batch_size)]
        return self._iter_batches(data_indices)

    def _iter_batches(self, data_indices):
        for batch_indices in self.batches:
            batch_set = Subset(self.dataset, batch_indices)
            rest_indices = np.setdiff1d(data_indices, batch_indices)
            rest_set = Subset(self.dataset, rest_indices)

            batch_loader = DataLoader(batch_set, batch_size=len(batch_set), shuffle=False)
            if len(rest_set) == 0:
                rest_loader = None
                rest_data = None
                rest_concepts = None
                rest_labels = None
            else:
                rest_loader = DataLoader(rest_set, batch_size=len(rest_set), shuffle=False)
                rest_data, rest_concepts, rest_labels = next(iter(rest_loader))

            batch_data, batch_concepts, batch_labels = next(iter(batch_loader))
            yield (batch_data, batch_concepts, batch_labels), (rest_data, rest_concepts, rest_labels)