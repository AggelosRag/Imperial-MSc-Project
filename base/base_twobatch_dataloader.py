import numpy as np
import torch
from torch.utils.data import DataLoader


class TwoBatchDataLoader(DataLoader):
    def __init__(self, dataset, ratio=0.5, shuffle=True):
        self.ratio = ratio
        self.shuffle = shuffle
        self.dataset = dataset
        self.num_samples = len(dataset)
        super(TwoBatchDataLoader, self).__init__(dataset,
                                                 batch_size=self.num_samples,
                                                 shuffle=self.shuffle)

    def __iter__(self):
        data_indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(data_indices)

        split_point = int(self.num_samples * self.ratio)
        first_batch_indices = data_indices[:split_point]
        second_batch_indices = data_indices[split_point:]

        first_batch = [self.dataset[i] for i in first_batch_indices]
        second_batch = [self.dataset[i] for i in second_batch_indices]

        first_batch_data = torch.stack([item[0] for item in first_batch])
        first_batch_labels = torch.stack([item[1] for item in first_batch])
        # check if the second batch is not an empty list
        if second_batch:
            second_batch_data = torch.stack([item[0] for item in second_batch])
            second_batch_labels = torch.stack([item[1] for item in second_batch])
        else:
            second_batch_data = None
            second_batch_labels = None

        yield ((first_batch_data, first_batch_labels),
               (second_batch_data, second_batch_labels))


class TwoBatchTripletDataLoader(DataLoader):
    def __init__(self, dataset, ratio=0.5, shuffle=True):
        self.ratio = ratio
        self.shuffle = shuffle
        self.dataset = dataset
        self.num_samples = len(dataset)
        super(TwoBatchTripletDataLoader, self).__init__(dataset,
                                                 batch_size=self.num_samples,
                                                 shuffle=self.shuffle)

    def __iter__(self):
        data_indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(data_indices)

        split_point = int(self.num_samples * self.ratio)
        first_batch_indices = data_indices[:split_point]
        second_batch_indices = data_indices[split_point:]

        first_batch = [self.dataset[i] for i in first_batch_indices]
        second_batch = [self.dataset[i] for i in second_batch_indices]

        first_batch_data = torch.stack([item[0] for item in first_batch])
        first_batch_concepts = torch.stack([item[1] for item in first_batch])
        first_batch_labels = torch.stack([item[2] for item in first_batch])
        # check if the second batch is not an empty list
        if second_batch:
            second_batch_data = torch.stack([item[0] for item in second_batch])
            second_batch_concepts = torch.stack([item[1] for item in second_batch])
            second_batch_labels = torch.stack([item[2] for item in second_batch])
        else:
            second_batch_data = None
            second_batch_concepts = None
            second_batch_labels = None

        yield ((first_batch_data, first_batch_concepts, first_batch_labels),
               (second_batch_data, second_batch_concepts, second_batch_labels))