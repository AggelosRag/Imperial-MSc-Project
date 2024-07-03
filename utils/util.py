import json

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])

        self.train_losses = []
        self.train_bce_losses = []
        self.train_target_losses = []
        self.val_losses = []
        self.val_bce_losses = []
        self.val_target_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_total_bce_losses = []
        self.val_total_bce_losses = []
        self.APLs_train = []
        self.APL_predictions_train = []
        self.APL_predictions_test = []
        self.APLs_test = []
        self.fidelities_train = []
        self.fidelities_test = []
        self.FI_train = []
        self.FI_test = []

        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class SimplerMetricTracker:

    def __init__(self, epochs, keys, writer=None, mode='train'):
        self._data = pd.DataFrame(index=[i for i in range(epochs)], columns=keys)
        self.writer = writer
        self.keys = keys
        self.mode = mode

        self.reset()

    def reset(self):
        self.batch_results = {key: [] for key in self.keys}

    def append_batch_result(self, key, value):
        self.batch_results[key].append(value)

    def append_epoch_result(self, epoch, key, value):
        # manually append a result to the epoch
        if key not in self._data.columns:
            self._data[key] = ''
        self._data.at[epoch, key] = value
        if self.writer is not None:
            if isinstance(value, list) == False:
                tb_key = self.mode + '_' + key
                self.writer.add_scalar(tb_key, value, epoch)

    def update_epoch(self, epoch):
        for key in self.keys:
            values = self.batch_results[key]
            total_value = sum(values)
            count = len(values)
            if key not in ['correct', 'total']:
                f_value = total_value / count if count > 0 else 0
                if self.writer is not None:
                    tb_key = self.mode + '_' + key
                    self.writer.add_scalar(tb_key, f_value, epoch)
            else:
                f_value = total_value
            self._data.loc[epoch, key] = f_value

        # fix for accuracy
        if (self._data.loc[epoch, 'correct'] is not None
                and self._data.loc[epoch, 'total'] is not None):
            self._data.loc[epoch, 'accuracy'] = (
                    self._data.loc[epoch, 'correct'] / self._data.loc[epoch, 'total'])
            tb_key = self.mode + '_' + "accuracy"
            self.writer.add_scalar(key, self._data.loc[epoch, 'accuracy'], epoch)
        else:
            raise ValueError('Both correct and total must be present in the epoch results.')

        # reset batch results for next epoch
        self.reset()

    def result(self, epoch):
        return dict(self._data.loc[epoch, :])

    def get_value(self, epoch, key):
        return self._data.loc[epoch, key]

    def all_results(self):
        return self._data
