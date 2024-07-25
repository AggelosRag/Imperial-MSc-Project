import json
import logging.config

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, average_precision_score


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

def get_correct(y_hat, y, num_classes):
    if num_classes == 1:
        y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
        correct = [1 if y_hat[i] == y[i] else 0 for i in range(len(y_hat))]
        return np.sum(correct)
    else:
        return y_hat.argmax(dim=1).eq(y).sum().item()

def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs

def setup_logging(save_dir, log_config='loggers/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

