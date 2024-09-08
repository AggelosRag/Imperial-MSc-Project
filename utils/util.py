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
    if num_classes == 2:
        y_hat = torch.sigmoid(y_hat)
        y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
        correct = [1 if y_hat[i] == y[i] else 0 for i in range(len(y_hat))]
        return np.sum(correct)
    else:
        return y_hat.argmax(dim=1).eq(y).sum().item()

def correct_predictions_per_class(logits, true_labels, num_classes,
                                  threshold=0.5):
    """
    Calculate the number of correct predictions per class.

    Parameters:
    logits (torch.Tensor): A tensor of predicted logits (shape: [batch_size, num_classes] or [batch_size]).
    true_labels (torch.Tensor): A tensor of true labels (shape: [batch_size]).
    num_classes (int): The number of classes.
    threshold (float): Threshold to convert logits to binary predictions for single class case.

    Returns:
    list: A list containing the number of correct predictions for each class.
    """
    if num_classes == 2:
        # Binary classification case
        probabilities = torch.sigmoid(logits)
        predicted_classes = torch.where(probabilities >= threshold, 1, 0)
    else:
        # Multi-class classification case
        predicted_classes = torch.argmax(logits, dim=1)

    # Initialize a list to store the number of correct predictions per class
    correct_counts = [0] * num_classes

    # Iterate over each class and count correct predictions
    for class_idx in range(num_classes):
        correct_counts[class_idx] = torch.sum(
            (predicted_classes == class_idx) & (
                        true_labels == class_idx)).item()

    return correct_counts

def column_get_correct(logits, labels, threshold=0.5):
    """
    Calculate accuracy per column for predicted logits.

    Parameters:
    logits (torch.Tensor): A tensor of predicted logits (shape: [batch_size, num_labels]).
    labels (torch.Tensor): A tensor of true labels (shape: [batch_size, num_labels]).
    threshold (float): Threshold to convert logits to binary predictions.

    Returns:
    torch.Tensor: A tensor containing the accuracy for each column.
    """
    # Apply sigmoid to logits to get probabilities
    probabilities = torch.sigmoid(logits)

    # Convert probabilities to binary predictions based on the threshold
    predictions = (probabilities >= threshold).float()

    # Calculate accuracy per column
    correct_predictions = (predictions == labels).float()
    correct_predictions = correct_predictions.sum(dim=0)

    return correct_predictions

def count_labels_per_class(y):
    """
    Count the number of ground truth labels per class.

    Parameters:
    y (torch.Tensor): A tensor of ground truth class labels (shape: [batch_size]).

    Returns:
    dict: A dictionary with the class labels as keys and the number of ground truth labels as values.
    """
    # Get unique class labels and their counts
    unique_labels, counts = torch.unique(y, return_counts=True)

    # Create a dictionary with class labels as keys and counts as values
    label_counts = {int(label.item()): int(count.item()) for label, count in
                    zip(unique_labels, counts)}

    return label_counts

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

def compute_class_weights(
    config,
    train_dl,
    n_classes,
):

    task_class_weights = None

    if config.get('use_task_class_weights', False):
        logging.info(
            f"Computing task class weights in the training dataset with "
            f"size {len(train_dl)}..."
        )
        attribute_count = np.zeros((max(n_classes, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            if len(data) == 2:
                (_, (y, _)) = data
            else:
                (_, y, _) = data
            if n_classes > 1:
                y = torch.nn.functional.one_hot(
                    y,
                    num_classes=n_classes,
                ).cpu().detach().numpy()
            else:
                y = torch.cat(
                    [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                    dim=-1,
                ).cpu().detach().numpy()
            attribute_count += np.sum(y, axis=0)
            samples_seen += y.shape[0]
        print("Class distribution is:", attribute_count / samples_seen)
        if n_classes > 1:
            task_class_weights = samples_seen / attribute_count - 1
        else:
            task_class_weights = np.array(
                [attribute_count[0]/attribute_count[1]]
            )
    return task_class_weights