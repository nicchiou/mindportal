import json
import os
import random

import numpy as np
import torch
from sklearn import metrics


def deterministic(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class NumpyEncoder(json.JSONEncoder):
    """ Encodes Numpy arrays for json dump. """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def run_inference(model, device, data_loader, criterion):
    """
    Given a model, device, and dataloader, this function runs inference and
    evaluates model performance on the given subset of data.
    """
    model.eval()
    running_loss = 0.0
    total = 0.0
    prob = list()
    pred = list()
    true = list()
    with torch.no_grad():
        for _, data, labels in data_loader:

            total += labels.shape[0]
            data = data.to(device) \
                if isinstance(data, torch.Tensor) \
                else [i.to(device) for i in data]
            labels = labels.to(device)

            outputs = model(data).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = probabilities > 0.5
            loss = criterion(outputs, labels)

            prob.extend(probabilities.data.tolist())
            pred.extend(predicted.data.tolist())
            true.extend(labels.data.tolist())

            running_loss += loss.item() * data.size(0)

    metrics = evaluate(true, pred, prob)
    metrics['loss'] = running_loss / total
    return metrics


def evaluate(true, pred, prob):
    """
    Given the true labels and the model's predictions and predicted
    probabilities of the positive class, this function evaluates the accuracy,
    precision, recall, F1-score, and ROC AUC.
    :param true: list of true labels
    :param pred: list of the model predictions
    :param prob: list of the model-predicted probabilities of the
                 positive class
    :return: dictionary containing the average metric values
    """

    eval_metrics = dict()

    try:
        _, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    except ValueError:
        if true[0] == 1 and pred[0] == 1:
            _, fp, fn, tp = 0, 0, 0, len(true)
        elif true[0] == 1 and pred[0] == 0:
            _, fp, fn, tp = 0, 0, len(true), 0
        elif true[0] == 0 and pred[0] == 1:
            _, fp, fn, tp = 0, len(true), 0, 0
        elif true[0] == 0 and pred[0] == 0:
            _, fp, fn, tp = len(true), 0, 0, 0
        print('true', true, len(true), flush=True)
        print('pred', pred, len(pred), flush=True)
    eval_metrics['accuracy'] = metrics.accuracy_score(true, pred)
    eval_metrics['precision'] = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    eval_metrics['recall'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    eval_metrics['f1'] = metrics.f1_score(true, pred)
    try:
        eval_metrics['auc'] = metrics.roc_auc_score(true, prob)
    except ValueError:
        pass

    return eval_metrics
