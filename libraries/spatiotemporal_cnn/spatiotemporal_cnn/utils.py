import json
import os
import random

import numpy as np
import pandas as pd
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


def evaluate(true: list, pred: list, prob: list):
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


def save_predictions(subject_id: str, montage: str,
                     true: list, pred: list, prob: list,
                     exp_dir: str, subset: str, checkpoint_suffix: str = '',
                     final: bool = False):
    """
    Writes a DataFrame containing the predictions for a given cross-validation
    iteration, along with the true labels and probabilities (confidence).
    """

    result_df = pd.DataFrame(
        columns=['subject_id', 'montage', 'cv_iter', 'true', 'pred', 'prob'])
    cv_iter = int(checkpoint_suffix) \
        if checkpoint_suffix.isdigit() else checkpoint_suffix

    result_df['true'] = true
    result_df['pred'] = pred
    result_df['prob'] = prob
    result_df.loc[:, 'subject_id'] = subject_id
    result_df.loc[:, 'montage'] = montage
    result_df.loc[:, 'cv_iter'] = cv_iter

    result_df.to_parquet(os.path.join(
        exp_dir, 'predictions' if not final else 'final_predictions',
        f'{subject_id}_{montage}_{subset}_{checkpoint_suffix}.parquet'),
        index=False)
