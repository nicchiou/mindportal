import copy
import json
import os
import random
from collections import defaultdict
from operator import add
from typing import List
import logging

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


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


def evaluate(true: dict, pred: dict, prob: dict):
    """
    Given the true labels and the model's predictions and predicted
    probabilities of the positive class, this function evaluates the accuracy,
    precision, recall, F1-score, and ROC AUC averaged over all trials.
    :param true: dictionary of true labels for each time step
    :param pred: dictionary of the model predictions for each time step
    :param prob: dictionary of the model-predicted probabilities of the
                 positive class for each time step
    :return: dictionary containing the average metric values
    """

    eval_metrics = dict()

    # Iterate in reverse to get predictions and labels from the end
    final_true = true[max(true.keys())]
    final_pred = pred[max(pred.keys())]
    final_prob = prob[max(prob.keys())]

    try:
        _, fp, fn, tp = metrics.confusion_matrix(
            final_true, final_pred).ravel()
    except ValueError:
        if final_true[0] == 1 and final_pred[0] == 1:
            _, fp, fn, tp = 0, 0, 0, len(final_true)
        elif final_true[0] == 1 and final_pred[0] == 0:
            _, fp, fn, tp = 0, 0, len(final_true), 0
        elif final_true[0] == 0 and final_pred[0] == 1:
            _, fp, fn, tp = 0, len(final_true), 0, 0
        elif final_true[0] == 0 and final_pred[0] == 0:
            _, fp, fn, tp = len(final_true), 0, 0, 0
        print('true', final_true, len(final_true), flush=True)
        print('pred', final_pred, len(final_pred), flush=True)
    eval_metrics['accuracy'] = metrics.accuracy_score(final_true, final_pred)
    eval_metrics['precision'] = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    eval_metrics['recall'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    eval_metrics['f1'] = metrics.f1_score(final_true, final_pred)
    try:
        eval_metrics['auc'] = metrics.roc_auc_score(final_true, final_prob)
    except ValueError:
        pass

    return eval_metrics


def evaluate_ts(data_loader: DataLoader, models: List[torch.nn.Module],
                device: torch.device, subset_name: str, criterion,
                logistic_threshold: float, exp_dir: str, metric:
                str = 'accuracy', max_seq: int = -1, aggregate: str = 'add',
                aggregate_or_threshold: float = 0.5):
    """
    Evaluates model predictions over time steps
    on data provided by the DataLoader.
    :param data_loader: PyTorch DataLoader that contains data to be evaluated
    :param models: List of PyTorch models to compute evaluation metrics for
    :param device: Torch device to use
    :param subset_name: train, valid, or test denotes the DataLoader subset
    :param criterion: PyTorch criterion used to evaluate loss
    :param logistic_threshold: provided by Args (default 0.5)
    :param exp_dir: experiment directory to write to
    :param max_seq: maximum length sequence for truncation
    :return: evaluated loss, averaged over all examples
    """
    assert aggregate in ['add', 'or']
    assert aggregate_or_threshold > 0 and aggregate_or_threshold <= 1
    assert metric in ['accuracy', 'f1']

    total = 0
    loss_total = 0
    [model.eval() for model in models]
    with torch.no_grad():
        # key: seq_step, value: list of preds
        predictions = [defaultdict(list) for _ in range(len(models))]
        trial_ids_from_start = defaultdict(list)
        true = defaultdict(list)
        first = True

        for model_idx, model in enumerate(models):
            for data in data_loader:
                trial_ids, dynamic_data, lengths, labels = \
                    (data[0], data[1], data[2], data[3].to(device))

                # Maximum sequence length - longer sequences are truncated from
                # the end (default: -1 -> do not truncate)
                if max_seq != -1:
                    new_dynamic_data = []
                    for data in dynamic_data:
                        if len(data) > max_seq:
                            new_dynamic_data.append(data[len(data) - max_seq:])
                        else:
                            new_dynamic_data.append(data)
                    dynamic_data = new_dynamic_data

                dynamic_data_padded = pad_sequence(
                    dynamic_data, batch_first=True, padding_value=0).to(device)

                # dynamic_data.shape = (B, T, *), B = batch_size and
                # T = maximum sequence length
                effective_lengths = torch.ones(
                    dynamic_data_padded.shape[0]).long()
                c_lengths = torch.tensor(
                    list(range(dynamic_data_padded.shape[1]))
                    ).long()
                outputs = torch.zeros(dynamic_data_padded.shape[0]).to(device)
                hidden = model.init_hidden(dynamic_data_padded.shape[0])
                max_seq_step = dynamic_data_padded.shape[1]

                # Initialize tensor to store history of dynamic data hidden
                # states
                dynamic_data_history = torch.zeros(
                    len(data[0]), dynamic_data_padded.shape[1],
                    model.hidden_size).to(device)

                # Step through time steps for the maximum sequence length
                for seq_step in range(max_seq_step):
                    step_events = dynamic_data_padded[:, seq_step, :]
                    non_zero = (effective_lengths != 0).nonzero(
                        as_tuple=False).squeeze()
                    lens = effective_lengths[non_zero]
                    events = step_events[non_zero]
                    if len(lens.shape) != 1:
                        lens = lens.unsqueeze(dim=0)
                        events = events.unsqueeze(dim=0)
                    events = events.unsqueeze(dim=1)

                    if model.arch != 'lstm':
                        if len(non_zero.shape) == 0:
                            (outputs[non_zero],
                             hidden[:, non_zero:non_zero + 1, :],
                             dynamic_data_event, _) = \
                                model((events, lens, hidden,
                                       dynamic_data_history), seq_step)
                        else:
                            (outputs[non_zero], hidden[:, non_zero, :],
                             dynamic_data_event, _) = \
                                model((events, lens, hidden,
                                       dynamic_data_history), seq_step)
                    else:
                        outputs[non_zero], h, dynamic_data_event, _ = \
                            model((events, lens, hidden, dynamic_data_history),
                                  seq_step)
                        if len(non_zero.shape) == 0:
                            hidden[0][:, non_zero:non_zero + 1, :] = h[0]
                            hidden[1][:, non_zero:non_zero + 1, :] = h[1]
                        else:
                            hidden[0][:, non_zero, :] = h[0]
                            hidden[1][:, non_zero, :] = h[1]

                    # Append predictions
                    if isinstance(non_zero.tolist(), list):
                        non_zero_indexes = non_zero.tolist()
                    else:
                        non_zero_indexes = [non_zero.tolist()]
                    # Append predictions and trial ids from start
                    # (left-aligned sequences)
                    for pred_idx in non_zero_indexes:
                        pred = torch.sigmoid(outputs[pred_idx]).clone().data
                        # pred_seq_len = lengths.tolist()[pred_idx] - 1
                        predictions[model_idx][seq_step].append(pred)

                        # Furthermore, store the trial_ids for each step
                        pid = trial_ids[pred_idx]
                        if int(pid) not in trial_ids_from_start[seq_step]:
                            trial_ids_from_start[seq_step].append(int(pid))

                    # Store history of dynamic data hidden states
                    dynamic_data_history[:, seq_step, :] = dynamic_data_event

                    # Store true labels
                    if first:
                        true_labels = labels[non_zero].clone().data.tolist()
                        if isinstance(true_labels, list):
                            true_labels = true_labels
                        else:
                            true_labels = [true_labels]
                        for label in true_labels:
                            true[seq_step].append(label)
                        total += 1 if len(non_zero.shape) == 0 \
                            else len(non_zero)

                    # Compute loss
                    if outputs[non_zero].size():
                        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                            loss_total += criterion(
                                outputs[non_zero].clone(),
                                labels[non_zero].float())
                        else:
                            loss_total += criterion(
                                torch.sigmoid(outputs[non_zero]).clone(),
                                labels[non_zero].float())

                    # lengths (B x 1) --> select samples that still have time
                    # steps (time step < length of sequence)
                    effective_lengths = \
                        (c_lengths[seq_step] < lengths - 1).long()
            first = False
    # average loss over all voting members in ensemble

    loss_total /= len(models)
    # Compute predictions and from end (right-aligned sequences) using the
    # sequence length for each prediction
    max_steps = len(predictions[0].keys())

    # Compute voted predictions
    def aggregate_or(votes):
        """
        Returns 1 if the proportion of voting models surpasses
        aggregate_or_threshold, otherwise 0, in addition to the
        representative probability over all aggregated models.
        """
        return (1 if len(list(filter(lambda x: x == 1, votes))) / len(votes)
                >= aggregate_or_threshold else 0, sum(votes) / len(votes))

    predicted = defaultdict(list)
    predicted_probs = defaultdict(list)
    for step in range(max_steps):
        # For each step, sum the prediction of each model in the ensemble
        preds_votes = []
        if aggregate == 'add':
            for model_idx in range(len(models)):
                if len(preds_votes) == 0:
                    preds_votes = [
                        pred.tolist() for pred in predictions[model_idx][step]]
                else:
                    preds_votes = list(
                        map(add, preds_votes, [pred.tolist() for pred in
                            predictions[model_idx][step]]))
            predicted[step] = [
                1 if pred >= logistic_threshold * len(models)
                else 0 for pred in preds_votes]
            predicted_probs[step] = [
                pred / len(models) for pred in preds_votes]
        else:
            preds_votes_to_aggregate = []
            for model_idx in range(len(models)):
                if len(preds_votes_to_aggregate) == 0:
                    preds_votes_to_aggregate = [
                        pred.tolist() for pred in predictions[model_idx][step]]
                    preds_votes_to_aggregate = [
                        [1 if pred >= logistic_threshold else 0 for pred in
                         preds_votes_to_aggregate]]
                else:
                    new_votes = [
                        pred.tolist() for pred in predictions[model_idx][step]]
                    new_votes = [1 if pred >= logistic_threshold
                                 else 0 for pred in new_votes]
                    preds_votes_to_aggregate.append(new_votes)
            pred_probs_or = []
            for idx_pred_ in range(len(preds_votes_to_aggregate[0])):
                preds_votes.append(
                    aggregate_or([preds[idx_pred_]
                                  for preds in preds_votes_to_aggregate]))
            for idx_pred_vote, pred_vote in enumerate(preds_votes):
                decision, probs = pred_vote
                preds_votes[idx_pred_vote] = decision
                pred_probs_or.append(probs)
            predicted[step] = preds_votes
            predicted_probs[step] = pred_probs_or

    # Populate predictions and labels from the start for each trial
    predictions = dict()
    prediction_probs = dict()
    labels = dict()
    for step in predicted.keys():
        ids_ = trial_ids_from_start[step]
        labels_ = true[step]
        preds_ = predicted[step]
        probs_ = predicted_probs[step]
        for id, label, prediction, prob in zip(ids_, labels_, preds_, probs_):
            if step == 0:
                predictions[id] = []
                prediction_probs[id] = []
                labels[id] = label  # final outcome day label
            predictions[id].append(prediction)
            prediction_probs[id].append(prob)

    # Iterate in reverse to get predictions and labels from the end
    predicted_from_end = defaultdict(list)
    predicted_probs_from_end = defaultdict(list)
    trial_ids_from_end = defaultdict(list)
    true_from_end = defaultdict(list)
    predictions_copy = copy.deepcopy(predictions)
    predictions_probs_copy = copy.deepcopy(prediction_probs)
    for step in range(max_steps):
        y_pred = []
        y_prob = []
        y_true = []
        trial_ids_step = []
        for id in predictions_copy:
            if len(predictions_copy[id]) > 0:
                y_pred.append(predictions_copy[id].pop())
                y_prob.append(predictions_probs_copy[id].pop())
                y_true.append(labels[id])
                trial_ids_step.append(id)
        trial_ids_from_end[step] = trial_ids_step
        predicted_from_end[step] = y_pred
        predicted_probs_from_end[step] = y_prob
        true_from_end[step] = y_true

    # Save predictions and corrects labels
    # eval_preds = {"predictions_from_start": predicted,
    #               "predictions_from_end": predicted_from_end,
    #               "trial_ids_from_start": trial_ids_from_start,
    #               "trial_ids_from_end": trial_ids_from_end,
    #               "predicted_probs_from_start": predicted_probs,
    #               "predicted_probs_from_end": predicted_probs_from_end,
    #               "labels": true,
    #               "labels_from_end": true_from_end}
    # with open(
    #     os.path.join(exp_dir,
    #                  'eval_preds_' + subset_name + '.json'), 'w') as pn:
    #     json.dump(eval_preds, pn, cls=NumpyEncoder)

    # Compute evaluation metrics and write report
    eval_metrics = {"from_start": defaultdict(), "from_end": defaultdict()}
    for step in range(max_steps):
        # Mean over all the correct predictions at given step
        assert (len(predicted[step]) == len(true[step]) and
               len(predicted_from_end[step]) == len(true_from_end[step])), \
            'number of labels different from number of predictions'

        cfmat = metrics.confusion_matrix(true[step], predicted[step]).ravel()
        if len(cfmat) == 4:
            tn, fp, fn, tp = cfmat
        elif 0 not in true[step]:
            tn, fp, fn, tp = (0, 0, 0, cfmat[0])
        else:
            tn, fp, fn, tp = (cfmat[0], 0, 0, 0)
        eval_metrics["from_start"][step] = {
            "accuracy": metrics.accuracy_score(true[step], predicted[step]),
            "precision": tp / (tp + fp) if (tp + fp) != 0 else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) != 0 else np.nan,
            "f1": metrics.f1_score(true[step], predicted[step])
            if (tp + fn) * (tn + fp) != 0 else np.nan,
            "auc": metrics.roc_auc_score(true[step], predicted_probs[step])
            if len(set(true[step])) > 1 else np.nan,
            "corrects": metrics.accuracy_score(
                true[step], predicted[step], normalize=False),
            "examples": len(predicted[step])}

        cfmat = metrics.confusion_matrix(
            true_from_end[step], predicted_from_end[step]).ravel()
        if len(cfmat) == 4:
            tn, fp, fn, tp = cfmat
        elif 0 not in true[step]:
            tn, fp, fn, tp = (0, 0, 0, cfmat[0])
        else:
            tn, fp, fn, tp = (cfmat[0], 0, 0, 0)
        eval_metrics["from_end"][step] = {
            "accuracy": metrics.accuracy_score(
                true_from_end[step], predicted_from_end[step]),
            "precision": tp / (tp + fp) if (tp + fp) != 0 else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) != 0 else np.nan,
            "f1": metrics.f1_score(
                true_from_end[step], predicted_from_end[step])
            if (tp + fn) * (tn + fp) != 0 else np.nan,
            "auc": metrics.roc_auc_score(
                true_from_end[step], predicted_probs_from_end[step])
            if len(set(true_from_end[step])) > 1 else np.nan,
            "corrects": metrics.accuracy_score(
                true_from_end[step], predicted_from_end[step],
                normalize=False),
            "examples": len(predicted_from_end[step])}

    # Compute the weighted average across time steps, weighing each time step's
    # metric by the number of samples present # for the time step (implicitly)
    predicted_all_scores = []
    predicted_all_scores_from_end = []
    predicted_all = []
    predicted_all_from_end = []
    true_all = []
    true_all_from_end = []
    for step in range(max_steps):
        predicted_all_scores.extend(predicted_probs[step])
        predicted_all_scores_from_end.extend(predicted_probs_from_end[step])
        predicted_all.extend(predicted[step])
        predicted_all_from_end.extend(predicted_from_end[step])
        true_all.extend(true[step])
        true_all_from_end.extend(true_from_end[step])

    tn, fp, fn, tp = metrics.confusion_matrix(true_all, predicted_all).ravel()
    tn_from_end, fp_from_end, fn_from_end, tp_from_end = \
        metrics.confusion_matrix(true_all, predicted_all).ravel()
    eval_metrics['accuracy_avg'] = \
        metrics.accuracy_score(true_all, predicted_all)
    eval_metrics['accuracy_avg_from_end'] = \
        metrics.accuracy_score(true_all_from_end, predicted_all_from_end)
    eval_metrics['precision_avg'] = \
        tp / (tp + fp) if (tp + fp) != 0 else np.nan
    eval_metrics['precision_avg_from_end'] = \
        tp_from_end / (tp_from_end + fp_from_end) \
        if (tp_from_end + fp_from_end) != 0 else np.nan
    eval_metrics['recall_avg'] = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    eval_metrics['recall_avg_from_end'] = \
        tp_from_end / (tp_from_end + fn_from_end) \
        if (tp_from_end + fn_from_end) != 0 else np.nan
    eval_metrics['f1_avg'] = metrics.f1_score(true_all, predicted_all)
    eval_metrics['f1_avg_from_end'] = \
        metrics.f1_score(true_all_from_end, predicted_all_from_end)
    eval_metrics['auc_avg'] = \
        metrics.roc_auc_score(true_all, predicted_all_scores)
    eval_metrics['auc_avg_from_end'] = \
        metrics.roc_auc_score(true_all_from_end, predicted_all_scores_from_end)

    # Write evaluation report to file
    write_evaluation_report(eval_metrics, max_steps, subset_name, exp_dir)

    # Write predictions and probabilities to .csv files (loadable by Pandas)
    start_probs = pd.DataFrame(
        columns=['trial_id'] + [f'{step}' for step in range(max_steps)])
    start_preds = pd.DataFrame(
        columns=['trial_id'] + [f'{step}' for step in range(max_steps)])
    end_probs = pd.DataFrame(
        columns=['trial_id'] +
                [f'{max_steps - step}' for step in range(max_steps)])
    end_preds = pd.DataFrame(
        columns=['trial_id'] +
                [f'{max_steps - step}' for step in range(max_steps)])
    start_probs = start_probs.assign(trial_id=trial_ids_from_start[0])
    start_preds = start_preds.assign(trial_id=trial_ids_from_start[0])
    end_probs = end_probs.assign(trial_id=trial_ids_from_end[0])
    end_preds = end_preds.assign(trial_id=trial_ids_from_end[0])
    for step in range(max_steps):
        start_probs.loc[
            start_probs['trial_id'].isin(trial_ids_from_start[step]),
            f'{step}'] = predicted_probs[step]
        start_preds.loc[
            start_preds['trial_id'].isin(trial_ids_from_start[step]),
            f'{step}'] = predicted[step]
        end_probs.loc[
            end_probs['trial_id'].isin(trial_ids_from_end[step]),
            f'{step}'] = predicted_probs_from_end[step]
        end_preds.loc[
            end_preds['trial_id'].isin(trial_ids_from_end[step]),
            f'{step}'] = predicted_from_end[step]
    # start_probs.to_csv(
    #     os.path.join(exp_dir, 'eval_probs_start_' + subset_name + '.csv'),
    #     index=False)
    # start_preds.to_csv(
    #     os.path.join(exp_dir, 'eval_preds_start_' + subset_name + '.csv'),
    #     index=False)
    # end_probs.to_csv(
    #     os.path.join(exp_dir, 'eval_probs_end_' + subset_name + '.csv'),
    #     index=False)
    # end_preds.to_csv(
    #     os.path.join(exp_dir, 'eval_preds_end_' + subset_name + '.csv'),
    #     index=False)

    return float(loss_total / total), eval_metrics


def write_evaluation_report(eval_metrics: dict, max_steps: int,
                            subset_name: str, exp_dir: str):

    eval_report = '\t'.join(['timesteps since beginning',
                             'corrects',
                             'examples',
                             'accuracy per ts',
                             'accuracy average',
                             'precision per ts',
                             'precision average',
                             'recall per ts',
                             'recall average',
                             'f1 per ts',
                             'f1 average',
                             'auc per ts',
                             'auc average'
                             ])

    for step in range(max_steps):
        eval_report += '\t'.join([
            f'\n{step}',
            f'{eval_metrics["from_start"][step]["corrects"]}',
            f'{eval_metrics["from_start"][step]["examples"]}',
            f'{eval_metrics["from_start"][step]["accuracy"] * 100:.2f}%',
            f'{eval_metrics["accuracy_avg"] * 100:.2f}%',
            f'{eval_metrics["from_start"][step]["precision"] * 100:.2f}%',
            f'{eval_metrics["precision_avg"] * 100:.2f}%',
            f'{eval_metrics["from_start"][step]["recall"] * 100:.2f}%',
            f'{eval_metrics["recall_avg"] * 100:.2f}%',
            f'{eval_metrics["from_start"][step]["f1"] * 100:.2f}%',
            f'{eval_metrics["f1_avg"] * 100:.2f}%',
            f'{eval_metrics["from_start"][step]["auc"] * 100:.2f}%',
            f'{eval_metrics["auc_avg"] * 100:.2f}%',
            ])

    eval_report += '\n'
    eval_report += '\t'.join(['timesteps before end',
                              'corrects',
                              'examples',
                              'accuracy per ts',
                              'accuracy average',
                              'precision per ts',
                              'precision average',
                              'recall per ts',
                              'recall average',
                              'f1 per ts',
                              'f1 average',
                              'auc per ts',
                              'auc average'
                              ])
    for step in range(max_steps):
        eval_report += '\t'.join([
            f'\n{step}',
            f'{eval_metrics["from_end"][step]["corrects"]}',
            f'{eval_metrics["from_end"][step]["examples"]}',
            f'{eval_metrics["from_end"][step]["accuracy"] * 100:.2f}%',
            f'{eval_metrics["accuracy_avg"] * 100:.2f}%',
            f'{eval_metrics["from_end"][step]["precision"] * 100:.2f}%',
            f'{eval_metrics["precision_avg"] * 100:.2f}%',
            f'{eval_metrics["from_end"][step]["recall"] * 100:.2f}%',
            f'{eval_metrics["recall_avg"] * 100:.2f}%',
            f'{eval_metrics["from_end"][step]["f1"] * 100:.2f}%',
            f'{eval_metrics["f1_avg"] * 100:.2f}%',
            f'{eval_metrics["from_end"][step]["auc"] * 100:.2f}%',
            f'{eval_metrics["auc_avg"] * 100:.2f}%',
            ])

    logging.info(eval_report)
    # with open(
    #     os.path.join(exp_dir,
    #                  'eval_report_' + subset_name + '.csv'), 'w') as fn:
    #     fn.writelines(eval_report)
