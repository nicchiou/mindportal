import argparse
import copy
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from queue import Empty

import numpy as np
import pandas as pd
import torch
from seq_rnn.dataset import (DatasetBuilder, MontagePretrainData,
                             SubjectMontageData, SubjectMontageDataset)
from seq_rnn.loss_fxns import FocalLoss
from seq_rnn.models import load_architecture
from seq_rnn.utils import (deterministic, evaluate, evaluate_ts,
                           save_predictions)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchsampler.imbalanced import ImbalancedDatasetSampler
from tqdm import tqdm
# (https://github.com/ufoym/imbalanced-dataset-sampler)
from utils import constants


def internal_model_runner(gpunum: int, args: argparse.Namespace, exp_dir: str,
                          input_queue: multiprocessing.Queue,
                          output_queue: multiprocessing.Queue):

    try:
        # Set up PyTorch device
        device = f'cuda:{gpunum}' if torch.cuda.is_available() else 'cpu'

        while not input_queue.empty():
            subject, montage = input_queue.get()

            # ========== PRE-TRAINING PHASE ========== #
            deterministic(args.train_seed)

            # Set up Datasets and DataLoaders for pre-training
            data = MontagePretrainData(
                os.path.join(
                    constants.SUBJECTS_DIR,
                    args.anchor,
                    'bandpass_only' if args.bandpass_only else 'rect_lowpass',
                    args.data_path),
                subject, montage,
                args.classification_task, args.n_montages,
                args.pool_sep_montages, args.filter_zeros, args.pool_ops,
                args.max_abs_scale)
            train_dataset = SubjectMontageDataset(data=data, subset='train',
                                                  stratified=args.stratified,
                                                  seed=args.seed)
            valid_dataset = SubjectMontageDataset(data=data, subset='valid',
                                                  stratified=args.stratified,
                                                  seed=args.seed)
            test_dataset = SubjectMontageDataset(data=data, subset='test',
                                                 stratified=args.stratified,
                                                 seed=args.seed)

            if args.use_imbalanced:
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size,
                    collate_fn=SubjectMontageDataset.collate,
                    sampler=ImbalancedDatasetSampler(train_dataset))
            else:
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True,
                    collate_fn=SubjectMontageDataset.collate)
            valid_loader = DataLoader(
                valid_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=SubjectMontageDataset.collate)
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=SubjectMontageDataset.collate)

            # If not provided, automatically set the dynamic input # size
            if not args.dynamic_input_size:
                args.dynamic_input_size = list(train_loader)[0][1][0].shape[-1]

            # Initialize PyTorch model with specified arguments and load to
            # device
            model = load_architecture(device, args)

            # Initialize optimizer
            if args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                             weight_decay=args.weight_decay)
            elif args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            momentum=0.9)
            elif args.optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                                                weight_decay=args.weight_decay,
                                                momentum=0.9)
            else:
                raise NotImplementedError('Optimizer not implemented')

            # Initialize criterion (binary)
            if args.criterion == 'bce':
                criterion = torch.nn.BCELoss()
            elif args.criterion == 'bce_logits':
                label_count = Counter(train_dataset.labels)
                pos_weight = torch.Tensor(
                    [label_count[0] / label_count[1]]).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif args.criterion == 'focal_loss':
                criterion = FocalLoss(args.focal_loss_gamma)
            else:
                raise NotImplementedError('Criterion not implemented')

            # Train
            trial_results = train(
                subject, montage,
                args, train_loader, valid_loader, test_loader, optimizer,
                criterion, model, device, exp_dir, verbose=args.verbose,
                checkpoint_suffix='pretrain')
            trial_results = {f'pretrain_{key}': value
                             for key, value in trial_results.items()}

            # ========== FINE-TUNING PHASE ========== #
            deterministic(args.train_seed)

            # Set up Datasets and DataLoaders for fine-tuning
            data = SubjectMontageData(
                os.path.join(
                    constants.SUBJECTS_DIR,
                    args.anchor,
                    'bandpass_only' if args.bandpass_only else 'rect_lowpass',
                    args.data_path),
                subject, montage,
                args.classification_task, args.n_montages,
                args.pool_sep_montages, args.filter_zeros, args.pool_ops,
                args.max_abs_scale)

            db = DatasetBuilder(data=data, seed=args.seed,
                                seed_cv=args.seed_cv)

            results = pd.DataFrame(columns=['cv_iter'])

            # Start with the same model initial state
            initialized_parameters = copy.deepcopy(model.state_dict())

            for i, (inner_train_valids, test_dataset) in enumerate(
                    db.build_datasets(cv=args.cross_val,
                                      nested_cv=args.nested_cross_val)):
                test_loader = DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False,
                    collate_fn=SubjectMontageDataset.collate)

                for j, (train_dataset, valid_dataset) in enumerate(
                        inner_train_valids):
                    train_loader = DataLoader(
                        train_dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=SubjectMontageDataset.collate)
                    valid_loader = DataLoader(
                        valid_dataset, batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=SubjectMontageDataset.collate)

                    if args.nested_cross_val > 1 and args.cross_val > 1:
                        checkpoint_suffix = f'{i}-{j}'
                    # Multiple learn / test splits -
                    # checkpoint named for the enumeration of learn / test
                    # splits
                    elif args.nested_cross_val > 1:
                        checkpoint_suffix = str(i)
                    # Multiple train / valid splits -
                    # checkpoint named for the enumeration of train / valid
                    # splits
                    elif args.cross_val > 1:
                        checkpoint_suffix = str(j)
                    # No cross-validation
                    else:
                        checkpoint_suffix = ''

                    # If not provided, automatically set the dynamic input #
                    # size
                    if not args.dynamic_input_size:
                        args.dynamic_input_size = list(
                            train_loader)[0][1][0].shape[-1]

                    # Initialize optimizer
                    if args.optimizer == 'Adam':
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
                    elif args.optimizer == 'SGD':
                        optimizer = torch.optim.SGD(
                            model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=0.9)
                    elif args.optimizer == 'RMSprop':
                        optimizer = torch.optim.RMSprop(
                            model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=0.9)
                    else:
                        raise NotImplementedError('Optimizer not implemented')

                    # Initialize criterion (binary)
                    if args.criterion == 'bce':
                        criterion = torch.nn.BCELoss()
                    elif args.criterion == 'bce_logits':
                        label_count = Counter(train_dataset.labels)
                        pos_weight = torch.Tensor(
                            [label_count[0] / label_count[1]]).to(device)
                        criterion = torch.nn.BCEWithLogitsLoss(
                            pos_weight=pos_weight)
                    elif args.criterion == 'focal_loss':
                        criterion = FocalLoss(args.focal_loss_gamma)
                    else:
                        raise NotImplementedError('Criterion not implemented')

                    # Train for one iteration
                    model.load_state_dict(
                        copy.deepcopy(initialized_parameters))
                    iter_results = copy.deepcopy(trial_results)
                    fine_tune_results = train(
                        subject, montage,
                        args, train_loader, valid_loader, test_loader,
                        optimizer, criterion, model, device, exp_dir,
                        checkpoint_suffix=checkpoint_suffix,
                        verbose=args.verbose)
                    iter_results.update(fine_tune_results)
                    iter_results['cv_iter'] = int(checkpoint_suffix) \
                        if checkpoint_suffix.isdigit() else checkpoint_suffix
                    iter_results['Status'] = 'PASS'
                    results = results.append(iter_results, ignore_index=True)

            output_queue.put(results)
            del model

    except Exception as e:
        del model
        traceback.print_exc()
        print(flush=True)
        raise e


def train(subject: str, montage: str,
          args: argparse.Namespace, train_loader: DataLoader,
          valid_loader: DataLoader, test_loader: DataLoader,
          optimizer: torch.optim, criterion: torch.nn.Module,
          model: torch.nn.Module, device, exp_dir: str,
          checkpoint_suffix: str = '', verbose=True):
    """
    Performs one iteration of training for the specified number of epochs
    (with early stopping if used).
    """

    # Save arguments as JSON
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logger.info(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Training {params} parameters')

    # Initialize training state and default settings
    epochs_without_improvement = 0
    trial_results = dict()
    best_epoch_results = dict()
    best_valid_metrics = dict()
    best_valid_metrics['loss'] = -1
    best_valid_metrics['accuracy'] = -1
    best_valid_metrics['precision'] = -1
    best_valid_metrics['recall'] = -1
    best_valid_metrics['f1'] = -1
    best_valid_metrics['auc'] = -1
    best_valid_loss = 1e6
    grad_norm = list()
    train_loss = list()
    train_acc = list()
    valid_loss = list()
    valid_acc = list()
    # keep track of last few models
    if args.early_stop != -1:
        last_models_list = [None] * args.early_stop
    else:
        last_models_list = [None]

    for _ in range(1, args.epochs):

        # ======== TRAIN ======== #
        model.train()
        running_loss_train = 0.0
        running_grad_norm = 0.0
        total = 0.0
        prob_train = defaultdict(list)
        pred_train = defaultdict(list)
        true_train = defaultdict(list)
        for _, data in enumerate(train_loader):
            _, dynamic_data, lengths, labels = \
                data[0], data[1], data[2], data[3].to(device)
            total += labels.shape[0]

            # Maximum sequence length - longer sequences are truncated from the
            # end (default: -1 -> do not truncate)
            if args.max_seq != -1:
                new_dynamic_data = []
                for data in dynamic_data:
                    if len(data) > args.max_seq:
                        new_dynamic_data.append(
                            data[len(data) - args.max_seq:])
                    else:
                        new_dynamic_data.append(data)
                dynamic_data = new_dynamic_data

            # Move to cuda and pad list of variable-length tensors with zero
            # (B x T x *)
            dynamic_data = [data.to(device) for data in dynamic_data]
            dynamic_data = pad_sequence(
                dynamic_data, batch_first=True, padding_value=0).to(device)

            model.zero_grad()
            loss = 0.0

            # dynamic_data.shape = (B, T, *), B = batch_size and T = maximum
            # sequence length
            effective_lengths = torch.ones(
                dynamic_data.shape[0]).long()
            c_lengths = torch.tensor(
                list(range(dynamic_data.shape[1]))).long()
            outputs = torch.zeros(dynamic_data.shape[0]).to(device)
            # initialized with zeros based on architecture
            hidden = model.init_hidden(dynamic_data.shape[0])
            max_seq_len = dynamic_data.shape[1]

            # Initialize tensor to store history of dynamic data hidden states
            dynamic_data_history = torch.zeros(
                dynamic_data.shape[0], dynamic_data.shape[1],
                args.hidden_size).to(device)

            # Step through time steps for the maximum sequence length, training
            # only those with data for the time step
            for seq_step in range(max_seq_len):
                step_events = dynamic_data[:, seq_step, :]
                non_zero = (effective_lengths != 0).nonzero(
                    as_tuple=False).squeeze()
                lens = effective_lengths[non_zero]
                events = step_events[non_zero]
                if len(lens.shape) != 1:
                    lens = lens.unsqueeze(dim=0)
                    events = events.unsqueeze(dim=0)
                events = events.unsqueeze(dim=1)

                if model.arch != 'lstm':
                    # 0-d array (because of squeeze) has only one element
                    if len(non_zero.shape) == 0:
                        (outputs[non_zero],
                         hidden[:, non_zero:non_zero + 1, :],
                         dynamic_data_event, _) = \
                            model((events, lens, hidden, dynamic_data_history),
                                  seq_step)
                    else:
                        (outputs[non_zero], hidden[:, non_zero, :],
                         dynamic_data_event, _) = \
                            model((events, lens, hidden, dynamic_data_history),
                                  seq_step)
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

                # Store history of dynamic data hidden states; update count of
                # samples for this time step
                dynamic_data_history[:, seq_step, :] = dynamic_data_event
                total += 1 if len(non_zero.shape) == 0 else len(non_zero)

                probability = torch.sigmoid(outputs[non_zero]).clone().data
                probability = probability.tolist()
                if isinstance(probability, list):
                    probability = probability
                else:
                    probability = [probability]
                for prob in probability:
                    prob_train[seq_step].append(prob)

                # Pass logits through a sigmoid and compare to preset
                # probability cutoff
                predicted = (torch.sigmoid(
                    outputs[non_zero]).clone().data >=
                        args.logistic_threshold).long()
                predicted = predicted.tolist()
                if isinstance(predicted, list):
                    predicted = predicted
                else:
                    predicted = [predicted]
                # Store predictions at each time step
                for pred in predicted:
                    pred_train[seq_step].append(pred)

                true_labels = labels[non_zero].clone().data.tolist()
                if isinstance(true_labels, list):
                    true_labels = true_labels
                else:
                    true_labels = [true_labels]
                # Store true labels at each time steps
                for label in true_labels:
                    true_train[seq_step].append(label)

                # Compute loss for time step
                if outputs[non_zero].size():
                    if args.criterion == 'bce_logits':
                        loss += criterion(outputs[non_zero].clone(),
                                          labels[non_zero].float())
                    else:
                        loss += criterion(
                            torch.sigmoid(outputs[non_zero]).clone(),
                            labels[non_zero].float())

                # lengths (B x 1) --> select samples that still have time steps
                # (time step < length of sequence)
                effective_lengths = (c_lengths[seq_step] < lengths - 1).long()

            if hasattr(loss, 'backward'):
                loss.backward()

            if args.clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clipping)

            if hasattr(loss, 'item'):
                optimizer.step()
                running_loss_train += loss.item() * labels.shape[0]

            # Calculate gradient norms
            for p in list(
                    filter(lambda p: p.grad is not None, model.parameters())):
                running_grad_norm += p.grad.data.norm(2).item()

        # Evaluate training predictions against ground truth labels
        metrics_train = evaluate(true_train, pred_train, prob_train)

        # Log training metrics
        loss_train_avg = running_loss_train / total
        grad_norm_avg = running_grad_norm / total
        train_loss.append(loss_train_avg)
        grad_norm.append(grad_norm_avg)
        train_acc.append(metrics_train['accuracy'])
        logger.info(f'train: avg_loss = {loss_train_avg:.5f}')
        for metric, value in metrics_train.items():
            logger.info(f'| {metric} = {value:.3f}')

        # ======== VALID ======== #
        model.eval()
        running_loss_valid = 0.0
        total = 0.0
        prob_valid = defaultdict(list)
        pred_valid = defaultdict(list)
        true_valid = defaultdict(list)
        with torch.no_grad():
            for data in valid_loader:
                _, dynamic_data, lengths, labels = \
                    data[0], data[1], data[2], data[3].to(device)
                total += labels.shape[0]
                dynamic_data = pad_sequence(
                    dynamic_data, batch_first=True, padding_value=0).to(device)

                loss = 0.0

                effective_lengths = torch.ones(
                    dynamic_data.shape[0]).long()
                c_lengths = torch.tensor(
                    list(range(dynamic_data.shape[1]))).long()
                outputs = torch.zeros(dynamic_data.shape[0]).to(device)
                hidden = model.init_hidden(dynamic_data.shape[0])
                max_seq_len = dynamic_data.shape[1]
                dynamic_data_history = torch.zeros(
                    len(data[0]), dynamic_data.shape[1], args.hidden_size
                    ).to(device)

                for seq_step in range(max_seq_len):
                    step_events = dynamic_data[:, seq_step, :]
                    non_zero = (effective_lengths != 0).nonzero(
                        as_tuple=False).squeeze()
                    lens = effective_lengths[non_zero]
                    events = step_events[non_zero]
                    if len(lens.shape) != 1:
                        lens = lens.unsqueeze(dim=0)
                        events = events.unsqueeze(dim=0)
                    events = events.unsqueeze(dim=1)

                    if args.arch != 'lstm':
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

                    # Store history of dynamic data hidden states; update count
                    # of samples for this time step
                    dynamic_data_history[:, seq_step, :] = dynamic_data_event
                    total += 1 if len(non_zero.shape) == 0 else len(non_zero)

                    probability = torch.sigmoid(outputs[non_zero]).clone().data
                    probability = probability.tolist()
                    if isinstance(probability, list):
                        probability = probability
                    else:
                        probability = [probability]
                    for prob in probability:
                        prob_valid[seq_step].append(prob)

                    # Pass logits through a sigmoid and compare to preset
                    # probability cutoff
                    predicted = (torch.sigmoid(
                        outputs[non_zero]).clone().data >=
                        args.logistic_threshold).long()
                    predicted = predicted.tolist()
                    if isinstance(predicted, list):
                        predicted = predicted
                    else:
                        predicted = [predicted]
                    for pred in predicted:
                        pred_valid[seq_step].append(pred)

                    true_labels = labels[non_zero].clone().data.tolist()
                    if isinstance(true_labels, list):
                        true_labels = true_labels
                    else:
                        true_labels = [true_labels]
                    for true in true_labels:
                        true_valid[seq_step].append(true)

                    # Compute loss for time step
                    if outputs[non_zero].size():
                        if args.criterion == 'bce_logits':
                            loss += criterion(
                                outputs[non_zero].clone(),
                                labels[non_zero].float())
                        else:
                            loss += criterion(
                                torch.sigmoid(outputs[non_zero]).clone(),
                                labels[non_zero].float())

                    # lengths (B x 1) --> select samples that still have time
                    # steps (time step < length of sequence)
                    effective_lengths = (
                        c_lengths[seq_step] < lengths - 1).long()

                if hasattr(loss, 'item'):
                    running_loss_valid += loss.item() * labels.shape[0]

        # Evaluate validation predictions against ground truth labels
        metrics_valid = evaluate(true_valid, pred_valid, prob_valid)

        # Log validation metrics
        loss_valid_avg = running_loss_valid / total
        valid_loss.append(loss_valid_avg)
        valid_acc.append(metrics_valid['accuracy'])
        logger.info(f'valid: avg_loss = {loss_valid_avg:.5f}')
        for metric, value in metrics_valid.items():
            logger.info(f'| {metric} = {value:.3f}')

        # Save checkpoints for the last few models
        if len(checkpoint_suffix) == 0:
            checkpoint_last_name = f'{subject}_{montage}_checkpoint_last.pt'
        else:
            checkpoint_last_name = \
                f'{subject}_{montage}_checkpoint_last_{checkpoint_suffix}.pt'
        torch.save(model.state_dict(),
                   os.path.join(exp_dir, 'checkpoints', checkpoint_last_name))

        # Save the best model, using validation accuracy or f1-score as the
        # metric
        eps = 0.001
        if loss_valid_avg < best_valid_loss and \
                best_valid_loss - loss_valid_avg >= eps:
            # Reset early stopping epochs w/o improvement
            epochs_without_improvement = 0
            # Record best validation metrics
            best_valid_loss = loss_valid_avg
            for metric, value in metrics_valid.items():
                best_valid_metrics[metric] = value
            best_valid_metrics['loss'] = loss_valid_avg
            # Update best epoch results
            best_epoch_results['train'] = {
                'true': true_train,
                'pred': pred_train,
                'prob': prob_train}
            best_epoch_results['valid'] = {
                'true': true_valid,
                'pred': pred_valid,
                'prob': prob_valid}
            # Save checkpoints
            if len(checkpoint_suffix) == 0:
                checkpoint_best_name = \
                    f'{subject}_{montage}_checkpoint_best.pt'
            else:
                checkpoint_best_name = \
                    f'{subject}_{montage}_checkpoint_best_{checkpoint_suffix}'\
                    '.pt'
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, 'checkpoints',
                                    checkpoint_best_name))
            # Save best model as a deepcopy
            best_model = copy.deepcopy(model)
            # Log best validation metrics
            logger.info(f'best valid loss = {best_valid_loss:.3f}')
            for metric, value in best_valid_metrics.items():
                logger.info(f'best valid {metric} = {value:.3f}')
        else:
            epochs_without_improvement += 1
            logger.info(f'best valid loss = {best_valid_loss:.3f}')
            for metric, value in best_valid_metrics.items():
                logger.info(f'best valid {metric} = {value:.3f}')
            if args.early_stop != -1 and \
                    epochs_without_improvement == args.early_stop:
                break
        # Remove the oldest model (last index) and add the latest model
        last_models_list.pop()
        last_models_list.insert(0, copy.deepcopy(model))

    # ======== EVALUATE FINAL MODEL ON TRAINING SET ======== #
    final_metrics_train = evaluate(true_train, pred_train, prob_train)
    save_predictions(subject, montage,
                     true_train, pred_train, prob_train,
                     exp_dir, 'train', checkpoint_suffix, final=True)

    # ======== EVALUTE FINAL MODEL ON VALIDATION SET ======== #
    final_metrics_valid = evaluate(true_valid, pred_valid, prob_valid)
    save_predictions(subject, montage,
                     true_valid, pred_valid, prob_valid,
                     exp_dir, 'valid', checkpoint_suffix, final=True)

    # ======== EVALUTE FINAL MODEL ON TEST SET ======== #
    model.eval()
    total = 0
    prob_test = defaultdict(list)
    pred_test = defaultdict(list)
    true_test = defaultdict(list)
    with torch.no_grad():
        for data in test_loader:
            _, dynamic_data, lengths, labels = \
                data[0], data[1], data[2], data[3].to(device)
            dynamic_data = pad_sequence(
                dynamic_data, batch_first=True, padding_value=0).to(device)

            effective_lengths = torch.ones(
                dynamic_data.shape[0]).long()
            c_lengths = torch.tensor(
                list(range(dynamic_data.shape[1]))).long()
            outputs = torch.zeros(dynamic_data.shape[0]).to(device)
            hidden = model.init_hidden(dynamic_data.shape[0])
            max_seq_len = dynamic_data.shape[1]
            dynamic_data_history = torch.zeros(
                len(data[0]), dynamic_data.shape[1], args.hidden_size
                ).to(device)

            for seq_step in range(max_seq_len):
                step_events = dynamic_data[:, seq_step, :]
                non_zero = (effective_lengths != 0).nonzero(
                    as_tuple=False).squeeze()
                lens = effective_lengths[non_zero]
                events = step_events[non_zero]
                if len(lens.shape) != 1:
                    lens = lens.unsqueeze(dim=0)
                    events = events.unsqueeze(dim=0)
                events = events.unsqueeze(dim=1)

                if args.arch != 'lstm':
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

                # Store history of dynamic data hidden states; update count of
                # samples for this time step
                dynamic_data_history[:, seq_step, :] = dynamic_data_event
                total += 1 if len(non_zero.shape) == 0 else len(non_zero)

                probability = torch.sigmoid(outputs[non_zero]).clone().data
                probability = probability.tolist()
                if isinstance(probability, list):
                    probability = probability
                else:
                    probability = [probability]
                for prob in probability:
                    prob_test[seq_step].append(prob)

                # Pass logits through a sigmoid and compare to preset
                # probability cutoff
                predicted = (torch.sigmoid(
                    outputs[non_zero]).clone().data >=
                        args.logistic_threshold).long()
                predicted = predicted.tolist()
                if isinstance(predicted, list):
                    predicted = predicted
                else:
                    predicted = [predicted]
                for pred in predicted:
                    pred_test[seq_step].append(pred)

                true_labels = labels[non_zero].clone().data.tolist()
                if isinstance(true_labels, list):
                    true_labels = true_labels
                else:
                    true_labels = [true_labels]
                for true in true_labels:
                    true_test[seq_step].append(true)

                # lengths (B x 1) --> select samples that still have time steps
                # (time step < length of sequence)
                effective_lengths = (c_lengths[seq_step] < lengths - 1).long()

    final_metrics_test = evaluate(true_test, pred_test, prob_test)
    save_predictions(subject, montage, true_test, pred_test, prob_test,
                     exp_dir, 'test', checkpoint_suffix, final=True)

    # ======== LOAD BEST MODEL ======== #
    model = load_architecture(device, args)
    model.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model.to(device)

    # ======== EVALUATE TRAIN ======== #
    metrics_train = evaluate(best_epoch_results['train']['true'],
                             best_epoch_results['train']['pred'],
                             best_epoch_results['train']['prob'])
    save_predictions(subject, montage,
                     best_epoch_results['train']['true'],
                     best_epoch_results['train']['pred'],
                     best_epoch_results['train']['prob'],
                     exp_dir, 'train', checkpoint_suffix)

    # ======== EVALUTE VALIDATION ======== #
    extra_name_parquet = f'_{checkpoint_suffix}' \
        if len(checkpoint_suffix) != 0 else ''
    _, _ = evaluate_ts(valid_loader, [model], device,
                       'valid' + extra_name_parquet, criterion,
                       args.logistic_threshold, exp_dir, max_seq=args.max_seq)
    metrics_valid = evaluate(best_epoch_results['valid']['true'],
                             best_epoch_results['valid']['pred'],
                             best_epoch_results['valid']['prob'])
    save_predictions(subject, montage,
                     best_epoch_results['valid']['true'],
                     best_epoch_results['valid']['pred'],
                     best_epoch_results['valid']['prob'],
                     exp_dir, 'valid', checkpoint_suffix)

    # ======== TEST ======== #
    model.eval()
    total = 0
    prob_test = defaultdict(list)
    pred_test = defaultdict(list)
    true_test = defaultdict(list)
    with torch.no_grad():
        for data in test_loader:
            _, dynamic_data, lengths, labels = \
                data[0], data[1], data[2], data[3].to(device)
            dynamic_data = pad_sequence(
                dynamic_data, batch_first=True, padding_value=0).to(device)

            effective_lengths = torch.ones(
                dynamic_data.shape[0]).long()
            c_lengths = torch.tensor(
                list(range(dynamic_data.shape[1]))).long()
            outputs = torch.zeros(dynamic_data.shape[0]).to(device)
            hidden = model.init_hidden(dynamic_data.shape[0])
            max_seq_len = dynamic_data.shape[1]
            dynamic_data_history = torch.zeros(
                len(data[0]), dynamic_data.shape[1], args.hidden_size
                ).to(device)

            for seq_step in range(max_seq_len):
                step_events = dynamic_data[:, seq_step, :]
                non_zero = (effective_lengths != 0).nonzero(
                    as_tuple=False).squeeze()
                lens = effective_lengths[non_zero]
                events = step_events[non_zero]
                if len(lens.shape) != 1:
                    lens = lens.unsqueeze(dim=0)
                    events = events.unsqueeze(dim=0)
                events = events.unsqueeze(dim=1)

                if args.arch != 'lstm':
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

                # Store history of dynamic data hidden states; update count of
                # samples for this time step
                dynamic_data_history[:, seq_step, :] = dynamic_data_event
                total += 1 if len(non_zero.shape) == 0 else len(non_zero)

                probability = torch.sigmoid(outputs[non_zero]).clone().data
                probability = probability.tolist()
                if isinstance(probability, list):
                    probability = probability
                else:
                    probability = [probability]
                for prob in probability:
                    prob_test[seq_step].append(prob)

                # Pass logits through a sigmoid and compare to preset
                # probability cutoff
                predicted = (torch.sigmoid(
                    outputs[non_zero]).clone().data >=
                        args.logistic_threshold).long()
                predicted = predicted.tolist()
                if isinstance(predicted, list):
                    predicted = predicted
                else:
                    predicted = [predicted]
                for pred in predicted:
                    pred_test[seq_step].append(pred)

                true_labels = labels[non_zero].clone().data.tolist()
                if isinstance(true_labels, list):
                    true_labels = true_labels
                else:
                    true_labels = [true_labels]
                for true in true_labels:
                    true_test[seq_step].append(true)

                # lengths (B x 1) --> select samples that still have time steps
                # (time step < length of sequence)
                effective_lengths = (c_lengths[seq_step] < lengths - 1).long()

    final_test_loss, _ = evaluate_ts(test_loader, [model], device,
                                     'test' + extra_name_parquet, criterion,
                                     args.logistic_threshold, exp_dir,
                                     max_seq=args.max_seq)
    metrics_test = evaluate(true_test, pred_test, prob_test)
    save_predictions(subject, montage, true_test, pred_test, prob_test,
                     exp_dir, 'test', checkpoint_suffix)

    # ======== SUMMARY ======== #
    trial_results['subject'] = subject
    trial_results['montage'] = montage
    trial_results['grad_norm'] = grad_norm
    trial_results['train_losses'] = train_loss
    trial_results['train_acc'] = train_acc
    trial_results['valid_losses'] = valid_loss
    trial_results['valid_acc'] = valid_acc
    trial_results['final_test_loss'] = final_test_loss
    # Save metrics of model selected by best validation accuracy
    for metric, value in metrics_train.items():
        trial_results['train_' + metric] = value
    for metric, value in metrics_valid.items():
        try:
            assert (value == best_valid_metrics[metric]) or \
                   (np.isnan(value) and np.isnan(best_valid_metrics[metric]))
        except AssertionError:
            print(value, best_valid_metrics[metric])
        trial_results['valid_' + metric] = value
    for metric, value in metrics_test.items():
        trial_results['test_' + metric] = value
    # Save metrics of final model at early stopping
    for metric, value in final_metrics_train.items():
        trial_results['final_train_' + metric] = value
    for metric, value in final_metrics_valid.items():
        trial_results['final_valid_' + metric] = value
    for metric, value in final_metrics_test.items():
        trial_results['final_test_' + metric] = value

    return trial_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('expt_name', type=str, help='Experiment name')
    parser.add_argument('--data_path', type=str, help='Path to data',
                        default='avg')
    parser.add_argument('--subset_subject_ids', action='store_true',
                        help='evaluate only a subset of the subjects (listed '
                        'in constants.py)')
    parser.add_argument('--subject', type=str, default='154',
                        help='evaluate a single subject (must be in '
                        'subset_subject_ids from constants.py)')
    parser.add_argument('--start_subject', type=str, default='127',
                        help='resume training at specific subject')
    parser.add_argument('--start_montage', type=str, default='a',
                        help='resume training at specific montage')
    parser.add_argument('--classification_task', type=str, default='motor_LR',
                        help='options include motor_LR (motor response), '
                        'stim_motor (stimulus modality and motor response) '
                        'and response_stim (response modality, stimulus '
                        'modality, and response polarity).')
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--n_montages', type=int, default=8,
                        help='number of montages to consider based on '
                        'grouped montages; options include 8 (a-h) or 4 '
                        '(grouped by trial)')
    parser.add_argument('--bandpass_only', action='store_true',
                        help='indicates whether to use the signal that has '
                        'not been rectified nor low-pass filtered')
    parser.add_argument('--filter_zeros', action='store_true',
                        help='Removes channels with all zeros from input.')
    parser.add_argument('--pool_sep_montages', action='store_true',
                        help='Specifies whether pooling operations should be '
                        'applied over paired montages (false) or pool over '
                        'separate montages.')
    parser.add_argument('--average_chan', action='store_true',
                        help='Average all input channels for each frequency '
                        'band before input into model.')
    parser.add_argument('--median_chan', action='store_true',
                        help='Take the median value of all input channels '
                        'for each frequency band before input into model.')
    parser.add_argument('--min_chan', action='store_true',
                        help='Take the minimum value of all input channels '
                        'for each frequency band before input into model.')
    parser.add_argument('--max_chan', action='store_true',
                        help='Take the maximum value of all input channels '
                        'for each frequency band before input into model.')
    parser.add_argument('--std_chan', action='store_true',
                        help='Take the standard deviation across all input '
                        'channels for each frequency band before input into '
                        'model.')
    parser.add_argument('--max_abs_scale', action='store_true')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention mechanisms for classification')
    parser.add_argument('--attention_type', type=str,
                        help='Attention type to be used', default='dot')
    parser.add_argument('--arch', type=str, help='Architecture',
                        default='lstm')
    parser.add_argument('--epochs', type=int, help='Number of epochs',
                        default=300)
    parser.add_argument('--lr', type=float, help='Learning Rate',
                        default=0.001)
    parser.add_argument('--optimizer', type=str, help='Optimizer',
                        default='Adam')
    parser.add_argument('--batch_size', type=int, help='Mini-batch size',
                        default=32)
    parser.add_argument('--criterion', type=str,
                        help='Possible options, "bce", "bce_logits", '
                        '"focal_loss"', default='bce')
    parser.add_argument('--focal_loss_gamma', type=float,
                        help='Gamma coefficient of focal loss', default=2)
    parser.add_argument('--early_stop', type=int,
                        help='Patience in early stop in validation set '
                        '(-1 -> no early stop)', default=50)
    parser.add_argument('--weight_decay', type=float, help='Weight decay',
                        default=0.0001)
    parser.add_argument('--dropout', type=float, help='Dropout in RNN and FC '
                        'layers', default=0.15)
    parser.add_argument('--dynamic_input_size', type=int, default=3,
                        help='Size of the dynamic input')
    parser.add_argument('--dynamic-embedding-size', type=int,
                        help='Size of the dynamic embedding', default=64)
    parser.add_argument('--hidden_size', type=int,
                        help='Hidden state size of the RNN', default=128)
    parser.add_argument('--rnn_layers', type=int,
                        help='Number of recurrent layers', default=1)
    parser.add_argument('--fc_layers', type=int,
                        help='Number of fully-connected layers after the '
                        'RNN output', default=1)
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN in the encoder')
    parser.add_argument('--clipping', type=float, help='Gradient clipping',
                        default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_cv', type=int, default=15)
    parser.add_argument('--train_seed', type=int,
                        help='Random seed for training', default=8)
    parser.add_argument('--logistic_threshold', type=float,
                        help='Threshold of the logistic regression',
                        default=0.5)
    parser.add_argument('--use_imbalanced', action='store_true',
                        help='Use imbalanced dataset for training')
    parser.add_argument('--no_stratified', action='store_true',
                        help='Disables stratified split')
    parser.add_argument('--cross_val', type=int,
                        help='K-Fold cross validation (if 1, ignored)',
                        default=1)
    parser.add_argument('--nested_cross_val', type=int,
                        help='Nested K-Fold cross validation (if 1, ignored)',
                        default=1)
    parser.add_argument('--metric', type=str, choices=['accuracy', 'f1'],
                        help='Metric to optimize and display',
                        default='accuracy')
    parser.add_argument('--max_seq', type=int,
                        help='Maximum sequence length (longer sequences are '
                        'truncated from the end default: -1 -> do not '
                        'truncate)', default=-1)
    parser.add_argument('--n_procs', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    args.stratified = not args.no_stratified
    args.pool_ops = {'mean': args.average_chan,
                     'median': args.median_chan,
                     'min': args.min_chan,
                     'max': args.max_chan,
                     'std': args.std_chan}

    # Set start method of multiprocessing library
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    mpl = multiprocessing.log_to_stderr()
    mpl.setLevel(logging.INFO)

    # Make experimental directories for output
    exp_dir = os.path.join(
        constants.RESULTS_DIR, args.classification_task, 'chan_avg_rnn',
        args.anchor,
        'bandpass_only' if args.bandpass_only else 'rect_lowpass',
        'max_abs_scale' if args.max_abs_scale else 'no_scale',
        f'{args.n_montages}_montages',
        args.arch, args.expt_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'final_predictions'), exist_ok=True)

    # Initialize queue objects
    input_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()

    # Assign montage list based on the desired number of montages
    if args.n_montages == 8:
        montage_list = constants.MONTAGES
    elif args.n_montages == 4:
        montage_list = constants.PAIRED_MONTAGES

    # Evaluate only a subset of subjects
    if args.subset_subject_ids:
        assert args.subject in constants.SUBSET_SUBJECT_IDS
        exp_dir = os.path.join(exp_dir, f's_{args.subject}')
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'final_predictions'), exist_ok=True)
        for montage in montage_list:
            input_queue.put((args.subject, montage))
    # Evaluate all subjects
    else:
        # Start training from a specific subject and montage
        subject_idx = np.argwhere(
            np.array(constants.SUBJECT_IDS) == args.start_subject)
        montage_idx = np.argwhere(
            np.array(montage_list) == args.start_montage)

        first_subject = True
        for subject in constants.SUBJECT_IDS[int(subject_idx):]:
            if first_subject:
                for montage in montage_list[int(montage_idx):]:
                    input_queue.put((subject, montage))
                first_subject = False
            else:
                for montage in montage_list:
                    input_queue.put((subject, montage))

    print(f'Approximate subject/montage queue size: {input_queue.qsize()}')
    prog_bar = tqdm(total=input_queue.qsize())

    # Create export DataFrame
    result_df = pd.DataFrame()

    # Set up processes
    gpus = [i // args.n_procs
            for i in range(args.n_procs * torch.cuda.device_count())]
    proclist = [multiprocessing.Process(
        target=internal_model_runner,
        args=(i, args, exp_dir, input_queue, results_queue)) for i in gpus]
    for proc in proclist:
        proc.start()

    procs_alive = True
    t0 = time.time()
    while procs_alive:
        try:
            trial_result = results_queue.get(timeout=20)
            result_df = result_df.append(trial_result, ignore_index=True)
            result_df.to_csv(
                os.path.join(exp_dir, 'trial_results.csv'), index=False)
            prog_bar.update(n=1)
        except Empty:
            pass
        except Exception:
            sys.exit(11)
        any_alive = False
        for proc in proclist:
            if proc.is_alive():
                any_alive = True
        procs_alive = any_alive

    for proc in proclist:
        proc.join()

    prog_bar.close()
    t1 = time.time()
    print(f'Finished training in {t1 - t0:.1f}s')
    sys.exit(0)
