import argparse
import copy
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from collections import Counter
from queue import Empty

import numpy as np
import pandas as pd
import torch
from SAnD.dataset import (DatasetBuilder, SubjectMontageData,
                          SubjectMontageDataset)
from SAnD.optim import ScheduledOptim
from SAnD.utils import (deterministic, evaluate, load_architecture,
                        save_predictions)
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
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

            deterministic(args.train_seed)

            # Set up Datasets and DataLoaders
            data = SubjectMontageData(
                os.path.join(
                    constants.SUBJECTS_DIR,
                    args.anchor,
                    'bandpass_only' if args.bandpass_only else 'rect_lowpass',
                    args.data_path),
                subject, montage,
                args.classification_task, args.seq_len, args.n_montages,
                args.pool_sep_montages, args.filter_zeros,
                args.pool_ops, args.max_abs_scale)

            db = DatasetBuilder(
                data=data, seed=args.seed, seed_cv=args.seed_cv)

            results = pd.DataFrame(columns=['cv_iter'])

            # Start with the same model initial state
            model = load_architecture(device, args)
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
                        shuffle=True,
                        collate_fn=SubjectMontageDataset.collate)
                    valid_loader = DataLoader(
                        valid_dataset, batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=SubjectMontageDataset.collate)

                    if args.nested_cross_val > 1 and args.cross_val > 1:
                        checkpoint_suffix = f'{i}-{j}'
                    # Multiple learn / test splits -
                    # checkpoint named for the enumeration of
                    # learn / test splits
                    elif args.nested_cross_val > 1:
                        checkpoint_suffix = str(i)
                    # Multiple train / valid splits -
                    # checkpoint named for the enumeration of
                    # train / valid splits
                    elif args.cross_val > 1:
                        checkpoint_suffix = str(j)
                    # No cross-validation
                    else:
                        checkpoint_suffix = ''

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
                    optimizer = ScheduledOptim(optimizer, 1, args.d_model,
                                               args.n_warmup_steps)

                    # Initialize criterion (binary)
                    if args.criterion == 'bce':
                        criterion = torch.nn.BCELoss()
                    elif args.criterion == 'bce_logits':
                        label_count = Counter(train_dataset.labels)
                        pos_weight = torch.Tensor(
                            [label_count[0] / label_count[1]]).to(device)
                        criterion = torch.nn.BCEWithLogitsLoss(
                            pos_weight=pos_weight)
                    else:
                        raise NotImplementedError('Criterion not implemented')

                    # Train for one iteration
                    model.load_state_dict(
                        copy.deepcopy(initialized_parameters))
                    trial_results = train(
                        subject, montage,
                        args, train_loader, valid_loader, test_loader,
                        optimizer, criterion, model, device, exp_dir,
                        verbose=args.verbose,
                        checkpoint_suffix=checkpoint_suffix)
                    trial_results['cv_iter'] = int(checkpoint_suffix) \
                        if checkpoint_suffix.isdigit() else checkpoint_suffix
                    trial_results['Status'] = 'PASS'
                    results = results.append(trial_results, ignore_index=True)

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

    start_time = time.time()

    # Initialize training state and default settings
    epochs_without_improvement = 0
    trial_results = dict()
    grad_norm = list()
    train_loss = list()
    train_acc = list()
    valid_loss = list()
    valid_acc = list()

    best_model = copy.deepcopy(model.state_dict())
    best_epoch_results = dict()
    best_valid_metrics = dict()
    best_valid_metrics['loss'] = -1
    best_valid_metrics['accuracy'] = -1
    best_valid_metrics['precision'] = -1
    best_valid_metrics['recall'] = -1
    best_valid_metrics['f1'] = -1
    best_valid_metrics['auc'] = -1
    best_valid_loss = 1e6
    best_valid_metric = -1.0

    for epoch in range(1, args.epochs + 1):

        if args.verbose:
            print('Epoch {}/{}'.format(epoch, args.epochs), flush=True)
            print('-' * 10, flush=True)

        # ======== TRAIN ======== #
        model.train()
        running_loss_train = 0.0
        running_corrects_train = 0.0
        running_grad_norm = 0.0
        total = 0.0
        prob_train = list()
        pred_train = list()
        true_train = list()

        for _, data, labels in train_loader:

            total += labels.shape[0]
            data = data.to(device) \
                if isinstance(data, torch.Tensor) \
                else [i.to(device) for i in data]
            labels = labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(data)
            outputs = outputs.squeeze(-1)
            probabilities = torch.sigmoid(outputs)
            predicted = probabilities > 0.5
            loss = criterion(outputs, labels)

            prob_train.extend(probabilities.data.tolist())
            pred_train.extend(predicted.data.tolist())
            true_train.extend(labels.data.tolist())

            loss.backward()
            optimizer.step()

            # Keep track of performance metrics (loss is mean-reduced)
            running_loss_train += loss.item() * data.size(0)
            running_corrects_train += torch.sum(
                predicted == labels.data).item()

            # Calculate gradient norms
            for p in list(
                    filter(lambda p: p.grad is not None, model.parameters())):
                running_grad_norm += p.grad.data.norm(2).item()

        # Evaluate training predictions against ground truth labels
        metrics_train = evaluate(true_train, pred_train, prob_train)

        # Log training metrics
        epoch_loss_train = running_loss_train / total
        epoch_acc_train = float(running_corrects_train) / total
        grad_norm_avg = running_grad_norm / total
        train_loss.append(epoch_loss_train)
        train_acc.append(epoch_acc_train)
        grad_norm.append(grad_norm_avg)
        auc_train = roc_auc_score(true_train, pred_train)

        if args.verbose:
            # Print metrics to stdout
            print('Train Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                epoch_loss_train, epoch_acc_train, auc_train), flush=True)

        # ======== VALID ======== #
        model.eval()
        running_loss_valid = 0.0
        running_corrects_valid = 0.0
        total = 0.0
        prob_valid = list()
        pred_valid = list()
        true_valid = list()
        with torch.no_grad():
            for _, data, labels in valid_loader:

                total += labels.shape[0]
                data = data.to(device) \
                    if isinstance(data, torch.Tensor) \
                    else [i.to(device) for i in data]
                labels = labels.to(device)

                outputs = model(data).squeeze(-1)
                probabilities = torch.sigmoid(outputs)
                predicted = probabilities > 0.5
                loss = criterion(outputs, labels)

                prob_valid.extend(probabilities.data.tolist())
                pred_valid.extend(predicted.data.tolist())
                true_valid.extend(labels.data.tolist())

                running_loss_valid += loss.item()

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss_valid += loss.item() * data.size(0)
                running_corrects_valid += torch.sum(
                    predicted == labels.data).item()

        # Evaluate validation predictions against ground truth labels
        metrics_valid = evaluate(true_valid, pred_valid, prob_valid)

        # Log validation metrics
        epoch_loss_valid = running_loss_valid / total
        epoch_acc_valid = float(running_corrects_valid) / total
        valid_loss.append(epoch_loss_valid)
        valid_acc.append(epoch_acc_valid)
        auc_valid = roc_auc_score(true_valid, pred_valid)

        if args.verbose:
            # Print metrics to stdout
            print('Valid Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                epoch_loss_valid, epoch_acc_valid, auc_valid), flush=True)

        # Save the best model at each epoch, using validation accuracy or
        # f1-score as the metric
        eps = 0.001
        if metrics_valid[args.metric] > best_valid_metric and \
                metrics_valid[args.metric] - best_valid_metric >= eps:
            # Reset early stopping epochs w/o improvement
            epochs_without_improvement = 0
            # Record best validation metrics
            best_valid_loss = epoch_loss_valid
            best_valid_metric = metrics_valid[args.metric]
            for metric, value in metrics_valid.items():
                best_valid_metrics[metric] = value
            best_valid_metrics['loss'] = epoch_loss_valid
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
        else:
            epochs_without_improvement += 1
            if args.early_stop != -1 and \
                    epochs_without_improvement >= args.early_stop:
                if args.verbose:
                    print('\nEarly stopping...\n', flush=True)
                trial_results['epoch_early_stop'] = epoch + 1
                break

        if args.verbose:
            print(flush=True)

    time_elapsed = time.time() - start_time
    if args.verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), flush=True)
        print('Best valid {}: {:4f}'.format(
            args.metric, best_valid_metric), flush=True)
        print('Best valid loss: {:4f}'.format(best_valid_loss), flush=True)
        print(flush=True)

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
    running_loss_test = 0.0
    total = 0.0
    prob_test = list()
    pred_test = list()
    true_test = list()
    with torch.no_grad():
        for _, data, labels in test_loader:

            total += labels.shape[0]
            data = data.to(device) \
                if isinstance(data, torch.Tensor) \
                else [i.to(device) for i in data]
            labels = labels.to(device)

            outputs = model(data).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = probabilities > 0.5
            loss = criterion(outputs, labels)

            prob_test.extend(probabilities.data.tolist())
            pred_test.extend(predicted.data.tolist())
            true_test.extend(labels.data.tolist())

            running_loss_test += loss.item() * data.size(0)

    final_metrics_test = evaluate(true_test, pred_test, prob_test)
    loss_test_avg = running_loss_test / total
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
    running_loss_test = 0.0
    total = 0.0
    prob_test = list()
    pred_test = list()
    true_test = list()
    with torch.no_grad():
        for _, data, labels in test_loader:

            total += labels.shape[0]
            data = data.to(device) \
                if isinstance(data, torch.Tensor) \
                else [i.to(device) for i in data]
            labels = labels.to(device)

            outputs = model(data).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = probabilities > 0.5
            loss = criterion(outputs, labels)

            prob_test.extend(probabilities.data.tolist())
            pred_test.extend(predicted.data.tolist())
            true_test.extend(labels.data.tolist())

            running_loss_test += loss.item() * data.size(0)

    metrics_test = evaluate(true_test, pred_test, prob_test)
    loss_test_avg = running_loss_test / total
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
    trial_results['final_test_loss'] = loss_test_avg
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
    parser.add_argument('--seq_len', type=int, default=156)
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
    parser.add_argument('--input_features', type=int, default=3)
    parser.add_argument('--epochs', type=int, help='Number of epochs',
                        default=200)
    parser.add_argument('--lr', type=float, help='Learning Rate',
                        default=0.001)
    parser.add_argument('--n_warmup_steps', type=int, default=10)
    parser.add_argument('--optimizer', type=str, help='Optimizer',
                        default='Adam')
    parser.add_argument('--batch_size', type=int, help='Mini-batch size',
                        default=32)
    parser.add_argument('--criterion', type=str, choices=['bce', 'bce_logits'],
                        default='bce')
    parser.add_argument('--early_stop', type=int,
                        help='Patience in early stop in validation set '
                        '(-1 -> no early stop)', default=50)
    parser.add_argument('--weight_decay', type=float, help='Weight decay',
                        default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_heads', type=int, default=3,
                        help='Number of attention heads to use')
    parser.add_argument('--factor', type=int, default=3,
                        help='M factor for dense interpolation')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of encoder layers to stack')
    parser.add_argument('--d_model', type=int,
                        help='Embedding size for the model', default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_cv', type=int, default=15)
    parser.add_argument('--train_seed', type=int,
                        help='Random seed for training', default=8)
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
    parser.add_argument('--n_procs', type=int, default=4)
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
        constants.RESULTS_DIR, args.classification_task,
        'chan_avg_transformer', args.anchor,
        'bandpass_only' if args.bandpass_only else 'rect_lowpass',
        'max_abs_scale' if args.max_abs_scale else 'no_scale',
        f'{args.n_montages}_montages',
        args.expt_name)
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
