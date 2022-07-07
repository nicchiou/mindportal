import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.zoopt import ZOOptSearch
from sklearn.metrics import roc_auc_score
from spatiotemporal_cnn.dataset_bci import DatasetBuilder, SingleSubjectData
from spatiotemporal_cnn.models_bci import SpatiotemporalCNN
from spatiotemporal_cnn.ray_tune import ExperimentPlateauAcrossTrialsStopper
from spatiotemporal_cnn.utils import (deterministic, evaluate, run_inference,
                                      save_predictions)
from torch.utils.data import DataLoader
# (https://github.com/ufoym/imbalanced-dataset-sampler)
from utils import constants


def train_with_configs(config, args: argparse.Namespace,
                       checkpoint_dir: str = None):
    """
    Trainable function for Ray Tune hyperparameter tuning with specified config
    parameters.
    """
    deterministic(args.train_seed)

    try:
        # Set up Datasets and DataLoaders
        data_dir = os.path.join(
            constants.PH_SUBJECTS_DIR
            if args.data_type == 'ph' else constants.DC_SUBJECTS_DIR,
            'bci', args.input_space, args.data_path
        )
        bci_data = SingleSubjectData(
            data_dir=data_dir,
            subject_id=args.subject,
            train_submontages=args.train_submontages,
            classification_task=args.classification_task,
            expt_type=args.expt_type,
            response_speed=args.response_speed,
            filter_zeros=args.filter_zeros,
            input_space=args.input_space,
            data_type=args.data_type
        )

        # Get number of available features
        args.num_features = bci_data.get_num_viable_features()

        db = DatasetBuilder(
            data=bci_data, seed=args.seed, seed_cv=args.seed_cv,
            max_abs_scale=args.max_abs_scale,
            impute=args.imputation_method)

        cv_results = pd.DataFrame(columns=['cv_fold'])

        # Start with the same model initial state
        model = SpatiotemporalCNN(
            C=args.num_features,  # number of input channels/voxels
            F1=config['F1'],      # number of temporal filters
            D=config['D'],        # number of spatial filters
            F2=config['F2'],      # number of pointwise filters
            p=config['dropout'],  # probability of dropout
            fs=config['fs'],      # sampling frequency
            T=config['T']         # sequence length
        )
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        if args.verbose:
            print(f'{params} trainable parameters', flush=True)
        initialized_parameters = copy.deepcopy(model.state_dict())

        # Put model on GPU
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        model.to(device)

        # Cross-validation loop
        for i, (inner_train_valids, test_dataset) in enumerate(
                db.build_datasets(cv=args.cross_val,
                                  nested_cv=args.nested_cross_val)):

            for j, (train_dataset, valid_dataset) in enumerate(
                    inner_train_valids):

                valid_dataset.impute_chan(train_dataset)
                test_dataset.impute_chan(train_dataset)

                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True)
                valid_loader = DataLoader(
                    valid_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False)

                if args.nested_cross_val > 1 and args.cross_val > 1:
                    raise NotImplementedError(
                        'Nested cross validation not supported.')
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
                    checkpoint_suffix = str(j)

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
                else:
                    raise NotImplementedError('Criterion not implemented')

                # Train for one cross-validation iteraion
                model.load_state_dict(
                    copy.deepcopy(initialized_parameters))
                fold_results, model = train(
                    args.subject, args, config,
                    train_loader, valid_loader, test_loader,
                    optimizer, criterion, model, device, exp_dir,
                    checkpoint_suffix=checkpoint_suffix)
                fold_results['cv_fold'] = int(checkpoint_suffix)
                fold_results['Status'] = 'PASS'
                cv_results = cv_results.append(fold_results, ignore_index=True)

                # Save trained model checkpoint
                with tune.checkpoint_dir(
                        step=checkpoint_suffix) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, 'checkpoint.pt')
                    torch.save(
                        (model.state_dict(), optimizer.state_dict()), path)

                # Save model predictions
                trial_dir = tune.get_trial_dir()
                os.makedirs(os.path.join(trial_dir, 'predictions'),
                            exist_ok=True)
                save_predictions(
                    args.subject, 'all',
                    fold_results['fetches']['train']['true'],
                    fold_results['fetches']['train']['pred'],
                    fold_results['fetches']['train']['prob'],
                    trial_dir, 'train', checkpoint_suffix)
                save_predictions(
                    args.subject, 'all',
                    fold_results['fetches']['valid']['true'],
                    fold_results['fetches']['valid']['pred'],
                    fold_results['fetches']['valid']['prob'],
                    trial_dir, 'valid', checkpoint_suffix)

        del model

    except Exception as e:
        if 'model' in locals():
            del model
        traceback.print_exc()
        print(flush=True)
        raise e

    # Save cross-validation results to .csv
    cv_results.to_csv(os.path.join(trial_dir, 'trial_results.csv'),
                      index=False)

    # Report result
    tune.report(**cv_results.mean().to_dict())


def train(subject: str, args: argparse.Namespace, config: dict,
          train_loader: DataLoader, valid_loader: DataLoader,
          test_loader: DataLoader, optimizer: torch.optim,
          criterion: torch.nn.Module, model: torch.nn.Module,
          device, exp_dir: str, checkpoint_suffix: str = ''):
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
    fold_results = dict()
    grad_norm = list()
    train_loss = list()
    train_acc = list()
    valid_loss = list()
    valid_acc = list()

    best_model = copy.deepcopy(model.state_dict())
    best_model_epoch = 0
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
            outputs = model(data).squeeze(-1)
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
                'prob': prob_train
            }
            best_epoch_results['valid'] = {
                'true': true_valid,
                'pred': pred_valid,
                'prob': prob_valid
            }
            # Save best model as a deepcopy
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
        else:
            epochs_without_improvement += 1
            if epoch > args.min_epochs and \
                    epochs_without_improvement >= args.early_stop:
                fold_results['epoch_early_stop'] = epoch
                if args.early_stop != -1:
                    if args.verbose:
                        print('\nEarly stopping...\n', flush=True)
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

    # ======== LOAD BEST MODEL ======== #
    model = SpatiotemporalCNN(
        C=args.num_features,
        F1=config['F1'],
        D=config['D'],
        F2=config['F2'],
        p=config['dropout'],
        fs=config['fs'],
        T=config['T']
    )
    model.load_state_dict(copy.deepcopy(best_model.state_dict()))
    model.to(device)

    # Evaluate model
    metrics_train = run_inference(model, device, train_loader, criterion)
    metrics_valid = run_inference(model, device, valid_loader, criterion)
    metrics_test = run_inference(model, device, test_loader, criterion)

    # ======== SUMMARY ======== #
    fold_results['subject_id'] = subject
    fold_results['grad_norm'] = grad_norm
    fold_results['train_losses'] = train_loss
    fold_results['train_acc'] = train_acc
    fold_results['valid_losses'] = valid_loss
    fold_results['valid_acc'] = valid_acc
    fold_results['epoch_early_stop'] = best_model_epoch
    # Save metrics of model selected by best validation accuracy
    for metric, value in metrics_train.items():
        fold_results['train_' + metric] = value
    for metric, value in metrics_valid.items():
        try:
            assert (value == best_valid_metrics[metric]) or \
                   (np.isnan(value) and np.isnan(best_valid_metrics[metric]))
        except AssertionError:
            print(metric, value, best_valid_metrics[metric])
        fold_results['valid_' + metric] = value
    for metric, value in metrics_test.items():
        fold_results['test_' + metric] = value
    # Save fetches from model
    fold_results['fetches'] = best_epoch_results

    return fold_results, model


def main(args: argparse.Namespace, exp_dir: str):
    """
    Sets up Ray Tune hyperparameter search and saves the best trial result.
    """
    # Search algorithm
    points_to_evaluate = [{
            'fs': args.fs,
            'T': args.seq_len,
            'F1': 8,
            'D': 8,
            'F2': 12,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
            'l2': args.weight_decay
    }]
    F1_search_space = tune.randint(4, 19)
    D_search_space = tune.randint(4, 25)
    F2_search_space = tune.randint(4, 18)
    lr_search_space = args.lr
    batch_size_search_space = args.batch_size  # tune.randint(8, 33)
    dropout_search_space = args.dropout  # tune.uniform(0.4, 0.6)
    l2_search_space = args.weight_decay  # tune.loguniform(1e-4, 1e-2)

    if args.search_algo == 'grid':
        algo = BasicVariantGenerator(random_state=0)
        F1_search_space = tune.grid_search([6, 8, 12])
        D_search_space = tune.grid_search([6, 8, 12])
        F2_search_space = tune.grid_search([6, 8, 12])
        lr_search_space = args.lr
        batch_size_search_space = args.batch_size
        dropout_search_space = args.dropout
        l2_search_space = args.weight_decay
    elif args.search_algo == 'bayes_opt':
        algo = AxSearch(
            metric='valid_accuracy', mode='max',
            enforce_sequential_optimization=False,
            points_to_evaluate=points_to_evaluate
        )
        algo = ConcurrencyLimiter(algo, max_concurrent=2, batch=True)
    elif args.search_algo == 'zoopt':
        algo = ZOOptSearch(
            budget=args.num_samples, metric='valid_accuracy', mode='max',
            parallel_num=2
        )
        algo = ConcurrencyLimiter(algo, max_concurrent=2, batch=True)
    elif args.search_algo == 'hyperopt':
        algo = HyperOptSearch(
            metric='valid_accuracy', mode='max',
            points_to_evaluate=points_to_evaluate, random_state_seed=0
        )
        algo = ConcurrencyLimiter(algo, max_concurrent=2, batch=True)

    # Search space
    config = {
        'fs': args.fs,
        'T': args.seq_len,
        'F1': F1_search_space,
        'D': D_search_space,
        'F2': F2_search_space,
        'lr': lr_search_space,
        'batch_size': batch_size_search_space,
        'dropout': dropout_search_space,
        'l2': l2_search_space,
    }
    # Reported metrics
    reporter = CLIReporter(
        metric_columns=['Trial name', 'status', 'D', 'F1', 'F2', 'iter',
                        'total time (s)', 'ts', 'C', 'train_loss',
                        'train_accuracy', 'valid_loss', 'valid_accuracy']
    )
    # Experiment stopper when trial performance stagnates
    stopper = ExperimentPlateauAcrossTrialsStopper(
        metric='valid_accuracy', total_iter=args.epochs, std=0.005, top=20,
        mode='max', patience=0
    )
    # Run experiments
    result = tune.run(
        partial(train_with_configs, args=args),
        resources_per_trial={'cpu': 1, 'gpu': args.gpus_per_trial},
        config=config,
        metric='valid_accuracy',
        mode='max',
        name=f'{args.search_algo}_search',
        search_alg=algo,
        num_samples=args.num_samples,
        local_dir=exp_dir,
        progress_reporter=reporter,
        sync_config=tune.SyncConfig(syncer=None),
        log_to_file=('my_stdout.log', 'my_stderr.log'),
        stop=stopper,
        fail_fast=True,
    )

    # Save the best config and model, load to device for evaluation
    best_trial = result.get_best_trial('valid_accuracy', 'max', 'last')
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial validation loss: {}'.format(
        best_trial.last_result['valid_loss']))
    print('Best trial validation accuracy: {}'.format(
        best_trial.last_result['valid_accuracy']))

    # Save best configuration
    best_config = result.get_best_config('valid_accuracy', 'max', 'last')
    with open(os.path.join(exp_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)

    # Get logging directory for the best trial and copy all trial files
    best_log_dir = result.get_best_logdir('valid_accuracy', 'max', 'last')
    files = os.listdir(best_log_dir)
    for fname in files:
        try:
            shutil.copy2(os.path.join(best_log_dir, fname), exp_dir)
        except IsADirectoryError:
            os.makedirs(os.path.join(exp_dir, fname), exist_ok=True)
            for f in os.listdir(os.path.join(best_log_dir, fname)):
                shutil.copy2(os.path.join(best_log_dir, fname, f),
                             os.path.join(exp_dir, fname))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('expt_name', type=str, help='Experiment name')

    # Data/input choices
    parser.add_argument('--data_path', type=str, help='Path to data')
    parser.add_argument('--expt_type', type=str, default='mot',
                        choices=['mot', 'gam'])
    parser.add_argument('--input_space', type=str, default='channel_space',
                        choices=['channel_space', 'voxel_space'],
                        help='specifies whether inputs are in channel-space '
                        'or voxel-space.')
    parser.add_argument('--data_type', type=str, choices=['ph', 'dc'],
                        default='ph')
    parser.add_argument('--subject', type=str,
                        help='evaluate a single subject (must be in '
                        'bci_subjects_{args.expt_type} from constants.py)')
    parser.add_argument('--train_submontages', nargs='+',
                        default=['a', 'b', 'c', 'abc'],
                        help='specify sub-montages include in dataset.')
    parser.add_argument('--classification_task', type=str, default='motor_LR',
                        choices=['motor_LR', 'motor_color'],
                        help='options include motor_LR (motor response), '
                        'motor_color (motor response + stimulus color.')
    parser.add_argument('--response_speed', type=str, default=None,
                        choices=['slow', 'fast'])

    # Model architecture
    parser.add_argument('--arch', type=str, default='spatiotemporal_cnn',
                        choices=['spatiotemporal_cnn'])
    parser.add_argument('--fs', type=int, default=52,
                        help='Sampling frequency of the data')
    parser.add_argument('--seq_len', type=int, default=75)
    parser.add_argument('--num_channels', type=int, default=480,
                        help='Number of input channels to the model')
    parser.add_argument('--num_temporal_filters', type=int, default=12,
                        help='Number of temporal/frequency filters')
    parser.add_argument('--num_spatial_filters', type=int, default=12,
                        help='Number of spatial filters per temporal filter')
    parser.add_argument('--num_pointwise_filters', type=int, default=12,
                        help='Number of pointwise filters')

    # Pre-processing steps
    parser.add_argument('--filter_zeros', action='store_true',
                        help='Removes channels with all zeros from input.')
    parser.add_argument('--max_abs_scale', action='store_true')
    parser.add_argument('--imputation_method', type=str, default='zero',
                        choices=['zero', 'mean', 'random'])

    # Optimization and model-fitting hyperparameters
    parser.add_argument('--epochs', type=int, help='Number of epochs',
                        default=300)
    parser.add_argument('--lr', type=float, help='Learning Rate',
                        default=0.001)
    parser.add_argument('--optimizer', type=str, help='Optimizer',
                        default='Adam')
    parser.add_argument('--batch_size', type=int, help='Mini-batch size',
                        default=32)
    parser.add_argument('--criterion', type=str, choices=['bce', 'bce_logits'],
                        default='bce_logits')
    parser.add_argument('--early_stop', type=int,
                        help='Patience in early stop in validation set '
                        '(-1 -> no early stop)', default=-1)
    parser.add_argument('--min_epochs', type=int, default=0,
                        help='Minimum number of epochs the model must train '
                        'for before starting early stopping patience')
    parser.add_argument('--weight_decay', type=float, help='Weight decay',
                        default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Determinsm
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_cv', type=int, default=15)
    parser.add_argument('--train_seed', type=int,
                        help='Random seed for training', default=8)

    # Data sampling and cross-validation
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

    # Hyperparameter search
    parser.add_argument('--hyperparameter_tune', action='store_true',
                        help='Performs hyperparameter tuning via the Ray Tune '
                        'library instead of using the specified '
                        'hyperparameters from args.')
    parser.add_argument('--search_algo', type=str, default='hyperopt',
                        choices=['grid', 'bayes_opt', 'zoopt', 'hyperopt'])
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--gpus_per_trial', type=float, default=0.5)

    # System
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    args.stratified = not args.no_stratified
    if args.hyperparameter_tune:
        assert args.early_stop == -1 and args.min_epochs == 0

    # Make experimental directories for output
    if args.expt_type == 'mot':
        assert args.subject in constants.BCI_SUBJECTS_MOT
    elif args.expt_type == 'gam':
        assert args.subject in constants.BCI_SUBJECTS_GAM
    exp_dir = os.path.join(
        constants.BCI_RESULTS_DIR, args.expt_type, args.classification_task,
        'spatiotemporal_cnn', args.input_space,
        'max_abs_scale' if args.max_abs_scale else 'no_scale',
        args.expt_name, f's_{args.subject}'
    )
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'predictions'), exist_ok=True)

    # Assign montage list based on the desired number of montages
    if args.input_space == 'channel_space':
        assert set(args.train_submontages).issubset(set(constants.SUBMONTAGES))
    elif args.input_space == 'voxel_space':
        assert (
            args.train_submontages[0] == 'abc' or
            set(args.train_submontages).issubset(set(constants.SUBMONTAGES)))

    t0 = time.time()
    main(args, exp_dir)
    t1 = time.time()

    print(f'Finished training in {t1 - t0:.1f}s')
    sys.exit(0)
