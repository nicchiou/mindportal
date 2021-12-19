import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from utils import constants

from csp_lib.CSP import run_csp_filtering
from csp_lib.data_splits import learn_eval_split_csp, learn_eval_split_non_csp
from csp_lib.features import select_features


def train_eval_clf(args, X_train, y_train, X_valid, y_valid, X_test, y_test):

    if args.scale_inputs:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    if args.clf_type == 'SVM':
        clf = svm.LinearSVC(penalty=args.penalty, loss=args.loss, C=args.C,
                            max_iter=args.max_iter,
                            random_state=args.train_seed,
                            dual=False if args.penalty == 'l1' and
                            args.loss == 'squared_hinge' else True)
    elif args.clf_type == 'RBF_SVM':
        clf = svm.SVC(C=args.C, max_iter=args.max_iter,
                      random_state=args.train_seed)
    elif args.clf_type == 'RF':
        clf = RandomForestClassifier(n_estimators=200, criterion='entropy',
                                     max_depth=args.max_depth,
                                     random_state=args.train_seed)
    else:
        raise NotImplementedError(
            'Classifier type not supported!')

    clf.fit(X_train, y_train)

    trial_results = dict()

    y_pred = clf.predict(X_train)
    class_rep = classification_report(y_train, y_pred, output_dict=True,
                                      zero_division=0)
    trial_results['Train Accuracy'] = class_rep['accuracy']
    trial_results['Train Precision'] = class_rep['macro avg']['precision']
    trial_results['Train Recall'] = class_rep['macro avg']['recall']
    trial_results['Train F1'] = class_rep['macro avg']['f1-score']

    y_pred = clf.predict(X_valid)
    class_rep = classification_report(y_valid, y_pred, output_dict=True,
                                      zero_division=0)
    trial_results['Valid Accuracy'] = class_rep['accuracy']
    trial_results['Valid Precision'] = class_rep['macro avg']['precision']
    trial_results['Valid Recall'] = class_rep['macro avg']['recall']
    trial_results['Valid F1'] = class_rep['macro avg']['f1-score']

    y_pred = clf.predict(X_test)
    class_rep = classification_report(y_test, y_pred, output_dict=True,
                                      zero_division=0)
    trial_results['Test Accuracy'] = class_rep['accuracy']
    trial_results['Test Precision'] = class_rep['macro avg']['precision']
    trial_results['Test Recall'] = class_rep['macro avg']['recall']
    trial_results['Test F1'] = class_rep['macro avg']['f1-score']
    for label in np.unique(y_test):
        for k, v in class_rep[str(label)].items():
            trial_results[f'{label} {k}'] = v
    trial_results['cfmat'] = confusion_matrix(y_test, y_pred).flatten()

    return trial_results


montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--clf_type', type=str, default='SVM',
                        help='options are: SVM, RBF_SVM, RF')
    parser.add_argument('--window', type=str, default='all',
                        help='options include 0-7')
    parser.add_argument('--unfilt', action='store_true',
                        help='indicates whether to use data without the '
                        'removal of zeroed channels')
    parser.add_argument('--csp', action='store_true',
                        help='indicates whether to use CSP filtering')
    parser.add_argument('--bandpass_only', action='store_true',
                        help='indicates whether to use the signal that has '
                        'not been rectified nor low-pass filtered')
    parser.add_argument('--log_variance_feats', action='store_true',
                        help='indicates whether to use CSP-derived '
                        'log-variance features instead of band-power features')
    parser.add_argument('--n_filters', type=int, default=16,
                        help='number of CSP filters used to generate features')
    parser.add_argument('--classification_task', type=str, default='motor_LR',
                        help='options include motor_LR (motor response), '
                        'stim_motor (stimulus modality and motor response) '
                        'and response_stim (response modality, stimulus '
                        'modality, and response polarity).')
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--selection_method', type=str, default='None',
                        help='options are: PCA, MI, tree, linear, RFE, SFS, '
                        'None')
    parser.add_argument('--selected_dir', type=str)
    parser.add_argument('--n_components', type=int, default=100,
                        help='used only for PCA')
    parser.add_argument('--n_neighbors', type=int, default=3,
                        help='used only for mutual information selection')
    parser.add_argument('--score_threshold', type=float, default=0.02,
                        help='used only for mutual information selection')
    parser.add_argument('--n_estimators', type=int, default=200,
                        help='used only for tree-based selection')
    parser.add_argument('--n_features', type=int, default=300,
                        help='maximum number of features to select')
    parser.add_argument('--scale_inputs', action='store_true')
    parser.add_argument('--penalty', type=str, default='l1')
    parser.add_argument('--loss', type=str, default='squared_hinge')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_cv', type=int, default=15)
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--n_splits', type=int, default=5, help='cv splits')
    parser.add_argument('--randomize_trials', action='store_true')

    args = parser.parse_args()
    if args.clf_type == 'SVM':
        model_name = 'linear_svm'
    elif args.clf_type == 'RBF_SVM':
        model_name = 'rbf_svm'
    else:
        model_name = 'random_forest'
    exp_dir = os.path.join(
        constants.RESULTS_DIR, args.classification_task,
        'csp_baseline' if args.csp else 'baseline',
        model_name, args.anchor,
        'bandpass_only' if args.bandpass_only else 'rect_lowpass',
        args.exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    results_df = pd.DataFrame()

    # Input file (if csp, contains signals; else, contains features)
    if args.csp:
        in_dir = os.path.join(
            constants.PHASE_DATA_DIR, args.anchor,
            'bandpass_only' if args.bandpass_only else 'rect_lowpass')
        if args.unfilt:
            if args.window == 'all':
                fname = 'phase_all_single_trial.parquet'
            else:
                fname = f'phase_win{args.window}_single_trial.parquet'
        else:
            if args.window == 'all':
                fname = 'phase_all_filt_chan.parquet'
            else:
                fname = f'phase_win{args.window}_filt_chan.parquet'
    else:
        in_dir = os.path.join(
            constants.BANDPOWER_DIR, args.anchor,
            'bandpass_only' if args.bandpass_only else 'rect_lowpass')
        if args.window == 'all':
            fname = 'all_simple_bandpower_features.parquet'
        else:
            fname = f'win{args.window}_simple_bandpower_features.parquet'
    df = pd.read_parquet(os.path.join(in_dir, fname))

    subjects = df['subject_id'].unique()

    for subject_id in subjects:
        for montage in montages:

            trial_results = dict()
            trial_results['subject_id'] = subject_id
            trial_results['montage'] = montage
            if args.csp:
                trial_results['n_filters'] = args.n_filters

            subset = df[(df['subject_id'] == str(subject_id)) &
                        (df['montage'] == montage)].copy().reset_index(
                            drop=True)

            if args.csp:
                X_learn, X_eval, y_learn, y_eval, X, y, learn_idx, eval_idx = \
                    learn_eval_split_csp(args, subset)
            else:
                X_learn, X_eval, y_learn, y_eval = \
                    learn_eval_split_non_csp(args, subset)

            # Stratified K-fold cross-validation
            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                                  random_state=args.seed_cv)
            for idx, (train_idx, valid_idx) in enumerate(
                    skf.split(X_learn, y_learn)):
                trial_results['idx'] = idx

                X_train, X_valid = X_learn[train_idx], X_learn[valid_idx]
                y_train, y_valid = y_learn[train_idx], y_learn[valid_idx]
                X_test, y_test = \
                    copy.deepcopy(X_eval), copy.deepcopy(y_eval)

                if args.csp:
                    X_train, y_train, X_valid, y_valid, X_test, y_test = \
                        run_csp_filtering(train_idx, valid_idx,
                                          learn_idx, eval_idx,
                                          X_train, y_train,
                                          X, y, subset, args)
                # Select features if applicable
                if args.selection_method == 'None':
                    trial_results['n_features'] = X_train.shape[1]
                else:
                    X_train, X_valid, X_test, trial_results = \
                        select_features(args, trial_results,
                                        X_train, y_train, X_valid, X_test)

                # Train and evaluate the model
                eval_results = train_eval_clf(
                    args, X_train, y_train, X_valid, y_valid, X_test, y_test)
                trial_results.update(eval_results)

                results_df = results_df.append(trial_results,
                                               ignore_index=True)

    results_df.to_parquet(
        os.path.join(exp_dir, f'{args.selection_method}_results.parquet'),
        index=False)
