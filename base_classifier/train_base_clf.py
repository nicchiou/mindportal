import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def combine_motor_LR(x):
    """
    Combines trial types for left motor responses and right motor responses.

    0 represents left and 1 represents right.
    """
    if x in [2, 4, 6, 8]:
        return 0
    elif x in [1, 3, 5, 7]:
        return 1
    else:
        return np.nan


def train_eval_clf(args, X_train, y_train, X_valid, y_valid, X_test, y_test):

    if args.scale_inputs:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    if args.clf_type == 'SVM':
        clf = svm.LinearSVC(penalty=args.penalty, loss=args.loss, C=args.C,
                            max_iter=args.max_iter, random_state=args.seed,
                            dual=False if args.penalty == 'l1' and \
                                          args.loss == 'squared_hinge' \
                                       else True)
    elif args.clf_type == 'RF':
        clf = RandomForestClassifier(n_estimators=200, criterion='entropy',
                                     max_depth=args.max_depth,
                                     random_state=args.seed)
    else:
        raise NotImplementedError(
            'Classifier type not supported!')

    clf.fit(X_train, y_train)

    trial_results = dict()
    y_pred = clf.predict(X_train)
    trial_results['Train Accuracy'] = accuracy_score(y_train, y_pred)
    trial_results['Train Precision'] = precision_score(y_train, y_pred,
        zero_division=0)
    trial_results['Train Recall'] = recall_score(y_train, y_pred,
        zero_division=0)
    trial_results['Train F1'] = f1_score(y_train, y_pred, zero_division=0)
    y_pred = clf.predict(X_valid)
    trial_results['Valid Accuracy'] = accuracy_score(y_valid, y_pred)
    trial_results['Valid Precision'] = precision_score(y_valid, y_pred,
        zero_division=0)
    trial_results['Valid Recall'] = recall_score(y_valid, y_pred,
        zero_division=0)
    trial_results['Valid F1'] = f1_score(y_valid, y_pred, zero_division=0)
    y_pred = clf.predict(X_test)
    trial_results['Test Accuracy'] = accuracy_score(y_test, y_pred)
    trial_results['Test Precision'] = precision_score(y_test, y_pred,
        zero_division=0)
    trial_results['Test Recall'] = recall_score(y_test, y_pred,
        zero_division=0)
    trial_results['Test F1'] = f1_score(y_test, y_pred, zero_division=0)

    return trial_results


data_dir = '/shared/rsaas/nschiou2/EROS/python/'
montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, default='linear_svm')
    parser.add_argument('--clf_type', type=str, default='SVM',
                        help='options are: SVM, RF')
    parser.add_argument('--RT', action='store_true',
                        help='use RT window signal')
    parser.add_argument('--selection_method', type=str, default=None,
                        help='options are: PCA, MI, tree, linear, RFE, SFS, ' \
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
    parser.add_argument('--sampler_seed', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--n_splits', type=int, default=5, help='cv splits')

    args = parser.parse_args()
    exp_dir = os.path.join('../experiments', args.exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    # Try to append to existing DataFrame of results
    try:
        results_df = pd.read_parquet(
            os.path.join(exp_dir, f'{args.selection_method}_results.parquet'))
    except OSError:
        results_df = pd.DataFrame()

    # Input features (not yet selected)
    if args.RT:
        df = pd.read_parquet(
            os.path.join(data_dir, 'simple_bandpower_features.parquet'))
    else:
        df = pd.read_parquet(
            os.path.join(data_dir, 'all_simple_bandpower_features.parquet'))
        
    
    subjects = df['subject_id'].unique()

    for subject_id in tqdm(subjects):
        for montage in montages:
        
            trial_results = dict()
            trial_results['subject_id'] = subject_id
            trial_results['montage'] = montage

            subset = df[(df['subject_id'] == str(subject_id)) & \
                        (df['montage'] == montage)].copy().reset_index(
                            drop=True)
            feat_cols = np.array([c for c in subset.columns if 'ph_' in c])
            discrete_feats = ((np.core.defchararray.find(
                                    feat_cols, 'samp_gt_zero') != -1) | 
                              (np.core.defchararray.find(
                                    feat_cols, 'zero_cross') != -1))
            # Combine trial types
            subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(
                combine_motor_LR)
            subset = subset.dropna(
                axis=0, subset=['label']).copy().reset_index(drop=True)
            
            # Remove features that are based on zeroed channels across all trials
            subset = subset.loc[:, \
                subset.isnull().sum(axis=0) != subset.shape[0]]

            # Keep track of features with zeroed channels for specific trials
            na_mask = subset.isna().any(axis=0).values
            na_cols = list(subset.columns[na_mask])
            feat_cols = np.array([c for c in subset.columns if 'ph_' in c])
            discrete_feats = ((np.core.defchararray.find(
                                    feat_cols, 'samp_gt_zero') != -1) | 
                              (np.core.defchararray.find(
                                    feat_cols, 'zero_cross') != -1))
            subset = subset.fillna(0)

            # Train-test split
            X = subset[feat_cols].values
            y = subset['label'].values
            X_learn, X_test, y_learn, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y,
                random_state=args.sampler_seed)

            if args.selection_method in ['RFE', 'SFS']:
                # Load selected features
                selected_feat_idx = np.load(
                    os.path.join(args.selected_dir,
                                f'{subject_id}_{montage}_support.npy'))
                trial_results['num_features'] = \
                    np.count_nonzero(selected_feat_idx)
                X_learn = X_learn[:, selected_feat_idx]
                X_test = X_test[:, selected_feat_idx]
            elif args.selection_method == 'PCA':
                pca = PCA(n_components=args.n_components)
                pca.fit(X_learn)
                exp_var = pca.explained_variance_ratio_
                trial_results['num_components'] = args.n_components
                trial_results['explained_variance'] = exp_var
                X_learn = pca.transform(X_learn)
                X_test = pca.transform(X_test)
            elif args.selection_method == 'MI':
                mi_scores = mutual_info_classif(X_learn, y_learn,
                    discrete_features=discrete_feats,
                    n_neighbors=args.n_neighbors)
                mi_score_selected_index = np.where(
                    mi_scores > args.score_threshold)[0]
                trial_results['n_neighbors'] = args.n_neighbors
                trial_results['score_treshold'] = args.score_threshold
                X_learn = X_learn[:, mi_score_selected_index]
                X_test = X_test[:, mi_score_selected_index]
            elif args.selection_method == 'tree':
                clf = ExtraTreesClassifier(n_estimators=args.n_estimators)
                clf = clf.fit(X_learn, y_learn)
                selector = SelectFromModel(clf, max_features=args.n_features,
                                           prefit=True)
                trial_results['n_estimators'] = args.n_estimators
                trial_results['max_features'] = args.n_features
                X_learn = selector.transform(X_learn)
                X_test = selector.transform(X_test)
                trial_results['n_features'] = X_learn.shape[1]
            elif args.selection_method == 'linear':
                scaler = StandardScaler()
                scaler.fit(X_learn)
                X_learn = scaler.transform(X_learn)
                clf = svm.LinearSVC(C=1, penalty='l1', dual=False,
                                    max_iter=10000)
                clf = clf.fit(X_learn, y_learn)
                selector = SelectFromModel(clf, max_features=args.n_features,
                                           prefit=True)
                trial_results['max_features'] = args.n_features
                X_learn = selector.transform(X_learn)
                X_test = selector.transform(X_test)
                trial_results['n_features'] = X_learn.shape[1]
            elif args.selection_method is None:
                trial_results['n_features'] = X_learn.shape[1]
            else:
                raise NotImplementedError(
                    'Feature selection method not implemented!')
                
            # Stratified K-fold cross-validation
            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                                  random_state=args.sampler_seed)
            for idx, (train_idx, valid_idx) in enumerate(
                    skf.split(X_learn, y_learn)):
                trial_results['idx'] = idx
                X_train, X_valid = X_learn[train_idx], X_learn[valid_idx]
                y_train, y_valid = y_learn[train_idx], y_learn[valid_idx]

                eval_results = train_eval_clf(args,
                    X_train, y_train, X_valid, y_valid, X_test, y_test)
                trial_results.update(eval_results)

                results_df = results_df.append(trial_results, ignore_index=True)
        
    results_df.to_parquet(
        os.path.join(exp_dir, f'{args.selection_method}_results.parquet'),
        index=False)
