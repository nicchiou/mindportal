import argparse

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.preprocessing import StandardScaler

montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def extract_features(df, window, csp=False):
    """ Extracts bandpower features from specific time sequence windows. """
    if csp:
        cols = [c for c in df.columns if 'csp' in c]
        orig = df[cols].values
        length = len(orig[0, 0])
    else:
        cols = [f'ph_win{window}_{m}_{c}_{f}'
                for m in montages for c in range(128)
                for f in ['04', '08', '13']]
        length = len(df.loc[0, f'ph_win{window}_e_0_04'])
        orig = df[cols].values
    empty = np.zeros((orig.shape[0], orig.shape[1], length))

    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            empty[i, j, :] = orig[i, j]
    max_feats = pd.DataFrame(np.max(empty.copy(), axis=-1),
                             columns=[f'max_{c}' for c in cols])
    min_feats = pd.DataFrame(np.min(empty.copy(), axis=-1),
                             columns=[f'min_{c}' for c in cols])
    mean_feats = pd.DataFrame(np.mean(empty.copy(), axis=-1),
                              columns=[f'mean_{c}' for c in cols])
    range_feats = pd.DataFrame(
        (np.max(empty.copy(), axis=-1) - np.min(empty.copy(), axis=-1)),
        columns=[f'range_{c}' for c in cols])
    avg_pwr_feats = pd.DataFrame(
        np.sum(empty.copy() ** 2,  axis=-1) / empty.shape[2],
        columns=[f'avg_pwr_{c}' for c in cols])

    nan_mask = np.isnan(empty.copy()).any(axis=-1)
    new_values = np.sum(empty.copy() > 0,  axis=-1)
    new_values = new_values.astype('float64')
    new_values[nan_mask] = np.nan

    samp_gt_zero_feats = pd.DataFrame(
        new_values,
        columns=[f'samp_gt_zero_{c}' for c in cols])

    def num_zero_crossing(arr):
        return len(np.where(np.diff(np.signbit(arr)))[0])

    nan_mask = np.isnan(empty.copy()).any(axis=-1)
    new_values = np.apply_along_axis(
        num_zero_crossing, axis=-1, arr=empty.copy())
    new_values = new_values.astype('float64')
    new_values[nan_mask] = np.nan

    zero_cross_feats = pd.DataFrame(
        new_values,
        columns=[f'zero_cross_{c}' for c in cols])

    feats_df = pd.concat(
        [max_feats, min_feats, mean_feats, range_feats, avg_pwr_feats,
         samp_gt_zero_feats, zero_cross_feats], axis=1)
    info_df = df[['trial_num', 'subject_id', 'trial_type', 'montage']]
    final_df = pd.concat([info_df, feats_df], axis=1)
    final_df.dropna(axis=1, how='all', inplace=True)

    return final_df


def extract_features_all(df, csp=False):
    """ Extracts bandpower features from the entire input signal. """
    if csp:
        cols = [c for c in df.columns if 'csp' in c]
        orig = df[cols].values
        length = len(orig[0, 0])
    else:
        cols = [f'ph_{m}_{c}_{f}'
                for m in montages for c in range(128)
                for f in ['04', '08', '13']]
        length = len(df.loc[0, 'ph_e_0_04'])
        orig = df[cols].values
    empty = np.zeros((orig.shape[0], orig.shape[1], length))

    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            empty[i, j, :] = orig[i, j]
    max_feats = pd.DataFrame(np.max(empty.copy(), axis=-1),
                             columns=[f'max_{c}' for c in cols])
    min_feats = pd.DataFrame(np.min(empty.copy(), axis=-1),
                             columns=[f'min_{c}' for c in cols])
    mean_feats = pd.DataFrame(np.mean(empty.copy(), axis=-1),
                              columns=[f'mean_{c}' for c in cols])
    range_feats = pd.DataFrame(
        (np.max(empty.copy(), axis=-1) - np.min(empty.copy(), axis=-1)),
        columns=[f'range_{c}' for c in cols])
    avg_pwr_feats = pd.DataFrame(
        np.sum(empty.copy() ** 2,  axis=-1) / empty.shape[2],
        columns=[f'avg_pwr_{c}' for c in cols])

    nan_mask = np.isnan(empty.copy()).any(axis=-1)
    new_values = np.sum(empty.copy() > 0,  axis=-1)
    new_values = new_values.astype('float64')
    new_values[nan_mask] = np.nan

    samp_gt_zero_feats = pd.DataFrame(
        new_values,
        columns=[f'samp_gt_zero_{c}' for c in cols])

    def num_zero_crossing(arr):
        return len(np.where(np.diff(np.signbit(arr)))[0])

    nan_mask = np.isnan(empty.copy()).any(axis=-1)
    new_values = np.apply_along_axis(
        num_zero_crossing, axis=-1, arr=empty.copy())
    new_values = new_values.astype('float64')
    new_values[nan_mask] = np.nan

    zero_cross_feats = pd.DataFrame(
        new_values,
        columns=[f'zero_cross_{c}' for c in cols])

    feats_df = pd.concat(
        [max_feats, min_feats, mean_feats, range_feats, avg_pwr_feats,
         samp_gt_zero_feats, zero_cross_feats], axis=1)
    info_df = df[['trial_num', 'subject_id', 'trial_type', 'montage']]
    final_df = pd.concat([info_df, feats_df], axis=1)
    final_df.dropna(axis=1, how='all', inplace=True)

    return final_df


def select_features(args: argparse.Namespace, trial_results: dict,
                    X_train, y_train, X_valid, X_test):
    """
    Selects features according to the selection method specified and adjusts
    the input feature matrices accordingly.
    """
    if args.selection_method == 'PCA':
        pca = PCA(n_components=args.n_components)
        pca.fit(X_train)
        exp_var = pca.explained_variance_ratio_
        trial_results['num_components'] = args.n_components
        trial_results['explained_variance'] = exp_var
        X_train = pca.transform(X_train)
        X_valid = pca.transform(X_valid)
        X_test = pca.transform(X_test)
    elif args.selection_method == 'MI':
        mi_scores = mutual_info_classif(
            X_train, y_train,
            n_neighbors=args.n_neighbors)
        mi_score_selected_index = np.where(
            mi_scores > args.score_threshold)[0]
        trial_results['n_neighbors'] = args.n_neighbors
        trial_results['score_treshold'] = args.score_threshold
        X_train = X_train[:, mi_score_selected_index]
        X_valid = X_valid[:, mi_score_selected_index]
        X_test = X_test[:, mi_score_selected_index]
    elif args.selection_method == 'tree':
        clf = ExtraTreesClassifier(n_estimators=args.n_estimators)
        clf = clf.fit(X_train, y_train)
        selector = SelectFromModel(
            clf, max_features=args.n_features, prefit=True)
        trial_results['n_estimators'] = args.n_estimators
        trial_results['max_features'] = args.n_features
        X_train = selector.transform(X_train)
        X_valid = selector.transform(X_valid)
        X_test = selector.transform(X_test)
        trial_results['n_features'] = X_train.shape[1]
    elif args.selection_method == 'linear':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        clf = svm.LinearSVC(C=1, penalty='l1', dual=False,
                            max_iter=10000)
        clf = clf.fit(X_train, y_train)
        selector = SelectFromModel(
            clf, max_features=args.n_features, prefit=True)
        trial_results['max_features'] = args.n_features
        X_train = selector.transform(X_train)
        X_valid = selector.transform(X_valid)
        X_test = selector.transform(X_test)
        trial_results['n_features'] = X_train.shape[1]
    elif args.selection_method == 'None':
        trial_results['n_features'] = X_train.shape[1]
    else:
        raise NotImplementedError(
            'Feature selection method not implemented!')

    return X_train, X_valid, X_test, trial_results
