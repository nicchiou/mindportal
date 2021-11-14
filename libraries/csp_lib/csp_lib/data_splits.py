import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.combine_trial_types import (combine_motor_LR,
                                       response_stim_modality_labels,
                                       stim_modality_motor_LR_labels)


def learn_eval_split_non_csp(args: argparse.Namespace, subset: pd.DataFrame):
    """
    Formats data from learn/eval split by combining trial types, removing NaN
    values, and seeding the split.
    """
    # Combine trial types
    # Classify left versus right motor response
    if args.classification_task == 'motor_LR':
        func = combine_motor_LR
    # Classify stimulus modality and motor response
    elif args.classification_task == 'stim_motor':
        func = stim_modality_motor_LR_labels
    # Classify response modality, stimulus modality, and response
    # polarity
    elif args.classification_task == 'response_stim':
        func = response_stim_modality_labels
    else:
        raise NotImplementedError('Classification task not supported')
    subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(func)
    subset = subset.dropna(
        axis=0, subset=['label']).copy().reset_index(drop=True)

    # Remove features based on zeroed channels across all trials
    subset = subset.loc[:, subset.isnull().sum(axis=0) != subset.shape[0]]

    # Keep track of features with zeroed channels
    # na_mask = subset.isna().any(axis=0).values
    # na_cols = list(subset.columns[na_mask])
    feat_cols = np.array([c for c in subset.columns if 'ph' in c])
    # discrete_feats = ((np.core.defchararray.find(
    #                         feat_cols, 'samp_gt_zero') != -1) |
    #                   (np.core.defchararray.find(
    #                         feat_cols, 'zero_cross') != -1))
    subset = subset.fillna(0)

    # Train-test split
    X = subset[feat_cols].values
    y = subset['label'].values
    X_learn, X_eval, y_learn, y_eval = train_test_split(
        X, y, test_size=0.2, stratify=y,
        random_state=args.sampler_seed)

    # Randomize trial types if applicable
    if args.randomize_trials:
        np.random.seed(42)
        np.random.shuffle(y_learn)

    return X_learn, X_eval, y_learn, y_eval


def learn_eval_split_csp(args: argparse.Namespace, subset: pd.DataFrame):
    """
    Formats data from learn/eval split by combining trial types, removing NaN
    values, reformating data for CSP, and seeding the split.

    Returns: Learn and eval matrices along with the original X and y data
    matrices and leran/eval indices.
    """
    # Drop all channels that have NaN (not viable) since CSP
    # uses the covariance matrix of all channels for a montage
    subset = subset.dropna(axis=1).reset_index(drop=True)

    # Combine trial types
    # Classify left versus right motor response
    if args.classification_task == 'motor_LR':
        func = combine_motor_LR
    # Classify stimulus modality and motor response
    elif args.classification_task == 'stim_motor':
        func = stim_modality_motor_LR_labels
    # Classify response modality, stimulus modality, and response
    # polarity
    elif args.classification_task == 'response_stim':
        func = response_stim_modality_labels
    else:
        raise NotImplementedError('Classification task not supported')
    subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(func)
    subset = subset.dropna(
        axis=0, subset=['label']).copy().reset_index(drop=True)

    # Re-define viable feature channels
    if args.window == 'all':
        chan_cols = np.array(
            [c for c in subset.columns if 'ph_' in c])
    else:
        chan_cols = np.array(
            [c for c in subset.columns if f'ph_win{args.window}' in c])
    label_col = 'label'

    # (trials x channels x timepoints)
    X = np.empty((subset.shape[0], len(chan_cols),
                  len(subset.loc[0, chan_cols[0]])))
    for i in subset.index:
        for j, c in enumerate(chan_cols):
            X[i, j, :] = subset.loc[i, c]
    y = subset.loc[:, label_col].values

    # Train-test split
    idx_learn, idx_eval = train_test_split(
        list(subset.index), test_size=0.2, stratify=y,
        random_state=args.sampler_seed)
    X_learn = X[idx_learn, :, :]
    X_eval = X[idx_eval, :, :]
    y_learn = y[idx_learn]
    y_eval = y[idx_eval]
    # Randomize training labels if applicable
    if args.randomize_trials:
        np.random.seed(42)
        np.random.shuffle(y_learn)
        y[idx_learn] = y_learn

    return X_learn, X_eval, y_learn, y_eval, X, y, idx_learn, idx_eval
