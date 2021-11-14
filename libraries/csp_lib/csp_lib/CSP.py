import numpy as np
import pandas as pd

from utils.combine_trial_types import (combine_motor_LR,
                                       response_stim_modality_labels,
                                       stim_modality_motor_LR_labels)
from csp_lib.features import extract_features, extract_features_all


def run_csp_filtering(train_idx, valid_idx, learn_idx, eval_idx,
                      X_train, y_train, X, y, subset, args):
    """
    Performs common spatial pattern (CSP) filtering by fitting the filters to
    the training set and using the coefficients to transform the training,
    validation, and testing sets. Supports multi-class classification by
    fitting a set of filters in a one-versus-rest setting.
    """

    # Drop all channels that have NaN (not viable) since CSP
    # uses the covariance matrix of all channels for a montage
    subset = subset.dropna(axis=1).reset_index(drop=True)

    # Combine trial types
    # Classify left versus right motor response
    if args.classification_task == 'motor_LR':
        func = combine_motor_LR
        classes = 2
    # Classify stimulus modality and motor response
    elif args.classification_task == 'stim_motor':
        func = stim_modality_motor_LR_labels
        classes = 4
    # Classify response modality, stimulus modality, and response
    # polarity
    elif args.classification_task == 'response_stim':
        func = response_stim_modality_labels
        classes = 8
    else:
        raise NotImplementedError('Classification task not supported')
    subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(func)
    subset = subset.dropna(
        axis=0, subset=['label']).copy().reset_index(drop=True)
    if args.randomize_trials:
        subset.loc[:, 'label'] = y

    # Re-define viable feature channels
    if args.window == 'all':
        feat_cols = np.array(
            [c for c in subset.columns if 'ph_' in c])
    else:
        feat_cols = np.array(
            [c for c in subset.columns if f'ph_win{args.window}' in c])

    # Binary classification
    if classes == 2:
        # Isolate trials by class (trial type)
        X_r = X_train[np.argwhere(y_train == 1).flatten()]
        X_l = X_train[np.argwhere(y_train == 0).flatten()]

        # Filter using common spatial patterns
        csp = CSP(X_l, X_r, n_filters=args.n_filters)
        transformed_df = pd.DataFrame(
            columns=[c for c in subset.columns if c not in feat_cols])
        for i in subset.index:
            curr_trial = subset.loc[
                i, [c for c in subset.columns if c not in feat_cols]]
            trial_result = curr_trial.to_dict()
            X_new = csp.transform(X[i, :, :])
            X_feat = csp.extract_features(X[i, :, :])
            for m in range(X_new.shape[0]):
                trial_result[f'csp_{m}'] = X_new[m, :]
            for k, f in enumerate(X_feat):
                trial_result[f'csp_feat_{k}'] = f
            transformed_df = transformed_df.append(trial_result,
                                                   ignore_index=True)
    # Multi-class classification
    else:
        # Keep track of metadata
        transformed_df = subset.loc[
            :, [c for c in subset.columns if c not in feat_cols]].copy()

        # One-versus-rest setting
        for tt in range(classes):

            # Isolate trials by class (trial type)
            X_j = X_train[np.argwhere(y_train == tt).flatten()].copy()
            X_i = X_train[np.argwhere(y_train != tt).flatten()].copy()

            # Filter using common spatial patterns
            csp = CSP(X_i, X_j, n_filters=args.n_filters)
            filtered_df = pd.DataFrame()
            for i in subset.index:
                trial_result = dict()
                X_new = csp.transform(X[i, :, :])
                X_feat = csp.extract_features(X[i, :, :])
                for m in range(X_new.shape[0]):
                    trial_result[f'csp_{tt}_{m}'] = X_new[m, :]
                for k, f in enumerate(X_feat):
                    trial_result[f'csp_feat_{tt}_{k}'] = f
                filtered_df = filtered_df.append(trial_result,
                                                 ignore_index=True)

            transformed_df = pd.concat([transformed_df, filtered_df], axis=1)

    # Extract features from CSP-filtered data
    if args.log_variance_feats:
        features = [c for c in transformed_df.columns if 'csp_feat' in c] + \
                   ['trial_num', 'subject_id', 'montage', 'trial_type']
        subset = transformed_df.loc[:, features]
    else:
        features = [c for c in transformed_df.columns if 'csp_feat' not in c]
        if args.window == 'all':
            subset = extract_features_all(
                transformed_df.loc[:, features], csp=True)
        else:
            subset = extract_features(
                transformed_df.loc[:, features], int(args.window), csp=True)

    feat_cols = np.array([c for c in subset.columns if 'csp_' in c])

    # Combine trial types
    subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(func)
    subset = subset.dropna(
        axis=0, subset=['label']).copy().reset_index(drop=True)

    # Train-test split
    X = subset[feat_cols].values
    X = X[:, ~np.isnan(X).any(axis=0)]
    X_learn, X_eval = X[learn_idx, :], X[eval_idx, :]
    y_learn, y_eval = y[learn_idx], y[eval_idx]
    # Sanity check that y_learn follows the same ordering expected as the
    # input arguments
    assert np.allclose(y_train, y_learn[train_idx])
    X_train, X_valid = X_learn[train_idx], X_learn[valid_idx]
    y_train, y_valid = y_learn[train_idx], y_learn[valid_idx]

    return X_train, y_train, X_valid, y_valid, X_eval, y_eval


class CSP:
    def __init__(self, X_l, X_r, n_filters=16):
        """
        X_l and X_r are the feature matrices for each class
        respectively.

        np.array of size (trial, chan, time) or (trial, N, T)
        """
        # Compute covariance matrices
        def cov(X):
            return X @ X.T / np.trace(X @ X.T)

        # Ramoser eq. 1
        R_r = np.zeros((X_r.shape[0], X_r.shape[1], X_r.shape[1]))
        for i in range(X_r.shape[0]):
            R_r[i, :, :] = cov(X_r[i, :, :])
        R_r_ = np.mean(R_r, axis=0)
        R_l = np.zeros((X_l.shape[0], X_l.shape[1], X_l.shape[1]))
        for i in range(X_l.shape[0]):
            R_l[i, :, :] = cov(X_l[i, :, :])
        R_l_ = np.mean(R_l, axis=0)

        # Ramoser eq. 2
        R_c = R_l_ + R_r_

        # Compute eigenvalues and sort in descending order
        eig_c, U_c = np.linalg.eig(R_c)
        descending_idx = np.argsort(eig_c)[::-1]
        eig_c = eig_c[descending_idx]
        U_c = U_c[:, descending_idx]

        # Find Whitening Transformation Matrix - Ramoser eq. 3
        P = np.sqrt(np.linalg.inv(np.diag(eig_c))) @ U_c.T

        # Whiten Data Using Whiting Transform - Ramoser eq. 4
        S_l = P @ R_l_ @ P.T
        S_r = P @ R_r_ @ P.T

        # Generalized eigenvectors/values
        eig_l, B_l = np.linalg.eig(S_l)
        eig_r, B_r = np.linalg.eig(S_r)
        descending_idx = np.argsort(eig_l)[::-1]
        eig_l = eig_l[descending_idx]
        B_l = B_l[:, descending_idx]
        ascending_idx = np.argsort(eig_r)
        eig_r = eig_r[ascending_idx]
        B_r = B_r[:, ascending_idx]
        # Simultaneous diagonalization - Ramoser eq. 5
        assert np.allclose(eig_l + eig_r, np.ones(len(eig_l)), atol=1e-3)

        B = B_l
        # Use only n_filter components for transform
        comp_idx = np.concatenate([np.arange(B.shape[1])[:n_filters],
                                   np.arange(B.shape[1])[-n_filters:]])
        B = B[:, comp_idx]

        # Resulting Projection Matrix - Ramoser eq. 6
        # These are the spatial filter coefficients
        W = B.T @ P
        self.W = W

    def transform(self, E):
        """
        Takes in E, a single trial matrix of size (N, T) and transforms it
        according to the spatial filter coefficients.

        Z is a (2M, T) matrix where M is the number of filters.
        """
        Z = self.W @ E
        return Z

    def extract_features(self, E):
        """
        Extract feature vectors for each trial. The log-transformation serves
        to approximate a normal distribution of the data.

        Returns a feature array of shape (2M,)
        """
        Z = self.transform(E)
        var_Z = np.var(Z, axis=1).flatten()
        return np.log(var_Z / np.sum(var_Z))
