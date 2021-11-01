import argparse
import os

import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from tqdm import tqdm

from utils import constants


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


montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(constants.CSP_DIR, 'motor_LR'))
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--unfilt', action='store_true')
    parser.add_argument('--window', type=str, default='all',
                        help='options include 0-7')
    parser.add_argument('--n_filters', type=int, default=16)
   
    args = parser.parse_args()

    in_dir = os.path.join(constants.PHASE_DATA_DIR, args.anchor)
    if args.unfilt:
        if args.window == 'all':
            fname = f'phase_all_single_trial.parquet'
        else:
            fname = f'phase_win{args.window}_single_trial.parquet'
    else:
        if args.window == 'all':
            fname = f'phase_all_filt_chan.parquet'
        else:
            fname = f'phase_win{args.window}_filt_chan.parquet'
    df = pd.read_parquet(os.path.join(in_dir, fname))

    out_dir = os.path.join(constants.CSP_DIR, 'motor_LR', args.anchor,
                           'unfilt' if args.unfilt else 'filt')
    os.makedirs(out_dir, exist_ok=True)

    subjects = df['subject_id'].unique()
    transformed_df = pd.DataFrame()

    for subject_id in tqdm(subjects):
        for montage in tqdm(montages, leave=False):
        
            subset = df[(df['subject_id'] == str(subject_id)) & \
                        (df['montage'] == montage)].copy()
            if args.window == 'all':
                feat_cols = np.array(
                    [c for c in subset.columns if 'ph_' in c])
            else:
                feat_cols = np.array(
                    [c for c in subset.columns if f'ph_win{args.window}' in c])
            label_col = 'trial_type'

            # Drop all channels that have NaN (not viable) since CSP
            # uses the covariance matrix of all channels for a montage
            subset = subset.dropna(axis=1).reset_index(drop=True)

            # Combine trial types
            subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(
                combine_motor_LR)
            subset = subset.dropna(
                axis=0, subset=['label']).copy().reset_index(drop=True)

            # Re-define viable feature channels
            if args.window == 'all':
                feat_cols = np.array(
                    [c for c in subset.columns if 'ph_' in c])
            else:
                feat_cols = np.array(
                    [c for c in subset.columns if f'ph_win{args.window}' in c])
            label_col = 'label'

            # (trials x feats x timepoints)
            X = np.empty((subset.shape[0], len(feat_cols),
                          len(subset.loc[0, feat_cols[0]])))
            for i in tqdm(subset.index, leave=False):
                for j, c in enumerate(feat_cols):
                    X[i, j, :] = subset.loc[i, c]
            y = subset.loc[:, label_col].values

            # Isolate trials by class (trial type)
            X_r = X[np.argwhere(y == 1).flatten()]
            X_l = X[np.argwhere(y == 0).flatten()]
            y_r = y[np.argwhere(y == 1).flatten()]
            y_l = y[np.argwhere(y == 0).flatten()]
            
            # Filter using common spatial patterns
            csp = CSP(X_l, X_r, n_filters=args.n_filters)
            intermediate_df = pd.DataFrame(
                columns=[c for c in subset.columns if c not in feat_cols])
            for i in tqdm(subset.index, leave=False):
                curr_trial = subset.loc[i,
                    [c for c in subset.columns if c not in feat_cols]]
                trial_result = curr_trial.to_dict()
                X_new = csp.transform(X[i, :, :])
                X_feat = csp.extract_features(X[i, :, :])
                for m in range(X_new.shape[0]):
                    trial_result[f'csp_{m}'] = X_new[m, :]
                for k, f in enumerate(X_feat):
                    trial_result[f'csp_feat_{k}'] = f
                intermediate_df = intermediate_df.append(trial_result,
                                                         ignore_index=True)

            transformed_df = transformed_df.append(intermediate_df,
                                                   ignore_index=True)
    
    if args.window == 'all':
        fname = f'CSP_filt_{args.n_filters}_all.parquet'
    else:
        fname = f'CSP_filt_{args.n_filters}_win{args.window}.parquet' 
    transformed_df.to_parquet(os.path.join(out_dir, fname), index=False)
