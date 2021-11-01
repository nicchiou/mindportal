import argparse
import os

import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from tqdm import tqdm

from utils import constants


def stim_modality_motor_LR_labels(x):
    """
    Combines trial types such that the problem is a multi-class classification
    problem with the goal to predict both the stimulus modality (visual or
    auditory) as well as the motor response (L/R).

    0: visual, right
    1: visual, left
    2: auditory, right
    3: auditory, left

    Returns np.nan for unsupported trial types.
    """
    if x in [1, 5]:
        return 0
    elif x in [2, 6]:
        return 1
    elif x in [3, 7]:
        return 2
    elif x in [4, 8]:
        return 3
    else:
        return np.nan


def response_stim_modality_labels(x):
    """
    Combines trial types such that the problem is a multi-class classification
    problem with the goal to predict the response modality (motor or vocal),
    the stimulus modality (visual or auditory) as well as the response
    polarity (L/R).

    0: motor, visual, right
    1: motor, visual, left
    2: motor, auditory, right
    3: motor, auditory, left
    4: vocal, visual, right
    5: vocal, visual, left
    6: vocal, auditory, right
    7: vocal, auditory, left

    Returns np.nan for unsupported trial types.
    """
    if x in [1, 5]:
        return 0
    elif x in [2, 6]:
        return 1
    elif x in [3, 7]:
        return 2
    elif x in [4, 8]:
        return 3
    elif x in [9, 13]:
        return 4
    elif x in [10, 14]:
        return 5
    elif x in [11, 15]:
        return 6
    elif x in [12, 16]:
        return 7
    else:
        return np.nan


class CSP:
    def __init__(self, X_i, X_j, n_filters=16):
        """
        X_i and X_j are the feature matrices for each class
        respectively.
        
        np.array of size (trial, chan, time) or (trial, N, T)
        """
        # Compute covariance matrices
        def cov(X):
            return X @ X.T / np.trace(X @ X.T)

        # Ramoser eq. 1
        R_j = np.zeros((X_j.shape[0], X_j.shape[1], X_j.shape[1]))
        for i in range(X_j.shape[0]):
            R_j[i, :, :] = cov(X_j[i, :, :])
        R_j_ = np.mean(R_j, axis=0)
        R_i = np.zeros((X_i.shape[0], X_i.shape[1], X_i.shape[1]))
        for i in range(X_i.shape[0]):
            R_i[i, :, :] = cov(X_i[i, :, :])
        R_i_ = np.mean(R_i, axis=0)
        
        # Ramoser eq. 2
        R_c = R_i_ + R_j_

        # Compute eigenvalues and sort in descending order
        eig_c, U_c = np.linalg.eig(R_c)
        descending_idx = np.argsort(eig_c)[::-1]
        eig_c = eig_c[descending_idx]
        U_c = U_c[:, descending_idx]

        # Find Whitening Transformation Matrix - Ramoser eq. 3
        P = np.sqrt(np.linalg.inv(np.diag(eig_c))) @ U_c.T

        # Whiten Data Using Whiting Transform - Ramoser eq. 4
        S_i = P @ R_i_ @ P.T
        S_j = P @ R_j_ @ P.T

        # Generalized eigenvectors/values
        eig_i, B_i = np.linalg.eig(S_i)
        eig_j, B_j = np.linalg.eig(S_j)
        descending_idx = np.argsort(eig_i)[::-1]
        eig_i = eig_i[descending_idx]
        B_i = B_i[:, descending_idx]
        ascending_idx = np.argsort(eig_j)
        eig_j = eig_j[ascending_idx]
        B_j = B_j[:, ascending_idx]
        # Simultaneous diagonalization - Ramoser eq. 5
        assert np.allclose(eig_i + eig_j, np.ones(len(eig_i)), atol=1e-3)

        B = B_i
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
    parser.add_argument('--classification_task', type=str,
                        default='stim_motor',
                        help='options include stim_motor (stimulus modality '
                        'and motor response) or response_stim (response '
                        'modality, stimulus modality, and response polarity).')
   
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

    out_dir = os.path.join(constants.CSP_DIR, args.classification_task,
                           args.anchor, 'unfilt' if args.unfilt else 'filt')
    os.makedirs(out_dir, exist_ok=True)
    classes = 4 if args.classification_task == 'stim_motor' else 8

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
            # Classify stimulus modality and motor response
            if args.classification_task == 'stim_motor':
                func = stim_modality_motor_LR_labels
            # Classify response modality, stimulus modality, and response
            # polarity
            elif args.classification_task == 'response_stim':
                func = response_stim_modality_labels
            else:
                raise NotImplementedError('Classification type not supported')
            subset.loc[:, 'label'] = subset.loc[:, 'trial_type'].apply(func)
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

            # Keep track of metadata
            intermediate_df = subset.loc[:, 
                [c for c in subset.columns if c not in feat_cols]].copy()

            # One-versus-rest setting
            for tt in tqdm(range(classes), leave=False):

                # Isolate trials by class (trial type)
                X_j = X[np.argwhere(y == tt).flatten()].copy()
                X_i = X[np.argwhere(y != tt).flatten()].copy()
                y_j = y[np.argwhere(y == tt).flatten()].copy()
                y_i = y[np.argwhere(y != tt).flatten()].copy()
            
                # Filter using common spatial patterns
                csp = CSP(X_i, X_j, n_filters=args.n_filters)
                filtered_df = pd.DataFrame()
                for i in tqdm(subset.index, leave=False):
                    trial_result = dict()
                    X_new = csp.transform(X[i, :, :])
                    X_feat = csp.extract_features(X[i, :, :])
                    for m in range(X_new.shape[0]):
                        trial_result[f'csp_{tt}_{m}'] = X_new[m, :]
                    for k, f in enumerate(X_feat):
                        trial_result[f'csp_feat_{tt}_{k}'] = f
                    filtered_df = filtered_df.append(trial_result,
                                                     ignore_index=True)
                
                intermediate_df = pd.concat([intermediate_df, filtered_df],
                                            axis=1)

            transformed_df = transformed_df.append(intermediate_df,
                                                   ignore_index=True)
    
    if args.window == 'all':
        fname = f'CSP_filt_{args.n_filters}_all.parquet'
    else:
        fname = f'CSP_filt_{args.n_filters}_win{args.window}.parquet' 
    transformed_df.to_parquet(os.path.join(out_dir, fname), index=False)
