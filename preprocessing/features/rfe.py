import argparse
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
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


data_dir = '/shared/rsaas/nschiou2/EROS/python/'
montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_features', type=int, default=100)
    parser.add_argument('--seed', type=int, default=8)

    args = parser.parse_args()
    exp_dir = os.path.join('RFE', f'{args.n_features}_features')
    os.makedirs(exp_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(data_dir,
                                      'simple_bandpower_features.parquet'))
    
    subjects = df['subject_id'].unique()

    for subject_id in tqdm(subjects):
        for montage in montages:
            subset = df[(df['subject_id'] == str(subject_id)) & \
                        (df['montage'] == montage)].copy().reset_index(
                            drop=True)
            feat_cols = np.array([c for c in subset.columns if 'ph_' in c])
            discrete_feats = ((np.core.defchararray.find(
                                    feat_cols, 'samp_gt_zero') != -1) | 
                              (np.core.defchararray.find(
                                    feat_cols, 'zero_cross') != -1))

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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=8)

            svc = svm.SVC(kernel='linear', C=1)
            rfe = RFE(estimator=svc, n_features_to_select=args.n_features,
                    step=1, verbose=False)
            rfe.fit(X_train, y_train)

            np.save(os.path.join(exp_dir,
                                 f'{subject_id}_{montage}_ranking.npy'),
                                    rfe.ranking_)
            np.save(os.path.join(exp_dir,
                                 f'{subject_id}_{montage}_support.npy'),
                                    rfe.support_)
