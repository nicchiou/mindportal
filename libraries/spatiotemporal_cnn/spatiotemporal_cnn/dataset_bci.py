import os
import random
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset

from utils import combine_trial_types, constants
from spatiotemporal_cnn.preprocessing import maxabs_scale


class BCIData:
    """ Abstract class for optical imaging data. """
    def __init__(self):
        pass


class IncludedExcludedSubjectData(BCIData):
    """
    Combines data for subjects specified as included/excluded into a single
    Dataset for leave-one-subject-out validation.
    """
    def __init__(self, data_dir: str,
                 included_subjects: list, excluded_subjects: list,
                 train_submontages: list,
                 classification_task: str, expt_type: str,
                 filter_zeros: bool = False, input_space: str = 'voxel_space',
                 data_type: str = 'ph'):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'

        # Make assertions about the montages and subjects included for training
        if input_space == 'channel_space':
            assert set(train_submontages).issubset(set(constants.SUBMONTAGES))
        elif input_space == 'voxel_space':
            assert (
                train_submontages[0] == 'abc' or
                set(train_submontages).issubset(set(constants.SUBMONTAGES)))
        assert set(included_subjects).issubset(set(constants.SUBJECT_IDS))
        assert set(excluded_subjects).issubset(set(constants.SUBJECT_IDS))

        if included_subjects and excluded_subjects:
            raise AttributeError('Cannot list included and excluded patients')
        elif included_subjects:
            subjects = included_subjects
        elif excluded_subjects:
            subjects = [s for s in constants.SUBJECT_IDS
                        if s != excluded_subjects]

        prev_trial_max = 0
        for s in subjects:
            prev_trial_nums = list()
            for i, m in enumerate(train_submontages):
                # Pandas DataFrame has format: timestep across trial numbers
                # (index), all possible channels + metadata (columns)
                try:
                    temp = pd.read_parquet(os.path.join(
                        data_dir,
                        f'{expt_type}_{s}_{m}_0.parquet'))
                except FileNotFoundError:
                    continue

                if i > 0:
                    assert temp.trial_num.values.tolist() == prev_trial_nums
                prev_trial_nums = temp.trial_num.values.tolist()

                # Add sub-montage data as additional features
                subject_data = pd.concat([self.data, temp], axis=1)

            # Create unique trial numbers for each subject's data
            subject_data.loc[:, 'trial_num'] = \
                subject_data.loc[:, 'trial_num'].values + prev_trial_max
            prev_trial_max = max(subject_data.loc[:, 'trial_num'].values) + 1

            # Add subject data as additional trials
            self.data = self.data.append(subject_data, ignore_index=True)

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        feat_cols = [c for c in self.data.columns
                     if self.data_type_prefix in c]
        if input_space == 'channel_space':
            expected_features = 0
            if 'a' in train_submontages:
                expected_features += 192
            if 'b' in train_submontages:
                expected_features += 192
            if 'c' in train_submontages:
                expected_features += 96
        elif input_space == 'voxel_space':
            expected_features = 2 * 49
        assert len(feat_cols) == expected_features
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, feat_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels by dropping features with all zeros across all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, feat_cols] = \
                self.dynamic_table.loc[:, feat_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            feat_cols = [c for c in self.dynamic_table.columns
                         if self.data_type_prefix in c]
        # How many viable features remain?
        viable_feat = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_feat)
        if self.num_viable == 0:
            raise NotImplementedError('num_viable features = 0')

        # Remove trials that contain all zero values for features
        self.dynamic_table.loc[:, feat_cols] = \
            self.dynamic_table.loc[:, feat_cols].replace(0, np.nan)
        self.feat_df = self.dynamic_table.loc[:, feat_cols]
        self.valid_rows = self.feat_df[
            ~self.feat_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_response
            self.labels['trial_type'] = self.labels['trial_type'].apply(func)
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'motor_color':
            self.classes = 4
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels.rename({'trial_type': 'label'}, axis=1, inplace=True)
        self.labels = self.labels.dropna(
            axis=0, subset=['label']).copy().reset_index(drop=True)

        # NumPy arrays of trial_num, labels, and indices
        self.trial_id = self.labels.loc[:, 'trial_num'].unique()
        self.labels = self.labels.loc[:, 'label'].values.astype(np.float32)
        self.idxs = list(range(len(self.labels)))

        # Dynamic table is a DataFrame containing trials present for the
        # specified trial type - assign corresponding idxs to match trial_id
        self.dynamic_table = self.dynamic_table.loc[
            self.dynamic_table['trial_num'].isin(self.trial_id), :
            ].reset_index(drop=True)
        trial_idx_map = dict(zip(self.trial_id, self.idxs))
        self.dynamic_table.loc[:, 'idx'] = self.dynamic_table.apply(
            lambda x: trial_idx_map[x.trial_num], axis=1)
        assert len(self.dynamic_table['trial_num'].unique()) == len(self.idxs)

    def get_num_viable_features(self):
        return self.num_viable


class SingleSubjectData(BCIData):
    """
    Creates a representation of the BCI data set that groups subjects' dynamic
    data for all sub-montages with their corresponding labels based on the
    classification task. Performs basic pre-processing to as specified by the
    input arguments.
    """
    def __init__(self, data_dir: str, subject_id: str, train_submontages: list,
                 classification_task: str, expt_type: str,
                 response_speed: str = None, filter_zeros: bool = False,
                 input_space: str = 'voxel_space', data_type: str = 'ph',
                 seq_start: int = 0, seq_end: int = None):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'
        self.seq_start = seq_start
        self.seq_end = seq_end

        # Make assertions about the montage list matching the number of
        # montages
        if input_space == 'channel_space':
            assert set(train_submontages).issubset(set(constants.SUBMONTAGES))
        elif input_space == 'voxel_space':
            assert (
                train_submontages[0] == 'abc' or
                set(train_submontages).issubset(set(constants.SUBMONTAGES)))

        prev_trial_nums = list()
        for i, submontage in enumerate(train_submontages):
            # Pandas DataFrame has format: timestep across trial numbers
            # (index), all possible channels + metadata (columns)
            try:
                temp = pd.read_parquet(os.path.join(
                    data_dir,
                    f'{expt_type}_{subject_id}_{submontage}_0.parquet'))
                feat_cols = [c for c in temp.columns
                             if self.data_type_prefix in c]
            except FileNotFoundError:
                continue

            if i > 0:
                assert temp.trial_num.values.tolist() == prev_trial_nums
            prev_trial_nums = temp.trial_num.values.tolist()

            # Add sub-montage data as additional features
            self.data = pd.concat([self.data, temp], axis=1)

        # Remove duplicate columns (i.e. trial_num, subject_id, etc.)
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        # Filter relevant trials based on response time
        if response_speed is not None:
            orig_num_trials = len(self.data)
            # Only use slow response trials
            if response_speed == 'slow':
                self.data = self.data[self.data['slow_response'] == 1]
            assert len(self.data) < orig_num_trials

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        feat_cols = [c for c in self.data.columns
                     if self.data_type_prefix in c]
        if input_space == 'channel_space':
            expected_features = 0
            if 'a' in train_submontages:
                expected_features += 192
            if 'b' in train_submontages:
                expected_features += 192
            if 'c' in train_submontages:
                expected_features += 96
        elif input_space == 'voxel_space':
            expected_features = 2 * 49
        assert len(feat_cols) == expected_features
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, feat_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels by dropping features with all zeros across all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, feat_cols] = \
                self.dynamic_table.loc[:, feat_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            feat_cols = [c for c in self.dynamic_table.columns
                         if self.data_type_prefix in c]
        # How many viable features remain?
        viable_feat = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_feat)
        if self.num_viable == 0:
            raise NotImplementedError('num_viable features = 0')

        # Remove trials that contain all zero values for features
        self.dynamic_table.loc[:, feat_cols] = \
            self.dynamic_table.loc[:, feat_cols].replace(0, np.nan)
        self.feat_df = self.dynamic_table.loc[:, feat_cols]
        self.valid_rows = self.feat_df[
            ~self.feat_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_response
            self.labels['trial_type'] = self.labels['trial_type'].apply(func)
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'motor_color':
            self.classes = 4
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels.rename({'trial_type': 'label'}, axis=1, inplace=True)
        self.labels = self.labels.dropna(
            axis=0, subset=['label']).copy().reset_index(drop=True)

        # NumPy arrays of trial_num, labels, and indices
        self.trial_id = self.labels.loc[:, 'trial_num'].unique()
        self.labels = self.labels.loc[:, 'label'].values.astype(np.float32)
        self.idxs = list(range(len(self.labels)))

        # Dynamic table is a DataFrame containing trials present for the
        # specified trial type - assign corresponding idxs to match trial_id
        self.dynamic_table = self.dynamic_table.loc[
            self.dynamic_table['trial_num'].isin(self.trial_id), :
            ].reset_index(drop=True)
        trial_idx_map = dict(zip(self.trial_id, self.idxs))
        self.dynamic_table.loc[:, 'idx'] = self.dynamic_table.apply(
            lambda x: trial_idx_map[x.trial_num], axis=1)
        assert len(self.dynamic_table['trial_num'].unique()) == len(self.idxs)

    def get_num_viable_features(self):
        return self.num_viable


class SubjectSubmontageData(BCIData):
    """
    Creates a representation of the BCI data set that groups subjects' dynamic
    data for a specifici sub-montage with their corresponding labels based on
    the classification task. Performs basic pre-processing to as specified by
    the input arguments.
    """
    def __init__(self, data_dir: str, subject_id: str, submontage: str,
                 classification_task: str, expt_type: str,
                 filter_zeros: bool = False, input_space: str = 'voxel_space',
                 data_type: str = 'ph',
                 seq_start: int = 0, seq_end: int = None):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'
        self.seq_start = seq_start
        self.seq_end = seq_end

        # Pandas DataFrame has format: timestep across trial numbers
        # (index), all possible channels + metadata (columns)
        try:
            self.data = pd.read_parquet(os.path.join(
                data_dir,
                f'{expt_type}_{subject_id}_{submontage}_0.parquet'))
        except FileNotFoundError as e:
            raise e

        # Remove duplicate columns (i.e. trial_num, subject_id, etc.)
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        feat_cols = [c for c in self.data.columns
                     if self.data_type_prefix in c]
        if input_space == 'channel_space':
            if submontage == 'a' or submontage == 'b':
                assert len(feat_cols) == 192
            elif submontage == 'c':
                assert len(feat_cols) == 96
        elif input_space == 'voxel_space':
            assert len(feat_cols) == 2 * 49
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, feat_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter features by dropping features with all zeros for all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, feat_cols] = \
                self.dynamic_table.loc[:, feat_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            feat_cols = [c for c in self.dynamic_table.columns
                         if self.data_type_prefix in c]
        # How many viable features remain?
        viable_feat = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_feat)
        if self.num_viable == 0:
            raise NotImplementedError('num_viable features = 0')

        # Remove trials that contain all zero values for features
        self.dynamic_table.loc[:, feat_cols] = \
            self.dynamic_table.loc[:, feat_cols].replace(0, np.nan)
        self.feat_df = self.dynamic_table.loc[:, feat_cols]
        self.valid_rows = self.feat_df[
            ~self.feat_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_response
            self.labels['trial_type'] = self.labels['trial_type'].apply(func)
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'motor_color':
            self.classes = 4
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels.rename({'trial_type': 'label'}, axis=1, inplace=True)
        self.labels = self.labels.dropna(
            axis=0, subset=['label']).copy().reset_index(drop=True)

        # NumPy arrays of trial_num, labels, and indices
        self.trial_id = self.labels.loc[:, 'trial_num'].unique()
        self.labels = self.labels.loc[:, 'label'].values.astype(np.float32)
        self.idxs = list(range(len(self.labels)))

        # Dynamic table is a DataFrame containing trials present for the
        # specified trial type - assign corresponding idxs to match trial_id
        self.dynamic_table = self.dynamic_table.loc[
            self.dynamic_table['trial_num'].isin(self.trial_id), :
            ].reset_index(drop=True)
        trial_idx_map = dict(zip(self.trial_id, self.idxs))
        self.dynamic_table.loc[:, 'idx'] = self.dynamic_table.apply(
            lambda x: trial_idx_map[x.trial_num], axis=1)
        assert len(self.dynamic_table['trial_num'].unique()) == len(self.idxs)

    def get_num_viable_features(self):
        return self.num_viable


class SubjectMontageDataset(Dataset):
    """
    Creates a PyTorch-loadable dataset that can be used for train/valid/test
    splits. Compatible with K-fold cross-validation and stratified splits.
    """
    def __init__(self, data: BCIData, subset: str = None,
                 seed: int = 42, props: Tuple[float] = (70, 10, 20),
                 stratified: bool = False, cv: int = 1, nested_cv: int = 1,
                 cv_idx: int = 0, nested_cv_idx: int = 0, seed_cv: int = 15,
                 **preprocessing):
        super().__init__()

        subsets = ['train', 'valid', 'test']
        assert subset in subsets
        assert sum(props) == 100

        self.data = data
        self.data_type_prefix = data.data_type_prefix
        self.props = props
        self.proportions = dict(zip(subsets, self.props))
        self.subset = subset
        self.seed = seed
        self.seed_cv = seed_cv
        self.stratified = stratified
        self.cv = cv
        self.nested_cv = nested_cv
        self.cv_idx = cv_idx
        self.nested_cv_idx = nested_cv_idx

        self.preprocessing = preprocessing
        self.impute_chan_dict = dict()

        self.labels = self.data.labels
        self.trial_id = self.data.trial_id
        self.dynamic_table = self.data.dynamic_table
        self.chan_cols = [c for c in self.dynamic_table.columns
                          if self.data_type_prefix in c]
        self.idxs = self.data.idxs

        self.seq_start = self.data.seq_start
        self.seq_end = self.data.seq_end

        pd.set_option('mode.chained_assignment', 'raise')

        # Stratify by class labels
        if self.stratified:

            # learn / test split
            sss_outer = StratifiedShuffleSplit(
                n_splits=self.nested_cv,
                test_size=self.proportions['test'] / 100,
                random_state=self.seed)
            learn_idx, test_idx = list(
                sss_outer.split(self.idxs, self.labels))[self.nested_cv_idx]
            learn = np.array(self.idxs)[learn_idx]
            learn_labels = self.labels[learn_idx]

            # train / valid split
            if self.cv == 1:
                sss_inner = StratifiedShuffleSplit(
                    n_splits=self.cv,
                    test_size=(
                        self.proportions['valid'] /
                        (self.proportions['train'] +
                         self.proportions['valid'])),
                    random_state=self.seed_cv)
            else:
                sss_inner = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.seed_cv)
            train_idx, valid_idx = list(
                sss_inner.split(learn, learn_labels))[self.cv_idx]
            train_idx = learn[train_idx]
            valid_idx = learn[valid_idx]

            if self.subset == 'train':
                self.idxs = list(train_idx)
            elif self.subset == 'valid':
                self.idxs = list(valid_idx)
            else:
                self.idxs = list(test_idx)
        else:
            if cv != 1 or nested_cv != 1:
                raise NotImplementedError(
                    'CV is not implemented without stratified split')
            if seed_cv != seed:
                raise NotImplementedError(
                    'Different seed for CV is not implemented without '
                    'stratified split')
            np.random.shuffle(self.idxs)
            if self.subset:
                self.idxs = [x for idx, x in enumerate(self.idxs)
                             if self._check_idx(idx + 1, self.subset)]

        # Filter relevent entries based on split
        self.trial_id = [self.trial_id[id_] for id_ in self.idxs]
        self.labels = self.labels[self.idxs]
        self.dynamic_table = self.dynamic_table.loc[
            self.dynamic_table['idx'].isin(self.idxs), :]

        # Fill zero channels with NaN
        feat_cols = [c for c in self.dynamic_table.columns
                     if self.data_type_prefix in c]
        self.dynamic_table.loc[:, feat_cols].replace(0, np.nan, inplace=True)

        # Get a list of NaN columns that require imputation
        self.nan_feats = list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])

        # Impute missing channels
        imputed_df = self.dynamic_table.copy()
        for feat in self.nan_feats:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [feat, 'trial_num']]

            # Zero imputation works for all splits
            if preprocessing['impute'] == 'zero':
                imputed_df.loc[:, feat] = subset.fillna(value=0.0)

            # Get imputed signal from train set and apply to other subsets
            elif preprocessing['impute'] == 'mean' \
                    and self.subset == 'train':
                running_sig = list()
                # Loop through trials to aggregate signal from viable channels
                for trial in subset.trial_num.unique():
                    # Single trial
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    running_sig.append(subset_subset.loc[:, feat].values)
                running_sig = np.vstack(running_sig).T
                if np.count_nonzero(~np.isnan(running_sig)) == 0:
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[0], :]
                    mean_sig = np.zeros_like(subset_subset.loc[:, feat].values)
                else:
                    mean_sig = np.nanmean(running_sig, axis=1)
                if self.subset == 'train':
                    self.impute_chan_dict[feat] = mean_sig
                # Impute with mean signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[feat].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, feat] = mean_sig

            elif preprocessing['impute'] == 'random' \
                    and self.subset == 'train':
                # Get non-NaN trials and corresponding indices to choose from
                non_nan_df = self.dynamic_table.dropna(
                    axis=0, how='any', subset=[feat])
                non_nan_trials = non_nan_df.trial_num.unique()
                non_nan_idx = np.zeros(len(subset.trial_num.unique()))
                for t in non_nan_trials:
                    non_nan_idx = np.logical_or(
                        non_nan_idx, subset.trial_num.unique() == t)
                assert len(non_nan_trials) == np.count_nonzero(non_nan_idx)
                non_nan_idx = np.argwhere(non_nan_idx).flatten()
                if len(non_nan_idx) == 0:
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[0], :]
                    rand_sig = np.zeros_like(subset_subset.loc[:, feat].values)
                else:
                    # Select random non-NaN trial used for imputation
                    rand = random.choice(non_nan_idx)
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[rand],
                        :]
                    rand_sig = subset_subset.loc[:, feat].values
                assert not np.isnan(rand_sig[0])
                if self.subset == 'train':
                    self.impute_chan_dict[feat] = rand_sig
                # Impute with random signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[feat].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, feat] = rand_sig
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        if preprocessing['impute'] == 'zero' or self.subset == 'train':
            assert len(list(
                self.dynamic_table.columns[self.dynamic_table.isna().any()])
                ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.seq_end - self.seq_start))

        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            # Slice data to desired sequence start and end
            data = data[:, self.seq_start:self.seq_end]
            # Maximum absolute value scaling
            if preprocessing['max_abs_scale']:
                data = maxabs_scale(data)
            self.dynamic_data[idx, :, :] = data

    def __getitem__(self, index: int) -> Tuple[int, np.ndarray, int]:
        return (self.trial_id[index], self.dynamic_data[index, :, :],
                self.labels[index])

    def __len__(self) -> int:
        return len(self.trial_id)

    def _check_idx(self, i: int, subset: str) -> bool:
        if subset == 'train' and i % 100 < self.proportions['train']:
            return True
        elif subset == 'valid' and self.proportions['train'] <= i % 100 < (
                self.proportions['train'] + self.proportions['valid']):
            return True
        elif subset == 'test' and i % 100 >= (
                self.proportions['train'] + self.proportions['valid']):
            return True
        return False

    def get_imputation(self):
        assert self.subset == 'train'
        return self.preprocessing['imputed_signal']

    def get_label(self, idx):
        return self.__getitem__(idx)[2]

    def get_labels(self):
        return self.labels

    def impute_chan(self, train_dataset: Dataset):

        impute_chan_dict = train_dataset.impute_chan_dict

        # Impute missing features
        imputed_df = self.dynamic_table.copy()
        for feat in self.nan_feats:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [feat, 'trial_num']]

            # Use pre-calculated imputed signal on training set to imput
            for trial in subset.trial_num.unique():
                subset_subset = subset.loc[subset['trial_num'] == trial, :]
                if np.isnan(subset_subset[feat].values[0]):
                    idxs_to_replace = subset_subset.index
                    imputed_df.loc[idxs_to_replace, feat] = \
                        impute_chan_dict[feat]
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        assert len(list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])
            ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.seq_end - self.seq_start))

        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            # Slice data to desired sequence start and end
            data = data[:, self.seq_start:self.seq_end]
            # Maximum absolute value scaling
            if self.preprocessing['max_abs_scale']:
                data = maxabs_scale(data)
            self.dynamic_data[idx, :, :] = data


class LeaveOneOutSplitSubjectMontageDataset(Dataset):
    """
    Creates a PyTorch-loadable dataset that can be used for train/valid
    and subject-specific test splits. Compatible with K-fold cross-validation
    and stratified splits with leave-one-out validation.
    """
    def __init__(self, learn_data: BCIData, test_data: BCIData,
                 subset: str = None, seed: int = 42,
                 props: Tuple[float] = (80, 20), stratified: bool = False,
                 cv: int = 1, nested_cv: int = 1, cv_idx: int = 0,
                 nested_cv_idx: int = 0, seed_cv: int = 15, **preprocessing):
        super().__init__()

        subsets = ['train', 'valid']
        assert (subset in subsets) or (subset == 'test')
        assert sum(props) == 100

        self.data = learn_data
        self.data_type_prefix = learn_data.data_type_prefix
        self.test_data = test_data
        self.props = props
        self.proportions = dict(zip(subsets, self.props))
        self.subset = subset
        self.seed = seed
        self.seed_cv = seed_cv
        self.stratified = stratified
        self.cv = cv
        self.nested_cv = nested_cv
        self.cv_idx = cv_idx
        self.nested_cv_idx = nested_cv_idx

        self.preprocessing = preprocessing
        self.impute_chan_dict = dict()

        self.labels = self.data.labels
        self.trial_id = self.data.trial_id
        self.dynamic_table = self.data.dynamic_table
        self.chan_cols = [c for c in self.dynamic_table.columns
                          if self.data_type_prefix in c]
        self.idxs = self.data.idxs

        self.seq_start = self.data.seq_start
        self.seq_end = self.data.seq_end

        self.test_labels = self.test_data.labels
        self.test_trial_id = self.test_data.trial_id
        self.test_dynamic_table = self.test_data.dynamic_table
        self.test_idxs = self.test_data.idxs

        pd.set_option('mode.chained_assignment', 'raise')

        # Stratify by class labels
        if self.stratified:

            # train / valid split
            if self.cv == 1:
                sss_inner = StratifiedShuffleSplit(
                    n_splits=self.cv,
                    test_size=(
                        self.proportions['valid'] / 100),
                    random_state=self.seed_cv)
            else:
                sss_inner = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.seed_cv)
            train_idx, valid_idx = list(
                sss_inner.split(self.idxs, self.labels))[self.cv_idx]

            if self.subset == 'train':
                self.idxs = list(train_idx)
            elif self.subset == 'valid':
                self.idxs = list(valid_idx)
            elif self.subset == 'test':
                self.idxs = list(self.test_idxs)

        else:
            if cv != 1:
                raise NotImplementedError(
                    'CV is not implemented without stratified split')
            if seed_cv != seed:
                raise NotImplementedError(
                    'Different seed for CV is not implemented without '
                    'stratified split')
            np.random.shuffle(self.idxs)
            if self.subset:
                if self.subset in ['train', 'valid']:
                    self.idxs = [x for idx, x in enumerate(self.idxs)
                                 if self._check_idx(idx + 1, self.subset)]
                elif self.subset == 'test':
                    self.idxs = list(self.test_idxs)

        # Filter relevent entries based on split
        if self.subset in ['train', 'valid']:
            self.trial_id = [self.trial_id[id_] for id_ in self.idxs]
            self.labels = self.labels[self.idxs]
            self.dynamic_table = self.dynamic_table.loc[
                self.dynamic_table['idx'].isin(self.idxs), :]
        elif self.subset == 'test':
            self.trial_id = [self.test_trial_id[id_] for id_ in self.idxs]
            self.labels = self.test_labels[self.idxs]
            self.dynamic_table = self.test_dynamic_table.loc[
                self.test_dynamic_table['idx'].isin(self.idxs), :]

        # Fill zero channels with NaN
        feat_cols = [c for c in self.dynamic_table.columns
                     if self.data_type_prefix in c]
        self.dynamic_table.loc[:, feat_cols].replace(0, np.nan, inplace=True)

        # Get a list of NaN columns that require imputation
        self.nan_feats = list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])

        # Impute missing channels
        imputed_df = self.dynamic_table.copy()
        for feat in self.nan_feats:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [feat, 'trial_num']]

            # Zero imputation works for all splits
            if preprocessing['impute'] == 'zero':
                imputed_df.loc[:, feat] = subset.fillna(value=0.0)

            # Get imputed signal from train set and apply to other subsets
            elif preprocessing['impute'] == 'mean' \
                    and self.subset == 'train':
                running_sig = list()
                # Loop through trials to aggregate signal from viable channels
                for trial in subset.trial_num.unique():
                    # Single trial
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    running_sig.append(subset_subset.loc[:, feat].values)
                running_sig = np.vstack(running_sig).T
                if np.count_nonzero(~np.isnan(running_sig)) == 0:
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[0], :]
                    mean_sig = np.zeros_like(subset_subset.loc[:, feat].values)
                else:
                    mean_sig = np.nanmean(running_sig, axis=1)
                if self.subset == 'train':
                    self.impute_chan_dict[feat] = mean_sig
                # Impute with mean signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[feat].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, feat] = mean_sig

            elif preprocessing['impute'] == 'random' \
                    and self.subset == 'train':
                # Get non-NaN trials and corresponding indices to choose from
                non_nan_df = self.dynamic_table.dropna(
                    axis=0, how='any', subset=[feat])
                non_nan_trials = non_nan_df.trial_num.unique()
                non_nan_idx = np.zeros(len(subset.trial_num.unique()))
                for t in non_nan_trials:
                    non_nan_idx = np.logical_or(
                        non_nan_idx, subset.trial_num.unique() == t)
                assert len(non_nan_trials) == np.count_nonzero(non_nan_idx)
                non_nan_idx = np.argwhere(non_nan_idx).flatten()
                if len(non_nan_idx) == 0:
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[0], :]
                    rand_sig = np.zeros_like(subset_subset.loc[:, feat].values)
                else:
                    # Select random non-NaN trial used for imputation
                    rand = random.choice(non_nan_idx)
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[rand],
                        :]
                    rand_sig = subset_subset.loc[:, feat].values
                assert not np.isnan(rand_sig[0])
                if self.subset == 'train':
                    self.impute_chan_dict[feat] = rand_sig
                # Impute with random signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[feat].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, feat] = rand_sig
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        if preprocessing['impute'] == 'zero' or self.subset == 'train':
            assert len(list(
                self.dynamic_table.columns[self.dynamic_table.isna().any()])
                ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.seq_end - self.seq_start))

        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            # Slice data to desired sequence start and end
            data = data[:, self.seq_start:self.seq_end]
            # Maximum absolute value scaling
            if preprocessing['max_abs_scale']:
                data = maxabs_scale(data)
            self.dynamic_data[idx, :, :] = data

    def __getitem__(self, index: int) -> Tuple[int, np.ndarray, int]:
        return (self.trial_id[index], self.dynamic_data[index, :, :],
                self.labels[index])

    def __len__(self) -> int:
        return len(self.trial_id)

    def _check_idx(self, i: int, subset: str) -> bool:
        if subset == 'train' and i % 100 < self.proportions['train']:
            return True
        elif subset == 'valid' and i % 100 >= self.proportions['train']:
            return True
        return False

    def get_imputation(self):
        assert self.subset == 'train'
        return self.preprocessing['imputed_signal']

    def get_label(self, idx):
        return self.__getitem__(idx)[2]

    def get_labels(self):
        return self.labels

    def impute_chan(self, train_dataset: Dataset):

        impute_chan_dict = train_dataset.impute_chan_dict

        # Impute missing channels
        imputed_df = self.dynamic_table.copy()
        for feat in self.nan_feats:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [feat, 'trial_num']]

            # Use pre-calculated imputed signal on training set to imput
            for trial in subset.trial_num.unique():
                subset_subset = subset.loc[subset['trial_num'] == trial, :]
                if np.isnan(subset_subset[feat].values[0]):
                    idxs_to_replace = subset_subset.index
                    imputed_df.loc[idxs_to_replace, feat] = \
                        impute_chan_dict[feat]
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        assert len(list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])
            ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.seq_end - self.seq_start))

        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            # Slice data to desired sequence start and end
            data = data[:, self.seq_start:self.seq_end]
            # Maximum absolute value scaling
            if self.preprocessing['max_abs_scale']:
                data = maxabs_scale(data)
            self.dynamic_data[idx, :, :] = data


class DatasetBuilder:
    def __init__(self, data: BCIData, seed: int = 42, seed_cv: int = 15,
                 **preprocessing):
        self.data = data
        self.seed = seed
        self.seed_cv = seed_cv
        self.preprocessing = preprocessing

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[Tuple[BCIData, BCIData]], BCIData]]:
        """
        Yields Datasets in the tuple form ((train, valid), test), where
        the inner tuple is iterated over for each cross-validation split.
        To be used with cross-validation. cv, number of cross-validation folds
        nested_cv, number of unique test sets to generate (typically just one)
        """
        seed = self.seed
        seed_cv = self.seed_cv

        # Iterate through possible test sets (typically just use one)
        for i in range(nested_cv):
            def _inner_loop(data: BCIData, seed: int, seed_cv: seed):
                # Iterate through cross-validation folds and yield train and
                # valid Datasets
                for j in range(cv):
                    train_dataset = SubjectMontageDataset(
                        data=data, subset='train', stratified=True,
                        cv=cv, nested_cv=nested_cv, cv_idx=j, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing)
                    valid_dataset = SubjectMontageDataset(
                        data=data, subset='valid', stratified=True,
                        cv=cv, nested_cv=nested_cv, cv_idx=j, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing)
                    yield (train_dataset, valid_dataset)

            yield ((_inner_loop(self.data, self.seed, self.seed_cv)),
                   SubjectMontageDataset(
                        data=self.data, subset='test', stratified=True, cv=cv,
                        nested_cv=nested_cv, cv_idx=0, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing))


class DualDatasetBuilder:
    def __init__(self, data_A: BCIData, data_B: BCIData,
                 seed: int = 42, seed_cv: int = 15, **preprocessing):
        self.data_A = data_A
        self.data_B = data_B
        self.seed = seed
        self.seed_cv = seed_cv
        self.preprocessing = preprocessing

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[Tuple[BCIData, BCIData]], BCIData]]:
        """
        Yields Datasets in the tuple form ((train, valid), test), where
        the inner tuple is iterated over for each cross-validation split.
        To be used with cross-validation. cv, number of cross-validation folds
        nested_cv, number of unique test sets to generate (typically just one)
        """
        seed = self.seed
        seed_cv = self.seed_cv

        # Iterate through possible test sets (typically just use one)
        for i in range(nested_cv):
            def _inner_loop(data: BCIData, seed: int, seed_cv: seed):
                # Iterate through cross-validation folds and yield train and
                # valid Datasets
                for j in range(cv):
                    train_dataset = SubjectMontageDataset(
                        data=data, subset='train', stratified=True,
                        cv=cv, nested_cv=nested_cv, cv_idx=j, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing)
                    valid_dataset = SubjectMontageDataset(
                        data=data, subset='valid', stratified=True,
                        cv=cv, nested_cv=nested_cv, cv_idx=j, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing)
                    yield (train_dataset, valid_dataset)

            test_dataset_A = SubjectMontageDataset(
                data=self.data_A, subset='test', stratified=True, cv=cv,
                nested_cv=nested_cv, cv_idx=0, nested_cv_idx=i,
                seed=seed, seed_cv=seed_cv, **self.preprocessing)
            test_dataset_B = SubjectMontageDataset(
                data=self.data_B, subset='test', stratified=True, cv=cv,
                nested_cv=nested_cv, cv_idx=0, nested_cv_idx=i,
                seed=seed, seed_cv=seed_cv, **self.preprocessing)
            yield ((_inner_loop(self.data_A, self.seed, self.seed_cv)),
                   test_dataset_A,
                   (_inner_loop(self.data_B, self.seed, self.seed_cv)),
                   test_dataset_B)


class LeaveOneOutDatasetBuilder:
    def __init__(self, learn_data: BCIData, test_data: BCIData,
                 seed: int = 42, seed_cv: int = 15, **preprocessing):
        self.learn_data = learn_data
        self.test_data = test_data
        self.seed = seed
        self.seed_cv = seed_cv
        self.preprocessing = preprocessing

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[Tuple[BCIData, BCIData]], BCIData]]:
        """
        Yields Datasets in the tuple form ((train, valid), test), where
        the inner tuple is iterated over for each cross-validation split.
        The test set consists of a witheld subject's data.
        """
        seed = self.seed
        seed_cv = self.seed_cv

        # Iterate through possible test sets (typically just use one)
        for i in range(nested_cv):
            def _inner_loop(learn_data: BCIData, test_data: BCIData,
                            seed: int, seed_cv: seed):
                # Iterate through cross-validation folds and yield train and
                # valid Datasets
                for j in range(cv):
                    train_dataset = LeaveOneOutSplitSubjectMontageDataset(
                        learn_data=learn_data, test_data=test_data,
                        subset='train', stratified=True,
                        cv=cv, nested_cv=nested_cv, cv_idx=j, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing)
                    valid_dataset = LeaveOneOutSplitSubjectMontageDataset(
                        learn_data=learn_data, test_data=test_data,
                        subset='valid', stratified=True,
                        cv=cv, nested_cv=nested_cv, cv_idx=j, nested_cv_idx=i,
                        seed=seed, seed_cv=seed_cv, **self.preprocessing)
                    yield (train_dataset, valid_dataset)

        yield ((_inner_loop(self.learn_data, self.test_data,
                            self.seed, self.seed_cv)),
               LeaveOneOutSplitSubjectMontageDataset(
                    learn_data=self.learn_data, test_data=self.test_data,
                    subset='test', stratified=True, cv=cv, nested_cv=nested_cv,
                    cv_idx=0, nested_cv_idx=i, seed=seed, seed_cv=seed_cv,
                    **self.preprocessing))
