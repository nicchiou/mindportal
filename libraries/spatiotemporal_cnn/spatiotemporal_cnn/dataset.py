import os
import random
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset

from utils import combine_trial_types, constants
from spatiotemporal_cnn.preprocessing import maxabs_scale


class FOSData:
    """ Abstract class for optical imaging data. """
    def __init__(self):
        pass


class CrossSubjectData(FOSData):
    """
    Combines selected montage data for subjects specified as included/excluded
    into a single Dataset for leave-one-subject-out validation.
    """
    def __init__(self, data_dir: str,
                 included_subjects: list, excluded_subjects: list,
                 train_montages: list,
                 classification_task: str, n_montages: int,
                 filter_zeros: bool = False, voxel_space: bool = True,
                 data_type: str = 'ph'):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'

        assert voxel_space

        # Make assertions about the montages and subjects included for training
        if n_montages == 4:
            assert set(train_montages).issubset(set(constants.PAIRED_MONTAGES))
        elif n_montages == 8:
            assert set(train_montages).issubset(set(constants.MONTAGES))
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
            for m in train_montages:
                # Pandas DataFrame has format: timestep across trial numbers
                # (index), all possible channels + metadata (columns)
                try:
                    temp = pd.read_parquet(os.path.join(
                        data_dir, f'{s}_{m}_0.parquet'))
                except FileNotFoundError:
                    continue

                # Rename montages to have common columns
                columns = list(temp.columns)
                for i, c in enumerate(columns):
                    if self.data_type_prefix not in c:
                        continue
                    splits = c.split('_')
                    del splits[1]
                    joined = '_'.join(splits)
                    columns[i] = joined
                temp.columns = columns

                # Create unique trial numbers
                temp.loc[:, 'trial_num'] = \
                    temp.loc[:, 'trial_num'].values + prev_trial_max
                prev_trial_max = max(temp.loc[:, 'trial_num'].values) + 1

                # Add montage data as additional trials
                self.data = self.data.append(temp, ignore_index=True)

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        vox_cols = [c for c in self.data.columns if self.data_type_prefix in c]
        assert len(vox_cols) == 84
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, vox_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels by dropping channels with all zeros across all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, vox_cols] = \
                self.dynamic_table.loc[:, vox_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            vox_cols = [c for c in self.dynamic_table.columns
                        if self.data_type_prefix in c]
        # How many viable channels remain?
        viable_chan = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_chan)

        # Remove trials that contain all zero values for voxels
        self.dynamic_table.loc[:, vox_cols] = \
            self.dynamic_table.loc[:, vox_cols].replace(0, np.nan)
        self.vox_df = self.dynamic_table.loc[:, vox_cols]
        self.valid_rows = self.vox_df[
            ~self.vox_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_LR
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'stim_motor':
            func = combine_trial_types.stim_modality_motor_LR_labels
            self.classes = 4
        # Classify response modality, stimulus modality, and response
        # polarity
        elif classification_task == 'response_stim':
            func = combine_trial_types.response_stim_modality_labels
            self.classes = 8
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels['trial_type'] = self.labels['trial_type'].apply(func)
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

    def get_num_viable_channels(self):
        return self.num_viable


class SubjectData(FOSData):
    """
    Creates a representation of the FOS data set that groups subjects' dynamic
    data with their corresponding labels based on the classification task and
    includes all montages available for that given subject (only makes sense
    in the voxel-space). Performs basic pre-processing to as specified by the
    input arguments.
    """
    def __init__(self, data_dir: str, subject: str, train_montages: list,
                 classification_task: str, n_montages: int,
                 filter_zeros: bool = False, voxel_space: bool = True,
                 data_type: str = 'ph'):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'

        assert voxel_space

        # Make assertions about the montage list matching the number of
        # montages
        if n_montages == 4:
            assert set(train_montages).issubset(set(constants.PAIRED_MONTAGES))
        elif n_montages == 8:
            assert set(train_montages).issubset(set(constants.MONTAGES))

        prev_trial_max = 0
        for m in train_montages:
            # Pandas DataFrame has format: timestep across trial numbers
            # (index), all possible channels + metadata (columns)
            try:
                temp = pd.read_parquet(os.path.join(
                    data_dir, f'{subject}_{m}_0.parquet'))
            except FileNotFoundError:
                print(f'Subject {subject} does not have montage {m} to '
                      'include in training')
                continue

            # Rename montages to have common columns (corresponding to voxels)
            columns = list(temp.columns)
            for i, c in enumerate(columns):
                if self.data_type_prefix not in c:
                    continue
                splits = c.split('_')
                del splits[1]
                joined = '_'.join(splits)
                columns[i] = joined
            temp.columns = columns

            # Create unique trial numbers
            temp.loc[:, 'trial_num'] = \
                temp.loc[:, 'trial_num'].values + prev_trial_max
            prev_trial_max = max(temp.loc[:, 'trial_num'].values) + 1

            # Add montage data as additional trials
            self.data = self.data.append(temp, ignore_index=True)

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        vox_cols = [c for c in self.data.columns if self.data_type_prefix in c]
        assert len(vox_cols) == 84
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, vox_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels by dropping channels with all zeros across all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, vox_cols] = \
                self.dynamic_table.loc[:, vox_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            vox_cols = [c for c in self.dynamic_table.columns
                        if self.data_type_prefix in c]
        # How many viable channels remain?
        viable_chan = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_chan)

        # Remove trials that contain all zero values for voxels
        self.dynamic_table.loc[:, vox_cols] = \
            self.dynamic_table.loc[:, vox_cols].replace(0, np.nan)
        self.vox_df = self.dynamic_table.loc[:, vox_cols]
        self.valid_rows = self.vox_df[
            ~self.vox_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_LR
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'stim_motor':
            func = combine_trial_types.stim_modality_motor_LR_labels
            self.classes = 4
        # Classify response modality, stimulus modality, and response
        # polarity
        elif classification_task == 'response_stim':
            func = combine_trial_types.response_stim_modality_labels
            self.classes = 8
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels['trial_type'] = self.labels['trial_type'].apply(func)
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

    def get_num_viable_channels(self):
        return self.num_viable


class SubjectMontageData(FOSData):
    """
    Creates a representation of the FOS data set that groups subjects' dynamic
    data with their corresponding labels based on the classification task.
    Performs basic pre-processing to as specified by the input arguments.
    """
    def __init__(self, data_dir: str, subject: str, montage: str,
                 classification_task: str, n_montages: int,
                 filter_zeros: bool = False, voxel_space: bool = False,
                 data_type: str = 'ph'):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'

        # Group montages in pairs based on trial num recorded (a-b, c-d, etc.)
        if n_montages == 8 or voxel_space:
            montages = [montage]
        elif n_montages == 4:
            paired_montages = {'a': 'A', 'b': 'A',
                               'c': 'B', 'd': 'B',
                               'e': 'C', 'f': 'C',
                               'g': 'D', 'h': 'D'}
            montages = [v for k, v in paired_montages.items() if v == montage]

        if voxel_space:
            assert len(montage) == 1
        for m in montages:
            # Pandas DataFrame has format: timestep across trial numbers
            # (index), all possible channels + metadata (columns)
            try:
                temp = pd.read_parquet(os.path.join(
                    data_dir, f'{subject}_{m}_0.parquet'))
            except FileNotFoundError as e:
                print(f'Montage {m} for subject {subject} does not exist!')
                raise e

            # Add channels as new features
            self.data = pd.concat([self.data, temp], axis=1)

            # Remove duplicate columns (i.e. trial_num, subject_id, etc.)
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        chan_cols = [c for c in self.data.columns
                     if self.data_type_prefix in c]
        if voxel_space:
            assert len(chan_cols) == 84
        else:
            assert ((len(chan_cols) == 256 and n_montages == 4) or
                    (len(chan_cols) == 128 and n_montages == 8))
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, chan_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels by dropping channels with all zeros across all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, chan_cols] = \
                self.dynamic_table.loc[:, chan_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            chan_cols = [c for c in self.dynamic_table.columns
                         if self.data_type_prefix in c]
        # How many viable channels remain?
        viable_chan = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_chan)
        if self.num_viable == 0:
            print(f'Montage {m} for subject {subject} has no viable channels!')
            raise NotImplementedError('num_viable channels = 0')

        # Remove trials that contain all zero values for channels
        self.dynamic_table.loc[:, chan_cols] = \
            self.dynamic_table.loc[:, chan_cols].replace(0, np.nan)
        self.chan_df = self.dynamic_table.loc[:, chan_cols]
        self.valid_rows = self.chan_df[
            ~self.chan_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_LR
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'stim_motor':
            func = combine_trial_types.stim_modality_motor_LR_labels
            self.classes = 4
        # Classify response modality, stimulus modality, and response
        # polarity
        elif classification_task == 'response_stim':
            func = combine_trial_types.response_stim_modality_labels
            self.classes = 8
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels['trial_type'] = self.labels['trial_type'].apply(func)
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

    def get_num_viable_channels(self):
        return self.num_viable


class MontagePretrainData(FOSData):
    """
    Creates a representation of the FOS data set that groups subjects' dynamic
    data with their corresponding labels based on the classification task.
    Performs basic pre-processing to as specified by the input arguments.

    Functions identically to the SubjectMontageDataset, except the montage
    of interest is excluded from the data and all other montages are used.
    """
    def __init__(self, data_dir: str, subject: str, montage: str,
                 classification_task: str, n_montages: int,
                 filter_zeros: bool = False, voxel_space: bool = False,
                 data_type: str = 'ph'):

        self.data_dir = data_dir
        self.data = pd.DataFrame()
        self.data_type_prefix = f'{data_type}_'

        prev_trial_max = 0
        paired_montages = {'a': 'A', 'b': 'A',
                           'c': 'B', 'd': 'B',
                           'e': 'C', 'f': 'C',
                           'g': 'D', 'h': 'D'}

        if n_montages == 8:
            for m in constants.MONTAGES:
                # Check whether we use the base montage for pre-training
                if m != montage:
                    # Pandas DataFrame has format: timestep across trial
                    # numbers (index), all possible channels + metadata
                    # (columns)
                    try:
                        temp = pd.read_parquet(
                            os.path.join(data_dir, f'{subject}_{m}_0.parquet'))
                    except FileNotFoundError:
                        print(f'Subject {subject} does not have montage {m} '
                              'for pre-training.')
                        continue

                    # Rename montages to have common columns
                    columns = list(temp.columns)
                    for i, c in enumerate(columns):
                        if self.data_type_prefix not in c:
                            continue
                        splits = c.split('_')
                        splits[1] = 'chan'
                        joined = '_'.join(splits)
                        columns[i] = joined
                    temp.columns = columns

                    # Create unique trial numbers
                    temp.loc[:, 'trial_num'] = \
                        temp.loc[:, 'trial_num'].values + prev_trial_max
                    prev_trial_max = max(temp.loc[:, 'trial_num'].values) + 1

                    self.data = self.data.append(temp, ignore_index=True)

        elif n_montages == 4:
            for pm in constants.PAIRED_MONTAGES:
                # Check whether we use the base montage for pre-training
                if pm != montage:
                    if voxel_space:
                        try:
                            temp = pd.read_parquet(os.path.join(
                                data_dir, f'{subject}_{pm}_0.parquet'))
                        except FileNotFoundError:
                            print(f'Subject {subject} does not have montage '
                                  f'{pm} for pre-training.')
                            continue

                        # Rename montages to have common columns
                        columns = list(temp.columns)
                        for i, c in enumerate(columns):
                            if self.data_type_prefix not in c:
                                continue
                            splits = c.split('_')
                            del splits[1]
                            joined = '_'.join(splits)
                            columns[i] = joined
                        temp.columns = columns

                        # Create unique trial numbers
                        temp.loc[:, 'trial_num'] = \
                            temp.loc[:, 'trial_num'].values + prev_trial_max
                        prev_trial_max = max(
                            temp.loc[:, 'trial_num'].values) + 1

                        # Add montage data as additional trials
                        self.data = self.data.append(temp, ignore_index=True)
                    else:
                        montages = [k for k, v in paired_montages.items()
                                    if v == pm]

                        trial = pd.DataFrame()

                        for montage_idx, m in enumerate(montages):
                            temp = pd.read_parquet(os.path.join(
                                data_dir, f'{subject}_{m}_0.parquet'))

                            # Rename montages to have common columns
                            columns = list(temp.columns)
                            for i, c in enumerate(columns):
                                if self.data_type_prefix not in c:
                                    continue
                                splits = c.split('_')
                                splits[1] = str(montage_idx)
                                joined = '_'.join(splits)
                                columns[i] = joined
                            temp.columns = columns

                            # Add channels as new features
                            trial = pd.concat([trial, temp], axis=1)

                            # Remove duplicate columns
                            trial = trial.loc[:, ~trial.columns.duplicated()]

                        # Create unique trial numbers
                        trial.loc[:, 'trial_num'] = \
                            trial.loc[:, 'trial_num'].values + prev_trial_max
                        prev_trial_max = max(
                            trial.loc[:, 'trial_num'].values) + 1

                        self.data = self.data.append(trial, ignore_index=True)

        # Separate DataFrame into metadata and dynamic phase data
        meta_cols = ['trial_num', 'subject_id', 'montage']
        chan_cols = [c for c in self.data.columns
                     if self.data_type_prefix in c]
        if voxel_space:
            assert len(chan_cols) == 84
        else:
            assert ((len(chan_cols) == 256 and n_montages == 4) or
                    (len(chan_cols) == 128 and n_montages == 8))
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, chan_cols + ['trial_num']]
        # Labels correspond to the trial type of a specific trial number
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels by dropping channels with all zeros across all trials
        # (across train/valid/test splits)
        # Step 1: replace zeros with NaN
        # Step 2: drop columns with all NaN values
        if filter_zeros:
            self.dynamic_table.loc[:, chan_cols] = \
                self.dynamic_table.loc[:, chan_cols].replace(0, np.nan)
            self.dynamic_table = self.dynamic_table.dropna(axis=1, how='all')
            chan_cols = [c for c in self.dynamic_table.columns
                         if self.data_type_prefix in c]
        # How many viable channels remain?
        viable_chan = [c for c in self.dynamic_table.columns
                       if self.data_type_prefix in c]
        self.num_viable = len(viable_chan)

        # Remove trials that contain all zero values for channels
        self.dynamic_table.loc[:, chan_cols] = \
            self.dynamic_table.loc[:, chan_cols].replace(0, np.nan)
        self.chan_df = self.dynamic_table.loc[:, chan_cols]
        self.valid_rows = self.chan_df[
            ~self.chan_df.isna().all(axis=1)].index
        self.dynamic_table = self.dynamic_table.loc[self.valid_rows, :]
        self.labels = self.labels.loc[
            self.labels['trial_num'].isin(
                set(self.dynamic_table['trial_num'].tolist()))]

        # Combine trial types to get class labels
        # Classify left versus right motor response
        if classification_task == 'motor_LR':
            func = combine_trial_types.combine_motor_LR
            self.classes = 2
        # Classify stimulus modality and motor response
        elif classification_task == 'stim_motor':
            func = combine_trial_types.stim_modality_motor_LR_labels
            self.classes = 4
        # Classify response modality, stimulus modality, and response
        # polarity
        elif classification_task == 'response_stim':
            func = combine_trial_types.response_stim_modality_labels
            self.classes = 8
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels['trial_type'] = self.labels['trial_type'].apply(func)
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

    def get_num_viable_channels(self):
        return self.num_viable


class SubjectMontageDataset(Dataset):
    """
    Creates a PyTorch-loadable dataset that can be used for train/valid/test
    splits. Compatible with K-fold cross-validation and stratified splits.
    """
    def __init__(self, data: FOSData, subset: str = None,
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
        chan_cols = [c for c in self.dynamic_table.columns
                     if self.data_type_prefix in c]
        self.dynamic_table.loc[:, chan_cols].replace(0, np.nan, inplace=True)

        # Get a list of NaN columns that require imputation
        self.nan_cols = list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])

        # Impute missing channels
        imputed_df = self.dynamic_table.copy()
        for col in self.nan_cols:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [col, 'trial_num']]

            # Zero imputation works for all splits
            if preprocessing['impute'] == 'zero':
                imputed_df.loc[:, col] = subset.fillna(value=0.0)

            # Get imputed signal from train set and apply to other subsets
            elif preprocessing['impute'] == 'mean' \
                    and self.subset == 'train':
                running_sig = list()
                # Loop through trials to aggregate signal from viable channels
                for trial in subset.trial_num.unique():
                    # Single trial
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    running_sig.append(subset_subset.loc[:, col].values)
                running_sig = np.vstack(running_sig).T
                if np.count_nonzero(~np.isnan(running_sig)) == 0:
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[0], :]
                    mean_sig = np.zeros_like(subset_subset.loc[:, col].values)
                else:
                    mean_sig = np.nanmean(running_sig, axis=1)
                if self.subset == 'train':
                    self.impute_chan_dict[col] = mean_sig
                # Impute with mean signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[col].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, col] = mean_sig

            elif preprocessing['impute'] == 'random' \
                    and self.subset == 'train':
                # Get non-NaN trials and corresponding indices to choose from
                non_nan_df = self.dynamic_table.dropna(
                    axis=0, how='any', subset=[col])
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
                    rand_sig = np.zeros_like(subset_subset.loc[:, col].values)
                else:
                    # Select random non-NaN trial used for imputation
                    rand = random.choice(non_nan_idx)
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[rand],
                        :]
                    rand_sig = subset_subset.loc[:, col].values
                assert not np.isnan(rand_sig[0])
                if self.subset == 'train':
                    self.impute_chan_dict[col] = rand_sig
                # Impute with random signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[col].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, col] = rand_sig
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        if preprocessing['impute'] == 'zero' or self.subset == 'train':
            assert len(list(
                self.dynamic_table.columns[self.dynamic_table.isna().any()])
                ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.dynamic_table.shape[0] // len(self.trial_id)))

        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            assert data.shape[1] == 56
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

        # Impute missing channels
        imputed_df = self.dynamic_table.copy()
        for col in self.nan_cols:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [col, 'trial_num']]

            # Use pre-calculated imputed signal on training set to imput
            for trial in subset.trial_num.unique():
                subset_subset = subset.loc[subset['trial_num'] == trial, :]
                if np.isnan(subset_subset[col].values[0]):
                    idxs_to_replace = subset_subset.index
                    imputed_df.loc[idxs_to_replace, col] = \
                        impute_chan_dict[col]
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        assert len(list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])
            ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.dynamic_table.shape[0] // len(self.trial_id)))
        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
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
    def __init__(self, learn_data: FOSData, test_data: FOSData,
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
        chan_cols = [c for c in self.dynamic_table.columns
                     if self.data_type_prefix in c]
        self.dynamic_table.loc[:, chan_cols].replace(0, np.nan, inplace=True)

        # Get a list of NaN columns that require imputation
        self.nan_cols = list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])

        # Impute missing channels
        imputed_df = self.dynamic_table.copy()
        for col in self.nan_cols:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [col, 'trial_num']]

            # Zero imputation works for all splits
            if preprocessing['impute'] == 'zero':
                imputed_df.loc[:, col] = subset.fillna(value=0.0)

            # Get imputed signal from train set and apply to other subsets
            elif preprocessing['impute'] == 'mean' \
                    and self.subset == 'train':
                running_sig = list()
                # Loop through trials to aggregate signal from viable channels
                for trial in subset.trial_num.unique():
                    # Single trial
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    running_sig.append(subset_subset.loc[:, col].values)
                running_sig = np.vstack(running_sig).T
                if np.count_nonzero(~np.isnan(running_sig)) == 0:
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[0], :]
                    mean_sig = np.zeros_like(subset_subset.loc[:, col].values)
                else:
                    mean_sig = np.nanmean(running_sig, axis=1)
                if self.subset == 'train':
                    self.impute_chan_dict[col] = mean_sig
                # Impute with mean signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[col].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, col] = mean_sig

            elif preprocessing['impute'] == 'random' \
                    and self.subset == 'train':
                # Get non-NaN trials and corresponding indices to choose from
                non_nan_df = self.dynamic_table.dropna(
                    axis=0, how='any', subset=[col])
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
                    rand_sig = np.zeros_like(subset_subset.loc[:, col].values)
                else:
                    # Select random non-NaN trial used for imputation
                    rand = random.choice(non_nan_idx)
                    subset_subset = subset.loc[
                        subset['trial_num'] == subset.trial_num.unique()[rand],
                        :]
                    rand_sig = subset_subset.loc[:, col].values
                assert not np.isnan(rand_sig[0])
                if self.subset == 'train':
                    self.impute_chan_dict[col] = rand_sig
                # Impute with random signal
                for trial in subset.trial_num.unique():
                    subset_subset = subset.loc[subset['trial_num'] == trial, :]
                    if np.isnan(subset_subset[col].values[0]):
                        idxs_to_replace = subset_subset.index
                        imputed_df.loc[idxs_to_replace, col] = rand_sig
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        if preprocessing['impute'] == 'zero' or self.subset == 'train':
            assert len(list(
                self.dynamic_table.columns[self.dynamic_table.isna().any()])
                ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.dynamic_table.shape[0] // len(self.trial_id)))

        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            assert data.shape[1] == 56
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
        for col in self.nan_cols:
            # Subset of columns
            subset = self.dynamic_table.loc[:, [col, 'trial_num']]

            # Use pre-calculated imputed signal on training set to imput
            for trial in subset.trial_num.unique():
                subset_subset = subset.loc[subset['trial_num'] == trial, :]
                if np.isnan(subset_subset[col].values[0]):
                    idxs_to_replace = subset_subset.index
                    imputed_df.loc[idxs_to_replace, col] = \
                        impute_chan_dict[col]
        self.dynamic_table = imputed_df
        # Check to make sure no more NaN columns exist
        assert len(list(
            self.dynamic_table.columns[self.dynamic_table.isna().any()])
            ) == 0

        # Stores the dynamic data in a (N, C, T) tensor
        self.dynamic_data = np.empty((
            len(self.trial_id), len(self.chan_cols),
            self.dynamic_table.shape[0] // len(self.trial_id)))
        for idx, trial in enumerate(self.trial_id):
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data.loc[:, [
                c for c in dynamic_trial_data.columns
                if c != 'trial_num' and c != 'idx']
                ].values.astype(np.float32).T  # data: (C, T)
            # Maximum absolute value scaling
            if self.preprocessing['max_abs_scale']:
                data = maxabs_scale(data)
            self.dynamic_data[idx, :, :] = data


class DatasetBuilder:
    def __init__(self, data: FOSData, seed: int = 42, seed_cv: int = 15,
                 **preprocessing):
        self.data = data
        self.seed = seed
        self.seed_cv = seed_cv
        self.preprocessing = preprocessing

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[Tuple[FOSData, FOSData]], FOSData]]:
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
            def _inner_loop(data: FOSData, seed: int, seed_cv: seed):
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
    def __init__(self, data_A: FOSData, data_B: FOSData,
                 seed: int = 42, seed_cv: int = 15, **preprocessing):
        self.data_A = data_A
        self.data_B = data_B
        self.seed = seed
        self.seed_cv = seed_cv
        self.preprocessing = preprocessing

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[Tuple[FOSData, FOSData]], FOSData]]:
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
            def _inner_loop(data: FOSData, seed: int, seed_cv: seed):
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
    def __init__(self, learn_data: FOSData, test_data: FOSData,
                 seed: int = 42, seed_cv: int = 15, **preprocessing):
        self.learn_data = learn_data
        self.test_data = test_data
        self.seed = seed
        self.seed_cv = seed_cv
        self.preprocessing = preprocessing

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[Tuple[FOSData, FOSData]], FOSData]]:
        """
        Yields Datasets in the tuple form ((train, valid), test), where
        the inner tuple is iterated over for each cross-validation split.
        The test set consists of a witheld subject's data.
        """
        seed = self.seed
        seed_cv = self.seed_cv

        # Iterate through possible test sets (typically just use one)
        for i in range(nested_cv):
            def _inner_loop(learn_data: FOSData, test_data: FOSData,
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
