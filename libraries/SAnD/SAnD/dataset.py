import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import maxabs_scale
from torch.utils.data import Dataset

from utils import combine_trial_types, constants


class FOSData:
    """ Abstract class for optical imaging data. """
    def __init__(self):
        pass


class SubjectMontageData(FOSData):
    """
    Creates a representation of the FOS data set that groups subjects' dynamic
    data into a single tensor along with  their corresponding labels based on
    the classification task. Performs basic pre-processing to as specified by
    the input arguments.
    """
    def __init__(self, data_dir: str, subject: str, montage: str,
                 classification_task: str, seq_len: int, n_montages: int,
                 pool_sep_montages: bool, filter_zeros: bool, pool_ops: bool,
                 max_abs_scale: bool):

        assert (pool_sep_montages and n_montages == 4) or not pool_sep_montages

        self.data_dir = data_dir
        self.seq_len = seq_len

        s04 = pd.DataFrame()
        s08 = pd.DataFrame()
        s13 = pd.DataFrame()

        # Group montages in pairs based on trial num recorded (a-b, c-d, etc.)
        if n_montages == 8:
            montages = [montage]
        elif n_montages == 4:
            paired_montages = {'a': 'A', 'b': 'A',
                               'c': 'B', 'd': 'B',
                               'e': 'C', 'f': 'C',
                               'g': 'D', 'h': 'D'}
            montages = [k for k, v in paired_montages.items() if v == montage]

        for m in montages:
            temp04 = pd.read_parquet(
                os.path.join(data_dir, f'{subject}_{m}_4.parquet'))
            temp08 = pd.read_parquet(
                os.path.join(data_dir, f'{subject}_{m}_8.parquet'))
            temp13 = pd.read_parquet(
                os.path.join(data_dir, f'{subject}_{m}_13.parquet'))

            # Add channels as new features
            s04 = pd.concat([s04, temp04], axis=1)
            s08 = pd.concat([s08, temp08], axis=1)
            s13 = pd.concat([s13, temp13], axis=1)

            # Remove duplicate columns (i.e. trial_num, subject_id, etc.)
            s04 = s04.loc[:, ~s04.columns.duplicated()]
            s08 = s08.loc[:, ~s08.columns.duplicated()]
            s13 = s13.loc[:, ~s13.columns.duplicated()]

        s04.columns = [f'{c}_04' for c in s04.columns]
        s08.columns = [f'{c}_08' for c in s08.columns]
        s13.columns = [f'{c}_13' for c in s13.columns]

        self.in_data = pd.concat([s04, s08, s13], axis=1)
        self.in_data = self.in_data.rename({'trial_num_04': 'trial_num',
                                            'subject_id_04': 'subject_id',
                                            'trial_type_04': 'trial_type',
                                            'montage_04': 'montage'}, axis=1)
        self.in_data = self.in_data.drop(['freq_band_04', 'event_04',
                                          'trial_num_08', 'subject_id_08',
                                          'trial_type_08', 'montage_08',
                                          'freq_band_08', 'event_08',
                                          'trial_num_13', 'subject_id_13',
                                          'trial_type_13', 'montage_13',
                                          'freq_band_13', 'event_13'], axis=1)

        meta_cols = ['trial_num', 'subject_id', 'montage']
        chan_cols = [c for c in self.in_data.columns if 'ph_' in c]
        assert ((len(chan_cols) == 3 * 256 and n_montages == 4) or
                (len(chan_cols) == 3 * 128 and n_montages == 8))
        self.meta_data = self.in_data.loc[:, meta_cols]
        self.dynamic_table = self.in_data.loc[:, chan_cols + ['trial_num']]
        self.labels = self.in_data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels with all zeros
        if filter_zeros:
            self.dynamic_table.loc[:, (self.dynamic_table != 0).any(axis=0)]

        # Perform pooling operations within frequency band
        dynamic_cols = list()
        if pool_sep_montages:
            for m in montages:
                for freq in ['04', '08', '13']:
                    if pool_ops['mean']:
                        prefix = 'avg'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == m)]
                                ].mean(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['median']:
                        prefix = 'med'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == m)]
                                ].median(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['min']:
                        prefix = 'min'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == m)]
                                ].min(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['max']:
                        prefix = 'max'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == m)]
                                ].max(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['std']:
                        prefix = 'std'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == m)]
                                ].std(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
        else:
            for freq in ['04', '08', '13']:
                if pool_ops['mean']:
                    prefix = 'avg'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].mean(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['median']:
                    prefix = 'med'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].median(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['min']:
                    prefix = 'min'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].min(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['max']:
                    prefix = 'max'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].max(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['std']:
                    prefix = 'std'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].std(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
        if True in pool_ops.values():
            self.dynamic_table = self.dynamic_table.loc[
                :, dynamic_cols + ['trial_num']]

        # Combine trial types
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
            func = combine_trial_types.sresponse_stim_modality_labels
            self.classes = 8
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels['trial_type'] = self.labels['trial_type'].apply(func)
        self.labels.rename({'trial_type': 'label'}, axis=1, inplace=True)
        self.labels = self.labels.dropna(
            axis=0, subset=['label']).copy().reset_index(drop=True)
        self.trial_id = self.labels.loc[:, 'trial_num'].unique()
        self.labels = self.labels.loc[:, 'label'].values.astype(np.float32)
        self.idxs = list(range(len(self.labels)))

        self.data = self.dynamic_table.loc[
            self.dynamic_table['trial_num'].isin(self.trial_id),
            [c for c in self.dynamic_table.columns if c != 'trial_num']
            ].values.astype(np.float32)
        self.data = self.data.reshape((-1, self.seq_len, self.data.shape[1]))
        # Maximum absolute value scaling
        if max_abs_scale:
            for idx, slice in enumerate(self.data):
                self.data[idx, :, :] = maxabs_scale(slice, axis=0)


class MontagePretrainData(FOSData):
    """
    Creates a representation of the FOS data set that groups subjects' dynamic
    data with their corresponding labels based on the classification task.
    Performs basic pre-processing to as specified by the input arguments.

    Functions identically to the SubjectMontageDataset, except the montage
    of interest is excluded from the data and all other montages are used.
    """
    def __init__(self, data_dir: str, subject: str, montage: str,
                 classification_task: str, seq_len: int, n_montages: int,
                 pool_sep_montages: bool, filter_zeros: bool, pool_ops: dict,
                 max_abs_scale: bool):

        assert (pool_sep_montages and n_montages == 4) or not pool_sep_montages

        self.data_dir = data_dir
        self.seq_len = seq_len

        s04 = pd.DataFrame()
        s08 = pd.DataFrame()
        s13 = pd.DataFrame()

        prev_trial_max = 0
        paired_montages = {'a': 'A', 'b': 'A',
                           'c': 'B', 'd': 'B',
                           'e': 'C', 'f': 'C',
                           'g': 'D', 'h': 'D'}

        if n_montages == 8:
            for m in constants.MONTAGES:
                # Check whether we use the base montage for pre-training
                if m != montage:
                    temp04 = pd.read_parquet(
                        os.path.join(data_dir, f'{subject}_{m}_4.parquet'))
                    temp08 = pd.read_parquet(
                        os.path.join(data_dir, f'{subject}_{m}_8.parquet'))
                    temp13 = pd.read_parquet(
                        os.path.join(data_dir, f'{subject}_{m}_13.parquet'))

                    # Rename montages to have common columns
                    columns = list(temp04.columns)
                    for i, c in enumerate(columns):
                        if 'ph_' not in c:
                            continue
                        splits = c.split('_')
                        splits[1] = 'chan'
                        joined = '_'.join(splits)
                        columns[i] = joined
                    temp04.columns = columns
                    temp08.columns = columns
                    temp13.columns = columns

                    # Create unique trial numbers
                    temp04.loc[:, 'trial_num'] = \
                        temp04.loc[:, 'trial_num'].values + prev_trial_max
                    temp08.loc[:, 'trial_num'] = \
                        temp08.loc[:, 'trial_num'].values + prev_trial_max
                    temp13.loc[:, 'trial_num'] = \
                        temp13.loc[:, 'trial_num'].values + prev_trial_max
                    prev_trial_max = max(temp04.loc[:, 'trial_num'].values) + 1

                    s04 = s04.append(temp04, ignore_index=True)
                    s08 = s08.append(temp08, ignore_index=True)
                    s13 = s13.append(temp13, ignore_index=True)

        elif n_montages == 4:
            for pm in constants.PAIRED_MONTAGES:
                # Check whether we use the base montage for pre-training
                if pm != montage:
                    montages = [k for k, v in paired_montages.items()
                                if v == pm]

                    trial04 = pd.DataFrame()
                    trial08 = pd.DataFrame()
                    trial13 = pd.DataFrame()

                    for montage_idx, m in enumerate(montages):
                        temp04 = pd.read_parquet(os.path.join(
                            data_dir, f'{subject}_{m}_4.parquet'))
                        temp08 = pd.read_parquet(os.path.join(
                            data_dir, f'{subject}_{m}_8.parquet'))
                        temp13 = pd.read_parquet(os.path.join(
                            data_dir, f'{subject}_{m}_13.parquet'))

                        # Rename montages to have common columns
                        columns = list(temp04.columns)
                        for i, c in enumerate(columns):
                            if 'ph_' not in c:
                                continue
                            splits = c.split('_')
                            splits[1] = str(montage_idx)
                            joined = '_'.join(splits)
                            columns[i] = joined
                        temp04.columns = columns
                        temp08.columns = columns
                        temp13.columns = columns

                        # Add channels as new features
                        trial04 = pd.concat([trial04, temp04], axis=1)
                        trial08 = pd.concat([trial08, temp08], axis=1)
                        trial13 = pd.concat([trial13, temp13], axis=1)

                        # Remove duplicate columns
                        trial04 = trial04.loc[:, ~trial04.columns.duplicated()]
                        trial08 = trial08.loc[:, ~trial08.columns.duplicated()]
                        trial13 = trial13.loc[:, ~trial13.columns.duplicated()]

                    # Create unique trial numbers
                    trial04.loc[:, 'trial_num'] = \
                        trial04.loc[:, 'trial_num'].values + prev_trial_max
                    trial08.loc[:, 'trial_num'] = \
                        trial08.loc[:, 'trial_num'].values + prev_trial_max
                    trial13.loc[:, 'trial_num'] = \
                        trial13.loc[:, 'trial_num'].values + prev_trial_max
                    prev_trial_max = max(
                        trial04.loc[:, 'trial_num'].values) + 1

                    s04 = s04.append(trial04, ignore_index=True)
                    s08 = s08.append(trial08, ignore_index=True)
                    s13 = s13.append(trial13, ignore_index=True)

        s04.columns = [f'{c}_04' for c in s04.columns]
        s08.columns = [f'{c}_08' for c in s08.columns]
        s13.columns = [f'{c}_13' for c in s13.columns]

        self.in_data = pd.concat([s04, s08, s13], axis=1)
        self.in_data = self.in_data.rename({'trial_num_04': 'trial_num',
                                            'subject_id_04': 'subject_id',
                                            'trial_type_04': 'trial_type',
                                            'montage_04': 'montage'}, axis=1)
        self.in_data = self.in_data.drop(['freq_band_04', 'event_04',
                                          'trial_num_08', 'subject_id_08',
                                          'trial_type_08', 'montage_08',
                                          'freq_band_08', 'event_08',
                                          'trial_num_13', 'subject_id_13',
                                          'trial_type_13', 'montage_13',
                                          'freq_band_13', 'event_13'], axis=1)

        meta_cols = ['trial_num', 'subject_id', 'montage']
        chan_cols = [c for c in self.in_data.columns if 'ph_' in c]
        assert ((len(chan_cols) == 3 * 256 and n_montages == 4) or
                (len(chan_cols) == 3 * 128 and n_montages == 8))
        self.meta_data = self.in_data.loc[:, meta_cols]
        self.dynamic_table = self.in_data.loc[:, chan_cols + ['trial_num']]
        self.labels = self.in_data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Filter channels with all zeros
        if filter_zeros:
            self.dynamic_table.loc[:, (self.dynamic_table != 0).any(axis=0)]

        # Perform pooling operations within frequency band
        dynamic_cols = list()
        if pool_sep_montages:
            for m in range(2):
                for freq in ['04', '08', '13']:
                    if pool_ops['mean']:
                        prefix = 'avg'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == str(m))]
                                ].mean(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['median']:
                        prefix = 'med'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == str(m))]
                                ].median(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['min']:
                        prefix = 'min'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == str(m))]
                                ].min(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['max']:
                        prefix = 'max'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == str(m))]
                                ].max(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
                    if pool_ops['std']:
                        prefix = 'std'
                        self.dynamic_table[f'{prefix}_{m}_{freq}'] = \
                            self.dynamic_table[
                                [c for c in self.dynamic_table
                                 if f'_{freq}' in c and 'ph_' in c
                                 and str(c.split('_')[1] == str(m))]
                                ].std(axis=1)
                        dynamic_cols.append(f'{prefix}_{m}_{freq}')
        else:
            for freq in ['04', '08', '13']:
                if pool_ops['mean']:
                    prefix = 'avg'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].mean(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['median']:
                    prefix = 'med'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].median(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['min']:
                    prefix = 'min'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].min(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['max']:
                    prefix = 'max'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].max(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
                if pool_ops['std']:
                    prefix = 'std'
                    self.dynamic_table[f'{prefix}_{freq}'] = \
                        self.dynamic_table[
                            [c for c in self.dynamic_table
                             if f'_{freq}' in c and 'ph_' in c]].std(axis=1)
                    dynamic_cols.append(f'{prefix}_{freq}')
        if True in pool_ops.values():
            self.dynamic_table = self.dynamic_table.loc[
                :, dynamic_cols + ['trial_num']]

        # Combine trial types
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
            func = combine_trial_types.sresponse_stim_modality_labels
            self.classes = 8
        else:
            raise NotImplementedError('Classification task not supported')
        self.labels['trial_type'] = self.labels['trial_type'].apply(func)
        self.labels.rename({'trial_type': 'label'}, axis=1, inplace=True)
        self.labels = self.labels.dropna(
            axis=0, subset=['label']).copy().reset_index(drop=True)
        self.trial_id = self.labels.loc[:, 'trial_num'].unique()
        self.labels = self.labels.loc[:, 'label'].values.astype(np.float32)
        self.idxs = list(range(len(self.labels)))

        self.data = self.dynamic_table.loc[
            self.dynamic_table['trial_num'].isin(self.trial_id),
            [c for c in self.dynamic_table.columns if c != 'trial_num']
            ].values.astype(np.float32)
        self.data = self.data.reshape((-1, self.seq_len, self.data.shape[1]))
        # Maximum absolute value scaling
        if max_abs_scale:
            for idx, slice in enumerate(self.data):
                self.data[idx, :, :] = maxabs_scale(slice, axis=0)


class SubjectMontageDataset(Dataset):
    """
    Creates a PyTorch-loadable dataset that can be used for train/valid/test
    splits. Compatible with K-fold cross-validation and stratified splits.
    """
    def __init__(self, data: FOSData, subset: str = None,
                 seed: int = 42, props: Tuple[float] = (70, 10, 20),
                 stratified: bool = False, cv: int = 1, nested_cv: int = 1,
                 cv_idx: int = 0, nested_cv_idx: int = 0, seed_cv: int = 15):
        super().__init__()

        subsets = ['train', 'valid', 'test']
        assert subset in subsets
        assert sum(props) == 100

        self.data_in = data
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

        self.labels = self.data_in.labels
        self.trial_id = self.data_in.trial_id
        self.data = self.data_in.data
        self.idxs = self.data_in.idxs

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
        self.data = self.data[self.idxs, :]
        self.labels = self.labels[self.idxs]

    def __getitem__(self, index: int) -> \
            Tuple[int, np.ndarray, np.ndarray]:
        return (self.trial_id[index], self.data[index], self.labels[index])

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

    def get_label(self, idx):
        return self.__getitem__(idx)[2]

    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(data: List) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        ids = [item[0] for item in data]
        signals = torch.tensor([item[1] for item in data])
        labels = torch.tensor([item[2] for item in data])
        return ids, signals, labels


class DatasetBuilder:
    def __init__(self, data: FOSData, seed: int = 42, seed_cv: int = 15):
        self.data = data
        self.seed = seed
        self.seed_cv = seed_cv

    def build_datasets(self, cv: int, nested_cv: int) -> Iterable[
            Tuple[Iterable[
                Tuple[SubjectMontageDataset, SubjectMontageDataset]],
                SubjectMontageDataset]]:
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
                    yield (SubjectMontageDataset(
                                data=data, subset='train', stratified=True,
                                cv=cv, nested_cv=nested_cv,
                                cv_idx=j, nested_cv_idx=i,
                                seed=seed, seed_cv=seed_cv),
                           SubjectMontageDataset(
                                data=data, subset='valid', stratified=True,
                                cv=cv, nested_cv=nested_cv,
                                cv_idx=j, nested_cv_idx=i,
                                seed=seed, seed_cv=seed_cv))

            yield (_inner_loop(self.data, self.seed, self.seed_cv),
                   SubjectMontageDataset(
                       data=self.data, subset='test', stratified=True, cv=cv,
                       nested_cv=nested_cv, cv_idx=0, nested_cv_idx=i,
                       seed=seed, seed_cv=seed_cv))
