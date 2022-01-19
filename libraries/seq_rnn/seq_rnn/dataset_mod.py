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
    data with their corresponding labels based on the classification task.
    Performs basic pre-processing to as specified by the input arguments.
    """
    def __init__(self, data_dir: str, subject: str, montage: str,
                 classification_task: str, n_montages: int,
                 filter_zeros: bool, average_chan: bool, max_abs_scale: bool):

        self.data_dir = data_dir

        s04 = pd.read_parquet(
            os.path.join(data_dir, f'{subject}_{montage}_4.parquet'))
        s08 = pd.read_parquet(
            os.path.join(data_dir, f'{subject}_{montage}_8.parquet'))
        s13 = pd.read_parquet(
            os.path.join(data_dir, f'{subject}_{montage}_13.parquet'))

        s04.columns = [f'{c}_04' for c in s04.columns]
        s08.columns = [f'{c}_08' for c in s08.columns]
        s13.columns = [f'{c}_13' for c in s13.columns]

        self.data = pd.concat([s04, s08, s13], axis=1)
        self.data = self.data.rename({'trial_num_04': 'trial_num',
                                      'subject_id_04': 'subject_id',
                                      'trial_type_04': 'trial_type',
                                      'montage_04': 'montage'}, axis=1)
        self.data = self.data.drop(['freq_band_04', 'event_04',
                                    'trial_num_08', 'subject_id_08',
                                    'trial_type_08', 'montage_08',
                                    'freq_band_08', 'event_08',
                                    'trial_num_13', 'subject_id_13',
                                    'trial_type_13', 'montage_13',
                                    'freq_band_13', 'event_13'], axis=1)

        meta_cols = ['trial_num', 'subject_id', 'montage']
        chan_cols = [c for c in self.data.columns if 'ph_' in c]
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, chan_cols + ['trial_num']]
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Group montages in pairs based on trial num recorded (a-b, c-d, etc.)
        if n_montages == 8:
            pass
        elif n_montages == 4:
            paired_montages = {'a': 'A', 'b': 'A',
                               'c': 'B', 'd': 'B',
                               'e': 'C', 'f': 'C',
                               'g': 'D', 'h': 'D'}
            columns = list(self.dynamic_table.columns)
            for i, c in enumerate(columns):
                if c == 'trial_num':
                    continue
                splits = c.split('_')
                splits[1] = paired_montages[splits[1]]
                joined = '_'.join(splits)
                columns[i] = joined
            self.dynamic_table.columns = columns

        # Filter channels with all zeros
        if filter_zeros:
            self.dynamic_table.loc[:, (self.dynamic_table != 0).any(axis=0)]
        # Average channels of the same frequency band
        if average_chan:
            self.dynamic_table['avg_04'] = self.dynamic_table[
                [c for c in self.dynamic_table if '_04' in c and 'ph_' in c]
                ].mean(axis=1)
            self.dynamic_table['avg_08'] = self.dynamic_table[
                [c for c in self.dynamic_table if '_08' in c and 'ph_' in c]
                ].mean(axis=1)
            self.dynamic_table['avg_13'] = self.dynamic_table[
                [c for c in self.dynamic_table if '_13' in c and 'ph_' in c]
                ].mean(axis=1)
            self.dynamic_table = self.dynamic_table.loc[
                :, ['avg_04', 'avg_08', 'avg_13'] + ['trial_num']]

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

        # Use dynamic_data to store a list of each trial's signal, where each
        # element of the list is a list of 1D numpy arrays where each array
        # represents a single recording data point.
        self.dynamic_data = []
        for trial in self.trial_id:
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data[
                [c for c in dynamic_trial_data.columns if c != 'trial_num']
                ].values.astype(np.float32)
            # Maximum absolute value scaling
            if max_abs_scale:
                data = maxabs_scale(data, axis=0)
            data = [row for row in data]
            self.dynamic_data.append(data)


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
                 filter_zeros: bool, average_chan: bool, max_abs_scale: bool):

        self.data_dir = data_dir

        s04 = pd.DataFrame()
        s08 = pd.DataFrame()
        s13 = pd.DataFrame()

        prev_trial_max = 0
        paired_montages = {'a': 'A', 'b': 'A',
                           'c': 'B', 'd': 'B',
                           'e': 'C', 'f': 'C',
                           'g': 'D', 'h': 'D'}
        for m in constants.MONTAGES:
            # Check whether we use the base montage for pre-training
            if (n_montages == 8 and m != montage) or \
               (n_montages == 4 and paired_montages[m] != montage):

                temp04 = pd.read_parquet(
                    os.path.join(data_dir, f'{subject}_{m}_4.parquet'))
                temp08 = pd.read_parquet(
                    os.path.join(data_dir, f'{subject}_{m}_8.parquet'))
                temp13 = pd.read_parquet(
                    os.path.join(data_dir, f'{subject}_{m}_13.parquet'))

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

        s04.columns = [f'{c}_04' for c in s04.columns]
        s08.columns = [f'{c}_08' for c in s08.columns]
        s13.columns = [f'{c}_13' for c in s13.columns]

        self.data = pd.concat([s04, s08, s13], axis=1)
        self.data = self.data.rename({'trial_num_04': 'trial_num',
                                      'subject_id_04': 'subject_id',
                                      'trial_type_04': 'trial_type',
                                      'montage_04': 'montage'}, axis=1)
        self.data = self.data.drop(['freq_band_04', 'event_04',
                                    'trial_num_08', 'subject_id_08',
                                    'trial_type_08', 'montage_08',
                                    'freq_band_08', 'event_08',
                                    'trial_num_13', 'subject_id_13',
                                    'trial_type_13', 'montage_13',
                                    'freq_band_13', 'event_13'], axis=1)

        meta_cols = ['trial_num', 'subject_id', 'montage']
        chan_cols = [c for c in self.data.columns if 'ph_' in c]
        self.meta_data = self.data.loc[:, meta_cols]
        self.dynamic_table = self.data.loc[:, chan_cols + ['trial_num']]
        self.labels = self.data.groupby('trial_num').mean().reset_index()[
            ['trial_num', 'trial_type']]
        self.labels.index.name = None
        self.labels = pd.DataFrame(self.labels,
                                   columns=['trial_num', 'trial_type'])

        # Group montages in pairs based on trial num recorded (a-b, c-d, etc.)
        if n_montages == 8:
            pass
        elif n_montages == 4:
            columns = list(self.dynamic_table.columns)
            for i, c in enumerate(columns):
                if c == 'trial_num':
                    continue
                splits = c.split('_')
                splits[1] = paired_montages[splits[1]]
                joined = '_'.join(splits)
                columns[i] = joined
            self.dynamic_table.columns = columns

        # Filter channels with all zeros
        if filter_zeros:
            self.dynamic_table.loc[:, (self.dynamic_table != 0).any(axis=0)]
        # Average channels of the same frequency band
        if average_chan:
            self.dynamic_table['avg_04'] = self.dynamic_table[
                [c for c in self.dynamic_table if '_04' in c and 'ph_' in c]
                ].mean(axis=1)
            self.dynamic_table['avg_08'] = self.dynamic_table[
                [c for c in self.dynamic_table if '_08' in c and 'ph_' in c]
                ].mean(axis=1)
            self.dynamic_table['avg_13'] = self.dynamic_table[
                [c for c in self.dynamic_table if '_13' in c and 'ph_' in c]
                ].mean(axis=1)
            self.dynamic_table = self.dynamic_table.loc[
                :, ['avg_04', 'avg_08', 'avg_13'] + ['trial_num']]

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

        # Use dynamic_data to store a list of each trial's signal, where each
        # element of the list is a list of 1D numpy arrays where each array
        # represents a single recording data point for multiple channels.
        self.dynamic_data = []
        for trial in self.trial_id:
            dynamic_trial_data = self.dynamic_table.loc[
                self.dynamic_table['trial_num'] == trial, :]
            data = dynamic_trial_data[
                [c for c in dynamic_trial_data.columns if c != 'trial_num']
                ].values.astype(np.float32)
            # Maximum absolute value scaling
            if max_abs_scale:
                data = maxabs_scale(data, axis=0)
            data = [row for row in data]
            self.dynamic_data.append(data)


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

        self.data = data
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

        self.labels = self.data.labels
        self.trial_id = self.data.trial_id
        self.dynamic_data = self.data.dynamic_data
        self.idxs = self.data.idxs

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
                         self.proportions['valid'])))
            else:
                sss_inner = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.seed_cv)
            train_idx, valid_idx = list(
                sss_inner.split(learn, learn_labels))[self.cv_idx]

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
        self.dynamic_data = [self.dynamic_data[id_] for id_ in self.idxs]
        self.labels = self.labels[self.idxs]
        self.lengths = [len(seq) for seq in self.dynamic_data]

    def __getitem__(self, index: int) -> \
            Tuple[int, np.ndarray, List, int, List[int]]:
        return (self.trial_id[index], self.dynamic_data[index],
                self.lengths[index], self.labels[index])

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
        return self.__getitem__(idx)[4]

    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(data: List) -> Tuple[List[int], torch.Tensor,
                                     List[torch.Tensor], torch.Tensor,
                                     torch.Tensor]:
        ids = [item[0] for item in data]
        dynamic_data = [torch.tensor(item[1]) for item in data]
        lengths = torch.tensor([item[2] for item in data])
        labels = torch.tensor([item[3] for item in data])
        return ids, dynamic_data, lengths, labels


class DatasetBuilder:
    def __init__(self, data: FOSData, seed: int = 42, seed_cv: int = 15):
        self.data = data
        self.seed = seed
        self.seed_cv = seed_cv

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