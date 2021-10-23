import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from utils import constants


def extract_features(data_dir, input_fname, key, output_fname, csp=False):

    df = pd.read_parquet(os.path.join(data_dir, input_fname))
    
    if csp:
        cols = [c for c in df.columns if 'csp' in c]
        orig = df[cols].values
        length = len(orig[0, 0])
    else:
        cols = [f'{window_mapping[key]}_{m}_{i}_{f}' for m in montages for i in range(128) for f in ['04', '08', '13']]
        length = len(df.loc[0, f'{window_mapping[key]}_e_0_04'])
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
    range_feats = pd.DataFrame((np.max(empty.copy(), axis=-1) - np.min(empty.copy(), axis=-1)),
        columns=[f'range_{c}' for c in cols])
    avg_pwr_feats = pd.DataFrame(np.sum(empty.copy() ** 2,  axis=-1) / empty.shape[2],
        columns=[f'avg_pwr_{c}' for c in cols])

    nan_mask = np.isnan(empty.copy()).any(axis=-1)
    new_values = np.sum(empty.copy() > 0,  axis=-1)
    new_values = new_values.astype('float64')
    new_values[nan_mask] = np.nan

    samp_gt_zero_feats = pd.DataFrame(new_values,
        columns=[f'samp_gt_zero_{c}' for c in cols])
    
    def num_zero_crossing(arr):
        return len(np.where(np.diff(np.signbit(arr)))[0])

    nan_mask = np.isnan(empty.copy()).any(axis=-1)
    new_values = np.apply_along_axis(num_zero_crossing, axis=-1, arr=empty.copy())
    new_values = new_values.astype('float64')
    new_values[nan_mask] = np.nan

    zero_cross_feats = pd.DataFrame(new_values,
        columns=[f'zero_cross_{c}' for c in cols])
    
    feats_df = pd.concat([max_feats, min_feats, mean_feats, range_feats, avg_pwr_feats, samp_gt_zero_feats, zero_cross_feats], axis=1)
    info_df = df[['trial_num', 'subject_id', 'trial_type', 'montage']]
    final_df = pd.concat([info_df, feats_df], axis=1)
    final_df.dropna(axis=1, how='all', inplace=True)

    print('Writing output file...')
    final_df.to_parquet(os.path.join(constants.BANDPOWER_DIR, output_fname),
                        index=False)


montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

window_mapping = {
    'all': 'ph',
    'rt': 'ph-rt',
    'pre_stim': 'ph-pre-stim',
    'init': 'ph-init',
    'pre_rt': 'ph-pre-rt',
    'post_rt': 'ph-post-rt'
}


if __name__ == '__main__':

    # Without CSP
    data_dir = os.path.join(constants.PYTHON_DIR, 'phase_data')

    for key in tqdm(window_mapping.keys()):
        input_fname = f'phase_{key}_filt_chan.parquet'
        output_fname = f'{key}_simple_bandpower_features.parquet'

        extract_features(data_dir, input_fname, key, output_fname)

    # With CSP
    data_dir = os.path.join(constants.PYTHON_DIR, 'csp_transform')

    for key in tqdm(window_mapping.keys()):
        for n_filters in [16, 8, 4, 2]:
            input_fname = f'CSP_filt_{n_filters}_{key}.parquet'
            output_fname = f'simple_bandpower_features_csp_{n_filters}_{key}.parquet'

            extract_features(data_dir, input_fname, key, output_fname, csp=True)