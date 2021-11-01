import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from utils import constants


def extract_features(data_dir, input_fname, window, csp=False):

    df = pd.read_parquet(os.path.join(data_dir, input_fname))
    
    if csp:
        cols = [c for c in df.columns if 'csp' in c]
        orig = df[cols].values
        length = len(orig[0, 0])
    else:
        cols = [f'ph_win{window}_{m}_{c}_{f}' for m in montages for c in range(128) for f in ['04', '08', '13']]
        length = len(df.loc[0, f'ph_win{window}_e_0_04'])
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

    return final_df


def extract_features_all(data_dir, input_fname, csp=False):

    df = pd.read_parquet(os.path.join(data_dir, input_fname))
    
    if csp:
        cols = [c for c in df.columns if 'csp' in c]
        orig = df[cols].values
        length = len(orig[0, 0])
    else:
        cols = [f'ph_{m}_{c}_{f}' for m in montages for c in range(128) for f in ['04', '08', '13']]
        length = len(df.loc[0, f'ph_e_0_04'])
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

    return final_df


montages = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str,
                        default=constants.BANDPOWER_DIR)
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--csp', action='store_true')
    parser.add_argument('--classification_task', type=str,
                        default='motor_LR',
                        help='options include motor_LR (motor response), '
                        'stim_motor (stimulus modality and motor response) or '
                        'response_stim (response modality, stimulus modality, '
                        'and response polarity).')

    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir, args.anchor)
    os.makedirs(out_dir, exist_ok=True)

    # Without CSP
    if not args.csp:
        data_dir = os.path.join(constants.PYTHON_DIR, 'phase_data', args.anchor)

        input_fname = f'phase_all_single_trial.parquet'
        output_fname = f'all_simple_bandpower_features.parquet'

        df = extract_features_all(data_dir, input_fname)
            
        print('Writing output file...')
        df.to_parquet(os.path.join(out_dir, output_fname), index=False)

        for i in tqdm(range(8)):
            input_fname = f'phase_win{i}_single_trial.parquet'
            output_fname = f'win{i}_simple_bandpower_features.parquet'

            df = extract_features(data_dir, input_fname, i)
            
            df.to_parquet(os.path.join(out_dir, output_fname), index=False)
    # With CSP
    else:
        data_dir = os.path.join(constants.PYTHON_DIR, 'csp_transform',
                                args.classification_task, args.anchor, 'filt')

        for n_filters in tqdm([16, 8, 4, 2]):

            input_fname = f'CSP_filt_{n_filters}_all.parquet'
            output_fname = f'all_simple_bandpower_features_csp_{n_filters}.parquet'

            df = extract_features_all(data_dir, input_fname, csp=True)
                
            print('Writing output file...')
            df.to_parquet(os.path.join(out_dir, output_fname), index=False)

            for i in tqdm(range(8), leave=False):
                input_fname = f'CSP_filt_{n_filters}_win{i}.parquet'
                output_fname = f'win{i}_simple_bandpower_features_csp_{n_filters}.parquet'

                df = extract_features(data_dir, input_fname, i, csp=True)
                
                df.to_parquet(os.path.join(out_dir, output_fname), index=False)
