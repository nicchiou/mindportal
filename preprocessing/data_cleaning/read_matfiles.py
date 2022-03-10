import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from utils import constants

###############################################################################
#
# Parses .mat files containing the band-filtered data into one DataFrame for
# each frequency band.
#
# Author: Nicole Chiou
#
###############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(constants.PYTHON_DIR, 'ac_dc_ph'))
    parser.add_argument('--anchor', type=str, default='pc',
                        choices=['pc', 'rs', 'rl'],
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--preprocessing_dir', type=str,
                        default='bandpass_only',
                        choices=['bandpass_only', 'rect_lowpass',
                                 'no_bandpass'])
    parser.add_argument('-l', '--input_dirs', nargs='+',
                        default=['pc00-04avg_rs', 'pc00-08avg_rs',
                                 'pc00-13avg_rs'],
                        help='directories to process from')
    parser.add_argument('--voxel_space', action='store_true',
                        help='specifies whether inputs are in channel-space '
                        'or voxel-space.')
    parser.add_argument('--n_montages', type=int, default=4,
                        help='number of montages to consider based on '
                        'grouped montages; options include 8 (a-h) or 4 '
                        '(grouped by trial)')
    parser.add_argument('--save_subject_ts', action='store_true',
                        help='dictates whether to save subject time series '
                        'data by time step.')
    parser.add_argument('--save_subject_dir', type=str,
                        help='output directory to save sequential data')

    args = parser.parse_args()

    baseline_dirs = args.input_dirs
    os.makedirs(os.path.join(
        args.output_dir,
        'voxel_space' if args.voxel_space else 'channel_space',
        args.anchor, args.preprocessing_dir), exist_ok=True)

    df = pd.DataFrame()
    montage_map = {
        'ab': 'A',
        'cd': 'B',
        'ef': 'C',
        'gh': 'D'
    }

    for d in baseline_dirs:

        df_freq = pd.DataFrame()

        # Parse directory name
        freq_band = d[2:7]

        for f in tqdm(os.listdir(os.path.join(
                constants.MATFILE_DIR,
                'voxel_space' if args.voxel_space else 'channel_space', d))):

            # Parse file name
            subject_id = f[3:6]
            if args.n_montages == 4:
                montage = f[6:8]
                if '.' in montage:
                    continue
            else:
                montage = f[6]

            # Read .mat file
            pc = sio.loadmat(os.path.join(
                constants.MATFILE_DIR,
                'voxel_space' if args.voxel_space else 'channel_space', d, f))
            if args.n_montages == 4:
                # Map to paired montage naming convention
                montage = montage_map[montage]

            # Parse header dictionary
            boxy_hdr = {k: v for (k, v) in zip(
                list(pc['boxy_hdr'].dtype.fields),
                [v[0][0] for v in pc['boxy_hdr'][0][0]])
                }
            fs = boxy_hdr['sample_rate']

            # Parse events and trial types for each of the trials
            # Shape is (num_trials, 2), type: pd.DataFrame
            events = pd.DataFrame(pc['trial_data'][0][0][0],
                                  columns=['event', 'trial_type'])

            if args.voxel_space:
                # Parse recording voxel-space data
                # Shape is (6, 7, 56, num_trials, 2), type: np.array
                dc_data = pc['voxel_avg'][0][0][0]
                ac_data = pc['voxel_avg'][0][0][1]
                ph_data = pc['voxel_avg'][0][0][2]

                # Rotate data matrix to align with diagram: y-coordinate
                # specifies row and x-coordinate specifies column
                dc_data = np.rot90(dc_data, axes=(0, 1))
                ac_data = np.rot90(ac_data, axes=(0, 1))
                ph_data = np.rot90(ph_data, axes=(0, 1))

                all_trials = pd.DataFrame()
                all_ph_signals = pd.DataFrame()

                for trial in range(events.shape[0]):

                    single_trial = pd.DataFrame()

                    # Isolate single-trial data and flip up/down so we can
                    # index starting from the bottom-left as opposed to the
                    # top-left
                    # Shape is (7, 6, 56, 2)
                    curr_dc_data = dc_data[::-1, :, :, trial, :]
                    curr_ac_data = ac_data[::-1, :, :, trial, :]
                    curr_ph_data = ph_data[::-1, :, :, trial, :]

                    # Transpose y (row) and x (col) dimensions so we can use a
                    # simple reshape operation to flatten 2D data matrix into
                    # 1D array of voxels
                    # Shape is (6, 7, 56, 2)
                    curr_dc_data = np.transpose(curr_dc_data, (1, 0, 2, 3))
                    curr_ac_data = np.transpose(curr_ac_data, (1, 0, 2, 3))
                    curr_ph_data = np.transpose(curr_ph_data, (1, 0, 2, 3))

                    # Flatten 2D voxel representation into 1D, taking voxels in
                    # the positive y-direction then positive x-direction
                    # Shape is (42, 56, 2)
                    dc_signal = curr_dc_data.reshape(
                        (-1, curr_dc_data.shape[2], curr_dc_data.shape[3]))
                    ac_signal = curr_ac_data.reshape(
                        (-1, curr_ac_data.shape[2], curr_ac_data.shape[3]))
                    ph_signal = curr_ph_data.reshape(
                        (-1, curr_ph_data.shape[2], curr_ph_data.shape[3]))

                    # Separate into left and right hemispheres
                    # Shape is (42, 56)
                    dc_L, dc_R = dc_signal[:, :, 0], dc_signal[:, :, 1]
                    ac_L, ac_R = ac_signal[:, :, 0], ac_signal[:, :, 1]
                    ph_L, ph_R = ph_signal[:, :, 0], ph_signal[:, :, 1]

                    full_signals = pd.DataFrame(
                        np.concatenate(
                            [dc_L, dc_R, ac_L, ac_R, ph_L, ph_R], axis=0).T,
                        columns=[f'dc_{montage}_L_{voxel}'
                                 for voxel in range(dc_L.shape[0])] +
                                [f'dc_{montage}_R_{voxel}'
                                 for voxel in range(dc_R.shape[0])] +
                                [f'ac_{montage}_L_{voxel}'
                                 for voxel in range(ac_L.shape[0])] +
                                [f'ac_{montage}_R_{voxel}'
                                 for voxel in range(ac_R.shape[0])] +
                                [f'ph_{montage}_L_{voxel}'
                                 for voxel in range(ph_L.shape[0])] +
                                [f'ph_{montage}_R_{voxel}'
                                 for voxel in range(ph_R.shape[0])]
                    )
                    if args.save_subject_ts:
                        ph_signals = pd.DataFrame(
                            np.concatenate([ph_L, ph_R], axis=0).T,
                            columns=[f'ph_{montage}_L_{voxel}'
                                     for voxel in range(ph_L.shape[0])] +
                                    [f'ph_{montage}_R_{voxel}'
                                     for voxel in range(ph_R.shape[0])]
                        )
                        ph_signals.loc[:, 'subject_id'] = subject_id
                        ph_signals.loc[:, 'trial_num'] = trial
                        ph_signals.loc[:, 'montage'] = montage
                        ph_signals.loc[:, 'freq_band'] = int(freq_band[-2:])
                        ph_signals.loc[:, 'event'] = events.loc[:, 'event']
                        ph_signals.loc[:, 'trial_type'] = \
                            events.loc[trial, 'trial_type']
                        all_ph_signals = all_ph_signals.append(
                            ph_signals, ignore_index=True)

                    # Collapse signals along time dimension
                    col_order = full_signals.columns
                    full_signals = full_signals.fillna(0)
                    full_signals = full_signals.stack().reset_index(
                        level=0, drop=True)
                    full_signals = full_signals.groupby(
                        full_signals.index).apply(list).to_frame().transpose()
                    full_signals = full_signals.loc[:, col_order]

                    single_trial = single_trial.append(
                        full_signals, sort=False, ignore_index=True)

                    all_trials = all_trials.append(
                        single_trial, sort=False, ignore_index=True)

                # Add information common across trials
                all_trials.loc[:, 'trial_num'] = np.arange(events.shape[0])
                all_trials.loc[:, 'subject_id'] = subject_id
                all_trials.loc[:, 'montage'] = montage
                all_trials.loc[:, 'freq_band'] = freq_band
                all_trials.loc[:, 'event'] = events.loc[:, 'event']
                all_trials.loc[:, 'trial_type'] = events.loc[:, 'trial_type']
                for key in boxy_hdr.keys():
                    if key != 'record':
                        all_trials.loc[:, key] = boxy_hdr[key]
                df = df.append(all_trials, ignore_index=True)
                df_freq = df_freq.append(all_trials, ignore_index=True)

                if args.save_subject_ts:
                    out_dir = os.path.join(
                        constants.SUBJECTS_DIR,
                        'voxel_space' if args.voxel_space else 'channel_space',
                        args.anchor,
                        args.preprocessing_dir, args.save_subject_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    suffix = int(freq_band[:2]) \
                        if args.preprocessing_dir == 'bandpass_only' \
                        else int(float(freq_band[-2:]))
                    all_ph_signals.to_parquet(os.path.join(
                        out_dir,
                        f'{subject_id}_{montage}_{suffix}.parquet'),
                        index=False)

            else:
                # Parse recording channel-space data
                # Shape is (156, 128, num_trials), type: np.array
                dc_data = pc['trial_data'][0][0][1]
                ac_data = pc['trial_data'][0][0][2]
                ph_data = pc['trial_data'][0][0][3]

                # Compute start and end of each window (in samples)
                win_start = [int(np.floor((float(x) / 1000) * fs))
                             for x in range(0, 4500, 500)]

                all_trials = pd.DataFrame()
                all_ph_signals = pd.DataFrame()

                for trial in range(events.shape[0]):

                    single_trial = pd.DataFrame()

                    full_signals = pd.DataFrame(
                        np.concatenate([
                            dc_data[:, np.arange(dc_data.shape[1]), trial],
                            ac_data[:, np.arange(ac_data.shape[1]), trial],
                            ph_data[:, np.arange(ph_data.shape[1]), trial]
                        ], axis=1),
                        columns=[f'dc_{montage}_{chan}'
                                 for chan in range(dc_data.shape[1])] +
                                [f'ac_{montage}_{chan}'
                                 for chan in range(ac_data.shape[1])] +
                                [f'ph_{montage}_{chan}'
                                 for chan in range(ph_data.shape[1])]
                    )
                    if args.save_subject_ts:
                        ph_signals = pd.DataFrame(
                            ph_data[:, np.arange(ph_data.shape[1]), trial],
                            columns=[f'ph_{montage}_{chan}'
                                     for chan in range(ph_data.shape[1])])
                        ph_signals.loc[:, 'subject_id'] = subject_id
                        ph_signals.loc[:, 'trial_num'] = trial
                        ph_signals.loc[:, 'montage'] = montage
                        ph_signals.loc[:, 'freq_band'] = int(freq_band[-2:])
                        ph_signals.loc[:, 'event'] = events.loc[:, 'event']
                        ph_signals.loc[:, 'trial_type'] = \
                            events.loc[trial, 'trial_type']
                        all_ph_signals = all_ph_signals.append(
                            ph_signals, ignore_index=True)

                    # Collapse signals along time dimension
                    full_signals = full_signals.stack().reset_index(
                        level=0, drop=True)
                    full_signals = full_signals.groupby(
                        full_signals.index).apply(list).to_frame().transpose()

                    single_trial = single_trial.append(
                        full_signals, ignore_index=True)

                    for i in range(len(win_start) - 1):

                        windowed_signals = pd.DataFrame(
                            np.concatenate([
                                dc_data[win_start[i]:win_start[i+1],
                                        np.arange(dc_data.shape[1]), trial],
                                ac_data[win_start[i]:win_start[i+1],
                                        np.arange(ac_data.shape[1]), trial],
                                ph_data[win_start[i]:win_start[i+1],
                                        np.arange(ph_data.shape[1]), trial]
                            ], axis=1),
                            columns=[f'dc_win{i}_{montage}_{chan}'
                                     for chan in range(dc_data.shape[1])] +
                                    [f'ac_win{i}_{montage}_{chan}'
                                     for chan in range(ac_data.shape[1])] +
                                    [f'ph_win{i}_{montage}_{chan}'
                                     for chan in range(ph_data.shape[1])]
                        )

                        # Collapse signals along time dimension
                        windowed_signals = windowed_signals.stack(
                            ).reset_index(level=0, drop=True)
                        windowed_signals = windowed_signals.groupby(
                            windowed_signals.index).apply(
                                list).to_frame().transpose()

                        single_trial = pd.concat(
                            [single_trial, windowed_signals], axis=1)

                    all_trials = all_trials.append(
                        single_trial, ignore_index=True)

                # Add information common across trials
                all_trials.loc[:, 'trial_num'] = np.arange(events.shape[0])
                all_trials.loc[:, 'subject_id'] = subject_id
                all_trials.loc[:, 'montage'] = montage
                all_trials.loc[:, 'freq_band'] = freq_band
                all_trials.loc[:, 'event'] = events.loc[:, 'event']
                all_trials.loc[:, 'trial_type'] = events.loc[:, 'trial_type']
                for key in boxy_hdr.keys():
                    if key != 'record':
                        all_trials.loc[:, key] = boxy_hdr[key]
                df = df.append(all_trials, ignore_index=True)
                df_freq = df_freq.append(all_trials, ignore_index=True)

                if args.save_subject_ts:
                    out_dir = os.path.join(
                        constants.SUBJECTS_DIR,
                        'voxel_space' if args.voxel_space else 'channel_space',
                        args.anchor,
                        args.preprocessing_dir, args.save_subject_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    suffix = freq_band[:2] \
                        if args.preprocessing_dir == 'bandpass_only' \
                        else freq_band[-2:]
                    all_ph_signals.to_parquet(os.path.join(
                        out_dir,
                        f'{subject_id}_{montage}_{int(suffix)}.parquet'),
                        index=False)

            df_freq.to_parquet(os.path.join(
                args.output_dir,
                'voxel_space' if args.voxel_space else 'channel_space',
                args.anchor, args.preprocessing_dir,
                f'{d}_all_single_trial.parquet'), index=False)
