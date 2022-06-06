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
                        default=os.path.join(constants.PYTHON_DIR,
                                             'ac_dc_ph', 'bci'))
    parser.add_argument('-l', '--input_dirs', nargs='+',
                        help='directories to process from')
    parser.add_argument('--input_space', type=str, default='channel_space',
                        choices=['channel_space', 'voxel_space'],
                        help='specifies whether inputs are in channel-space '
                        'or voxel-space.')
    parser.add_argument('--expt_type', type=str, default='mot',
                        choices=['mot', 'gam'])
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--save_subject_dir', type=str,
                        help='output directory to save sequential data')

    args = parser.parse_args()

    baseline_dirs = args.input_dirs
    os.makedirs(os.path.join(args.output_dir, args.input_space), exist_ok=True)

    for d in baseline_dirs:

        try:
            df_freq = pd.read_parquet(
                os.path.join(
                    args.output_dir, args.input_space,
                    f'{d}_{args.expt_type}_all_single_trial.parquet'))
        except OSError:
            df_freq = pd.DataFrame()

        files = os.listdir(os.path.join(
                           constants.MATFILE_DIR, 'bci', args.input_space, d))
        files = [f for f in files if f.startswith(args.expt_type)]
        files = [f for f in files if int(f[3:7]) in args.subjects]

        # Parse directory name
        freq_band = d[2:7]

        for f in tqdm(files, leave=False):

            # Parse file name
            exp_name = f[:3]
            subject_id = f[3:7]
            montage = f.split('.')[0][7:]
            assert montage in constants.SUBMONTAGES or montage == 'abc'

            # Read .mat file
            pc = sio.loadmat(os.path.join(
                constants.MATFILE_DIR, 'bci', args.input_space, d, f))

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

            if args.input_space == 'channel_space':
                # Parse recording channel-space data
                # Shape is (T, C, num_trials), type: np.array
                dc_data = pc['trial_data'][0][0][1]
                ac_data = pc['trial_data'][0][0][2]
                ph_data = pc['trial_data'][0][0][3]

                # Keep track of signals over all trials to save to a file
                all_trials = pd.DataFrame()
                all_dc_signals = pd.DataFrame()
                all_ac_signals = pd.DataFrame()
                all_ph_signals = pd.DataFrame()

                # Iterate over trials to store relevant signals
                for trial in range(events.shape[0]):

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
                    # Save as a sequence for input into neural methods
                    dc_signals = pd.DataFrame(
                        dc_data[:, np.arange(dc_data.shape[1]), trial],
                        columns=[f'dc_{montage}_{chan}'
                                 for chan in range(dc_data.shape[1])])
                    dc_signals.loc[:, 'subject_id'] = subject_id
                    dc_signals.loc[:, 'trial_num'] = trial
                    dc_signals.loc[:, 'montage'] = montage
                    dc_signals.loc[:, 'freq_band'] = int(freq_band[-2:])
                    dc_signals.loc[:, 'event'] = events.loc[:, 'event']
                    dc_signals.loc[:, 'trial_type'] = \
                        events.loc[trial, 'trial_type']
                    all_dc_signals = all_dc_signals.append(
                        dc_signals, ignore_index=True)

                    ac_signals = pd.DataFrame(
                        ac_data[:, np.arange(ac_data.shape[1]), trial],
                        columns=[f'ac_{montage}_{chan}'
                                 for chan in range(ac_data.shape[1])])
                    ac_signals.loc[:, 'subject_id'] = subject_id
                    ac_signals.loc[:, 'trial_num'] = trial
                    ac_signals.loc[:, 'montage'] = montage
                    ac_signals.loc[:, 'freq_band'] = int(freq_band[-2:])
                    ac_signals.loc[:, 'event'] = events.loc[:, 'event']
                    ac_signals.loc[:, 'trial_type'] = \
                        events.loc[trial, 'trial_type']
                    all_ac_signals = all_ac_signals.append(
                        ac_signals, ignore_index=True)

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

                    all_trials = all_trials.append(
                        full_signals, ignore_index=True)

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
                if len(df_freq) > 0 and \
                        int(subject_id) in df_freq.subject_id.unique():
                    print(f'Subject {subject_id} already pre-processed.')
                else:
                    print(f'Adding subject {subject_id}...')
                    df_freq = df_freq.append(all_trials, ignore_index=True)

            elif args.input_space == 'voxel_space':
                # Enforce different pre-processing steps for individual sub-
                # montage data v.s. averaged data
                voxel_key = 'voxel_avg' if montage == 'abc' else 'voxel'
                # Parse recording voxel-space data
                # Shape is (w, h, T, num_trials, 2), type: np.array
                dc_data = pc[voxel_key][0][0][0]
                ac_data = pc[voxel_key][0][0][1]
                ph_data = pc[voxel_key][0][0][2]

                # Rotate data matrix to align with diagram: y-coordinate
                # specifies row and x-coordinate specifies column
                dc_data = np.rot90(dc_data, axes=(0, 1))
                ac_data = np.rot90(ac_data, axes=(0, 1))
                ph_data = np.rot90(ph_data, axes=(0, 1))

                # Keep track of signals over all trials to save to a file
                all_trials = pd.DataFrame()
                all_dc_signals = pd.DataFrame()
                all_ac_signals = pd.DataFrame()
                all_ph_signals = pd.DataFrame()

                # Iterate over trials to store relevant signals
                for trial in range(events.shape[0]):

                    single_trial = pd.DataFrame()

                    # Isolate single-trial data and flip up/down so we can
                    # index starting from the bottom-left as opposed to the
                    # top-left
                    # Shape is (w, h, T, 2)
                    if montage == 'abc':
                        curr_dc_data = dc_data[::-1, :, :, trial, :]
                        curr_ac_data = ac_data[::-1, :, :, trial, :]
                        curr_ph_data = ph_data[::-1, :, :, trial, :]
                    else:
                        curr_dc_data = dc_data[::-1, :, :, trial, :, 0]
                        curr_ac_data = ac_data[::-1, :, :, trial, :, 0]
                        curr_ph_data = ph_data[::-1, :, :, trial, :, 0]

                    # Transpose y (row) and x (col) dimensions so we can use a
                    # simple reshape operation to flatten 2D data matrix into
                    # 1D array of voxels
                    # Shape is (h, w, T, 2)
                    curr_dc_data = np.transpose(curr_dc_data, (1, 0, 2, 3))
                    curr_ac_data = np.transpose(curr_ac_data, (1, 0, 2, 3))
                    curr_ph_data = np.transpose(curr_ph_data, (1, 0, 2, 3))

                    # Flatten 2D voxel representation into 1D, taking voxels in
                    # the positive y-direction then positive x-direction
                    # Shape is (h * w, T, 2)
                    dc_signal = curr_dc_data.reshape(
                        (-1, curr_dc_data.shape[2], curr_dc_data.shape[3]))
                    ac_signal = curr_ac_data.reshape(
                        (-1, curr_ac_data.shape[2], curr_ac_data.shape[3]))
                    ph_signal = curr_ph_data.reshape(
                        (-1, curr_ph_data.shape[2], curr_ph_data.shape[3]))

                    # Separate into left and right hemispheres
                    # Shape is (h * w, T)
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
                    # Save as a sequence for input into neural methods
                    dc_signals = pd.DataFrame(
                        np.concatenate([dc_L, dc_R], axis=0).T,
                        columns=[f'dc_{montage}_L_{voxel}'
                                 for voxel in range(dc_L.shape[0])] +
                                [f'dc_{montage}_R_{voxel}'
                                 for voxel in range(dc_R.shape[0])]
                    )
                    dc_signals.loc[:, 'subject_id'] = subject_id
                    dc_signals.loc[:, 'trial_num'] = trial
                    dc_signals.loc[:, 'montage'] = montage
                    dc_signals.loc[:, 'freq_band'] = int(freq_band[-2:])
                    dc_signals.loc[:, 'event'] = events.loc[:, 'event']
                    dc_signals.loc[:, 'trial_type'] = \
                        events.loc[trial, 'trial_type']
                    all_dc_signals = all_dc_signals.append(
                        dc_signals, ignore_index=True)

                    ac_signals = pd.DataFrame(
                        np.concatenate([ac_L, ac_R], axis=0).T,
                        columns=[f'ac_{montage}_L_{voxel}'
                                 for voxel in range(ac_L.shape[0])] +
                                [f'ac_{montage}_R_{voxel}'
                                 for voxel in range(ac_R.shape[0])]
                    )
                    ac_signals.loc[:, 'subject_id'] = subject_id
                    ac_signals.loc[:, 'trial_num'] = trial
                    ac_signals.loc[:, 'montage'] = montage
                    ac_signals.loc[:, 'freq_band'] = int(freq_band[-2:])
                    ac_signals.loc[:, 'event'] = events.loc[:, 'event']
                    ac_signals.loc[:, 'trial_type'] = \
                        events.loc[trial, 'trial_type']
                    all_ac_signals = all_ac_signals.append(
                        ac_signals, ignore_index=True)

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
                if len(df_freq) > 0 and \
                        int(subject_id) in df_freq.subject_id.unique():
                    print(f'Subject {subject_id} already pre-processed.')
                else:
                    print(f'Adding subject {subject_id}...')
                    df_freq = df_freq.append(all_trials, ignore_index=True)

            # Save sequential data
            out_dir = os.path.join(
                constants.DC_SUBJECTS_DIR, 'bci', args.input_space,
                args.save_subject_dir)
            os.makedirs(out_dir, exist_ok=True)
            suffix = int(freq_band[:2])
            all_dc_signals.to_parquet(os.path.join(
                out_dir,
                f'{exp_name}_{subject_id}_{montage}_{suffix}.parquet'),
                index=False)

            out_dir = os.path.join(
                constants.AC_SUBJECTS_DIR, 'bci', args.input_space,
                args.save_subject_dir)
            os.makedirs(out_dir, exist_ok=True)
            suffix = int(freq_band[:2])
            all_ac_signals.to_parquet(os.path.join(
                out_dir,
                f'{exp_name}_{subject_id}_{montage}_{suffix}.parquet'),
                index=False)

            out_dir = os.path.join(
                constants.PH_SUBJECTS_DIR, 'bci', args.input_space,
                args.save_subject_dir)
            os.makedirs(out_dir, exist_ok=True)
            suffix = int(freq_band[:2])
            all_ph_signals.to_parquet(os.path.join(
                out_dir,
                f'{exp_name}_{subject_id}_{montage}_{suffix}.parquet'),
                index=False)

            df_freq.to_parquet(os.path.join(
                args.output_dir, args.input_space,
                f'{d}_{exp_name}_all_single_trial.parquet'), index=False)
