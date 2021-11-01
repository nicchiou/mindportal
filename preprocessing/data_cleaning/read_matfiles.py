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

baseline_dirs = [
    'pc00-04avg',
    'pc00-08avg',
    'pc00-13avg'
]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(constants.PYTHON_DIR, 'ac_dc_ph'))
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
   
    args = parser.parse_args()

    baseline_dirs = [f'{d}_{args.anchor}' for d in baseline_dirs]

    df = pd.DataFrame()

    for d in baseline_dirs:

        df_freq = pd.DataFrame()
        
        # Parse directory name
        freq_band = d[2:7]

        for f in tqdm(os.listdir(os.path.join(constants.MATFILE_DIR, d))):

            # Parse file name
            subject_id = f[3:6]
            montage = f[6]
            
            # Read .mat file
            pc = sio.loadmat(os.path.join(constants.MATFILE_DIR, d, f))

            # Parse header dictionary
            boxy_hdr = {k: v for (k, v) in zip(
                list(pc['boxy_hdr'].dtype.fields),
                [v[0][0] for v in pc['boxy_hdr'][0][0]])
                }
            fs = boxy_hdr['sample_rate']
            
            # Parse events and trial types for each of the 460 trials
            # Shape is (460, 2), type: pd.DataFrame
            events = pd.DataFrame(pc['trial_data'][0][0][0],
                                  columns=['event', 'trial_type'])

            # Parse recording data
            # Shape is (156, 128, 460), type: np.array
            dc_data = pc['trial_data'][0][0][1]
            ac_data = pc['trial_data'][0][0][2]
            ph_data = pc['trial_data'][0][0][3]

            # Compute start and end of each window (in samples)
            win_start = [int(np.floor((float(x) / 1000) * fs)) for x in range(0, 4500, 500)]

            all_trials = pd.DataFrame()

            for trial in range(events.shape[0]):

                single_trial = pd.DataFrame()

                full_signals = pd.DataFrame(
                    np.concatenate([
                        dc_data[:, np.arange(dc_data.shape[1]), trial],
                        ac_data[:, np.arange(ac_data.shape[1]), trial],
                        ph_data[:, np.arange(ph_data.shape[1]), trial]
                    ], axis=1),
                    columns=
                    [f'dc_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                    [f'ac_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                    [f'ph_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                )
                # Collapse signals along time dimension
                full_signals = full_signals.stack().reset_index(level=0, drop=True)
                full_signals = full_signals.groupby(full_signals.index).apply(list).to_frame().transpose()

                single_trial = single_trial.append(full_signals, ignore_index=True)
                
                for i in range(len(win_start) - 1):

                    windowed_signals = pd.DataFrame(
                        np.concatenate([
                            dc_data[win_start[i]:win_start[i+1], np.arange(dc_data.shape[1]), trial],
                            ac_data[win_start[i]:win_start[i+1], np.arange(ac_data.shape[1]), trial],
                            ph_data[win_start[i]:win_start[i+1], np.arange(ph_data.shape[1]), trial]
                        ], axis=1),
                        columns=
                        [f'dc_win{i}_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                        [f'ac_win{i}_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                        [f'ph_win{i}_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                    )

                    # Collapse signals along time dimension
                    windowed_signals = windowed_signals.stack().reset_index(level=0, drop=True)
                    windowed_signals = windowed_signals.groupby(windowed_signals.index).apply(list).to_frame().transpose()

                    single_trial = pd.concat([single_trial, windowed_signals], axis=1)

                all_trials = all_trials.append(single_trial, ignore_index=True)
           
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

        df_freq.to_parquet(
            os.path.join(args.output_dir, args.anchor, f'{d}_all_single_trial.parquet'),
            index=False)
