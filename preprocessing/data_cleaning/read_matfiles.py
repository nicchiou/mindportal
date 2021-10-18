import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.io as sio

from utils import constants

###############################################################################
# 
# Parses .mat files containing the band-filtered data into one DataFrame for
# each frequency band.
#
# Author: Nicole Chiou
#  
############################################################################### 

output_dir = os.path.join(constants.PYTHON_DIR, 'ac_dc_ph')

baseline_dirs = [
    'pc00-04baseline',
    'pc00-08baseline',
    'pc00-13baseline'
]


if __name__ == '__main__':

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

            # Compute start and end of the response-time window (in samples)
            # This is 300-900 ms after the response stimulus
            rt_start = int(np.floor((1 + 300. / 1000) * fs))
            rt_end = int(np.ceil((1 + 900. / 1000) * fs))

            # Follow the same logic for other time windows
            stim_time = int(1 * fs)
            pre_rt_start = int(np.floor((1 + 100. / 1000) * fs))

            all_trials = pd.DataFrame()

            for trial in range(events.shape[0]):

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
                
                # RT signals
                rt_signals = pd.DataFrame(
                    np.concatenate([
                        dc_data[rt_start:rt_end, np.arange(dc_data.shape[1]), trial],
                        ac_data[rt_start:rt_end, np.arange(ac_data.shape[1]), trial],
                        ph_data[rt_start:rt_end, np.arange(ph_data.shape[1]), trial]
                    ], axis=1),
                    columns=
                    [f'dc-rt_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                    [f'ac-rt_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                    [f'ph-rt_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                    )
                # Collapse signals along time dimension
                rt_signals = rt_signals.stack().reset_index(level=0, drop=True)
                rt_signals = rt_signals.groupby(rt_signals.index).apply(list).to_frame().transpose()

                # Pre-stimulus signals
                pre_stim_signals = pd.DataFrame(
                    np.concatenate([
                        dc_data[:stim_time, np.arange(dc_data.shape[1]), trial],
                        ac_data[:stim_time, np.arange(ac_data.shape[1]), trial],
                        ph_data[:stim_time, np.arange(ph_data.shape[1]), trial]
                    ], axis=1),
                    columns=
                    [f'dc-pre-stim_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                    [f'ac-pre-stim_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                    [f'ph-pre-stim_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                    )
                # Collapse signals along time dimension
                pre_stim_signals = pre_stim_signals.stack().reset_index(level=0, drop=True)
                pre_stim_signals = pre_stim_signals.groupby(pre_stim_signals.index).apply(list).to_frame().transpose()

                # initial response signals
                init_signals = pd.DataFrame(
                    np.concatenate([
                        dc_data[stim_time:pre_rt_start, np.arange(dc_data.shape[1]), trial],
                        ac_data[stim_time:pre_rt_start, np.arange(ac_data.shape[1]), trial],
                        ph_data[stim_time:pre_rt_start, np.arange(ph_data.shape[1]), trial]
                    ], axis=1),
                    columns=
                    [f'dc-init_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                    [f'ac-init_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                    [f'ph-init_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                    )
                # Collapse signals along time dimension
                init_signals = init_signals.stack().reset_index(level=0, drop=True)
                init_signals = init_signals.groupby(init_signals.index).apply(list).to_frame().transpose()

                # pre-RT signals
                pre_rt_signals = pd.DataFrame(
                    np.concatenate([
                        dc_data[pre_rt_start:rt_start, np.arange(dc_data.shape[1]), trial],
                        ac_data[pre_rt_start:rt_start, np.arange(ac_data.shape[1]), trial],
                        ph_data[pre_rt_start:rt_start, np.arange(ph_data.shape[1]), trial]
                    ], axis=1),
                    columns=
                    [f'dc-pre-rt_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                    [f'ac-pre-rt_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                    [f'ph-pre-rt_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                    )
                # Collapse signals along time dimension
                pre_rt_signals = pre_rt_signals.stack().reset_index(level=0, drop=True)
                pre_rt_signals = pre_rt_signals.groupby(pre_rt_signals.index).apply(list).to_frame().transpose()

                # post-RT signals
                post_rt_signals = pd.DataFrame(
                    np.concatenate([
                        dc_data[rt_end:, np.arange(dc_data.shape[1]), trial],
                        ac_data[rt_end:, np.arange(ac_data.shape[1]), trial],
                        ph_data[rt_end:, np.arange(ph_data.shape[1]), trial]
                    ], axis=1),
                    columns=
                    [f'dc-post-rt_{montage}_{chan}' for chan in range(dc_data.shape[1])] +
                    [f'ac-post-rt_{montage}_{chan}' for chan in range(ac_data.shape[1])] + 
                    [f'ph-post-rt_{montage}_{chan}' for chan in range(ph_data.shape[1])]
                    )
                # Collapse signals along time dimension
                post_rt_signals = post_rt_signals.stack().reset_index(level=0, drop=True)
                post_rt_signals = post_rt_signals.groupby(post_rt_signals.index).apply(list).to_frame().transpose()

                all_trials = all_trials.append(pd.concat(
                    [full_signals,
                     rt_signals,
                     pre_stim_signals,
                     init_signals,
                     pre_rt_signals,
                     post_rt_signals], axis=1), ignore_index=True)
           
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
            os.path.join(output_dir, f'{d}_all_single_trial.parquet'),
            index=False)
