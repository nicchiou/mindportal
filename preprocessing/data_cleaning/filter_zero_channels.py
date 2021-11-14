import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import constants

###############################################################################
#
# Filters the channels with all zero values and replaces them with NaN in the
# DataFrame cell for each frequency band.
#
# Author: Nicole Chiou
#
###############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(constants.PYTHON_DIR, 'ac_dc_ph'))
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--bandpass_only', action='store_true')
    parser.add_argument('-l', '--input_dirs', nargs='+',
                        default=['pc00-04avg', 'pc00-08avg', 'pc00-13avg'],
                        help='directories to process from')
    parser.add_argument('--filter_zeros', action='store_true')
    parser.add_argument('--concat_unfilt', action='store_true')
    parser.add_argument('--concat_filt', action='store_true')
    parser.add_argument('--save_phase_data', action='store_true')

    args = parser.parse_args()

    baseline_dirs = [f'{d}_{args.anchor}' for d in args.input_dirs]
    out_dir = os.path.join(
        args.output_dir, args.anchor,
        'bandpass_only' if args.bandpass_only else 'rect_lowpass')
    os.makedirs(out_dir, exist_ok=True)

    # Filter zero channels from "pc00- all single trial" files
    if args.filter_zeros:
        for d in baseline_dirs:
            pc = pd.read_parquet(
                os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            for i in tqdm(range(pc.shape[0])):
                for c in pc.columns:
                    if isinstance(pc.loc[i, c], np.ndarray) and \
                            np.count_nonzero(pc.loc[i, c]) == 0:
                        pc.loc[i, c] = np.nan
            pc.to_parquet(
                os.path.join(out_dir, f'{d}_filt_chan.parquet'), index=False)

    # Aggregate over all frequencies' unfiltered channel data
    if args.concat_unfilt:
        for d in baseline_dirs:
            if '00-04' in d:
                pc04 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            elif '04-07' in d:
                pc04 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            elif '00-08' in d:
                pc08 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            elif '08-13' in d:
                pc08 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            elif '00-13' in d:
                pc13 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            elif '13-20' in d:
                pc13 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))

        pc04.columns = [f'{c}_04' for c in pc04.columns]
        pc08.columns = [f'{c}_08' for c in pc08.columns]
        pc13.columns = [f'{c}_13' for c in pc13.columns]
        all_data = pd.concat([pc04, pc08, pc13], axis=1)

        all_data.to_parquet(
            os.path.join(out_dir, 'all_single_trial.parquet'), index=False)

        # Filter phase features only
        if args.save_phase_data:
            data_dir = os.path.join(constants.PHASE_DATA_DIR, args.anchor)
            ph_data = all_data[
                [c for c in all_data.columns
                 if ('ac_' not in c) and ('dc_' not in c)]]
            ph_data.to_parquet(
                os.path.join(data_dir, 'phase_single_trial.parquet'),
                index=False)

            # Isolate 500 ms windows
            ph_data = ph_data.rename({'trial_num_04': 'trial_num',
                                      'subject_id_04': 'subject_id',
                                      'trial_type_04': 'trial_type',
                                      'montage_04': 'montage'}, axis=1)
            ph_data = ph_data.drop(['trial_num_08', 'subject_id_08',
                                    'trial_type_08', 'montage_08',
                                    'trial_num_13', 'subject_id_13',
                                    'trial_type_13', 'montage_13'], axis=1)

            # Utilize only the full signal
            ph_all = ph_data[
                [c for c in ph_data.columns
                 if len(c.split('_')) == 4 and 'ph_' in c]
                ['trial_num', 'subject_id', 'trial_type', 'montage']]
            ph_all.to_parquet(
                os.path.join(data_dir, 'phase_all_single_trial.parquet'),
                index=False)

            # Filter windows of 500 ms
            for i in tqdm(range(8)):
                ph_win = ph_data[
                    [c for c in ph_data.columns if f'win{i}' in c]
                    ['trial_num', 'subject_id', 'trial_type', 'montage']]
                ph_win.to_parquet(
                    os.path.join(data_dir,
                                 f'phase_win{i}_single_trial.parquet'),
                    index=False)

    # Aggregate over all frequencies' filtered channel data
    if args.concat_filt:
        for d in baseline_dirs:
            if '00-04' in d:
                pc04 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '04-07' in d:
                pc04 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '00-08' in d:
                pc08 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '08-13' in d:
                pc08 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '00-13' in d:
                pc13 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '13-20' in d:
                pc13 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))

        pc04.columns = [f'{c}_04' for c in pc04.columns]
        pc08.columns = [f'{c}_08' for c in pc08.columns]
        pc13.columns = [f'{c}_13' for c in pc13.columns]
        all_data = pd.concat([pc04, pc08, pc13], axis=1)

        all_data.to_parquet(
            os.path.join(out_dir, 'all_filt_chan.parquet'), index=False)

        # Filter phase features only
        if args.save_phase_data:
            data_dir = os.path.join(constants.PHASE_DATA_DIR, args.anchor)
            ph_data = all_data[
                [c for c in all_data.columns
                 if ('ac_' not in c) and ('dc_' not in c)]]
            ph_data.to_parquet(
                os.path.join(data_dir, 'phase_filt_chan.parquet'),
                index=False)

            # Isolate 500 ms windows
            ph_data = ph_data.rename({'trial_num_04': 'trial_num',
                                      'subject_id_04': 'subject_id',
                                      'trial_type_04': 'trial_type',
                                      'montage_04': 'montage'}, axis=1)
            ph_data = ph_data.drop(['trial_num_08', 'subject_id_08',
                                    'trial_type_08', 'montage_08',
                                    'trial_num_13', 'subject_id_13',
                                    'trial_type_13', 'montage_13'], axis=1)

            # Utilize only the full signal
            ph_all = ph_data[
                [c for c in ph_data.columns
                 if len(c.split('_')) == 4 and 'ph_' in c] +
                ['trial_num', 'subject_id', 'trial_type', 'montage']]
            ph_all.to_parquet(
                os.path.join(data_dir, 'phase_all_filt_chan.parquet'),
                index=False)

            # Filter windows of 500 ms
            for i in tqdm(range(8)):
                ph_win = ph_data[
                    [c for c in ph_data.columns if f'win{i}' in c] +
                    ['trial_num', 'subject_id', 'trial_type', 'montage']]
                ph_win.to_parquet(
                    os.path.join(data_dir, f'phase_win{i}_filt_chan.parquet'),
                    index=False)
