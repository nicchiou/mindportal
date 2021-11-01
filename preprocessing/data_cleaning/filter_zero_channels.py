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
    parser.add_argument('-l', '--input_dirs', nargs='+',
                        default=['pc00-04avg', 'pc00-08avg', 'pc00-13avg'],
                        help='directories to process from')
    parser.add_argument('--filter_zeros', action='store_true')
    parser.add_argument('--concat_unfilt', action='store_true')
    parser.add_argument('--concat_filt', action='store_true')
   
    args = parser.parse_args()

    baseline_dirs = [f'{d}_{args.anchor}' for d in args.input_dirs]
    out_dir = os.path.join(args.output_dir, args.anchor)
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
            elif '00-08' in d:
                pc08 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))
            elif '00-13' in d:
                pc13 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_all_single_trial.parquet'))

        pc04.columns = [f'{c}_04' for c in pc04.columns]
        pc08.columns = [f'{c}_08' for c in pc08.columns]
        pc13.columns = [f'{c}_13' for c in pc13.columns]
        all_data = pd.concat([pc04, pc08, pc13], axis=1)

        all_data.to_parquet(
            os.path.join(out_dir, 'all_single_trial.parquet'), index=False)

    
    # Aggregate over all frequencies' filtered channel data
    if args.concat_filt:
        for d in baseline_dirs:
            if '00-04' in d:
                pc04 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '00-08' in d:
                pc08 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))
            elif '00-13' in d:
                pc13 = pd.read_parquet(
                    os.path.join(out_dir, f'{d}_filt_chan.parquet'))

        pc04.columns = [f'{c}_04' for c in pc04.columns]
        pc08.columns = [f'{c}_08' for c in pc08.columns]
        pc13.columns = [f'{c}_13' for c in pc13.columns]
        all_data = pd.concat([pc04, pc08, pc13], axis=1)

        all_data.to_parquet(
            os.path.join(out_dir, 'all_filt_chan.parquet'), index=False)
