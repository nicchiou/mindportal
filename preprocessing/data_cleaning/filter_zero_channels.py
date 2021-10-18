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

root_dir = os.path.join(constants.PYTHON_DIR, 'ac_dc_ph')

baseline_dirs = [
    'pc00-04baseline',
    'pc00-08baseline',
    'pc00-13baseline'
]


if __name__ == '__main__':

    # Filter zero channels from "baseline all single trial" files
    for d in baseline_dirs:
        pc = pd.read_parquet(os.path.join(root_dir,
                                          f'{d}_all_single_trial.parquet'))
        for i in tqdm(range(pc.shape[0])):
            for c in pc.columns:
                if isinstance(pc.loc[i, c], np.ndarray) and \
                        np.count_nonzero(pc.loc[i, c]) == 0:
                    pc.loc[i, c] = np.nan
        pc.to_parquet(os.path.join(root_dir,
                                   f'{d}_filt_chan.parquet'), index=False)

    # Aggregate over all frequencies' filtered channel data
    for d in baseline_dirs:
        if '00-04' in d:
            pc04 = pd.read_parquet(os.path.join(root_dir,
                                                f'{d}_filt_chan.parquet'))
        elif '00-08' in d:
            pc08 = pd.read_parquet(os.path.join(root_dir,
                                                f'{d}_filt_chan.parquet'))
        elif '00-13' in d:
            pc13 = pd.read_parquet(os.path.join(root_dir,
                                                f'{d}_filt_chan.parquet'))

    pc04.columns = [f'{c}_04' for c in pc04.columns]
    pc08.columns = [f'{c}_08' for c in pc08.columns]
    pc13.columns = [f'{c}_13' for c in pc13.columns]
    all_data = pd.concat([pc04, pc08, pc13], axis=1)

    all_data.to_parquet(os.path.join(root_dir,
                                     'all_filt_chan.parquet'), index=False)
