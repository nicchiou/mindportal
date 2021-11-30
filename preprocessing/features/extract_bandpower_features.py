import argparse
import os

import pandas as pd
from csp_lib.features import extract_features, extract_features_all
from tqdm import tqdm
from utils import constants

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str,
                        default=constants.BANDPOWER_DIR)
    parser.add_argument('--anchor', type=str, default='pc',
                        help='pre-cue (pc) or response stimulus (rs)')
    parser.add_argument('--csp', action='store_true',
                        help='indicates whether to use CSP filtering')
    parser.add_argument('--bandpass_only', action='store_true',
                        help='indicates whether to use the signal that has '
                        'not been rectified nor low-pass filtered')
    parser.add_argument('--unfilt', action='store_true',
                        help='indicates whether to use data without the '
                        'removal of zeroed channels')
    parser.add_argument('--classification_task', type=str,
                        default='motor_LR',
                        help='options include motor_LR (motor response), '
                        'stim_motor (stimulus modality and motor response) or '
                        'response_stim (response modality, stimulus modality, '
                        'and response polarity).')

    args = parser.parse_args()
    out_dir = os.path.join(
        args.output_dir, args.anchor,
        'bandpass_only' if args.bandpass_only else 'rect_lowpass')
    os.makedirs(out_dir, exist_ok=True)

    # Without CSP
    if not args.csp:
        data_dir = os.path.join(
            constants.PYTHON_DIR, 'phase_data', args.anchor,
            'bandpass_only' if args.bandpass_only else 'rect_lowpass')

        if args.unfilt:
            input_fname = 'phase_all_single_trial.parquet'
        else:
            input_fname = 'phase_all_filt_chan.parquet'
        output_fname = 'all_simple_bandpower_features.parquet'

        df = pd.read_parquet(os.path.join(data_dir, input_fname))

        df = extract_features_all(df, csp=False)

        print('Writing output file...')
        df.to_parquet(os.path.join(out_dir, output_fname), index=False)

        for i in tqdm(range(8)):
            if args.unfilt:
                input_fname = f'phase_win{i}_single_trial.parquet'
            else:
                input_fname = f'phase_win{i}_filt_chan.parquet'
            output_fname = f'win{i}_simple_bandpower_features.parquet'

            df = pd.read_parquet(os.path.join(data_dir, input_fname))

            df = extract_features(df, i, csp=False)

            df.to_parquet(os.path.join(out_dir, output_fname), index=False)
