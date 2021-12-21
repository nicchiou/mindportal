import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = '/shared/rsaas/nschiou2/'

DATA_DIR = os.path.join(SHARED_DIR, 'FOS')

MATFILE_DIR = os.path.join(DATA_DIR, 'matfiles')
PYTHON_DIR = os.path.join(DATA_DIR, 'python')

ALL_SIGNAL_DIR = os.path.join(PYTHON_DIR, 'ac_dc_ph')
PHASE_DATA_DIR = os.path.join(PYTHON_DIR, 'phase_data')
BANDPOWER_DIR = os.path.join(PYTHON_DIR, 'bandpower_features')
CSP_DIR = os.path.join(PYTHON_DIR, 'csp_transform')
SUBJECTS_DIR = os.path.join(PHASE_DATA_DIR, 'subject_time_series')

RESULTS_DIR = os.path.join(SHARED_DIR, 'mindportal')

SUBJECT_IDS = ['127', '130', '146', '149', '150', '151', '152', '153', '154',
               '155', '157', '505', '516', '527', '534']
SUBSET_SUBJECT_IDS = ['154', '534']
MONTAGES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
PAIRED_MONTAGES = ['A', 'B', 'C', 'D']
