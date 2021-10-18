import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = '/shared/rsaas/nschiou2/'

DATA_DIR = os.path.join(SHARED_DIR, 'EROS')

MATFILE_DIR = os.path.join(DATA_DIR, 'matfiles')
PYTHON_DIR = os.path.join(DATA_DIR, 'python')

ALL_SIGNAL_DIR = os.path.join(PYTHON_DIR, 'ac_dc_ph')
PHASE_DATA_DIR = os.path.join(PYTHON_DIR, 'phase_data')
BANDPOWER_DIR = os.path.join(PYTHON_DIR, 'bandpower_features')
CSP_DIR = os.path.join(PYTHON_DIR, 'csp_transform')