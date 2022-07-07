import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = '/shared/rsaas/nschiou2/'

DATA_DIR = os.path.join(SHARED_DIR, 'FOS')

MATFILE_DIR = os.path.join(DATA_DIR, 'matfiles')
PYTHON_DIR = os.path.join(DATA_DIR, 'python')

ALL_SIGNAL_DIR = os.path.join(PYTHON_DIR, 'ac_dc_ph')
DC_DATA_DIR = os.path.join(PYTHON_DIR, 'dc_data')
AC_DATA_DIR = os.path.join(PYTHON_DIR, 'ac_data')
PHASE_DATA_DIR = os.path.join(PYTHON_DIR, 'phase_data')
BANDPOWER_DIR = os.path.join(PYTHON_DIR, 'bandpower_features')
CSP_DIR = os.path.join(PYTHON_DIR, 'csp_transform')
DC_SUBJECTS_DIR = os.path.join(DC_DATA_DIR, 'subject_time_series')
AC_SUBJECTS_DIR = os.path.join(AC_DATA_DIR, 'subject_time_series')
PH_SUBJECTS_DIR = os.path.join(PHASE_DATA_DIR, 'subject_time_series')

RESULTS_DIR = os.path.join(SHARED_DIR, 'mindportal')
PSC_RESULTS_DIR = os.path.join(RESULTS_DIR, 'psc')
BCI_RESULTS_DIR = os.path.join(RESULTS_DIR, 'bci')

SUBJECT_IDS = ['127', '146', '149', '150', '151', '152', '153', '154', '155',
               '157', '505', '516']
SUBSET_SUBJECT_IDS = ['127', '146', '152', '154', '155', '505']
MONTAGES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
PAIRED_MONTAGES = ['A', 'B', 'C', 'D']

BCI_SUBJECTS_MOT = ['9552', '9555', '9564', '9566',
                    '2801', '2802', '2803', '2804', '2805',
                    '2806', '2807', '2808', '2809', '2810',
                    '3036']
BCI_SUBJECTS_GAM = ['9552', '9564', '9566',
                    '2801', '2802', '2803', '2804', '2805',
                    '2806', '2807', '2808', '2810',
                    '3036']
SUBMONTAGES = ['a', 'b', 'c']
