import numpy as np


def combine_motor_LR(x):
    """
    Combines trial types for left motor responses and right motor responses.

    0 represents left and 1 represents right.
    """
    if x in [2, 4, 6, 8]:
        return 0
    elif x in [1, 3, 5, 7]:
        return 1
    else:
        return np.nan


def stim_modality_motor_LR_labels(x):
    """
    Combines trial types such that the problem is a multi-class classification
    problem with the goal to predict both the stimulus modality (visual or
    auditory) as well as the motor response (L/R).

    0: visual, right
    1: visual, left
    2: auditory, right
    3: auditory, left

    Returns np.nan for unsupported trial types.
    """
    if x in [1, 5]:
        return 0
    elif x in [2, 6]:
        return 1
    elif x in [3, 7]:
        return 2
    elif x in [4, 8]:
        return 3
    else:
        return np.nan


def response_stim_modality_labels(x):
    """
    Combines trial types such that the problem is a multi-class classification
    problem with the goal to predict the response modality (motor or vocal),
    the stimulus modality (visual or auditory) as well as the response
    polarity (L/R).

    0: motor, visual, right
    1: motor, visual, left
    2: motor, auditory, right
    3: motor, auditory, left
    4: vocal, visual, right
    5: vocal, visual, left
    6: vocal, auditory, right
    7: vocal, auditory, left

    Returns np.nan for unsupported trial types.
    """
    if x in [1, 5]:
        return 0
    elif x in [2, 6]:
        return 1
    elif x in [3, 7]:
        return 2
    elif x in [4, 8]:
        return 3
    elif x in [9, 13]:
        return 4
    elif x in [10, 14]:
        return 5
    elif x in [11, 15]:
        return 6
    elif x in [12, 16]:
        return 7
    else:
        return np.nan
