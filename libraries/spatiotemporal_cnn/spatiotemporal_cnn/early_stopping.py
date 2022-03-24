import numpy as np


def simple_exp_smoothing(x, gamma=0.9, c=5):
    """ Performs clipped exponential smoothing on the validation metric """
    try:
        last_epochs = x[-c:]
        beta = 0.0
        w = 1
        for i in range(c):
            beta = beta * gamma + last_epochs[i]
            w = w * gamma + 1
        return beta / w
    except IndexError:
        return np.nan
