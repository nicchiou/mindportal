import numpy as np


def maxabs_scale(x):
    """ Scales 2D data (dim_features, dim_time) in the range [-1, 1] """
    x_max, x_min = np.nanmax(x), np.nanmin(x)
    x = (x - x_min) / (x_max - x_min)  # [ 0, 1]
    x = 2 * x - 1                      # [-1, 1]
    return x
