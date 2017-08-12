import numpy as np


def add_dummy_dim(v):
    assert len(v.shape) == 1
    return np.reshape(v, newshape=(v.shape[0], 1))
