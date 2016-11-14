import numpy as np


def fastcount(x):
    unq, inv = np.unique(x, return_inverse=1)
    m = np.arange(len(unq))[:, None] == inv
    return (m.cumsum(1) * m).sum(0)