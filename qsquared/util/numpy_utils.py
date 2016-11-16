import numpy as np


def fastcount(x):
    """junk algorith that scales at :math:`O(n^2)`

    Is supposed to cumulative count the observations
    of each unique type in x.

    :param x: list
    :type x: list

    :rtype: list
    """
    unq, inv = np.unique(x, return_inverse=1)
    m = np.arange(len(unq))[:, None] == inv
    return (m.cumsum(1) * m).sum(0)