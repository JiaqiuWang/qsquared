import pandas as pd
import numpy as np
import itertools
import re


def cumprod(df):
    """Assumes the values of `df` are returns and cumulates them by
    adding one and then performing a cumulative product.

    :param df: input DataFrame assumed to be time series or returns
      for groups *or* securities.
    :type df: pd.DataFrame

    :rtype: pd.DataFrame
    """
    return df.add(1).cumprod()


def cumplot(df, *args, **kwargs):
    """Assumes the values of `df` are returns and cumulates them by
    adding one and then performing a cumulative product.  Subsequently
    pass resulting DataFrame to pd.DataFrame.plot

    :param df:
    :param args: passed to pd.DataFrame.plot
    :type args: pd.DataFrame.plot

    :param kwargs: passed to pd.DataFrame.plot
    :type kwargs: pd.DataFrame.plot

    :rtype: matplotlib.AxesSubplot
    """
    return cumprod(df).plot(*args, **kwargs)


def cumprod_ibyk(returns, i, k):
    """groups rows starting at :math:`i^{th}` position by `k` at a time
    and calculates the cumulative return.

    :param returns: returns DataFrame to cumulate
    :type returns: pd.DataFrame

    :param i: starting position
    :type i: int

    :param k: number of rows per group
    :type k: int

    :rtype: pd.DataFrame
    """
    r = returns.iloc[i:].add(1)
    a = np.arange(len(r)) // k
    return r.groupby(a).cumprod()


def floated(weights, returns, normalize=True):
    """weights is an arbitrary length DataFrame with the same columns as
    the returns DataFrame.  we'll fill in the gaps in weights DataFrame by
    "floating" the previous weights forward assuming the asset or security's
    return from the returns DataFrame.

    :param weights: time series of weights for assets or securities
    :type weights: pd.DataFrame

    :param returns: time series of returns for assets or securities
    :type returns: pd.DataFrame

    :param normalize: flag to determine if we ensure each row sums to one
    :type normalize: bool

    :rtype: pd.DataFrame
    """
    weights = weights.reindex(returns.index)
    mask = weights.notnull()
    cum_returns = cumprod(returns).shift().fillna(1)
    break_fills = cum_returns[mask].ffill().fillna(1)
    float_returns = cum_returns.div(break_fills)
    weights = weights.ffill().mul(float_returns)
    if normalize:
        weights = weights.div(weights.sum(1), 0)
    return weights

