import pandas as pd
import numpy as np
import itertools
import re


def cumprod(df):
    return df.add(1).cumprod()


def cumplot(df, *args, **kwargs):
    return cumprod(df).plot(*args, **kwargs)


def cumprod_ibyk(returns, i, k):
    r = returns.iloc[i:].add(1)
    a = np.arange(len(r)) // k
    return r.groupby(a).cumprod()


def cumprod_pnlk(returns, k):
    return pd.Panel({i: cumprod_ibyk(returns, i, k) for i in range(k)})


def floated(weights, returns, normalize=True):
    weights = weights.reindex(returns.index)
    mask = weights.notnull()
    cum_returns = cumprod(returns).shift().fillna(1)
    break_fills = cum_returns[mask].ffill().fillna(1)
    float_returns = cum_returns.div(break_fills)
    weights = weights.ffill().mul(float_returns)
    if normalize:
        weights = weights.div(weights.sum(1), 0)
    return weights

