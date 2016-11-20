import pandas as pd
import numpy as np


def estimate_frequency(tidx):
    """calculate the esitmated numeric frequency of time series

    :param tidx: series of dates
    :type tidx: DatetimeIndex

    :rtype: float

    Examples

    >>> import pandas as pd
    >>> import numpy as np
    >>> import qsquared.util.simutil as simutil
    >>> tidx = pd.date_range('2008-01-01', '2019-12-31', freq='B')
    >>> simutil.estimate_frequency(tidx)
    261.0

    """
    tidx = pd.to_datetime(tidx)
    first = tidx.min()
    last = tidx.max()
    days_in_range = (last - first).days
    num_periods = tidx.shape[0] - 1
    days_in_year = 365.25
    return round(days_in_year * num_periods / days_in_range, 0)


def random_returns(securities, tidx, annual_return=.04, annual_volatility=.12):
    """Simulate log normal returns.
    First row will be all zeros..

    Assumptions

    - Annual expected return of `4%`
    - Annual volatility of `12%`
    - Returns are log normal

    .. math:: \\ln{(1 + r)} \\sim N(\\mu, \\sigma)

    With:

    .. math:: E(annual\_return) = 4\%
    .. math:: f = annual\_frequency

    we can estimate :math:`\\mu`

    .. math:: \\mu = \\frac{\\ln{(1 + E(annual\_return))}}{f}

    and estimate :math:`\\sigma`

    .. math:: \\sigma = \\frac{12\%}{\\sqrt{f}}

    :param securities: list of security Id's
    :type securities: list

    :param tidx: series of dates
    :type tidx: DatetimeIndex

    :param annual_return: expected annual return
    :type annual_return: float

    :param annual_volatility: expected annual volatility
    :type annual_volatility: float

    :rtype: DataFrame

    Examples

    >>> import numpy as np
    >>> import qsquared.util.simutil as simutil
    >>> np.random.seed([3,1415])
    >>> securities = list('AB')
    >>> tidx = ['2011-12-31', '2012-06-30',
    >>>         '2012-12-31', '2013-06-30',
    >>>         '2013-12-31', '2014-06-30']
    >>> simutil.random_returns(list('AB'), tidx)
    Id                 A         B
    2011-12-31  0.000000  0.000000
    2012-06-30 -0.148795 -0.084260
    2012-12-31 -0.137217 -0.158086
    2013-06-30 -0.009977  0.017474
    2013-12-31  0.047539  0.050436
    2014-06-30  0.083624  0.088730

    """
    tidx = pd.to_datetime(tidx)
    securities = pd.Index(securities, name='Id')
    freq = estimate_frequency(tidx)
    mu = np.log(1 + annual_return) / freq
    sigma = annual_volatility / np.sqrt(freq)
    a = np.random.lognormal(mean=mu, sigma=sigma,
                            size=(tidx.shape[0], securities.shape[0]))
    return pd.DataFrame(a - 1, tidx, securities, dtype=np.float64)


def qcats(n, values=None, prefix='q', suffix='', naught=1):
    """Produce pandas CategoricalIndex of length n

    * each category will start with prefix and end with suffix
    * the first category starts with naught (initial index)
    * categories will be ordered and sorted by range(naught, n + naught)


    :param n: number of categories
    :type n: int

    :param values: list of values to use in the construction of
      categorical index.  If values is None, will default to producing
      n unique values
    :type values: list

    :param prefix: what to put in front, defaults to 'q'
    :type prefix: str

    :param suffix: what to put in back, defaults to '' (empty string)
    :type suffix: str

    :param naught: where to start numeric indexing, defaults to 1
    :type naught: int

    :rtype: pd.CategoricalIndex
    """
    f = '{}{}{}'.format
    g = range(naught, n + naught)
    categories = [f(prefix, i, suffix) for i in g]
    if values is None:
        values = categories
    return pd.Categorical(values, categories, True)


def annualized_returns(returns):
    """calculate annualized returns.

    :param returns: time series of returns
    :type returns: pd.DataFrame or pd.Series

    :rtype: pd.Series
    """
    tidx = returns.index
    freq = estimate_frequency(tidx)
    num = len(tidx)
    return returns.add(1).prod() ** (freq / num) - 1


def annualized_risk(returns):
    """calculate annualized risk.

    :param returns: time series of returns
    :type returns: pd.DataFrame or pd.Series

    :rtype: pd.Series
    """
    tidx = returns.index
    freq = estimate_frequency(tidx)
    return returns.std().mul(np.sqrt(freq))


def sharpe_ratio(returns):
    """calculate sharpe ratio.
    this is a simplistic approach where we take the annualized return
    divided by the annualized risk.

    :param returns: time series of returns
    :type returns: pd.DataFrame or pd.Series

    :rtype: pd.Series
    """
    return annualized_returns(returns) / annualized_risk(returns)


def annualized_active_returns(returns, benchmarks):
    """get difference in annualized returns of `returns` and `benchmarks`

    :param returns: time series of returns
    :type returns: pd.DataFrame

    :param benchmarks: time series of returns
    :type benchmarks: pd.DataFrame

    :rtype: pd.Series
    """
    ann_returns = annualized_returns(returns)
    bm_returns = annualized_returns(benchmarks)
    return ann_returns.sub(bm_returns)


def annualized_active_risk(returns, benchmarks):
    """get annualized standard deviation of active returns or `returns`
    relative to `benchmarks`

    :param returns: time series of returns
    :type returns: pd.DataFrame

    :param benchmarks: time series of returns
    :type benchmarks: pd.DataFrame

    :rtype: pd.Series
    """
    axis = 0 if isinstance(benchmarks, pd.Series) else 1
    return annualized_risk(returns.sub(benchmarks, axis=axis))


def max_draw_down_absolute(returns):
    """calculate the maximum draw down of a time series of returns.
    also include the beginning and end of the draw down period.

    :param returns: time series of returns
    :type returns: pd.DataFrame

    :rtype: pd.DataFrame
    """
    if returns.values.ndim == 1:
        returns = returns.to_frame()
    r = (returns.values + 1).cumprod(0)
    dd = (r / np.maximum.accumulate(r)) - 1
    mdd = dd.min(0)
    end = dd.argmin(0)
    rng = np.arange(len(dd))
    idx = returns.index.values
    msk = end >= rng[:, None]
    beg = np.nanargmax(np.where(msk, r, np.nan), 0)
    return pd.DataFrame(
        dict(
            MaxDrawDown=mdd,
            Begin=idx[beg],
            End=idx[end]
        ),
        returns.columns)


def max_draw_down_relative(returns, benchmarks):
    """find the magnitude of the largest decline in active
    return as well as when it began and ended.

    The current draw down is the return from the most recent high point to
    the current position.  We start by identifying where the most recent high
    point was for each point in time.  That gets tracked with
    `argcummax_active`.

    We'll use `argcummax_active` to find out what the cumulative returns
    and cumulative benchmarks were at those high points.  We'll call these
    `p0` and `b0`.

    For any two points in time, the active return is define as

    .. math::  \\frac{p}{p_0} - \\frac{b}{b_0}

    This converts to

    .. math:: \\frac{p \cdot b_0 - b \cdot p_0}{p_0 \cdot b_0}

    We use this to calculate the current draw downs and subsequently identify
    the maximum draw down.

    :param returns: absolute returns
    :type returns: pd.DataFrame

    :param benchmarks: used to compare against returns
    :type benchmarks: pd.DataFrame

    :rtype: pd.DataFrame
    """
    # re-assign variables for convenience
    # `p` is for portfolio and `b` is for benchmark
    p, b = returns.values, benchmarks.values
    # add 1 and accumulate to get cumulative return
    # use reshape to ensure both `p` and `b` are 2 dimensional
    p = (p + 1).cumprod(0).reshape(len(p), -1)
    b = (b + 1).cumprod(0).reshape(len(b), -1)

    # if `p` and `b` both have one column then `cum_active`
    # will be one column of active returns. if
    # `p` is multi-column and `b` is one column, the
    # subtraction will be broadcast across all columns of `p`.
    cum_active = p - b

    # keep track of the position where the cumulative maximum
    # has occured.  We'll use it to slice `p` and `b`
    argcummax_active = cumargmax(cum_active)

    # arrays of ordinal posistions used for slicing
    p_col_rng = np.arange(p.shape[1])
    b_col_rng = np.arange(b.shape[1])
    row_rng = np.arange(p.shape[0])

    # when a maximum occurs in `cum_active` we need to capture
    # what `p` and `b` were.  We keep track of that
    # with `p0` and `b0`.  The track the max because the max
    # draw down must begin with what ever the max was prior
    p0 = p[argcummax_active, p_col_rng]
    b0 = b[argcummax_active, b_col_rng]
    dd = (p * b0 - b * p0) / (p0 * b0)

    mdd = dd.min(0)
    end = dd.argmin(0)
    beg = argcummax_active[end, p_col_rng]

    return pd.DataFrame(
        dict(
            Begin=returns.index.values[beg],
            End=returns.index.values[end],
            MaxDD=mdd
        ), returns.columns)


def max_draw_down(returns, benchmarks=None):
    """pass through function.  determine if we need absolute or active draw
    down.

    :param returns: absolute returns
    :type returns: pd.DataFrame

    :param benchmarks: used to compare against returns
    :type benchmarks: pd.DataFrame

    :rtype: pd.DataFrame
    """
    if benchmarks is None:
        return max_draw_down_absolute(returns)
    else:
        return max_draw_down_relative(returns, benchmarks)


def cumargmax(a, return_cummax=False):
    """takes a numpy array and returns the cumulative argmax per columns.

    credit @ajcr on stackoverflow.com
    http://stackoverflow.com/a/40675969/2336654

    :param a: array to calculate cumargmax on
    :type a: np.ndarray

    :param return_cummax: flag on whether to return the cummax array

    :rtype: np.ndarray
    """
    m = np.maximum.accumulate(a)
    x = np.repeat(
        np.arange(a.shape[0])[:, None],
        a.shape[1], axis=1)
    x[1:] *= m[:-1] < m[1:]
    np.maximum.accumulate(x, axis=0, out=x)
    if return_cummax:
        return x, m
    else:
        return x