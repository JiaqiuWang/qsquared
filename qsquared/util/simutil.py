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

