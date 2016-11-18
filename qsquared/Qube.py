import pandas as pd
import numpy as np

from .util import simutil
from .util import util


class Qube(object):
    def __init__(self, signals, returns):
        """Qube main container for all relevant data to conduct quantile
        analysis.

        :param signals: This is a pd.DataFrame with a pd.DatetimeIndex for
          the index and identifiers in the column index.

          The column identifiers are expected to align with those in the
          returns parameter

        :type signal: pd.DataFrame

        :param returns: This is a pd.DataFrame with a pd.DatetimeIndex for
          the index and and identifiers in the column index.

          The column identifiers are expected to align with those in the
          signals parameter.

        :type returns: pd.DataFrame

        :rtype: Qube
        """
        self.signals = signals
        self.returns = returns

        self.__cuts = {}
        self.__pnls = {}

    def quintile(self):
        """convenience method to return self.qcut(n=5)

        :rtype: pd.DataFrame
        """
        return self.qcut(5)

    def decile(self):
        """convenience method to return self.qcut(n=10)

        :rtype: pd.DataFrame
        """
        return self.qcut(10)

    def qcut(self, n=5):
        """divide each row of signals into n quanitles and return a
        DataFrame with each signal replace by its quantile assignment.

        :param n: number of quantiles to divide into
        :type n: int

        :rtype: pd.DataFrame
        """
        if n not in self.__cuts:
            labels = ['q{}'.format(i) for i in range(1, n + 1)]
            self.__cuts[n] = self.signals.apply(
                pd.qcut, q=n, labels=labels, axis=1)

        return self.__cuts[n]

    def greturns(self, n=5):
        """Group returns by results of self.qcut(n=n) and aggregate them
        with mean.  Will return a DataFrame with the same number of rows
        as returns and one column for each quantile.  This gives the
        mean return per quantile for each date.

        :param n: number of quantiles to use in self.qcut
        :type n: int

        :rtype: pd.DataFrame
        """
        qs = self.qcut(n).stack()
        rs = self.returns.stack()
        ds = rs.index.get_level_values(0)
        gr = rs.groupby([ds, qs]).mean().unstack()
        gr.columns = simutil.qcats(n, values=gr.columns)
        return gr.sort_index(1)

    def by_quintile(self):
        """convenience method to return self.greturns(n=5)

        :rtype: pd.DataFrame
        """
        return self.greturns(n=5)

    def by_decile(self):
        """convenience method to return self.greturns(n=10)

        :rtype: pd.DataFrame
        """
        return self.greturns(n=10)

    def by_rank(self):
        """ranks the rows of the signal DataFrame and constructs
        a new DataFrame with ranks as columns and values are the
        corresponding returns from the returns DataFrame for the
        security with that rank for that period.

        :rtype: pd.DataFrame
        """
        ks = self.signals.rank(axis=1, method='first').stack().astype(int)
        rs = self.returns.stack()
        ds = rs.index.get_level_values(0)
        return rs.groupby([ds, ks]).mean().unstack()

    def holdings(self, n, i, k):
        """Produce the equal weights for all `n` buckets for every `k` period
        starting with the :math:`i^{th}` period.  Final holdings DataFrame
        is restricted to this subset of rows and is expected to be reindexed
        later.

        :param n: number of quantiles or groups
        :type n: int

        :param i: which of the k starting points are we beginning the
          rebalances.  ranges from 0 to k - 1
        :type i: int

        :param k: how many periods are we holding.  if we have monthly returns
          and we want to rebalance every 6 months, we use `k = 6`
        :type k: int

        :rtype: pd.DataFrame
        """
        r = self.greturns(n)
        h = r.iloc[i::k]
        h.values.fill(1 / n)
        return h

    def floated(self, n, i, k):
        """
        * generate `n` quantiles by signal
        * group returns by quantiles
        * assume rebalancing starts on :math:`i^{th}` period for `k` periods
          at a time
        * float holdings over periods with no rebalancing

        :param n: number of quantiles (5 for quintiles, 10 for deciles)
        :type n: int

        :param i: starting period
        :type i: int

        :param k: periods between rebalances (horizon)
        :type k: int

        :rtype: pd.DataFrame
        """
        r = self.greturns(n)
        h = self.holdings(n, i, k)
        return util.floated(h, r, normalize=True)

    def floated_returns(self, n, i, k):
        """float holdings then calculate the time series of returns

        :param n: number of quantiles (5 for quintiles, 10 for deciles)
        :type n: int

        :param i: starting period
        :type i: int

        :param k: periods between rebalances (horizon)
        :type k: int

        :rtype: pd.DataFrame
        """
        r = self.greturns(n)
        h = self.floated(n, i, k)
        return r.mul(h)

    def floated_panel_returns(self, n, k):
        """get floated returns for each `i` from `0` to `k-1` and return
        all `k` DataFrames in a Panel

        :param n: number of quantiles (5 for quintiles, 10 for deciles)
        :type n: int

        :param k: periods between rebalances (horizon)
        :type k: int

        :rtype: pd.Panel
        """
        key = (n, k) = int(n), int(k)
        if key not in self.__pnls:
            rng = range(k)
            pnl = pd.Panel({i: self.floated_returns(n, i, k) for i in rng})
            self.__pnls[key] = pnl
        return self.__pnls[key]

    def horizon_returns(self, n, k):
        """generate the floated panel of returns and take the mean return
        across the `k` iterations of the `k` period horizon.

        :param n: number of quantiles (5 for quintiles, 10 for deciles)
        :type n: int

        :param k: periods between rebalances (horizon)
        :type k: int

        :rtype: pd.DataFrame
        """
        pnl = self.floated_panel_returns(n, k)
        return pnl.mean('items', skipna=False).dropna()



