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
        ks = self.signals.rank(axis=1, method='first').stack().astype(int)
        rs = self.returns.stack()
        ds = rs.index.get_level_values(0)
        return rs.groupby([ds, ks]).mean().unstack()

    def holdings(self, n, i, k):
        r = self.greturns(n)
        h = r.iloc[i::k]
        h.values.fill(1 / n)
        return h

    def floated(self, n, i, k):
        r = self.greturns(n)
        h = self.holdings(n, i, k)
        return util.floated(h, r, normalize=True)

    def floated_aggregate_returns(self, n, i, k):
        r = self.greturns(n)
        h = self.floated(n, i, k)
        return r.mul(h).sum(1)





