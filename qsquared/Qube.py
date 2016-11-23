import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .util import simutil
from .util import util


class Qube(object):
    def __init__(self, signals, returns, bench=None):
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
        # clean data
        keep = signals.fillna(0).eq(0).sum(1).le(1)
        signals = signals.loc[keep]
        returns = returns.loc[signals.index]

        self.signals = signals
        self.returns = returns
        if bench is None:
            bench = returns.mean(1).rename('Benchmark')
        bench = bench.loc[signals.index]
        self.bench = bench

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

    def horizon_returns(self, n, k, dropna=True):
        """generate the floated panel of returns and take the mean return
        across the `k` iterations of the `k` period horizon.

        :param n: number of quantiles (5 for quintiles, 10 for deciles)
        :type n: int

        :param k: periods between rebalances (horizon)
        :type k: int

        :param dropna: boolean flag whether to drop na rows (default True)
        :type dropna: bool

        :rtype: pd.DataFrame
        """
        pnl = self.floated_panel_returns(n, k)
        df = pnl.mean('items', skipna=False)
        if dropna:
            df = df.dropna()
        return df

    def horizon_stats(self, n, k):
        """calculate stats for n groups, rebalancing every k periods

        :param n: number of groups
        :type n: int

        :param k: number of periods between rebalances (horizon)
        :type k: int

        :type: pd.DataFrame
        """
        hr = self.horizon_returns(n, k)
        df = pd.DataFrame([], columns=hr.columns)
        df = df.append(simutil.annualized_returns(hr))
        df = df.append(simutil.annualized_risk(hr))
        df = df.append(simutil.sharpe_ratio(hr))
        df = df.append(simutil.annualized_active_returns(hr, self.bench))
        df = df.append(simutil.annualized_active_risk(hr, self.bench))
        df = df.append(simutil.information_ratio(hr, self.bench))
        return df

    def horizons_stats(self, n):
        """iterate over pre-specified set of horizons and calculate
        horizon stats with n groups

        :param n: number of groups
        :type n: int

        :rtype: pd.DataFrame
        """
        k = (1, 2, 3, 6, 12)
        hdf = pd.concat(
            [self.horizon_stats(n, i) for i in k],
            keys=k).rename_axis(['Horizon', 'Stat'])
        return hdf

    def horizons_plot(self, n, figsize=(10, 5), suptitle=None):
        """plot signal frequencies and cumulative plots for different horizons

        :param n: number of groups
        :type n: int

        :param figsize: tuple specifying figure size width by height
        :type figsize: (int, int)

        :param suptitle: title of figure
        :type suptitle: str

        :rtype: plt.Figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(suptitle, fontsize=12)
        self.sig_frequency(axes[0, 0])
        for i, k in enumerate((1, 2, 3, 6, 12), 1):
            r, c = i // 3, i % 3
            self.cumplot(n, k, ax=axes[r, c])
        fig.autofmt_xdate()
        return fig

    def cumplot(self, n, k, ax=None, title=True):
        """plot cumulative returns for n groups with horizon k

        :param n: number of groups
        :type n: int

        :param k: periods between rebalancing (horizon)
        :type k: int

        :param ax: option axes object to plot onto
        :type ax: matplotlib.axes._subplots.AxesSubplot

        :param title: title of axes object
        :type title: str

        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        hr = self.horizon_returns(n, k, dropna=False)
        title_fmt = 'Cum Return - Horizon = {}'.format
        if title:
            title = title_fmt(k)
        ax = util.cumplot(
            hr, ax=ax, colormap='jet', title=title
        )
        ax.title.set(size=10)
        ax.get_xaxis().label.set_visible(False)
        return ax

    def sig_frequency(self, ax=None):
        """plot bar chart showing the frequency that each security spends
        at each rank

        :param ax: option axes object to plot onto
        :type ax: matplotlib.axes._subplots.AxesSubplot

        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        ax = self.signals.rank(1).apply(
            pd.value_counts,
            normalize=True
        ).T.plot.barh(
            ax=ax,
            stacked=True,
            colormap='jet',
            legend=False,
            xlim=[0, 1],
            title='Rank Frequency',
            width=.95,
        )
        ax.get_xaxis().set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=6)
        return ax

    def max_dd_plot(self, n, k):
        """plot maximum draw downs for n groups over horizon k


        :param n: number of groups
        :type n: int

        :param k: number of periods between rebalances (horizon)
        :type k: int

        :rtype: matplotlib.axes._subplots.AxesSubplot
        """
        hr = self.horizon_returns(n, k, False)
        return simutil.max_dd_plot(hr, self.bench)