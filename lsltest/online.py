"""Causal synchronization on the receiver clock; never extrapolates samples."""
import numpy as np


class OnlineAligner:
    def __init__(self, sensors, rate, lag=.25):
        self.rate, self.lag, self.next_time = rate, lag, None
        self.rates = {s['name']: s['rate'] for s in sensors}
        self.data = {s['name']: (np.empty(0), np.empty((0, s['channels']))) for s in sensors}

    def add(self, name, timestamps, values):
        old_t, old_y = self.data[name]
        if not len(timestamps):
            return
        # A backward correction supersedes buffered data at the new timestamps.
        cut = np.searchsorted(old_t, timestamps[0], side='left')
        old_t, old_y = old_t[:cut], old_y[:cut]
        t = np.concatenate((old_t, timestamps))
        y = np.concatenate((old_y, values))
        # A clock correction step can produce nonmonotonic timestamps. Keep the
        # newest strictly increasing sequence; do not interpolate across a reversal.
        keep = np.r_[True, t[1:] > np.maximum.accumulate(t[:-1])]
        self.data[name] = t[keep], y[keep]

    def pull(self, now):
        if any(len(t) < 2 for t, _ in self.data.values()):
            return None
        if self.next_time is None:
            self.next_time = np.ceil(max(t[0] for t, _ in self.data.values())*self.rate)/self.rate
        end = min(now-self.lag, *(t[-1] for t, _ in self.data.values()))
        n = int(np.floor((end-self.next_time)*self.rate))+1
        if n <= 0:
            return None
        grid = self.next_time + np.arange(n)/self.rate
        result = {}
        for name, (t, y) in self.data.items():
            values = np.column_stack([np.interp(grid, t, y[:, c], left=np.nan, right=np.nan)
                                      for c in range(y.shape[1])])
            right = np.searchsorted(t, grid, side='right').clip(1, len(t)-1)
            gap = (t[right]-t[right-1]) > 1.5/self.rates[name]
            exact = np.isclose(grid, t[right-1], rtol=0, atol=1e-9) | np.isclose(grid, t[right], rtol=0, atol=1e-9)
            gap &= ~exact
            values[gap] = np.nan
            result[name] = values
            cut = max(0, np.searchsorted(t, grid[-1])-1)
            self.data[name] = t[cut:], y[cut:]
        self.next_time += n/self.rate
        return grid, result
